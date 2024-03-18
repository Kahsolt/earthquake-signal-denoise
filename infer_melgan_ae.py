#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/03/14

import os
import yaml
from zipfile import ZipFile, ZIP_DEFLATED
from argparse import ArgumentParser

from models import GeneratorAE, Audio2Spec
from utils import *


@torch.inference_mode
@timer
def infer(args):
  ''' Model & Ckpt '''
  logdir = Path(args.load)
  with open(logdir / 'args.yml', 'r', encoding='utf-8') as fh:
    hp = yaml.unsafe_load(fh)
  model = GeneratorAE(hp.n_mel_channels, hp.ngf, hp.n_residual_layers).to(device)
  model.load_state_dict(torch.load(logdir / 'best_netG.pt'))
  fft = Audio2Spec(N_FFT, HOP_LEN, WIN_LEN, device)

  ''' Data & Infer '''
  x_test = get_data_test()
  namelist, lendict = get_data_test_meta()

  fp_submit = SUBMIT_PATH.parent / (SUBMIT_PATH.stem + f'_{now_ts()}' + SUBMIT_PATH.suffix)
  with ZipFile(fp_submit, 'w', compression=ZIP_DEFLATED, compresslevel=9) as zf:
    for name, x in tqdm(zip(namelist, x_test), total=len(namelist)):
      # noisy signal -> noisy stft -> denoised stft -> inverted signal
      x_noisy = torch.from_numpy(x).unsqueeze_(dim=0).unsqueeze_(dim=0).to(device)
      logS_noisy, P_noisy = fft(x_noisy, ret_mag=True, ret_phase=True)
      logS_denoised = model(logS_noisy)
      S_denoised = 10 ** logS_denoised
      x_denoised = griffinlim_hijack(S_denoised, P_noisy, n_iter=32, **FFT_PARAMS, length=len(x)).squeeze_().cpu().numpy()

      if args.debug and 'cmp spec denoise':
        import librosa.display as LD
        # https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html
        plt.clf()
        plt.subplot(221) ; plt.title('1. x')               ; plt.plot(x)
        plt.subplot(222) ; plt.title('4. denoised x')      ; plt.plot(x_denoised)
        plt.subplot(223) ; plt.title('2. x_logS')          ; LD.specshow(logS_noisy.squeeze().cpu().numpy(),    sr=SR, hop_length=HOP_LEN, win_length=WIN_LEN, cmap='plasma')
        plt.subplot(224) ; plt.title('3. denoised x_logS') ; LD.specshow(logS_denoised.squeeze().cpu().numpy(), sr=SR, hop_length=HOP_LEN, win_length=WIN_LEN, cmap='plasma')
        plt.suptitle('infer melgan-ae model')
        plt.show()

      # truncate length
      if name in lendict:
        x_denoised = x_denoised[:lendict[name]]

      # save signal result
      with zf.open(f'{name}_C.txt', 'w') as fh:
        np.savetxt(fh, x_denoised, fmt='%.6f')

  print(f'>> save to {fp_submit}')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--load', default='lightning_logs/melgan_ae', type=Path, help='pretrained melgan-ae ckpt')
  parser.add_argument('--debug', action='store_true', help='plot debug results')
  args = parser.parse_args()

  args.debug = args.debug or os.getenv('DEBUG_INFER', False)

  infer(args)
