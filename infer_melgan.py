#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/03/14

import os
import yaml
from zipfile import ZipFile, ZIP_DEFLATED
from argparse import ArgumentParser

from utils import *
from models import Generator, Audio2Spec

device = 'cpu'


@torch.inference_mode()
@timer
def infer(args):
  ''' Model & Ckpt '''
  logdir = Path(args.load_M)
  with open(logdir / 'args.yml', 'r', encoding='utf-8') as fh:
    hp = yaml.unsafe_load(fh)
  modelM = Generator(hp.n_mel_channels, hp.ngf, hp.n_residual_layers).to(device)
  modelM.load_state_dict(torch.load(logdir / 'best_netG.pt'))
  fft = Audio2Spec(N_FFT, HOP_LEN, WIN_LEN, SR, device)

  ''' Data & Infer '''
  x_test = get_data_test()
  namelist, lendict = get_data_test_meta()

  fp_submit = SUBMIT_PATH.parent / (SUBMIT_PATH.stem + f'_{now_ts()}' + SUBMIT_PATH.suffix)
  with ZipFile(fp_submit, 'w', compression=ZIP_DEFLATED, compresslevel=9) as zf:
    for name, x in tqdm(zip(namelist, x_test), total=len(namelist)):
      # norm domain: noisy signal -> noisy stft -> denoised signal
      x = wav_norm(x)
      x_noisy = torch.from_numpy(x).unsqueeze(dim=0).unsqueeze(dim=0)
      logM_noisy = fft(x_noisy)
      x_denoised = modelM(logM_noisy)
      logM_denoised = fft(x_denoised)
      x_norm_denoised = x_denoised.squeeze().cpu().numpy()

      if args.debug and 'cmp spec denoise':
        # https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html
        plt.clf()
        plt.subplot(221) ; plt.title('1. x_norm')          ; plt.plot(x)
        plt.subplot(222) ; plt.title('4. denoised x_norm') ; plt.plot(x_norm_denoised)
        plt.subplot(223) ; plt.title('2. x_logM')          ; LD.specshow(logM_noisy.squeeze().cpu().numpy(),    sr=SR, hop_length=HOP_LEN, win_length=WIN_LEN, cmap='plasma')
        plt.subplot(224) ; plt.title('3. denoised x_logM') ; LD.specshow(logM_denoised.squeeze().cpu().numpy(), sr=SR, hop_length=HOP_LEN, win_length=WIN_LEN, cmap='plasma')
        plt.suptitle('infer denoise model')
        plt.show()

      # shift envolope
      x_denoised_shifted = x_norm_denoised

      # truncate length
      if name in lendict:
        x_denoised_shifted = x_denoised_shifted[:lendict[name]]

      # save signal result
      with zf.open(f'{name}_C.txt', 'w') as fh:
        np.savetxt(fh, x_denoised_shifted, fmt='%.6f')

  print(f'>> save to {fp_submit}')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--load_M', default='lightning_logs/melgan', type=Path, help='pretrained melgan ckpt')
  parser.add_argument('--debug', action='store_true', help='plot debug results')
  args = parser.parse_args()

  args.debug = args.debug or os.getenv('DEBUG_INFER', False)

  infer(args)
