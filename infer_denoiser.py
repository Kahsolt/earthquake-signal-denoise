#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/03/06

import os
from zipfile import ZipFile, ZIP_DEFLATED
from argparse import ArgumentParser

import torchaudio.functional as TAF
from train_denoiser import LitModel as DenoiserLitModel
from models import DenoiseModel, Audio2Spec
from utils import *


if 'configs':
  N_FRAME_INFER = N_FRAME * 2
  MAXL = NLEN // HOP_LEN   # 750
  N_INFER = int(np.ceil(MAXL / N_FRAME_INFER))
  N_OVERLAP: float = (N_INFER * N_FRAME_INFER - MAXL) / (N_INFER - 1)
  assert N_OVERLAP.is_integer()
  N_OVERLAP = int(N_OVERLAP)


@torch.inference_mode
@timer
def infer(args):
  ''' Model & Ckpt '''
  model: DenoiseModel = DenoiserLitModel.load_from_checkpoint(args.load, model=DenoiseModel()).model.eval().to(device)
  model.remove_weight_norm()
  fft = Audio2Spec(N_FFT, HOP_LEN, WIN_LEN, SR).to(device)

  def denoise(M:Tensor) -> Tensor:
    ''' sliced inference to match model I/O size '''
    X = M                                      # [F, L]
    E = torch.arange(M.shape[-1]).to(device)   # [L]

    # slice & pack
    slicers = []
    X_segs, E_segs = [], []
    sp = 0
    for _ in range(N_INFER):
      seg_slicer = slice(sp, sp + N_FRAME_INFER)
      slicers.append(seg_slicer)
      X_segs.append(X[:, seg_slicer])
      E_segs.append(E[   seg_slicer])
      sp += N_FRAME_INFER - N_OVERLAP
    X_segs = torch.stack(X_segs, dim=0)
    E_segs = torch.stack(E_segs, dim=0)
    # forward
    M_segs = model(X_segs, E_segs)
    # unpack & unslice
    M_d = torch.zeros_like(X, dtype=torch.float32)   # [F, L]
    C_d = torch.zeros_like(X, dtype=torch.uint8)     # [F, L]
    for seg_slicer, M_seg in zip(slicers, M_segs):
      M_d[:, seg_slicer] += M_seg
      C_d[:, seg_slicer] += 1
    return M_d / C_d

  ''' Data & Infer '''
  X = get_data_test()
  namelist, lendict = get_data_test_meta()

  fp_submit = SUBMIT_PATH.parent / (SUBMIT_PATH.stem + f'_{now_ts()}' + SUBMIT_PATH.suffix)
  with ZipFile(fp_submit, 'w', compression=ZIP_DEFLATED, compresslevel=9) as zf:
    for name, x in tqdm(zip(namelist, X), total=len(namelist)):
      # noisy signal -> (stft) -> sliced denoise -> (istft) -> denoised signal
      x = torch.from_numpy(x).unsqueeze_(0).unsqueeze_(0).to(device)
      logS, P = fft(x, ret_mag=True, ret_phase=True)  # [F=65, L=750]

      # denoise
      logS_low, logS_high = logS[:-1], np.ones_like(logS[-1:]) * EPS    # suppress hifreq
      logS_low_denoised = denoise(logS_low)   # [F=64, L=750]
      logS_denoised = torch.cat([logS_low_denoised, logS_high], dim=0)  # [F=65, L=750]
      S_denoised = 10 ** logS_denoised

      # inv_wav
      sel = 1
      if sel == 0:
        x_denoised: Tensor = TAF.griffinlim(S_denoised, fft.window, **FFT_PARAMS, length=len(x))
      elif sel == 1:
        x_denoised: Tensor = griffinlim_hijack(S_denoised, P, **FFT_PARAMS, length=len(x))
      elif sel == 2:
        x_denoised: Tensor = torch.istft(S_denoised * P, **FFT_PARAMS, window=fft.window, length=len(x))

      if args.debug and 'cmp denoise':
        # https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html
        plt.clf()
        plt.subplot(221) ; plt.title('1. x')               ; plt.plot(x.cpu().numpy())
        plt.subplot(222) ; plt.title('4. denoised x')      ; plt.plot(x_denoised.cpu().numpy())
        plt.subplot(223) ; plt.title('2. x_logM')          ; LD.specshow(logS.cpu().numpy(),          sr=SR, hop_length=HOP_LEN, win_length=WIN_LEN, cmap='plasma')
        plt.subplot(224) ; plt.title('3. denoised x_logM') ; LD.specshow(logS_denoised.cpu().numpy(), sr=SR, hop_length=HOP_LEN, win_length=WIN_LEN, cmap='plasma')
        plt.suptitle('infer denoise model')
        plt.show()

      # truncate length
      x_denoised = x_denoised.cpu().numpy()
      if name in lendict:
        x_denoised = x_denoised[:lendict[name]]

      # save signal result
      #print(f'>> writing to {name}_C.txt')
      with zf.open(f'{name}_C.txt', 'w') as fh:
        np.savetxt(fh, x_denoised, fmt='%.6f')

  print(f'>> save to {fp_submit}')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--load', required=True, type=Path, help='pretrained denoiser ckpt')
  parser.add_argument('--debug', action='store_true', help='plot debug results')
  args = parser.parse_args()

  args.debug = args.debug or os.getenv('DEBUG_INFER', False)

  infer(args)
