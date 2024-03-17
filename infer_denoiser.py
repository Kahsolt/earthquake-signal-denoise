#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/03/06

import os
from zipfile import ZipFile, ZIP_DEFLATED
from argparse import ArgumentParser

from train_denoiser import LitModel as DenoiserLitModel
from models import DenoiseModel
from utils import *


if 'configs':
  N_FRAME_INFER = N_FRAME * 2
  MAXL = NLEN // HOP_LEN   # 750
  N_INFER = int(np.ceil(MAXL / N_FRAME_INFER))
  N_OVERLAP: float = (N_INFER * N_FRAME_INFER - MAXL) / (N_INFER - 1)
  assert N_OVERLAP.is_integer()
  N_OVERLAP = int(N_OVERLAP)


def griffinlim_hijack(
  S:ndarray,
  angles:ndarray,
  n_iter=32,
  hop_length=None,
  win_length=None,
  n_fft=None,
  window="hann",
  center=True,
  dtype=None,
  length=None,
  pad_mode="constant",
  momentum=0.9,
):
  from librosa.core.spectrum import griffinlim
  from librosa import util

  # using complex64 will keep the result to minimal necessary precision
  eps = util.tiny(angles)

  # And initialize the previous iterate to 0
  rebuilt = 0.0
  for _ in range(n_iter):
    # Store the previous iterate
    tprev = rebuilt
    # Invert with our current estimate of the phases
    inverse = L.istft(
      S * angles,
      hop_length=hop_length,
      win_length=win_length,
      n_fft=n_fft,
      window=window,
      center=center,
      dtype=dtype,
      length=length,
    )
    # Rebuild the spectrogram
    rebuilt = L.stft(
      inverse[:-1],
      n_fft=n_fft,
      hop_length=hop_length,
      win_length=win_length,
      window=window,
      center=center,
      pad_mode=pad_mode,
    )
    # Update our phase estimates
    angles[:] = rebuilt - (momentum / (1 + momentum)) * tprev
    angles[:] /= np.abs(angles) + eps

  # Return the final phase estimates
  return L.istft(
    S * angles,
    hop_length=hop_length,
    win_length=win_length,
    n_fft=n_fft,
    window=window,
    center=center,
    dtype=dtype,
    length=length,
  )


@torch.inference_mode
@timer
def infer(args):
  ''' Model & Ckpt '''
  model: DenoiseModel = DenoiserLitModel.load_from_checkpoint(args.load, model=DenoiseModel()).model.eval().to(device)
  model.remove_weight_norm()

  def denoise(M:ndarray) -> ndarray:
    ''' sliced inference to match model I/O size '''
    X = torch.from_numpy(M)      .to(device)   # [F, L]
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
    return (M_d / C_d).cpu().numpy()

  ''' Data & Infer '''
  X = get_data_test()
  namelist, lendict = get_data_test_meta()

  fp_submit = SUBMIT_PATH.parent / (SUBMIT_PATH.stem + f'_{now_ts()}' + SUBMIT_PATH.suffix)
  with ZipFile(fp_submit, 'w', compression=ZIP_DEFLATED, compresslevel=9) as zf:
    for name, x in tqdm(zip(namelist, X), total=len(namelist)):
      # noisy signal -> (stft) -> sliced denoise -> (istft) -> denoised signal
      D = L.stft(x[:-1], **FFT_PARAMS)
      M, P = L.spectrum.magphase(D, power=1)
      logM = np.clip(np.log10(M + 1e-15), a_min=EPS, a_max=None)  # [F=65, L=750]

      # denoise
      logM_low, logM_high = logM[:-1], np.ones_like(logM[-1:]) * EPS    # suppress hifreq
      logM_low_denoised = denoise(logM_low)   # [F=64, L=750]
      logM_denoised = np.concatenate([logM_low_denoised, logM_high], axis=0)  # [F=65, L=750]
      M_denoised = 10 ** logM_denoised

      # inv_wav
      sel = 1
      if sel == 0:
        x_denoised: ndarray = L.griffinlim(M_denoised, **FFT_PARAMS, length=len(x)-1)
        x_denoised = np.pad(x_denoised, (0, 1), mode='reflect')
      elif sel == 1:
        x_denoised: ndarray = griffinlim_hijack(M_denoised, P, **FFT_PARAMS, length=len(x))
      elif sel == 2:
        x_denoised: ndarray = L.istft(M_denoised * P, **FFT_PARAMS, length=len(x))

      if args.debug and 'cmp denoise':
        # https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html
        plt.clf()
        plt.subplot(221) ; plt.title('1. x')               ; plt.plot(x)
        plt.subplot(222) ; plt.title('4. denoised x')      ; plt.plot(x_denoised)
        plt.subplot(223) ; plt.title('2. x_logM')          ; LD.specshow(logM,          sr=SR, hop_length=HOP_LEN, win_length=WIN_LEN, cmap='plasma')
        plt.subplot(224) ; plt.title('3. denoised x_logM') ; LD.specshow(logM_denoised, sr=SR, hop_length=HOP_LEN, win_length=WIN_LEN, cmap='plasma')
        plt.suptitle('infer denoise model')
        plt.show()

      # truncate length
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
