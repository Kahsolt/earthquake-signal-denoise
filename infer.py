#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/03/06

import os
from copy import deepcopy
from zipfile import ZipFile, ZIP_DEFLATED
from argparse import ArgumentParser

from train_denoiser import LitModel as DenoiserLitModel
from train_envolope import LitModel as EnvolopeLitModel
from model import DenoiseModel, EnvolopeModel, EnvolopeExtractor
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


@torch.inference_mode()
@timer
def infer(args):
  ''' Model & Ckpt '''
  model_D: DenoiseModel = DenoiserLitModel.load_from_checkpoint(args.load_D, model=DenoiseModel()).model.to(device)
  model_E: EnvolopeModel = EnvolopeLitModel.load_from_checkpoint(args.load_E, model=EnvolopeModel()).model.to(device)
  model_D.remove_weight_norm()
  model_E.remove_weight_norm()
  envolope_extractor = EnvolopeExtractor().to(device)

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
    M_segs = model_D(X_segs, E_segs)
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
      # norm domain: noisy signal -> (stft) -> sliced denoise -> (istft) -> denoised signal
      x_norm = wav_norm(x)
      D = L.stft(x_norm[:-1], **FFT_PARAMS)
      M, P = L.spectrum.magphase(D, power=1)
      logM = np.clip(np.log(M + 1e-15), a_min=EPS, a_max=None)  # [F=65, L=750]

      # denoise
      sel = 0
      if sel == 0:
        logM_low, logM_high = logM[:-1], np.ones_like(logM[-1:]) * EPS    # suppress hifreq
        logM_low_denoised = denoise(logM_low)   # [F=64, L=750]
        logM_denoised = np.concatenate([logM_low_denoised, logM_high], axis=0)  # [F=65, L=750]
      elif sel == 1:
        logM_denoised = deepcopy(logM)
        logM_denoised[:2, :] = EPS
      M_denoised = np.exp(logM_denoised)

      # inv_wav
      sel = 1
      if sel == 0:
        x_norm_denoised: ndarray = L.griffinlim(M_denoised, **FFT_PARAMS, length=len(x_norm)-1)
        x_norm_denoised = np.pad(x_norm_denoised, (0, 1), mode='reflect')
      elif sel == 1:
        x_norm_denoised: ndarray = griffinlim_hijack(M_denoised, P, **FFT_PARAMS, length=len(x_norm))
      elif sel == 2:
        x_norm_denoised: ndarray = L.istft(M_denoised * P, **FFT_PARAMS, length=len(x_norm))

      if args.debug and 'cmp spec denoise':
        # https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html
        plt.clf()
        plt.subplot(221) ; plt.title('1. x_norm')          ; plt.plot(x_norm)
        plt.subplot(222) ; plt.title('4. denoised x_norm') ; plt.plot(x_norm_denoised)
        plt.subplot(223) ; plt.title('2. x_logM')          ; LD.specshow(logM,          sr=SR, hop_length=HOP_LEN, win_length=WIN_LEN, cmap='plasma')
        plt.subplot(224) ; plt.title('3. denoised x_logM') ; LD.specshow(logM_denoised, sr=SR, hop_length=HOP_LEN, win_length=WIN_LEN, cmap='plasma')
        plt.suptitle('infer denoise model')
        plt.show()

      if not 'shift envolope':
        # log1p domain: noisy signal -> denoised envolope
        x_lop1p: ndarray = wav_log1p(x)
        z_lop1p_upper, z_lop1p_lower = model_E(torch.from_numpy(x_lop1p).to(device).unsqueeze(0).unsqueeze(0))[0]
        z_upper, z_lower = torch.expm1(z_lop1p_upper), -torch.expm1(-z_lop1p_lower)

        if args.debug and 'cmp envolope mapping':
          plt.clf()
          plt.subplot(221) ; plt.title('1. x')                ; plt.plot(x)
          plt.subplot(222) ; plt.title('2. x_log1p')          ; plt.plot(x_lop1p)
          plt.subplot(223) ; plt.title('4. target env')       ; plt.plot(z_upper      .cpu().numpy()) ; plt.plot(z_lower      .cpu().numpy())
          plt.subplot(224) ; plt.title('3. target env_log1p') ; plt.plot(z_lop1p_upper.cpu().numpy()) ; plt.plot(z_lop1p_lower.cpu().numpy())
          plt.suptitle('infer envolope model')
          plt.show()

        # shift predicted `x_norm_denoised` to predicted envolope of `[z_upper, z_lower]`
        y_upper, y_lower = envolope_extractor(torch.from_numpy(x_norm_denoised).to(device).unsqueeze(0).unsqueeze(0))[0]
        upper_scale_shift = torch.mean(z_upper / y_upper)
        lower_scale_shift = torch.mean(z_lower / y_lower)
        scale_shift: float = ((upper_scale_shift + lower_scale_shift) / 2).item()
        x_denoised_shifted: ndarray = x_norm_denoised * scale_shift

        if args.debug and 'cmp envolope shift':
          plt.clf()
          plt.subplot(221) ; plt.title('0. x')       ; plt.plot(x)
          plt.subplot(222) ; plt.title('2. y_final') ; plt.plot(x_denoised_shifted) ; plt.plot(z_upper.cpu().numpy()) ; plt.plot(z_lower.cpu().numpy())
          plt.subplot(224) ; plt.title('1. y')       ; plt.plot(x_norm_denoised)    ; plt.plot(y_upper.cpu().numpy()) ; plt.plot(y_lower.cpu().numpy())
          plt.suptitle('envolope shift')
          plt.show()

      else:
        x_denoised_shifted = x_norm_denoised
  
      # truncate length
      if name in lendict:
        x_denoised_shifted = x_denoised_shifted[:lendict[name]]

      # save signal result
      #print(f'>> writing to {name}_C.txt')
      with zf.open(f'{name}_C.txt', 'w') as fh:
        np.savetxt(fh, x_denoised_shifted, fmt='%.6f')

  print(f'>> save to {fp_submit}')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--load_D', required=True, type=Path, help='pretrained denoiser ckpt')
  parser.add_argument('--load_E', required=True, type=Path, help='pretrained envolope ckpt')
  parser.add_argument('--debug', action='store_true', help='plot debug results')
  args = parser.parse_args()

  args.debug = args.debug or os.getenv('DEBUG_INFER', False)

  infer(args)
