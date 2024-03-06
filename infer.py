#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/03/06

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


@torch.inference_mode()
def infer(args):
  ''' Model & Ckpt '''
  model_D = DenoiserLitModel.load_from_checkpoint(args.load_D, model=DenoiseModel()).model.to(device)
  model_E = EnvolopeLitModel.load_from_checkpoint(args.load_E, model=EnvolopeModel()).model.to(device)
  envolope_extractor = EnvolopeExtractor().to(device)

  def denoise(M:ndarray) -> ndarray:
    ''' sliced inference to match model I/O size '''
    X = torch.from_numpy(M)      .to(device)   # [F, L]
    E = torch.arange(M.shape[-1]).to(device)   # [L]

    M_d = np.zeros_like(M, dtype=np.float32)   # [F, L]
    C_d = np.zeros_like(M, dtype=np.uint8)     # [F, L]
    sp = 0
    for _ in range(N_INFER):
      seg_slicer = slice(sp, sp + N_FRAME_INFER)
      X_seg = X[:, seg_slicer]
      E_seg = E[   seg_slicer]
      M_d[:, seg_slicer] += model_D(X_seg.unsqueeze(0), E_seg.unsqueeze(0))[0].cpu().numpy()
      C_d[:, seg_slicer] += 1
      sp += N_FRAME_INFER - N_OVERLAP
    return M_d / C_d

  ''' Data & Infer '''
  X = get_data_test()
  namelist, lendict = get_data_test_meta()

  fp_submit = SUBMIT_PATH.parent / (SUBMIT_PATH.stem + f'_{now_ts()}' + SUBMIT_PATH.suffix)
  with ZipFile(fp_submit, 'w', compression=ZIP_DEFLATED, compresslevel=9) as zf:
    for name, x in tqdm(zip(namelist, X)):
      # norm domain: noisy signal -> (stft) -> sliced denoise -> (istft) -> denoised signal
      x_norm = wav_norm(x)
      D = L.stft(x_norm[:-1], **FFT_PARAMS)
      M, P = L.spectrum.magphase(D, power=1)
      logM = np.clip(np.log(M + 1e-15), a_min=EPS, a_max=None)  # [F=65, L=750]
      logM_low, logM_high = logM[:-1], logM[-1:]
      logM_low_denoised = denoise(logM_low)   # [F=64, L=750]
      logM_denoised = np.concatenate([logM_low_denoised, logM_high], axis=0)  # [F=65, L=750]
      D_denoised = np.exp(logM_denoised) * P
      x_norm_denoised = L.istft(D_denoised, **FFT_PARAMS, length=len(x_norm))

      if args.debug and 'cmp spec denoise':
        # https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html
        plt.clf()
        plt.subplot(221) ; plt.title('1. x_norm')          ; plt.plot(x_norm)
        plt.subplot(222) ; plt.title('4. denoised x_norm') ; plt.plot(x_norm_denoised)
        plt.subplot(223) ; plt.title('2. x_logM')          ; LD.specshow(logM,          sr=SR, hop_length=HOP_LEN, win_length=WIN_LEN, cmap='plasma')
        plt.subplot(224) ; plt.title('3. denoised x_logM') ; LD.specshow(logM_denoised, sr=SR, hop_length=HOP_LEN, win_length=WIN_LEN, cmap='plasma')
        plt.suptitle('infer denoise model')
        plt.show()

      # log1p domain: noisy signal -> denoised envolope
      x_lop1p = wav_log1p(x)
      envolope = model_E(torch.from_numpy(x_lop1p).unsqueeze(0).unsqueeze(0).to(device))[0].cpu().numpy()
      z_lop1p_upper, z_lop1p_lower = envolope
      z_upper, z_lower = np.expm1(z_lop1p_upper), -np.expm1(-z_lop1p_lower)

      if args.debug and 'cmp envolope mapping':
        plt.clf()
        plt.subplot(221) ; plt.title('1. x')                ; plt.plot(x)
        plt.subplot(222) ; plt.title('2. x_log1p')          ; plt.plot(x_lop1p)
        plt.subplot(223) ; plt.title('4. target env')       ; plt.plot(z_upper)       ; plt.plot(z_lower)
        plt.subplot(224) ; plt.title('3. target env_log1p') ; plt.plot(z_lop1p_upper) ; plt.plot(z_lop1p_lower)
        plt.suptitle('infer envolope model')
        plt.show()

      # shift predicted `x_norm_denoised` to predicted envolope of `[z_upper, z_lower]`
      y_upper, y_lower = envolope_extractor(torch.from_numpy(x_norm_denoised).unsqueeze(0).unsqueeze(0).to(device))[0].cpu().numpy()
      upper_scale_shift = (z_upper / y_upper).mean()
      lower_scale_shift = (z_lower / y_lower).mean()
      scale_shift = (upper_scale_shift + lower_scale_shift) / 2
      x_denoised_shifted = x_norm_denoised * scale_shift

      if args.debug and 'cmp envolope shift':
        plt.clf()
        plt.subplot(221) ; plt.title('0. x')       ; plt.plot(x)
        plt.subplot(222) ; plt.title('2. y_final') ; plt.plot(x_denoised_shifted) ; plt.plot(z_upper) ; plt.plot(z_lower)
        plt.subplot(224) ; plt.title('1. y')       ; plt.plot(x_norm_denoised)    ; plt.plot(y_upper) ; plt.plot(y_lower)
        plt.suptitle('envolope shift')
        plt.show()

      # truncate length
      if name in lendict:
        x_denoised_shifted = x_denoised_shifted[:lendict[name]]

      # save signal result
      print(f'>> writing to {name}_C.txt')
      with zf.open(f'{name}_C.txt', 'w') as fh:
        np.savetxt(fh, x_denoised_shifted)

  print(f'>> save to {fp_submit}')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--load_D', required=True, type=Path, help='pretrained denoiser ckpt')
  parser.add_argument('--load_E', required=True, type=Path, help='pretrained envolope ckpt')
  parser.add_argument('--debug', action='store_true', help='plot debug results')
  args = parser.parse_args()

  infer(args)
