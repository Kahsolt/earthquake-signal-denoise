#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/03/15

from infer_melgan import *
from infer import griffinlim_hijack


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
  X, Y = get_data_train()
  to_np = lambda x: x.squeeze().cpu().numpy()

  score_list = []
  for x, y in tqdm(zip(X, Y), total=len(X)):
    # norm domain: noisy signal -> noisy stft -> denoised signal
    x = wav_norm(x)
    y = wav_norm(y)
    x_spec = fft(torch.from_numpy(x).unsqueeze(dim=0).unsqueeze(dim=0))
    y_spec = fft(torch.from_numpy(y).unsqueeze(dim=0).unsqueeze(dim=0))
    x_gen = modelM(x_spec)
    x_gen_spec = fft(x_gen)
    x_gen_S = to_np(10**x_gen_spec)
    if not 'use original gl':
      x_gl = L.griffinlim(x_gen_S, n_iter=32, **FFT_PARAMS, length=len(x)-1)
      x_gl = np.pad(x_gl, (0, 1), mode='reflect')
    else:
      P = to_np(fft(torch.from_numpy(x).unsqueeze(dim=0).unsqueeze(dim=0), ret_phase=True))
      x_gl = griffinlim_hijack(x_gen_S, P, n_iter=32, **FFT_PARAMS, length=len(x))
    x_gl_spec = fft(torch.from_numpy(x_gl).unsqueeze(dim=0).unsqueeze(dim=0))

    sc_gen = get_score(to_np(x_gen), y)
    sc_ref = get_score(x, y)
    sc_gl  = get_score(x_gl, y)
    score_list.append([sc_gen, sc_ref, sc_gl])

    if not 'cmp spec denoise':
      # https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html
      plt.clf()
      plt.subplot(421) ; plt.title('x')                     ; plt.plot(x)
      plt.subplot(423) ; plt.title(f'x_gen ({sc_gen:.3f})') ; plt.plot(to_np(x_gen))
      plt.subplot(425) ; plt.title(f'y ({sc_ref:.3f})')     ; plt.plot(y)
      plt.subplot(427) ; plt.title(f'x_gl ({sc_gl:.3f})')   ; plt.plot(x_gl)
      plt.subplot(422) ; plt.title('x_spec')     ; LD.specshow(to_np(x_spec)    , sr=SR, hop_length=HOP_LEN, win_length=WIN_LEN, cmap='plasma')
      plt.subplot(424) ; plt.title('x_gen_spec') ; LD.specshow(to_np(x_gen_spec), sr=SR, hop_length=HOP_LEN, win_length=WIN_LEN, cmap='plasma')
      plt.subplot(426) ; plt.title('y_spec')     ; LD.specshow(to_np(y_spec)    , sr=SR, hop_length=HOP_LEN, win_length=WIN_LEN, cmap='plasma')
      plt.subplot(428) ; plt.title('x_gl_spec')  ; LD.specshow(to_np(x_gl_spec) , sr=SR, hop_length=HOP_LEN, win_length=WIN_LEN, cmap='plasma')
      plt.suptitle('infer melgan')
      plt.show()

  #                 sc_gen       sc_ref     sc_gl
  # gl_original: -0.09826729 -0.22446253 -0.09697114
  # gl_hijack:   -0.09826729 -0.22446253 -0.08164768
  score_list = np.asarray(score_list)
  print(score_list.mean(axis=0))


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--load_M', default='lightning_logs/melgan', type=Path, help='pretrained melgan ckpt')
  args = parser.parse_args()

  infer(args)
