#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/03/15

from infer import *

# 测试评分函数的性质


@timer
def run():
  X, Y = get_data_train()
  score_list = []
  for x, y in tqdm(zip(X, Y), total=len(X)):
    if '*10':
      x_m10 = x * 10
    if '/10':
      x_d10 = x / 10
    if 'norm':
      x_norm = wav_norm(x)
    if 'isft':
      D = L.stft(x[:-1], **FFT_PARAMS)
      M, P = L.spectrum.magphase(D, power=1)
      x_isft = L.istft(M * P, **FFT_PARAMS, length=len(x))
    if 'gl':
      x_gl = L.griffinlim(M, n_iter=32, **FFT_PARAMS, length=len(x)-1)
      x_gl = np.pad(x_gl, (0, 1), mode='reflect')
    if 'gl-hijack':
      D = L.stft(x[:-1], **FFT_PARAMS)
      M, P = L.spectrum.magphase(D, power=1)
      x_gl_ex = griffinlim_hijack(M, P, n_iter=32, **FFT_PARAMS, length=len(x))

    score_list.append([
      get_score(x, y),
      get_score(x_m10, y),
      get_score(x_d10, y),
      get_score(x_norm, y),
      get_score(x_isft, y),
      get_score(x_gl, y),
      get_score(x_gl_ex, y),
    ])

  # ref: 0.05933891
  # *10: 0.05413401
  # /10: 0.06204821
  # norm: 0.0643403
  # isft: 0.05933908
  # gl: -0.01090131
  # gl_ex: 0.05933908
  print(np.asarray(score_list).mean(axis=0))


if __name__ == '__main__':
  run()
