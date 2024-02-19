#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/19

import seaborn as sns

from utils import *

n_fft = 256
hop_len = 64
win_len = 256


def get_mag_phase(y:ndarray) -> Tuple[ndarray, ndarray]:
  D = L.stft(y, n_fft=n_fft, hop_length=hop_len, win_length=win_len)
  mag, phase = L.spectrum.magphase(D)
  M = mag
  P = (np.log(phase) / 1j).real
  return M, P


def test_stft_power2_istft(x:ndarray, y:ndarray):
  D = L.stft(x, n_fft=n_fft, hop_length=hop_len, win_length=win_len)
  x_inv = L.istft(D, n_fft=n_fft, hop_length=hop_len, win_length=win_len, length=len(y))
  mag, phase = L.spectrum.magphase(D, power=2)
  D2 = mag * phase
  x2_inv = L.istft(D2, n_fft=n_fft, hop_length=hop_len, win_length=win_len, length=len(y))

  plt.clf()
  plt.subplot(411) ; plt.title('x') ; plt.plot(x)
  plt.subplot(412) ; plt.title('y') ; plt.plot(y)
  plt.subplot(413) ; plt.title('x_inv (mag^1)') ; plt.plot(x_inv)
  plt.subplot(414) ; plt.title('x_inv (mag^2)') ; plt.plot(x2_inv)
  plt.suptitle('test_stft_power2_istft')
  plt.tight_layout()
  plt.show()


def test_mag_sqrt(x:ndarray, y:ndarray):
  M1, P1 = get_mag_phase(x)
  M2, P2 = get_mag_phase(y)
  M2 = M2**0.5
  M1 = np.log(M1)
  M2 = np.log(M2)

  dM = np.abs(M1 - M2)
  dP = np.abs(P1 - P2)
  print('dM.mean:', dM.mean())
  print('dP.mean:', dP.mean())

  plt.clf()
  plt.subplot(321) ; plt.hist(M1.flatten(), bins=50)
  plt.subplot(322) ; plt.hist(M2.flatten(), bins=50)
  ax = plt.subplot(323) ; sns.heatmap(M1, ax=ax) ; ax.invert_yaxis()
  ax = plt.subplot(324) ; sns.heatmap(M2, ax=ax) ; ax.invert_yaxis()
  ax = plt.subplot(325) ; sns.heatmap(P1, ax=ax) ; ax.invert_yaxis()
  ax = plt.subplot(326) ; sns.heatmap(P2, ax=ax) ; ax.invert_yaxis()
  plt.suptitle('test_mag_sqrt')
  plt.tight_layout()
  plt.show()


if __name__ == '__main__':
  X, Y = get_data_train()
  for i, (x, y) in enumerate(zip(X, Y)):
    test_stft_power2_istft(x, y)
    test_mag_sqrt(x, y)
