#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/19

'''
观察到原始数据的数值和动态范围有很大差异，且这种差异貌似**非简单线性**，可知数据中可能存在多个模态源。例举一些样本的峰值对比如下：

| 强震QZ | 测震 CZ |
| :-: | :-: |
|  ~3 |  ~3000 |
|  ~5 |  ~1000 |
|  ~5 |  ~2000 |
| ~10 |  ~5000 |
| ~20 |  ~6000 |
| ~20 | ~20000 |
| ~40 |  ~7500 |
| ~50 | ~20000 |
| ~100 | ~5000 |
| ~400 | ?? |

并且这种数据的一致性放大效应可能来自给定数据处理中的错误操作：

```
假设数据集的制备方式如下：
  wav -(stft)-> mag/phase -(f_denoise)-> mag_denoise/phase -(istft)-> wav_denoise
                 ↑                             ↑
以上两处谱幅度的 power 若未能保持一致，典型错误是把 能量谱(p=2) 当作了 幅度谱(p=1)，则此时 istft 导出的信号即会产生这种异常大值
```

写了个脚本来验证这一点，似乎又不那么符合假设，tmd...
'''

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
