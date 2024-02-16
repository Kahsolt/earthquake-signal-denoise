#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/19

from utils import *


def signal_noise_ratio(y_hat:ndarray, y:ndarray):
  ''' 信噪比对数值缩放敏感 '''
  return 1.0 * np.log10(np.sum(np.power(y, 2)) / np.sum(np.power(y - y_hat, 2)))


def cross_correlation_coefficient(y_hat:ndarray, y:ndarray):
  ''' 互相关系数对数值缩放完全不敏感 '''
  y_shift = y - y.mean()
  y_hat_shift = y_hat - y_hat.mean()
  return np.sum(y_shift * y_hat_shift) / np.sqrt(np.sum(np.power(y_shift, 2)) * np.sum(np.power(y_hat_shift, 2)))


X, Y = get_data_train()
for i, (x, y) in enumerate(zip(X, Y)):
  ''' [正常评估中] y_hat: 测振(CZ)预测值, y: 测振(CZ)真值 '''

  snr = signal_noise_ratio(x, y)
  ccc = cross_correlation_coefficient(x, y)
  print(f'[{i}] snr: {snr:.3f}, ccc: {ccc:.3f}')
