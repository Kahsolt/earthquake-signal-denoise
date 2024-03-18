#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/01

import os
import sys
import json
from pathlib import Path
from time import time
from datetime import datetime
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module as Model
from torch import Tensor
import numpy as np
from numpy import ndarray

from tqdm import tqdm
import matplotlib.pyplot as plt

DEBUG_SHAPE = os.getenv('DEBUG_SHAPE', False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.set_float32_matmul_precision('highest')

BASE_PATH = Path(__file__).parent.relative_to(Path.cwd())
DATA_PATH = BASE_PATH / 'data'
LOG_PATH = BASE_PATH / 'log' ; LOG_PATH.mkdir(exist_ok=True)
REPO_PATH = BASE_PATH  / 'repo'
SUBMIT_PATH = LOG_PATH / 'result.zip'
MELGAN_PATH = REPO_PATH / 'melgan-neurips'
sys.path.append(str(MELGAN_PATH))

NLEN = 24000  # 4min (1min before + 3min after)
SR   = 100    # 100Hz
N_FFT   = 128
HOP_LEN = 32
WIN_LEN = 128
FFT_PARAMS = {
  'n_fft': N_FFT, 
  'hop_length': HOP_LEN, 
  'win_length': WIN_LEN,
}
N_SEG   = 4096              # 4096, in samples
N_FRAME = N_SEG // HOP_LEN  # 128, in frames
N_SPEC  = N_FFT // 2 + 1    # 65, n_freq_band

EPS = 1e-5
SEED = 114514


''' misc '''

def timer(fn):
  def wrapper(*args, **kwargs):
    start = time()
    r = fn(*args, **kwargs)
    end = time()
    print(f'[Timer]: {fn.__name__} took {end - start:.3f}s')
    return r
  return wrapper

def now_ts():   # 2024-03-06#15-58-36
  return str(datetime.now()).split('.')[0].replace(' ', '_').replace(':', '-')

def stat_tensor(x:Tensor, name:str='X'):
  print(f'[{name}] {x.shape}')
  print('  max:', x.max())
  print('  min:', x.min())
  print('  avg:', x.mean())
  print('  std:', x.std())


''' data '''

@timer
def get_data_train() -> Tuple[ndarray, ndarray]:
  data = np.load(DATA_PATH / 'train.npz')
  return data['X'], data['Y']

@timer
def get_data_test() -> ndarray:
  data = np.load(DATA_PATH / 'test.npz')
  return data['X']

def get_data_test_meta() -> Tuple[List[str], Dict[str, int]]:
  with open(DATA_PATH / 'test.txt', 'r', encoding='utf-8') as fh:
    namelist = fh.read().strip().split('\n')
  with open(DATA_PATH / 'test.json', 'r', encoding='utf-8') as fh:
    lendict = json.load(fh)
  return namelist, lendict

@timer
def get_submit_pred_maybe(nlen:int, fp:Path=None) -> ndarray:
  fp = fp or SUBMIT_PATH
  if not fp.exists():
    print(f'>> warn: file {fp} not exits')
    return [None] * nlen

  from mk_data import process_zipfile
  X, fns, lens = process_zipfile(fp)
  return X


''' audio '''

def wav_log1p(x:ndarray) -> ndarray:
  mask = x >= 0
  pos = x *  mask
  neg = x * ~mask
  pos_log1p =  np.log1p( pos)
  neg_log1p = -np.log1p(-neg)
  return np.where(mask, pos_log1p, neg_log1p)

def wav_norm(x:ndarray, C:float=1.0, remove_DC:bool=True) -> ndarray:
  X_min = x.min(axis=-1, keepdims=True)
  X_max = x.max(axis=-1, keepdims=True)
  x = (x - X_min) / (X_max - X_min)   # [0, 1]
  x = (x - 0.5) * 2 * C   # ~[-C, C]
  if remove_DC: x -= x.mean(axis=-1, keepdims=True)   # remove DC offset
  return x

def griffinlim_hijack(
  specgram: Tensor,
  init_phase: Tensor,
  n_iter: int = 32,
  n_fft: int = N_FFT,
  hop_length: int = HOP_LEN,
  win_length: int = WIN_LEN,
  momentum: float = 0.99,
  length: Optional[int] = None,
) -> Tensor:
  from torchaudio.functional import griffinlim
  from torchaudio.transforms import GriffinLim

  window = torch.hann_window(win_length).float().to(specgram.device)
  momentum = momentum / (1 + momentum)
  # pack batch
  shape = specgram.size()
  specgram = specgram.reshape([-1] + list(shape[-2:]))
  # initialize the phase
  angles = init_phase
  # And initialize the previous iterate to 0
  tprev = torch.tensor(0.0, dtype=specgram.dtype, device=specgram.device)
  for _ in range(n_iter):
    # Invert with our current estimate of the phases
    inverse = torch.istft(specgram * angles, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, length=length)
    # Rebuild the spectrogram
    rebuilt = torch.stft(
      input=inverse,
      n_fft=n_fft,
      hop_length=hop_length,
      win_length=win_length,
      window=window,
      center=True,
      pad_mode="reflect",
      normalized=False,
      onesided=True,
      return_complex=True,
    )
    # Update our phase estimates
    angles = rebuilt
    if momentum:
      angles -= tprev.mul_(momentum)
    angles = angles.div(angles.abs().add(1e-16))
    # Store the previous iterate
    tprev = rebuilt

  # Return the final phase estimates
  waveform = torch.istft(specgram * angles, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, length=length)
  # unpack batch
  waveform = waveform.reshape(shape[:-2] + waveform.shape[-1:])
  return waveform


''' metrics '''

def signal_noise_ratio(y_hat:ndarray, y:ndarray) -> float:
  ''' 信噪比对数值缩放敏感 '''
  if np.allclose(y_hat, y): pass    # 评测代码特殊处理了一下除零异常
  return 1.0 * np.log10(np.sum(np.power(y, 2)) / np.sum(np.power(y - y_hat, 2)))

def cross_correlation_coefficient(y_hat:ndarray, y:ndarray) -> float:
  ''' 互相关系数对数值缩放完全不敏感 '''
  y_shift = y - y.mean()
  y_hat_shift = y_hat - y_hat.mean()
  return np.sum(y_shift * y_hat_shift) / np.sqrt(np.sum(np.power(y_shift, 2)) * np.sum(np.power(y_hat_shift, 2)))

def pearson_correlation(y_hat:ndarray, y:ndarray) -> float:
  ''' 评测代码直接使用了scipy库的皮尔逊相关系数实现 '''
  from scipy.stats import pearsonr
  return pearsonr(y_hat, y).correlation

def get_score(y_hat:ndarray, y:ndarray) -> float:
  return (signal_noise_ratio(y_hat, y) + cross_correlation_coefficient(y_hat, y)) / 2


if __name__ == '__main__':
  X, Y = get_data_train()
  for i, (x, y) in enumerate(zip(X, Y)):
    ''' [正常评估中] y_hat: 测振(CZ)预测值, y: 测振(CZ)真值 '''
    snr = signal_noise_ratio(x, y)
    ccc = cross_correlation_coefficient(x, y)
    pc = pearson_correlation(x, y)
    print(f'[{i}] snr: {snr:.7f}, ccc: {ccc:.7f}, pc: {pc:.7f}')
    # 总分为两项指标的平均值
    score = (snr + ccc) / 2
