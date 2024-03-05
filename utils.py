#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/01

from pathlib import Path
from time import time
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module as Model
from torch import Tensor
import numpy as np
from numpy import ndarray
import librosa as L
import librosa.display as LD
from tqdm import tqdm
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.set_float32_matmul_precision('medium')

BASE_PATH = Path(__file__).parent.relative_to(Path.cwd())
DATA_PATH = BASE_PATH / 'data'
LOG_PATH = BASE_PATH / 'log' ; LOG_PATH.mkdir(exist_ok=True)
SUBMIT_PATH = LOG_PATH / 'result.zip'

NLEN = 24000  # 4min (1min before + 3min after)
SR   = 100    # 100Hz
SEED = 114514


def timer(fn):
  def wrapper(*args, **kwargs):
    start = time()
    r = fn(*args, **kwargs)
    end = time()
    print(f'[Timer]: {fn.__name__} took {end - start:.3f}s')
    return r
  return wrapper


@timer
def get_data_train(is_log1p:bool=True) -> Tuple[ndarray, ndarray]:
  if is_log1p:
    data = np.load(DATA_PATH / 'train-log.npz')
  else:
    data = np.load(DATA_PATH / 'train.npz')
  return data['X'], data['Y']

@timer
def get_data_test(is_log1p:bool=True) -> ndarray:
  if is_log1p:
    data = np.load(DATA_PATH / 'test-log.npz')
  else:
    data = np.load(DATA_PATH / 'test.npz')
  return data['X']

def get_submit_pred_maybe(nlen:int, fp:Path=None) -> ndarray:
  fp = fp or SUBMIT_PATH
  if not fp.exists(): return [None] * nlen
  # TODO: load & parse result.zip'
  return np.loadtxt(fp, dtype=np.int32)


def wav_norm(X:ndarray) -> ndarray:
  X_min = X.min(axis=-1, keepdims=True)
  X_max = X.max(axis=-1, keepdims=True)
  X = (X - X_min) / (X_max - X_min)
  X -= 0.5    # [-0.5, 0.5]
  X -= X.mean(axis=-1, keepdims=True)   # remove DC offset
  return X    # ~[-0.5, 0.5]


def get_spec(y:ndarray, n_fft:int=256, hop_len:int=16, win_len:int=64) -> ndarray:
  D = L.stft(y, n_fft=n_fft, hop_length=hop_len, win_length=win_len)
  M = np.clip(np.log(np.abs(D) + 1e-15), a_min=1e-5, a_max=None)
  return M
