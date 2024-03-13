#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/16

import random
from torch.utils.data import Dataset, DataLoader

from utils import *


def make_split(X:ndarray, Y:ndarray, ratio:float=0.1) -> Tuple[List[Tuple[ndarray, ndarray]]]:
  data = [(x, y) for x, y in zip(X, Y)]
  random.seed(SEED)
  random.shuffle(data)
  cp = int(len(data) * ratio)
  return data[:-cp], data[-cp:]


class SignalDataset(Dataset):

  def __init__(self, XY:List[Tuple[ndarray, ndarray]], transform:Callable=None, n_seg:int=-1, aug:bool=False):
    self.n_seg = n_seg
    self.aug = aug

    if transform:
      XY = [(transform(x), transform(y)) for x, y in XY]
    self.data = XY

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx) -> Tuple[ndarray, ndarray]:
    X, Y = self.data[idx]
    id_rng = [0, len(X)]
    if self.n_seg > 0:
      cp = random.randrange(len(X) - self.n_seg)
      id_rng = [cp, cp + self.n_seg]
      slicer = slice(*id_rng)
      X, Y = X[slicer], Y[slicer]
    if self.aug:
      X = X * np.random.uniform(low=0.8, high=1.2)

    X = np.expand_dims(X, axis=0)
    Y = np.expand_dims(Y, axis=0)
    ids = np.arange(*[e // HOP_LEN for e in id_rng])
    return X, Y, ids


class SpecDataset(SignalDataset):

  def __init__(self, XY:List[Tuple[ndarray]], transform:Callable=None, n_seg:int=N_SEG):
    super().__init__(XY, transform, n_seg)

    self.get_spec_ = lambda x: get_spec(x.squeeze(0)[:-1], **FFT_PARAMS)[:-1]   # ignore last band (hifreq ~1e-5)

  def __getitem__(self, idx) -> Tuple[ndarray, ndarray, ndarray]:
    X, Y, ids = super().__getitem__(idx)
    MX = self.get_spec_(X)   # [F=64, L=128]
    MY = self.get_spec_(Y)
    return MX, MY, ids


if __name__ == '__main__':
  dataset = SignalDataset(transform=wav_norm)
  for X, Y, ids in iter(dataset):
    stat_tensor(X, 'X')   # [1, 24000]
    stat_tensor(Y, 'Y')
    stat_tensor(ids, 'ids')
    break

  dataset = SpecDataset(transform=wav_norm)
  for X, Y, ids in iter(dataset):
    stat_tensor(X, 'X')   # [64, 128]
    stat_tensor(Y, 'Y')
    stat_tensor(ids, 'ids')
    break
