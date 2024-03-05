#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/16

import random
from torch.utils.data import Dataset, DataLoader

from utils import *


def make_split(X:ndarray, Y:ndarray, split:str='train', ratio:float=0.1) -> List[Tuple[ndarray, int]]:
  data = [(x, y) for x, y in zip(X, Y)]
  random.seed(SEED)
  random.shuffle(data)
  cp = int(len(data) * ratio)
  if split == 'train': data = data[:-cp]
  else:                data = data[-cp:]
  return data


class SignalDataset(Dataset):

  def __init__(self, split:str='train', transform:Callable=None, n_seg:int=-1, ratio:float=0.1):
    self.split = split
    self.n_seg = n_seg

    if self.is_train:
      X, Y = get_data_train()
      if transform:
        X = transform(X)
        Y = transform(Y)
      self.data = make_split(X, Y, split, ratio)
    else:
      X = get_data_test()
      if transform: X = transform(X)
      self.data = X

  @property
  def is_train(self):
    return self.split in ['train', 'valid']

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    if self.is_train:
      X, Y = self.data[idx]
      if self.n_seg > 0:
        cp = random.randrange(len(X) - self.n_seg)
        slicer = slice(cp, cp + self.n_seg)
        X, Y = X[slicer], Y[slicer]
      return [np.expand_dims(e, axis=0) for e in [X, Y]]
    else:
      X = self.data[idx]
      return np.expand_dims(X, axis=0)


class SpecDataset(SignalDataset):

  def __init__(self, split:str='train', transform:Callable=None, n_seg:int=N_SEG, ratio:float=0.1):
    super().__init__(split, transform, n_seg, ratio)

    self.get_spec_ = lambda x: get_spec(x.squeeze(0)[:-1], N_FFT, HOP_LEN, WIN_LEN)

  def __getitem__(self, idx):
    if self.is_train:
      X, Y = super().__getitem__(idx)
      MX = self.get_spec_(X)   # [F=65, L=128]
      MY = self.get_spec_(Y)
      return MX, MY
    else:
      X = super().__getitem__(idx)
      MX = self.get_spec_(X)
      return MX


if __name__ == '__main__':
  dataset = SignalDataset(transform=wav_norm)
  for X, Y in iter(dataset):
    stat_tensor(X, 'X')
    stat_tensor(Y, 'Y')
    break

  dataset = SpecDataset()
  for X, Y in iter(dataset):
    stat_tensor(X, 'X')
    stat_tensor(Y, 'Y')
    break
