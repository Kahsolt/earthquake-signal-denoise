#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/16

import random
from torch.utils.data import Dataset, DataLoader

from utils import *


def make_split(X:ndarray, Y:ndarray, split:str='train', ratio:float=0.3) -> List[Tuple[ndarray, int]]:
  data = [(x, y) for x, y in zip(X, Y)]
  random.seed(SEED)
  random.shuffle(data)
  cp = int(len(data) * ratio)
  if split == 'train': data = data[:-cp]
  else:                data = data[-cp:]
  return data

def sample_to_XY(data:Union[Tuple[ndarray, int], ndarray]) -> Tuple[ndarray, int]:
  return data if isinstance(data, tuple) else (data, -1)


class SignalDataset(Dataset):

  def __init__(self, split:str='train', transform:Callable=None, ratio:float=0.3):
    self.split = split
    self.is_train = split in ['train', 'valid']

    if self.is_train:
      X, Y = get_data_train()
      if transform: X = transform(X)
      self.data = make_split(X, Y, split, ratio)
    else:
      X = get_data_test()
      if transform: X = transform(X)
      self.data = X

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    data = self.data[idx]
    X, Y = sample_to_XY(data)
    return np.expand_dims(X, axis=0), np.expand_dims(Y, axis=0)


class SpecDataset(SignalDataset):

  def __getitem__(self, idx):
    data = self.data[idx]
    X, Y = sample_to_XY(data)
    MX = get_spec(X[:-1])
    MY = get_spec(Y[:-1])
    return MX, MY


if __name__ == '__main__':
  dataset = SignalDataset()
  for X, Y in iter(dataset):
    print('X:', X)
    print('X.shape:', X.shape)
    print('Y:', Y)
    break
