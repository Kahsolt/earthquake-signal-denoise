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
  sdata = data[:-cp] if split == 'train' else data[-cp:]
  return sdata


class SignalDataset(Dataset):

  def __init__(self, split:str='train', transform:Callable=None, n_seg:int=-1, ratio:float=0.1):
    self.n_seg = n_seg
    self.id_rng = None

    X, Y = get_data_train()
    if transform:
      X = transform(X)
      Y = transform(Y)
    self.data = make_split(X, Y, split, ratio)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx) -> Tuple[ndarray, ndarray]:
    X, Y = self.data[idx]
    if self.n_seg > 0:
      cp = random.randrange(len(X) - self.n_seg)
      self.id_rng = [cp, cp + self.n_seg]
      slicer = slice(*self.id_rng)
      X, Y = X[slicer], Y[slicer]
    else:
      self.id_rng = [0, len(X)]
    return [np.expand_dims(e, axis=0) for e in [X, Y]]


class SpecDataset(SignalDataset):

  def __init__(self, split:str='train', transform:Callable=None, n_seg:int=N_SEG, ratio:float=0.1):
    super().__init__(split, transform, n_seg, ratio)

    self.get_spec_ = lambda x: get_spec(x.squeeze(0)[:-1], **FFT_PARAMS)[:-1]   # ignore last band (hifreq ~1e-5)

  @property
  def id_rng_spec(self) -> Tuple[int]:
    x, y = self.id_rng
    return x // HOP_LEN, y // HOP_LEN

  def __getitem__(self, idx) -> Tuple[ndarray, ndarray, ndarray]:
    X, Y = super().__getitem__(idx)
    MX = self.get_spec_(X)   # [F=64, L=128]
    MY = self.get_spec_(Y)
    return MX, MY, np.arange(*self.id_rng_spec)


if __name__ == '__main__':
  dataset = SignalDataset(transform=wav_norm)
  for X, Y in iter(dataset):
    stat_tensor(X, 'X')   # [1, 24000]
    stat_tensor(Y, 'Y')
    break

  dataset = SpecDataset(transform=wav_norm)
  for X, Y, ids in iter(dataset):
    stat_tensor(X, 'X')   # [64, 128]
    stat_tensor(Y, 'Y')
    stat_tensor(ids, 'ids')
    break
