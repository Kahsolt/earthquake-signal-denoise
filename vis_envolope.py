#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/19

from argparse import ArgumentParser

from models import EnvolopeExtractor
from utils import *


envolope_extractor = EnvolopeExtractor(n_avg=2).to(device)

def get_env_dyn(x:ndarray) -> Tuple[Tuple[ndarray, ndarray], Tensor]:
  x_upper, x_lower = envolope_extractor(torch.from_numpy(x).to(device).unsqueeze(dim=0).unsqueeze(dim=0))[0]
  x_dyn = x_upper - x_lower
  return [(e.cpu().numpy()) for e in [x_upper, x_lower]], x_dyn


def plot_value_hist(X:ndarray, Y:ndarray):
  ''' 对比 QZ 和 CZ 的数值取值分布 '''

  X_flat = X.flatten()
  Y_flat = Y.flatten()
  plt.clf()
  plt.subplot(211) ; plt.hist(X_flat.clip(-5,    5),    bins=200)
  plt.subplot(212) ; plt.hist(Y_flat.clip(-1000, 1000), bins=200)
  plt.suptitle('train-hist')
  plt.show()


def plot_dyn_dists(X:ndarray, Y:ndarray):
  ''' 训练集 QZ-CZ 在归一化后，有些样本对的包络非常相似，我们来看看是否存在某些数据模态：好像没有 '''

  get_dyn = lambda x: get_env_dyn(x)[-1]

  def cut_seg(x:Tensor) -> Tensor:
    return x
    #return x[6000:12000]   # the pulse second

  dist = []
  for x, y in tqdm(zip(X, Y)):
    x_dyn = get_dyn(cut_seg(x))
    y_dyn = get_dyn(cut_seg(y))
    dist.append(F.l1_loss(x_dyn, y_dyn).item())

  plt.clf()
  plt.hist(dist, bins=50)
  plt.suptitle('dynamic L1 distance')
  plt.tight_layout()
  plt.show()


def vis_cmp(X:ndarray, Y:ndarray):
  for x, y in tqdm(zip(X, Y)):
    (x_upper, x_lower), x_dyn = get_env_dyn(x)
    (y_upper, y_lower), y_dyn = get_env_dyn(y)

    plt.clf()
    plt.subplot(211) ; plt.plot(x_upper, 'b') ; plt.plot(x_lower, 'b') ; plt.plot(x, 'r') ; plt.title('QZ')
    plt.subplot(212) ; plt.plot(y_upper, 'b') ; plt.plot(y_lower, 'b') ; plt.plot(y, 'r') ; plt.title('CZ')
    plt.suptitle(f'dyn L1 dist: {F.l1_loss(x_dyn, y_dyn).item():.5f}')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-T', choices=['', 'norm', 'log1p'], default='raw signal transform')
  args = parser.parse_args()

  X, Y = get_data_train()
  if args.T == 'norm':
    X = np.stack([wav_norm(x) for x in X], axis=0)
    Y = np.stack([wav_norm(y) for y in Y], axis=0)
  elif args.T == 'log1p':
    X = np.stack([wav_log1p(x) for x in X], axis=0)
    Y = np.stack([wav_log1p(y) for y in Y], axis=0)

  #plot_value_hist(X, Y)
  #plot_dyn_dists(X, Y)

  vis_cmp(X, Y)
