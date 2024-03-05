#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/19

from argparse import ArgumentParser

from utils import *


def plot_value_hist(X:ndarray, Y:ndarray):
  ''' 对比 QZ 和 CZ 的数值取值分布 '''
  
  X_flat = X.flatten()
  Y_flat = Y.flatten()
  plt.clf()
  plt.subplot(211) ; plt.hist(X_flat.clip(-5,    5),    bins=200)
  plt.subplot(212) ; plt.hist(Y_flat.clip(-1000, 1000), bins=200)
  plt.suptitle('train-hist')
  plt.show()


def vis_cmp(X:ndarray, Y:ndarray):
  ''' 对比 QZ 和 CZ 的包络 '''

  maxpool = nn.MaxPool1d(kernel_size=160, stride=1, padding=80)
  avgpool = nn.AvgPool1d(kernel_size=80,  stride=1, padding=40)

  for i, (x, y) in enumerate(zip(X, Y)):
    data = torch.from_numpy(y).unsqueeze(dim=0).cuda()
    upper =  maxpool( data)
    lower = -maxpool(-data)
    for i in range(2):
      upper = avgpool(upper)
      lower = avgpool(lower)
    upper = upper.squeeze(0).cpu().numpy()
    lower = lower.squeeze(0).cpu().numpy()

    plt.clf()
    plt.plot(upper, 'r')
    plt.plot(lower, 'r')
    plt.plot(y, 'b')
    plt.show()


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--norm', action='store_true')
  parser.add_argument('--log1p', action='store_true')
  args = parser.parse_args()

  X, Y = get_data_train(is_log1p=args.log1p)
  if args.norm:
    X = np.stack([wav_norm(x) for x in X], axis=0)
    X = np.stack([wav_norm(y) for y in Y], axis=0)

  plot_value_hist(X, Y)
  vis_cmp(X, Y)
