#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/19

from utils import *

maxpool = nn.MaxPool1d(kernel_size=160, stride=1, padding=80)
avgpool = nn.AvgPool1d(kernel_size=80,  stride=1, padding=40)


def plot_value_hist(X:ndarray, Y:ndarray):
  X_flat = X.flatten()
  Y_flat = Y.flatten()
  plt.clf()
  plt.subplot(211) ; plt.hist(X_flat.clip(-5,    5),    bins=200)
  plt.subplot(212) ; plt.hist(Y_flat.clip(-1000, 1000), bins=200)
  plt.suptitle('train-hist')
  plt.show()


X, Y = get_data_train()
#plot_value_hist(X, Y)

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
