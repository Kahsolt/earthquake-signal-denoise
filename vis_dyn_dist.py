#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/03/05

from utils import *
from model_envolope import EnvolopeExtractor

''' 训练集 QZ-CZ 在归一化后，有些样本对的包络非常相似，我们来看看是否存在某些数据模态 '''


X, Y = get_data_train(is_log1p=False)
envolope_extractor = EnvolopeExtractor(n_avg=2).to(device)


def wav_norm(X:Tensor, remove_DC:bool=True) -> Tensor:
  assert len(X.shape) == 1, 'signal must be 1d-tensor'

  X_min, X_max = X.min(), X.max()
  X = (X - X_min) / (X_max - X_min)
  X -= 0.5    # [-0.5, 0.5]
  if remove_DC: X -= X.mean()   # remove DC offset
  return X    # ~[-0.5, 0.5]

def get_dyn(x:Tensor) -> Tensor:
  x_env = envolope_extractor(x.unsqueeze(dim=0).unsqueeze(dim=0))[0]
  return x_env[0] - x_env[1]

def cut_seg(x:Tensor) -> Tensor:
  return x
  #return x[6000:12000]   # the pulse second


dist1, dist2 = [], []
for i, (x, y) in enumerate(tqdm(zip(X, Y))):
  x = torch.from_numpy(x).to(device)
  y = torch.from_numpy(y).to(device)

  x_dyn = get_dyn(cut_seg(wav_norm(x, remove_DC=False)))
  y_dyn = get_dyn(cut_seg(wav_norm(y, remove_DC=False)))
  d1 = F.l1_loss(x_dyn, y_dyn)
  dist1.append(d1.item())

  x_dyn = get_dyn(cut_seg(wav_norm(x, remove_DC=True)))
  y_dyn = get_dyn(cut_seg(wav_norm(y, remove_DC=True)))
  d2 = F.l1_loss(x_dyn, y_dyn)
  dist2.append(d2.item())

plt.clf()
plt.scatter(dist1, dist2, s=1, c='r')
plt.xlabel('signal')
plt.ylabel('signal (remove_DC)')
plt.suptitle('dynamic L1 distance')
plt.tight_layout()
plt.show()
