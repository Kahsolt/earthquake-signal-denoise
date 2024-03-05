#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/20  

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from utils import *


def get_padding(kernel_size:int, stride:int):
  return (kernel_size - stride) // 2


class EnvolopeModel(nn.Module):

  def __init__(self):
    super().__init__()

    # x192 downsample: 24000 => 125
    self.pre_conv = weight_norm(nn.Conv1d(1, 8, kernel_size=41, padding=20))
    self.downs = nn.ModuleList([
      weight_norm(nn.Conv1d( 8,  32, kernel_size=16, stride=8, groups=4,  padding=get_padding(16, 8))),
      weight_norm(nn.Conv1d(32,  64, kernel_size=12, stride=6, groups=16, padding=get_padding(12, 6))),
      weight_norm(nn.Conv1d(64, 128, kernel_size=8,  stride=4, groups=32, padding=get_padding(8,  4))),
    ])
    self.ini = nn.ModuleList([
      weight_norm(nn.Conv1d(128, 128, kernel_size=1)),
      weight_norm(nn.Conv1d(128, 128, kernel_size=1, groups=16)),
    ])
    self.ups = nn.ModuleList([
      weight_norm(nn.ConvTranspose1d(128*2, 64, kernel_size=8,  stride=4, padding=get_padding(8,  4))),
      weight_norm(nn.ConvTranspose1d( 64*2, 32, kernel_size=12, stride=6, padding=get_padding(12, 6))),
      weight_norm(nn.ConvTranspose1d( 32*2, 16, kernel_size=16, stride=8, padding=get_padding(16, 8))),
    ])
    self.post_conv = weight_norm(nn.Conv1d(16, 2, kernel_size=41, padding=20))

  def forward(self, x:Tensor) -> Tensor:
    DEBUG_SHAPE = False

    x = self.pre_conv(x)      # [B, D=1, L=24000]
    if DEBUG_SHAPE: print('pre_conv:', x.shape)
    hs = []
    for i, layer in enumerate(self.downs):
      x = F.relu(x)
      x = layer(x)
      if DEBUG_SHAPE: print(f'downs-[{i}]:', x.shape)
      hs.append(x)
    for i, layer in enumerate(self.ini):
      x = F.relu(x)
      x = layer(x)
      if DEBUG_SHAPE: print(f'ini-[{i}]:', x.shape)
    for i, layer in enumerate(self.ups):
      x = F.relu(x)
      h = hs[len(self.ups) - 1 - i]
      fused = torch.cat([x, h], dim=1)
      if DEBUG_SHAPE: print(f'fused-[{i}]:', x.shape)
      x = layer(fused)
      if DEBUG_SHAPE: print(f'ups-[{i}]:', x.shape)
    x = F.tanh(x)
    x = self.post_conv(x)
    if DEBUG_SHAPE: print(f'post_conv:', x.shape)
    return x


class EnvolopeExtractor(nn.Module):

  def __init__(self, n_avg:int=2):
    super().__init__()

    self.n_avg = n_avg
    self.maxpool = nn.MaxPool1d(kernel_size=161, stride=1, padding=80)
    self.avgpool = nn.AvgPool1d(kernel_size=81,  stride=1, padding=40)

  @torch.no_grad
  def forward(self, x:Tensor) -> Tensor:
    upper =  self.maxpool( x)
    lower = -self.maxpool(-x)
    for _ in range(self.n_avg):
      upper = self.avgpool(upper)
      lower = self.avgpool(lower)
    envolope = torch.cat([upper, lower], dim=1)
    return envolope


if __name__ == '__main__':
  # QZ signal -> CZ envolope (upper & lower)
  X = torch.rand([4, 1, NLEN])
  model = EnvolopeModel()
  out = model(X)
  print(out.shape)  # [B, C=2, L]

  # signal -> envolope (upper & lower)
  Y = torch.rand([4, 1, NLEN])
  extractor = EnvolopeExtractor()
  out = extractor(Y)
  print(out.shape)  # [B, C=2, L]
