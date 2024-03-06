#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/20  

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations

remove_weight_norm = lambda mod: remove_parametrizations(mod, 'weight')

from utils import *


def get_padding(kernel_size:int, stride:int=2):
  return (kernel_size - stride) // 2


class EnvolopeModel(nn.Module):

  ''' log1p(QZ signal) -> log1p(CZ envolope) '''

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

  def remove_weight_norm(self):
    remove_weight_norm(self.pre_conv)
    for conv in self.downs: remove_weight_norm(conv)
    for conv in self.ini: remove_weight_norm(conv)
    for conv in self.ups: remove_weight_norm(conv)
    remove_weight_norm(self.post_conv)


class EnvolopeExtractor(nn.Module):

  ''' signal -> dynamic envolope '''

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
    return torch.cat([upper, lower], dim=1)


class ResBlock(nn.Module):

  def __init__(self, dim:int, ks:List[int]=[3, 1]):
    super().__init__()

    self.convs = nn.ModuleList([weight_norm(nn.Conv2d(dim, dim, kernel_size=k, padding=k//2)) for k in ks])

  def forward(self, x:Tensor) -> Tensor:
    r = x
    for conv in self.convs:
      o = F.leaky_relu(x)
      o = conv(o)
      x = x + o
    return x + r

  def remove_weight_norm(self):
    for conv in self.convs: remove_weight_norm(conv)


class GatedActivation(nn.Module):

  def __init__(self, dim:int, k:int=5):
    super().__init__()

    self.conv = weight_norm(nn.Conv2d(dim, dim*2, kernel_size=k, padding=k//2))

  def forward(self, x:Tensor) -> Tensor:
    o = self.conv(x)
    fx, gx = torch.chunk(o, 2, dim=1)
    return fx * F.sigmoid(gx)

  def remove_weight_norm(self):
    remove_weight_norm(self.conv)


class DenoiseModel(nn.Module):

  ''' spectrogram -> denoised spectrogram '''

  def __init__(self):
    super().__init__()

    # learnable posenc for index L=128
    embed_dim = 32
    self.posenc = nn.Embedding(NLEN//HOP_LEN, embed_dim)

    # x8 downsample: [1+embed_dim, 64, 128] => [256, 8, 16] => [1, 64, 128]
    self.pre_conv = weight_norm(nn.Conv2d(1+embed_dim, 32, kernel_size=7, padding=3))
    self.downs = nn.ModuleList([
      ResBlock(32),
      nn.LeakyReLU(),
      weight_norm(nn.Conv2d( 32,  64, kernel_size=4, stride=2, groups=2, padding=get_padding(4))),
      ResBlock(64),
      nn.LeakyReLU(),
      weight_norm(nn.Conv2d( 64, 128, kernel_size=4, stride=2, groups=4, padding=get_padding(4))),
      ResBlock(128),
      nn.LeakyReLU(),
      weight_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, groups=8, padding=get_padding(4))),
    ])
    self.ini = nn.ModuleList([
      nn.LeakyReLU(),
      weight_norm(nn.Conv2d(256, 256, kernel_size=1)),
      GatedActivation(256, k=5),
      weight_norm(nn.Conv2d(256, 256, kernel_size=1, groups=16)),
    ])
    self.ups = nn.ModuleList([
      ResBlock(256),
      nn.LeakyReLU(),
      weight_norm(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=get_padding(4))),
      ResBlock(128),
      nn.LeakyReLU(),
      weight_norm(nn.ConvTranspose2d(128,  64, kernel_size=4, stride=2, padding=get_padding(4))),
      ResBlock(64),
      nn.LeakyReLU(),
      weight_norm(nn.ConvTranspose2d( 64,  32, kernel_size=4, stride=2, padding=get_padding(4))),
    ])
    self.post_conv = weight_norm(nn.Conv2d(32, 1, kernel_size=7, padding=3))

  def forward(self, x:Tensor, ids:Tensor) -> Tensor:
    DEBUG_SHAPE = False

    if DEBUG_SHAPE: print('x:', x.shape)      # [B, F=65, L=128]
    if DEBUG_SHAPE: print('ids:', ids.shape)  # [B, L]

    pe: Tensor = self.posenc(ids)   # [B, L=128, D=32]
    if DEBUG_SHAPE: print('pe:', pe.shape)
    pe_ex = pe.swapaxes(1, 2).unsqueeze(dim=-2).expand(-1, -1, x.shape[-2], -1)
    if DEBUG_SHAPE: print('pe_ex:', pe_ex.shape)
    x = torch.cat([x.unsqueeze(dim=1), pe_ex], dim=1)
    if DEBUG_SHAPE: print('x_cat:', x.shape)

    x = self.pre_conv(x)
    if DEBUG_SHAPE: print('pre_conv:', x.shape)
    for i, layer in enumerate(self.downs):
      x = layer(x)
      if DEBUG_SHAPE: print(f'downs-{i}:', x.shape)
    for i, layer in enumerate(self.ini):
      x = layer(x)
      if DEBUG_SHAPE: print(f'ini-{i}:', x.shape)
    for i, layer in enumerate(self.ups):
      x = layer(x)
      if DEBUG_SHAPE: print(f'ups-{i}:', x.shape)
    x = F.leaky_relu(x)
    x = self.post_conv(x)
    if DEBUG_SHAPE: print('post_conv:', x.shape)

    x = x.squeeze(dim=1)
    if DEBUG_SHAPE: print('out:', x.shape)
    return x

  def remove_weight_norm(self):
    remove_weight_norm(self.pre_conv)
    for layer in self.downs:
      if isinstance(layer, ResBlock):
        layer.remove_weight_norm()
      elif isinstance(layer, nn.Conv2d):
        remove_weight_norm(layer)
    for layer in self.ini:
      if isinstance(layer, GatedActivation):
        layer.remove_weight_norm()
      elif isinstance(layer, nn.Conv2d):
        remove_weight_norm(layer)
    for layer in self.ups:
      if isinstance(layer, ResBlock):
        layer.remove_weight_norm()
      elif isinstance(layer, nn.Conv2d):
        remove_weight_norm(layer)
    remove_weight_norm(self.post_conv)


if __name__ == '__main__':
  # QZ signal -> CZ envolope (upper & lower)
  X = torch.rand([4, 1, NLEN])
  model = EnvolopeModel()
  model.remove_weight_norm()
  out = model(X)
  print(out.shape)  # [B, C=2, L]

  # signal -> envolope (upper & lower)
  Y = torch.rand([4, 1, NLEN])
  extractor = EnvolopeExtractor()
  out = extractor(Y)
  print(out.shape)  # [B, C=2, L]

  # spec -> denosied spec
  M = torch.rand([4, N_SPEC-1, N_FRAME])
  denoiser = DenoiseModel()
  denoiser.remove_weight_norm()
  ids = torch.arange(N_FRAME).unsqueeze(dim=0).expand(M.shape[0], -1)
  out = denoiser(M, ids)
  print(out.shape)  # [B, C=1, F, L]
