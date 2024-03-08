#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/20  

from models.utils import *


act = lambda x: F.leaky_relu_(x, negative_slope=0.02)


class ResBlock(nn.Module):

  ''' alike HiFiGAN ResBlock2 '''

  def __init__(self, ch:int, k:int=3, dilation:Tuple[int]=(1, 3)):
    super().__init__()

    self.convs = nn.ModuleList([
      weight_norm(nn.Conv2d(ch, ch, k, dilation=dilation[0], padding=get_padding_dilated(k, dilation[0]))),
      weight_norm(nn.Conv2d(ch, ch, k, dilation=dilation[1], padding=get_padding_dilated(k, dilation[1]))),
    ])
    self.convs.apply(init_weights)

  def forward(self, x:Tensor) -> Tensor:
    for c in self.convs:
      xt = act(x)
      xt = c(xt)
      x = xt + x
    return x

  def remove_weight_norm(self):
    for l in self.convs: remove_weight_norm(l)


class GatedActivation(nn.Module):

  def __init__(self, ch:int, k:int=5):
    super().__init__()

    self.conv = weight_norm(nn.Conv2d(ch, ch*2, k, padding=k//2))
    self.conv.apply(init_weights)

  def forward(self, x:Tensor) -> Tensor:
    o = self.conv(x)
    fx, gx = torch.chunk(o, 2, dim=1)
    return fx * F.sigmoid(gx)

  def remove_weight_norm(self):
    remove_weight_norm(self.conv)


class DenoiseModel(nn.Module):

  ''' alike HiFiGAN Generator: spectrogram -> denoised spectrogram '''

  def __init__(self):
    super().__init__()

    # learnable posenc for index L=128
    embed_dim = 32
    self.posenc = nn.Embedding(NLEN//HOP_LEN, embed_dim)

    # x8 downsample: [1+embed_dim, 64, 128] => [256, 8, 16] => [1, 64, 128]
    self.pre_conv = weight_norm(nn.Conv2d(1+embed_dim, 16, kernel_size=7, padding=3))
    self.downs = nn.ModuleList([
      weight_norm(nn.Conv2d(16,  32, kernel_size=4, stride=2, groups=4,  padding=get_padding_strided(4))),
      weight_norm(nn.Conv2d(32,  64, kernel_size=4, stride=2, groups=8,  padding=get_padding_strided(4))),
      weight_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, groups=16, padding=get_padding_strided(4))),
    ])
    self.downs_resblocks = nn.ModuleList([
      ResBlock(32),
      ResBlock(64),
      ResBlock(128),
    ])
    self.mid = nn.ModuleList([
      weight_norm(nn.Conv2d(128, 256, kernel_size=1)),
      GatedActivation(256, k=7),
      weight_norm(nn.Conv2d(256, 128, kernel_size=1, groups=16)),
    ])
    self.ups = nn.ModuleList([
      weight_norm(nn.ConvTranspose2d(128+128, 128, kernel_size=4, stride=2, padding=get_padding_strided(4))),
      weight_norm(nn.ConvTranspose2d(128+ 64,  64, kernel_size=4, stride=2, padding=get_padding_strided(4))),
      weight_norm(nn.ConvTranspose2d( 64+ 32,  32, kernel_size=4, stride=2, padding=get_padding_strided(4))),
    ])
    self.ups_resblocks = nn.ModuleList([
      ResBlock(128),
      ResBlock(64),
      ResBlock(32),
    ])
    self.post_conv = weight_norm(nn.Conv2d(32, 1, kernel_size=7, padding=3))

    self.pre_conv.apply(init_weights)
    self.downs.apply(init_weights)
    self.mid.apply(init_weights)
    self.ups.apply(init_weights)
    self.post_conv.apply(init_weights)

  def forward(self, x:Tensor, ids:Tensor) -> Tensor:
    if DEBUG_SHAPE: print('x:', x.shape)      # [B, F=65, L=128]
    if DEBUG_SHAPE: print('ids:', ids.shape)  # [B, L]

    pe: Tensor = self.posenc(ids)   # [B, L=128, D=32]
    if DEBUG_SHAPE: print('pe:', pe.shape)
    pe_ex = pe.swapaxes(1, 2).unsqueeze(dim=-2).expand(-1, -1, x.shape[-2], -1)   # expand index `F`
    if DEBUG_SHAPE: print('pe_ex:', pe_ex.shape)
    x = torch.cat([x.unsqueeze(dim=1), pe_ex], dim=1)
    if DEBUG_SHAPE: print('x_cat:', x.shape)

    x = self.pre_conv(x)
    if DEBUG_SHAPE: print('pre_conv:', x.shape)
    hs = []
    for i, layer in enumerate(self.downs):
      x = act(x)
      x = layer(x)
      hs.append(x)
      if DEBUG_SHAPE: print(f'downs-{i}:', x.shape)
      x = self.downs_resblocks[i](x)
      if DEBUG_SHAPE: print(f'downs_rblk-{i}:', x.shape)
    for i, layer in enumerate(self.mid):
      x = act(x)
      x = layer(x)
      if DEBUG_SHAPE: print(f'mid-{i}:', x.shape)
    for i, layer in enumerate(self.ups):
      x = act(x)
      h = hs[len(self.downs) - 1 - i]
      fused = torch.cat([x, h], dim=1)
      if DEBUG_SHAPE: print(f'fused-{i}:', fused.shape)
      x = layer(fused)
      if DEBUG_SHAPE: print(f'ups-{i}:', x.shape)
      x = self.ups_resblocks[i](x)
      if DEBUG_SHAPE: print(f'ups_rblk-{i}:', x.shape)
    x = act(x)
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
    for layer in self.mid:
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
  # spec -> denosied spec
  M = torch.rand([4, N_SPEC-1, N_FRAME])
  denoiser = DenoiseModel()
  denoiser.remove_weight_norm()
  ids = torch.arange(N_FRAME).unsqueeze(dim=0).expand(M.shape[0], -1)
  out = denoiser(M, ids)
  print(out.shape)  # [B, C=1, F, L]
