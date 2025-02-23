#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/20  

'''
modeling: log10(stft(QZ)) -> wav_norm(CZ)
  - 优点: 直出波形，谱看似合理
  - 缺点: 相位差异大、也不知道振幅该如何反归一化
'''

from mel2wav.modules import WNConv1d, WNConvTranspose1d, ResnetBlock, Discriminator, weights_init

from models.fft import Audio2Spec
from models.utils import *


class Generator(nn.Module):
  
  def __init__(self, input_size, ngf, n_residual_layers):
    super().__init__()

    ratios = [4, 4, 2]
    self.hop_length = np.prod(ratios)
    mult = int(2 ** len(ratios))

    model = [
      nn.ReflectionPad1d(3),
      WNConv1d(input_size, mult * ngf, kernel_size=7, padding=0),
    ]

    # Upsample to raw audio scale
    for i, r in enumerate(ratios):
      model += [
        nn.LeakyReLU(0.2),
        WNConvTranspose1d(
          mult * ngf,
          mult * ngf // 2,
          kernel_size=r * 2,
          stride=r,
          padding=r // 2 + r % 2,
          output_padding=r % 2,
        ),
      ]
      for j in range(n_residual_layers):
        model += [ResnetBlock(mult * ngf // 2, dilation=3 ** j)]
      mult //= 2

    model += [
      nn.LeakyReLU(0.2),
      nn.ReflectionPad1d(3),
      WNConv1d(ngf, 1, kernel_size=7, padding=0),
      nn.Tanh(),
    ]

    self.model = nn.Sequential(*model)
    self.apply(weights_init)

  def forward(self, x):
      return self.model(x)


class GeneratorTE(Generator):

  def __init__(self, input_size, ngf, n_residual_layers):
    super().__init__(input_size + 32, ngf, n_residual_layers)

    embed_dim = 32
    self.posenc = nn.Embedding(NLEN//HOP_LEN, embed_dim)

  def forward(self, x:Tensor, ids:Tensor) -> Tensor:
    pe: Tensor = self.posenc(ids).swapaxes(1, 2)   # [B, D=32, L=128]
    if DEBUG_SHAPE: print('pe:', pe.shape)
    x = torch.cat([x, pe], dim=1)
    if DEBUG_SHAPE: print('x_cat:', x.shape)
    return self.model(x)


if __name__ == '__main__':
  from argparse import Namespace
  args = Namespace()
  args.n_mel_channels = N_SPEC
  args.ngf = 16
  args.n_residual_layers = 3
  args.num_D = 3
  args.ndf = 16
  args.n_layers_D = 4
  args.downsamp_factor = 4
  args.seq_len = N_SEG

  netG = Generator(args.n_mel_channels, args.ngf, args.n_residual_layers)
  netD = Discriminator(args.num_D, args.ndf, args.n_layers_D, args.downsamp_factor)
  fft = Audio2Spec(N_FFT, HOP_LEN, WIN_LEN, device='cpu')

  wav = torch.randn([4, 1, args.seq_len])
  print('wav.shape:', wav.shape)
  mel = fft(wav)
  print('mel.shape:', mel.shape)
  wav_hat = netG(mel)
  print('wav_hat.shape:', wav_hat.shape)
  scores = netD(wav_hat)
  print([[e.shape for e in d] for d in scores])

  netG_te = GeneratorTE(args.n_mel_channels, args.ngf, args.n_residual_layers)
  ids = torch.arange(N_FRAME).unsqueeze(dim=0).expand(mel.shape[0], -1)
  wav_te_hat = netG_te(mel, ids)
  print('wav_te_hat.shape:', wav_te_hat.shape)
