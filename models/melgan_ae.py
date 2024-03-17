#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/03/17  

from models.utils import *

from mel2wav.modules import WNConv1d, WNConvTranspose1d, ResnetBlock, weights_init
from models import Audio2Spec


class GeneratorAE(nn.Module):

  ''' spec -> pseudo-signal -> spec '''

  def __init__(self, input_size, ngf, n_residual_layers):
    super().__init__()

    ratios = [4, 4, 2]
    self.hop_length = np.prod(ratios)
    mult = int(2 ** len(ratios))

    # prenet
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

    # Downsample to spec scale
    for i, r in enumerate(reversed(ratios)):
      model += [
        nn.LeakyReLU(0.2),
        WNConv1d(
          mult * ngf,
          mult * ngf * 2,
          kernel_size=r * 2,
          stride=r,
          padding=r // 2 + r % 2,
        ),
      ]
      for j in range(n_residual_layers):
        model += [ResnetBlock(mult * ngf * 2, dilation=3 ** j)]
      mult *= 2

    # posnet
    model += [
      nn.LeakyReLU(0.2),
      nn.ReflectionPad1d(3),
      WNConv1d(mult * ngf, input_size, kernel_size=7, padding=0),
    ]

    self.model = nn.Sequential(*model)
    self.apply(weights_init)

  def forward(self, x):
    return self.model(x)


class NLayerDiscriminatorAE(nn.Module):

  def __init__(self, input_size, ndf, n_layers, downsampling_factor):
    super().__init__()

    model = nn.ModuleDict()
    model["layer_0"] = nn.Sequential(
      nn.ReflectionPad1d(7),
      WNConv1d(input_size, ndf, kernel_size=15),
      nn.LeakyReLU(0.2, True),
    )
    nf = ndf
    stride = downsampling_factor
    for n in range(1, n_layers + 1):
      nf_prev = nf
      nf = min(nf * stride, 1024)

      model["layer_%d" % n] = nn.Sequential(
        WNConv1d(
          nf_prev,
          nf,
          kernel_size=stride * 10 + 1,
          stride=stride,
          padding=stride * 5,
          groups=nf_prev // 4,
        ),
        nn.LeakyReLU(0.2, True),
      )
    nf = min(nf * 2, 1024)
    model["layer_%d" % (n_layers + 1)] = nn.Sequential(
      WNConv1d(nf_prev, nf, kernel_size=5, stride=1, padding=2),
      nn.LeakyReLU(0.2, True),
    )
    model["layer_%d" % (n_layers + 2)] = WNConv1d(
      nf, 1, kernel_size=3, stride=1, padding=1
    )
    self.model = model

  def forward(self, x):
    results = []
    for key, layer in self.model.items():
      x = layer(x)
      results.append(x)
    return results


class DiscriminatorAE(nn.Module):

  def __init__(self, input_size, num_D, ndf, n_layers, downsampling_factor):
    super().__init__()

    self.model = nn.ModuleDict()
    for i in range(num_D):
      self.model[f"disc_{i}"] = NLayerDiscriminatorAE(input_size, ndf, n_layers, downsampling_factor)
    self.downsample = nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)
    self.apply(weights_init)

  def forward(self, x):
    results = []
    for key, disc in self.model.items():
      results.append(disc(x))
      x = self.downsample(x)
    return results


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

  netG = GeneratorAE(args.n_mel_channels, args.ngf, args.n_residual_layers)
  netD = DiscriminatorAE(args.n_mel_channels, args.num_D, args.ndf, args.n_layers_D, args.downsamp_factor)
  fft = Audio2Spec(N_FFT, HOP_LEN, WIN_LEN, SR, device='cpu')

  wav = torch.randn([4, 1, args.seq_len])
  print('wav.shape:', wav.shape)
  spec = fft(wav)
  print('spec.shape:', spec.shape)
  spec_hat = netG(spec)
  print('spec_hat.shape:', spec_hat.shape)
  scores = netD(spec_hat)
  print([[e.shape for e in d] for d in scores])
