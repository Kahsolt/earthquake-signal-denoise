#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/20  

from models.utils import *

from mel2wav.modules import WNConv1d, WNConvTranspose1d, ResnetBlock, Discriminator, weights_init


class Audio2Spec(nn.Module):

  def __init__(self, n_fft=N_FFT, hop_length=HOP_LEN, win_length=WIN_LEN, sampling_rate=SR, device=device):
    super().__init__()

    self.n_fft = n_fft
    self.hop_length = hop_length
    self.win_length = win_length
    self.sampling_rate = sampling_rate
    self.window = torch.hann_window(win_length).float().to(device)

  def forward(self, audio:Tensor) -> Tensor:
    p = (self.n_fft - self.hop_length) // 2
    audio = F.pad(audio, (p, p), "reflect").squeeze(1)
    fft = torch.stft(
      audio,
      n_fft=self.n_fft,
      hop_length=self.hop_length,
      win_length=self.win_length,
      window=self.window,
      center=False,
      return_complex=True,
    )
    return torch.log10(torch.clamp(torch.abs(fft), min=EPS))


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
  fft = Audio2Spec(N_FFT, HOP_LEN, WIN_LEN, SR)

  wav = torch.randn([4, 1, args.seq_len])
  print('wav.shape:', wav.shape)
  mel = fft(wav)
  print('mel.shape:', mel.shape)
  wav_hat = netG(mel)
  print('wav_hat.shape:', wav_hat.shape)
  scores = netD(wav_hat)
  print([[e.shape for e in d] for d in scores])
