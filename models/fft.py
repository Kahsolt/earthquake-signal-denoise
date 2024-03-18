#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/20  

from models.utils import *


class Audio2Spec(nn.Module):

  def __init__(self, n_fft=N_FFT, hop_length=HOP_LEN, win_length=WIN_LEN, device=device):
    super().__init__()

    self.n_fft = n_fft
    self.hop_length = hop_length
    self.win_length = win_length
    self.window = torch.hann_window(win_length).float().to(device)

  def forward(self, audio:Tensor, ret_mag:bool=True, ret_phase:bool=False) -> Union[Tensor, List[Tensor]]:
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
    ret = []
    if ret_mag:     # logS
      ret.append(torch.log10(torch.clamp(torch.abs(fft), min=EPS)))
    if ret_phase:   # phase: e^iθ
      ret.append(torch.exp(1.0j * torch.angle(fft)))
    return ret[0] if len(ret) == 1 else ret


if __name__ == '__main__':
  fft = Audio2Spec(N_FFT, HOP_LEN, WIN_LEN, device='cpu')

  wav = torch.randn([4, 1, NLEN])
  print('wav.shape:', wav.shape)
  spec = fft(wav)
  print('spec.shape:', spec.shape)

  wav = torch.randn([4, 1, N_SEG])
  print('wav.shape:', wav.shape)
  spec = fft(wav)
  print('spec.shape:', spec.shape)
