#### modeling comments

```
由于 输入QZ 和 输出CZ 的数据量级差异很大，我一度以为应该分解为三个任务来做
  - spectrogram denoising: UNet-like encoder-decoder
  - wavform reconstruction： istft / griffin-lim / melgan
  - envolope rescaling: naive CNN or UNet-like encoder-decoder

最终我还是屈服于文献搜索，发现现存工作都是直接用 AutoEncoder 在频域降噪；再仔细一想，输入和输出的频谱并不一定需要能量幂等 (可以让nn自己去rescale)，而频谱在取log之后数量级差异没有想象中那么大，我可能还是想复杂了
  - denoise: log10(stft(QZ)) -> log10(stft(CZ))
  - inv_wav: griffin-lim with init phase

相位问题：
  - 评价指标仿佛要求保相位不变
  - GAN 类方法都很难做到保相位，比如在 phase 上加判别器毫无卵用；WORLD 方法也有一些疑难杂症
  - 姑且只能考虑唯一已知的 griffin-lim 方法，即使它重构质量堪忧 :(
```


#### modeling domains note

```
noisy signal -> noisy spec -> denoised spec -> denoised signal
    x_n      ->    s_n     ->     s_h       ->      x_h
1.      log10(stft)       CNN          Griffin-Lim
2.      log10(stft)                       MelGAN  norm(x_h)
```
