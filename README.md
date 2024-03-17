# earthquake-signal-denoise

    Contest solution for 2023中国华录杯数据湖算法大赛 - 地震预警站网记录波形降噪分析算法赛

----

Contest page: [http://practice.tanzhonghedasai.com/top/index.html#/good?id=1725046816823468033](http://practice.tanzhonghedasai.com/top/index.html#/good?id=1725046816823468033)  
Team Name: 御霊一片祈八百万救给  


### Quickstart

- install pytorch
- `pip install -r requirements.txt`
- refer to `run.cmd`


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


#### refenrence

- 官方赛题解读: [https://pan.baidu.com/s/14Se5Kc4iRGIhVHWaymTHWA?pwd=2pgf](https://pan.baidu.com/s/14Se5Kc4iRGIhVHWaymTHWA?pwd=2pgf)
- General-Cross-Validation-denoising-Forward: [https://github.com/smousavi05/General-Cross-Validation-denoising-Forward](https://github.com/smousavi05/General-Cross-Validation-denoising-Forward)
- Denoising_GCVwavaletF: [https://github.com/tanxn5/Denoising_GCVwavaletF](https://github.com/tanxn5/Denoising_GCVwavaletF)

----
by Armit
2024/2/16 
