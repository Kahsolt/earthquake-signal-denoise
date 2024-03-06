@ECHO OFF

:preprocess
python mk_data.py

:train
python train_envolope.py -E 35
python train_denoiser.py -E 1500

:infer
python infer.py ^
  --load_E lightning_logs\version_3\checkpoints\epoch=34-step=4340.ckpt ^
  --load_D lightning_logs\version_2\checkpoints\epoch=999-step=124000.ckpt
