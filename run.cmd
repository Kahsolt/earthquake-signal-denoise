@ECHO OFF

:preprocess
python mk_data.py

:train
python train_envolope.py -E 500
python train_denoiser.py -E 3000

python train_melgan.py --save_path lightning_logs\melgan

:infer
python infer.py ^
  --load_E lightning_logs\version_0\checkpoints\epoch=54-step=6820.ckpt ^
  --load_D lightning_logs\version_1\checkpoints\epoch=96-step=12028.ckpt
