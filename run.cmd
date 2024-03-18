@ECHO OFF

:preprocess
python mk_data.py


:: simple denoiser
:train
python train_denoiser.py -E 3000

:infer
python infer.py --load lightning_logs\version_1\checkpoints\epoch=96-step=12028.ckpt


:: melgan
:train
python train_melgan.py -M melgan --save_path lightning_logs\melgan
python train_melgan.py -M melgan-te --save_path lightning_logs\melgan-te

:infer
python infer_melgan.py --load lightning_logs\melgan


:: melgan-ae
:train
python train_melgan_ae.py --save_path lightning_logs\melgan_ae

:infer
python infer_melgan_ae.py --load lightning_logs\melgan_ae
