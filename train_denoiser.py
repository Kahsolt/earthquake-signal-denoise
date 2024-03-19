#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/03/05

# log10(stft(QZ)) -> log10(stft(CZ))

import sys
from argparse import ArgumentParser

import torch.nn.functional as F
from torch.optim import Optimizer, Adam
from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

from data import SignalDataset, DataLoader, make_split
from models import DenoiseModel, GeneratorAE, Audio2Spec
from utils import *


class LitModel(LightningModule):

  def __init__(self, model:Model):
    super().__init__()

    self.model = model
    self.fft = Audio2Spec(N_FFT, HOP_LEN, WIN_LEN, device)

    # ↓↓ training specified ↓↓
    self.args = None
    self.epochs = -1
    self.lr = 2e-4

  def setup_train_args(self, args):
    self.args = args
    self.epochs = args.epochs
    self.lr = args.lr

  def configure_optimizers(self) -> Optimizer:
    return Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)

  def get_losses(self, batch:Tuple[Tensor], prefix:str) -> Tensor:
    x_wav, y_wav, ids = batch
    x_spec = self.fft(x_wav)
    y_spec = self.fft(y_wav)

    if self.args.model == 'simple':
      x_spec = x_spec[:, :-1, :]
      y_spec = y_spec[:, :-1, :]
      output = self.model(x_spec, ids)
    elif self.args.model == 'melgan_ae':
      output = self.model(x_spec)
    loss = F.l1_loss(output, y_spec)

    if prefix == 'train':
      self.log('train/loss', loss.item(), on_step=True, on_epoch=True)
    else:
      self.log('valid/loss', loss.item(), on_step=False, on_epoch=True)

    return loss

  def training_step(self, batch:Tuple[Tensor], batch_idx:int) -> Tensor:
    return self.get_losses(batch, 'train')

  def validation_step(self, batch:Tuple[Tensor], batch_idx:int) -> Tensor:
    return self.get_losses(batch, 'valid')


def train(args):
  seed_everything(args.seed)
  print('>> cmd:', ' '.join(sys.argv))
  print('>> args:', vars(args))

  ''' Data '''
  dataloader_kwargs = {
    'num_workers': 0,
    'persistent_workers': False,
    'pin_memory': True,
  }
  X, Y = get_data_train()
  trainset, validset = make_split(X, Y, ratio=0.01)
  n_seg = N_SEG if args.model == 'simple' else -1
  trainloader = DataLoader(SignalDataset(trainset, n_seg=n_seg, aug=True),  args.batch_size, shuffle=True,  drop_last=True,  **dataloader_kwargs)
  validloader = DataLoader(SignalDataset(validset, n_seg=n_seg, aug=False), args.batch_size, shuffle=False, drop_last=False, **dataloader_kwargs)

  ''' Model & Optim '''
  if args.model == 'simple':
    model = DenoiseModel()
  elif args.model == 'melgan_ae':
    model = GeneratorAE(N_SPEC, 32, 3)
  lit = LitModel(model)
  if args.load:
    lit = LitModel.load_from_checkpoint(args.load, model=model)
  lit.setup_train_args(args)

  ''' Train '''
  model_ckpt_callback = ModelCheckpoint(monitor='valid/loss', mode='min')
  trainer = Trainer(
    max_epochs=args.epochs,
    precision='32',
    benchmark=True,
    enable_checkpointing=True,
    log_every_n_steps=5,
    callbacks=[model_ckpt_callback],
  )
  trainer.fit(lit, trainloader, validloader)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-M', '--model', default='melgan_ae', choices=['simple', 'melgan_ae'])
  parser.add_argument('-B', '--batch_size', type=int, default=16)
  parser.add_argument('-E', '--epochs',     type=int, default=7000)
  parser.add_argument('-lr', '--lr',        type=eval, default=1e-4)
  parser.add_argument('--load', type=Path, help='ckpt to resume from')
  parser.add_argument('--seed', type=int, default=114514)
  args = parser.parse_args()

  train(args)
