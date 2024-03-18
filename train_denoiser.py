#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/03/05

# log10(stft(QZ)) -> log10(stft(CZ))

import sys
from argparse import ArgumentParser

import torch.nn.functional as F
from torch.optim import Optimizer, Adam
from lightning import LightningModule, Trainer, seed_everything
from torchmetrics.regression import MeanAbsoluteError

from data import SignalDataset, DataLoader, make_split
from models import DenoiseModel, Audio2Spec
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
    self.train_mae = None
    self.valid_mae = None

  def setup_train_args(self, args):
    self.args = args
    self.epochs = args.epochs
    self.lr = args.lr
    self.train_mae = MeanAbsoluteError()
    self.valid_mae = MeanAbsoluteError()

  def configure_optimizers(self) -> Optimizer:
    return Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)

  def on_after_backward(self):
    found_nan_or_inf = False
    for param in self.parameters():
      if not found_nan_or_inf: break
      if param.grad is not None:
        found_nan_or_inf = torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
    if found_nan_or_inf:
      self.zero_grad()
      print(f'>> detected inf or nan values in gradients. not updating model parameters')
      breakpoint()

  def get_losses(self, batch:Tuple[Tensor], prefix:str) -> Tensor:
    x_wav, y_wav, ids = batch
    x_spec = self.fft(x_wav)[:, :-1, :]
    y_spec = self.fft(y_wav)[:, :-1, :]

    if 'debug gradient NaN':
      self.x_wav, self.y_wav, self.ids = x_wav, y_wav, ids
      self.x_spec, self.y_spec = x_spec, y_spec
      if any([
        torch.isnan(x_wav).any(),  torch.isinf(x_wav).any(),
        torch.isnan(y_wav).any(),  torch.isinf(y_wav).any(),
        torch.isnan(x_spec).any(), torch.isinf(x_spec).any(),
        torch.isnan(y_spec).any(), torch.isinf(y_spec).any(),
      ]):
        print(f'>> detected inf or nan values in inputs')
        breakpoint()

    output = self.model(x_spec, ids)
    loss = F.l1_loss(output, y_spec)

    if prefix == 'train':
      self.train_mae(output, y_spec)
      self.log('train/mae', self.train_mae, on_step=True, on_epoch=True)
      self.log('train/loss', loss.item(), on_step=True, on_epoch=True)
    else:
      self.valid_mae(output, y_spec)
      self.log('valid/mae', self.valid_mae, on_step=False, on_epoch=True)
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
  trainloader = DataLoader(SignalDataset(trainset, n_seg=N_SEG, aug=True),  args.batch_size, shuffle=True,  drop_last=True,  **dataloader_kwargs)
  validloader = DataLoader(SignalDataset(validset, n_seg=N_SEG, aug=False), args.batch_size, shuffle=False, drop_last=False, **dataloader_kwargs)

  ''' Model & Optim '''
  model = DenoiseModel()
  lit = LitModel(model)
  if args.load:
    lit = LitModel.load_from_checkpoint(args.load, model=model)
  lit.setup_train_args(args)

  ''' Train '''
  trainer = Trainer(
    max_epochs=args.epochs,
    precision='32',
    benchmark=True,
    enable_checkpointing=True,
    log_every_n_steps=5,
  )
  trainer.fit(lit, trainloader, validloader)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-B', '--batch_size', type=int, default=16)
  parser.add_argument('-E', '--epochs',     type=int, default=7000)
  parser.add_argument('-lr', '--lr',        type=eval, default=1e-4)
  parser.add_argument('--load', type=Path, help='ckpt to resume from')
  parser.add_argument('--seed', type=int, default=114514)
  args = parser.parse_args()

  train(args)
