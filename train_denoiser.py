#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/03/05

import sys
from argparse import ArgumentParser

import torch.nn.functional as F
from torch.optim import Optimizer, SGD, Adam
from lightning import LightningModule, Trainer, seed_everything
from torchmetrics.regression import MeanAbsoluteError

from data import SpecDataset, DataLoader
from model import DenoiseModel
from utils import *


class LitModel(LightningModule):

  def __init__(self, model:Model):
    super().__init__()

    self.model = model

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
    sel = 0
    if sel == 0:
      optim = Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
    else:
      optim = SGD(self.model.parameters(), lr=self.lr, weight_decay=1e-5, momentum=0.9)
    return optim

  def optimizer_step(self, epoch:int, batch_idx:int, optim:Optimizer, optim_closure:Callable):
    super().optimizer_step(epoch, batch_idx, optim, optim_closure)
    if batch_idx % 10 == 0:
      self.log_dict({f'lr/{i}': group['lr'] for i, group in enumerate(optim.param_groups)})

  def get_losses(self, batch:Tuple[Tensor], prefix:str) -> Tuple[Tensor, Dict[str, float]]:
    x, y, ids = batch
    output = self.model(x, ids)
    loss = F.l1_loss(output, y)

    if prefix == 'train':
      self.train_mae(output, y)
      self.log('train/mae', self.train_mae, on_step=True, on_epoch=True)
      self.log('train/loss', loss.item(), on_step=True, on_epoch=True)
    else:
      self.valid_mae(output, y)
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
  trainloader = DataLoader(SpecDataset('train', transform=wav_norm), args.batch_size, shuffle=True,  drop_last=True,  **dataloader_kwargs)
  validloader = DataLoader(SpecDataset('valid', transform=wav_norm), args.batch_size, shuffle=False, drop_last=False, **dataloader_kwargs)

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
  )
  trainer.fit(lit, trainloader, validloader)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-B', '--batch_size', type=int, default=64)
  parser.add_argument('-E', '--epochs',     type=int, default=1000)
  parser.add_argument('-lr', '--lr',        type=eval, default=2e-4)
  parser.add_argument('--load', type=Path, help='ckpt to resume from')
  parser.add_argument('--seed', type=int, default=114514)
  args = parser.parse_args()

  train(args)
