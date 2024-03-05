#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/19

import sys
from argparse import ArgumentParser

import torch.nn.functional as F
from torch.optim import Optimizer, SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from lightning import LightningModule, Trainer, seed_everything
from torchmetrics.regression import MeanAbsoluteError

from data import SignalDataset, DataLoader
from model_envolope import EnvolopeModel, EnvolopeExtractor
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
    self.envolope_extractor = EnvolopeExtractor()

  def setup_train_args(self, args):
    self.args = args
    self.epochs = args.epochs
    self.lr = args.lr
    self.train_mae = MeanAbsoluteError()
    self.valid_mae = MeanAbsoluteError()

  def configure_optimizers(self) -> Optimizer:
    optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
    return optimizer
    #scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs, verbose=True)
    #return {
    #  'optimizer': optimizer,
    #  'lr_scheduler': scheduler,
    #}

  def optimizer_step(self, epoch:int, batch_idx:int, optim:Optimizer, optim_closure:Callable):
    super().optimizer_step(epoch, batch_idx, optim, optim_closure)
    if batch_idx % 10 == 0:
      self.log_dict({f'lr/{i}': group['lr'] for i, group in enumerate(optim.param_groups)})

  def get_losses(self, batch:Tuple[Tensor], prefix:str) -> Tuple[Tensor, Dict[str, float]]:
    x, y = batch
    output = self.model(x)
    envolope = self.envolope_extractor(y)
    loss = F.mse_loss(output, envolope)

    if prefix == 'train':
      self.train_mae(output, envolope)
      self.log('train/mae', self.train_mae, on_step=True, on_epoch=True)
      self.log('train/loss', loss.item(), on_step=True, on_epoch=True)
    else:
      self.valid_mae(output, envolope)
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
  dataset_cls = SignalDataset
  dataloader_kwargs = {
    'num_workers': 0,
    'persistent_workers': False,
    'pin_memory': True,
  }
  trainloader = DataLoader(dataset_cls('train'), args.batch_size, shuffle=True,  drop_last=True,  **dataloader_kwargs)
  validloader = DataLoader(dataset_cls('valid'), args.batch_size, shuffle=False, drop_last=False, **dataloader_kwargs)

  ''' Model & Optim '''
  model = EnvolopeModel()
  lit = LitModel(model)
  lit.setup_train_args(args)
  if args.load:
    lit = LitModel.load_from_checkpoint(args.load, model=model)
  lit.setup_train_args(args)

  ''' Train '''
  trainer = Trainer(
    max_epochs=args.epochs,
    precision='16-mixed',
    benchmark=True,
    enable_checkpointing=True,
  )
  trainer.fit(lit, trainloader, validloader)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-B', '--batch_size', type=int, default=64)
  parser.add_argument('-E', '--epochs',     type=int, default=30)
  parser.add_argument('-lr', '--lr',        type=eval, default=2e-4)
  parser.add_argument('--load', type=Path, help='ckpt to resume from')
  parser.add_argument('--seed', type=int, default=114514)
  args = parser.parse_args()

  train(args)
