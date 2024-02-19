#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/19

import sys
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch.optim import Optimizer, SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from lightning import LightningModule, Trainer, seed_everything
from torchmetrics.regression import MeanAbsoluteError


from data import *
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
    self.mae = None
    self.envolope_extractor = EnvolopeExtractor()

  def setup_train_args(self, args):
    self.args = args
    self.epochs = args.epochs
    self.lr = args.lr
    self.mae = MeanAbsoluteError()

  def configure_optimizers(self) -> Optimizer:
    optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs, verbose=True)
    return {
      'optimizer': optimizer,
      'lr_scheduler': scheduler,
    }

  def optimizer_step(self, epoch:int, batch_idx:int, optim:Optimizer, optim_closure:Callable):
    super().optimizer_step(epoch, batch_idx, optim, optim_closure)
    if batch_idx % 10 == 0:
      self.log_dict({f'lr-{i}': group['lr'] for i, group in enumerate(optim.param_groups)})

  def get_losses(self, batch:Tuple[Tensor], batch_idx:int) -> Tuple[Tensor, Dict[str, float]]:
    x, y = batch

    output = self.model(x)
    envolope = self.envolope_extractor(y)
    loss = F.mse_loss(output, envolope)

    if batch_idx % 100 == 0:
      with torch.no_grad():
        self.mae(output, envolope)
        log_dict = {
          'l_total': loss.item(),
        }
    else: log_dict = None
    return loss, log_dict

  def on_train_epoch_end(self):
    self.log('train/mae/epoch', self.mae, on_step=False, on_epoch=True)

  def on_validation_epoch_end(self):
    self.log('valid/mae/epoch', self.mae, on_step=False, on_epoch=True)

  def log_losses(self, log_dict:Dict[str, float], prefix:str='log'):
    self.log_dict({f'{prefix}/{k}': v for k, v in log_dict.items()})

  def training_step(self, batch:Tuple[Tensor], batch_idx:int) -> Tensor:
    loss, log_dict = self.get_losses(batch, batch_idx)
    if log_dict: self.log_losses(log_dict, 'train')
    return loss

  def validation_step(self, batch:Tuple[Tensor], batch_idx:int) -> Tensor:
    loss, log_dict = self.get_losses(batch, batch_idx)
    if log_dict: self.log_losses(log_dict, 'valid')
    return loss


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
  parser.add_argument('-B', '--batch_size', type=int, default=16)
  parser.add_argument('-E', '--epochs',     type=int, default=35)
  parser.add_argument('-lr', '--lr',        type=eval, default=2e-4)
  parser.add_argument('--load', type=Path, help='ckpt to resume from')
  parser.add_argument('--seed', type=int, default=114514)
  args = parser.parse_args()

  train(args)
