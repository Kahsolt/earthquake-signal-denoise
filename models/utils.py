#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/03/08  

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations

from utils import *

remove_weight_norm = lambda mod: remove_parametrizations(mod, 'weight')


def init_weights(m:nn.Conv2d, mean:float=0.0, std:float=0.01):
  classname = m.__class__.__name__
  if 'Conv' in classname:
    m.weight.data.normal_(mean, std)

def get_padding_strided(kernel_size:int, stride:int=2):
  return (kernel_size - stride) // 2

def get_padding_dilated(kernel_size:int, dilation:int=1):
  return int((kernel_size * dilation - dilation) / 2)
