#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 10:42:57 2024

@author: user
"""
import numpy as np
import torch

from network_bf_unet3D import BF_UNet as model1
from network_RCAN3D import RCAN3D as model2
from network_RCAU3D import RCAU_test3 as model3


net1 = model1(in_nc=1, out_nc=1)
net2 = model2(in_nc=1, out_nc=1)
net3 = model3(in_channels=1,num_classes=1)
print(sum(p.numel() for p in net1.parameters() if p.requires_grad))
print(sum(p.numel() for p in net2.parameters() if p.requires_grad))
print(sum(p.numel() for p in net3.parameters() if p.requires_grad))