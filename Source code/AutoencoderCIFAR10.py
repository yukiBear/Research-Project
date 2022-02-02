# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 14:21:53 2020

@author: LENOVO
"""
"""
Autoencoder for CIFAR10.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mish as mh
import math
# 1500 1000 200
batch_size = 50
class AutoencoderCIFAR10(nn.Module):
    def __init__(self):
        super(AutoencoderCIFAR10, self).__init__()
        self.mishAct = mh.Mish()
        self.conv1 = nn.Conv2d(3, 12, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(12, 24, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(24, 48, 4, stride=2, padding=1)

        self.conv4 = nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1)
        self.conv5 = nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1)
        self.conv6 = nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1)

    def forward(self, x):
        x = self.mishAct(self.conv1(x))
        x = self.mishAct(self.conv2(x))
        encoded = self.mishAct(self.conv3(x))
        
        x = self.mishAct(self.conv4(encoded))
        x = self.mishAct(self.conv5(x))
        decoded = torch.sigmoid(self.conv6(x))
        return encoded, decoded
    
    def encode(self, x):
        x = self.mishAct(self.conv1(x))
        x = self.mishAct(self.conv2(x))
        encoded = self.mishAct(self.conv3(x))
        
        return encoded
    def decode(self, x):
        x = self.mishAct(self.conv4(x))
        x = self.mishAct(self.conv5(x))
        decoded = torch.sigmoid(self.conv6(x))
        
        return decoded