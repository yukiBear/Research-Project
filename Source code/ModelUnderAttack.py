# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 14:50:29 2020

@author: LENOVO
"""
"""
Target model for MNIST.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mish as mish

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.cov1 = nn.Conv2d(1, 32, kernel_size = 5)
        
        self.cov2 = nn.Conv2d(32, 64, kernel_size = 5)
        
        self.fc1 = nn.Linear(4 * 4 * 64, 50)
        self.fc2 = nn.Linear(50, 10)
        self.m = mish.Mish()
         
    def forward(self, x):
        x = torch.max_pool2d(self.m(self.cov1(x)), 2)
        x = torch.max_pool2d(self.m(self.cov2(x)), 2)
        x = x.view(-1, 4 * 4 * 64)
        x = self.m(self.fc1(x))
        x = self.fc2(x)
        
        return F.log_softmax(x)
    
class SimpleMLPOnLatentCode(nn.Module):
    def __init__(self):
        super(SimpleMLPOnLatentCode, self).__init__()
        self.fc1 = nn.Linear(128, 10)
        #self.fc2 = nn.Linear(64, 10)
        self.m = mish.Mish()
         
    def forward(self, x):
        x = self.fc1(x)
        
        
        return F.log_softmax(x)
    