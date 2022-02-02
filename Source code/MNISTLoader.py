# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 10:43:44 2021

@author: LENOVO
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import random

path = '/files/'

# this code is used to make sure each batch has samples from all classes
class MNISTLoader:
    def LoadMNIST(self, train = True):
        train_data = torchvision.datasets.MNIST('../Datasets/', train=train, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor()
                                 ]))
        return train_data
    
    def WeightedSampler(self, labels):
        index = []
        validation_idx = []
        for i in range(0, 10):
            idx = np.where(labels == i)
            index.append(idx[0][0:-1])
            validation_idx.append(idx[0][0:-1])
        return index, validation_idx
    
    
    def GenerateBatches(self, data, labels, index, batch_size):
        n_batch = int(len(data) / batch_size)
        n_class = len(np.unique(labels))
        
        # number of samples per class per batch
        n_samples = int(batch_size / n_class)
        
        batches = []
        for i in range(0, n_class):
            random.shuffle(index[i])
        for i in range (0, n_batch):
            batch_samples = []
            for j in range (0, 10):
                if (i * n_samples < len(index[j]) and (i + 1) * n_samples <= len(index[j])):
                    batch_samples.extend(index[j][(i * n_samples) : ((i + 1) * n_samples)])
            # shuffle this batch
            random.shuffle(batch_samples)
            if(len(batch_samples) == n_class * n_samples):
                batches.append(batch_samples)
        
        return batches
        
    def GetBatchesAndData(self, train = True, batch_size = 50):
        train = self.LoadMNIST(train)
        train_data = train.data.numpy().astype(np.float32)
        print("Training data len: ", len(train_data))
        
        self.train_data = train_data
        train_targets = train.train_labels.numpy().astype(np.float32)
        self.train_targets = train_targets
        # convert them into float numpy array
        index, validation_idx = self.WeightedSampler(train_targets)
        self.index = index
        
        
        n = 0
        for i in range(0, len(index)):
            n = n + len(np.unique(index[i]))
        print(n)
        
        batches = self.GenerateBatches(train_data, train_targets, index, batch_size)
        
        # use a part of batches
        batches = batches
        
        return batches, torch.tensor(train_data / 255), torch.FloatTensor(train_targets), validation_idx

    def ShuffleBatches(self, batch_size):
        new_batches = self.GenerateBatches(self.train_data, self.train_targets, self.index, batch_size)
        
        
        return new_batches
