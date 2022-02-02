# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 14:21:53 2020

@author: LENOVO
"""
"""
Autoencoder for MNIST.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mish as mh
import math
batch_size = 50
class AutoEncoder(nn.Module):
    def __init__(self): 
        self.inputD = 784
        self.o1 = 512
        self.o2 = 264
        self.o3 = 128
        self.classDistributions = dict()
        
        super(AutoEncoder, self).__init__()
        self.enFC1 = nn.Linear(self.inputD, self.o1, bias = False)
        self.norm1 = nn.LayerNorm(self.o1)
        self.drop1 = nn.Dropout()
        self.enFC2 = nn.Linear(self.o1, self.o2, bias = False)
        self.norm2 = nn.LayerNorm(self.o2)
        self.drop2 = nn.Dropout()
        self.enFC3 = nn.Linear(self.o2, self.o3)
        self.normh = nn.LayerNorm(self.o3)
        
        self.deFC1 = nn.Linear(self.o3, self.o2, bias = False)
        self.norm3   = nn.LayerNorm(self.o2)
        self.drop3 = nn.Dropout()
        self.deFC2 = nn.Linear(self.o2, self.o1, bias = False)
        self.norm4  = nn.LayerNorm(self.o1)
        self.drop4 = nn.Dropout()
        self.deFC3 = nn.Linear(self.o1, self.inputD)
        
        self.mishAct = mh.Mish()
        
        self.class_means = []
        for i in range(10):
            self.class_means.append(-1)
        self.nsample_per_cluster = 0
    
    def forward(self, x):
        # encoder
        x = self.mishAct(self.norm1(self.enFC1(x)))
        #x = self.drop1(x)

        x = self.mishAct(self.norm2(self.enFC2(x)))
        #x = self.drop2(x)
        
        x_latent = self.mishAct(self.normh(self.enFC3(x)))
        #x_latent = F.relu(self.normh(self.enFC3(x)))
        #x = self.drop3(x)
        # decoder
        x = self.mishAct(self.norm3(self.deFC1(x_latent)))
        
        
        x = self.mishAct(self.norm4(self.deFC2(x)))
        #x = self.drop4(x)
        
        x = torch.sigmoid(self.deFC3(x))
        
        return x, torch.sum(torch.abs(x_latent))
    
    def encode(self, x):
        x = self.mishAct(self.norm1(self.enFC1(x)))
        #x = self.drop1(x)
        #x = x.mul(0.5)
        
        x = self.mishAct(self.norm2(self.enFC2(x)))
        #x = x.mul(0.5)
        #x = self.drop2(x)
        
        x = self.mishAct(self.normh(self.enFC3(x)))
        #x = F.relu(self.normh(self.enFC3(x)))
        return x
    
    def decode(self, x):
        #x = self.drop3(x)
        x = self.mishAct(self.norm3(self.deFC1(x)))
        #x = x.mul(0.5)
        
        
        x = self.mishAct(self.norm4(self.deFC2(x)))
        #x = x.mul(0.5)
        #x = self.drop4(x)
        
        x = torch.sigmoid(self.deFC3(x))
        return x
    
    # l1-norm of outputs of activation functions
    # each sample should use as less as features as possible
    # therefore these are the real important features for this sample
    def sparseLoss(self, x):
        o = self.encode(x)
        loss = torch.sum(torch.abs(o))
        
        return loss
        

    # not used
    # variance of times of each hidden node get fired
    # different samples should use different features
    # the more features are usd by a batch of samples the better
    # therefore the new feature space can model the problem well
    def batchActivationLoss(self, batch):
        #loss = torch.cuda.FloatTensor(200)
        acts = torch.zeros(self.o3).to('cuda')
        for i in range(0, len(batch)):
            o = self.encode(batch[i])
            acts.add_(torch.abs(o[0]))
        return acts
    
    # not used
    def GaussianLoss(self, batch, labels):
        loss = torch.zeros(self.o3).to('cuda')
        for i in range(0, len(batch)):
            o = self.encode(batch[i])
            label = labels[i]
            (mean,var) = self.classDistributions[label]
            loss.add_(o.minus(mean))
        return loss
    
    
    # variance of mean of clusters using label information
    def batchVarianceLoss(self, batch, labels):
        acts = torch.zeros(self.o3).to('cuda')
        class_means = []
        for i in range(0, 10):
            class_means.append([])
        for i in range(0, len(batch)):
            o = self.encode(batch[i])
            #print("Label is: ", labels[i])
            class_means[(int)(labels[i].item())].append(o[0])
        
        # calculate means for each class
        final_means = []
        # variance inside a class
        class_variance = torch.zeros(1).to('cuda')
        for i in range(10):
            final_means.append(torch.zeros(self.o3).to('cuda'))
            class_data = torch.cat(class_means[i], dim = 0).reshape(-1, self.o3)
            #print("Class data var: ", torch.var(class_data, dim = 0))
            temp_tensor1 = torch.var(class_data, dim = 0)
            temp_tensor1.mul_(len(class_means[i]) - 1)
            # temp_tensor is the sum of distances to mean
            temp_tensor = torch.pow(temp_tensor1, 0.5)
            temp_tensor.div_(len(class_means[i]) - 1)
            class_variance.add_(torch.sum(temp_tensor))
            # for each sample of class i
            for j in range(len(class_means[i])):
                final_means[i].add_(class_means[i][j])
            final_means[i].div_(len(class_means[i]))
        
        # calculate variance for these means
        var_tensor = torch.cat(final_means, dim=0).reshape(-1, self.o3)
        
        result = torch.var(var_tensor, dim=0)
        
        
        result = torch.pow(result.mul(len(class_means) - 1), 0.5)
        return torch.sum(result.div(len(class_means) - 1)), class_variance
        #return torch.sum(result), class_variance
        
        
        
    # not used, build mixture gaussian distrubution in latent space    
    # variance of mean of clusters using label information
    def ClassVarianceLoss(self, batch, labels):
        acts = torch.zeros(self.o3).to('cuda')
        class_means = []
        for i in range(0, 10):
            class_means.append([])
        for i in range(0, len(batch)):
            o = self.encode(batch[i])
            #print("Label is: ", labels[i])
            class_means[(int)(labels[i].item())].append(o[0])
        
        # calculate means for each class
        final_mean_error = torch.zeros(1)
        class_variance = torch.zeros(1).to('cuda')
        for i in range(10):
            class_data = torch.cat(class_means[i], dim = 0).reshape(-1, self.o3)
            
            class_mean = torch.mean(class_data, dim = 0)
            # distance between class mean and batch mean
            if (isinstance(self.class_means[i], int)):
                self.class_means[i] = class_mean
            else:
                final_mean_error.add_(
                    torch.cdist(self.class_means[i], class_mean))
                print(final_mean_error)
            # Recalculate class means
            self.class_means[i].mul_(self.nsample_per_cluster)
            
            
            self.class_means[i].add_(class_mean)
            self.nsample_per_cluster += 50
            self.class_means[i].div_(self.nsample_per_cluster)
        #print(self.class_means)
            # Variance of class means as penalty
        var_error = torch.var(
            torch.cat(self.class_means).reshape(-1, self.o3), dim = 0)
        tmp_var = var_error.mul(len(class_means) - 1)
        tmp_var = tmp_var.pow(0.5)
        tmp_var = tmp_var.div(len(class_means) - 1)
        tmp_var = torch.sum(tmp_var)
        print(final_mean_error)
        return final_mean_error
    