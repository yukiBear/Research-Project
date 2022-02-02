# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 16:35:13 2021

@author: LENOVO
"""
"""
Generate attacks with other methods with foolbox.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
from torchvision.datasets import CIFAR10
import foolbox as fb
import eagerpy as ep
import time
from PyTorch_CIFAR10_master.cifar10_models.vgg import vgg11_bn

# model under attack
mua = vgg11_bn(pretrained = True).to('cuda')


mean = [0.4914, 0.4822, 0.4465]
std = [0.2471, 0.2435, 0.2616]
mean_tensor = torch.tensor(mean).to('cuda')
std_tensor = torch.tensor(std).to('cuda')

train_data = CIFAR10('../Datasets/CIFAR10/', train = True,
                               transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                                ]), 
                                  download = True)

training_data = []
training_targets = []
for i in range(0, 6):
    training_data.append(train_data[i][0].numpy())
    training_targets.append(train_data[i][1])

#%% #prepare seeds
images = torch.tensor(training_data).reshape(-1, 3, 32, 32).float().to('cuda')
labels = torch.tensor(training_targets).to('cuda')
preprocessing = dict(mean=mean, std=std, axis=-3)

images = ep.astensors(images)[0]
labels = ep.astensors(labels)[0]

#%% choose an attack
#attack = fb.attacks.LinfFastGradientAttack()
attack = fb.attacks.LinfBasicIterativeAttack()
#attack = fb.attacks.PGD()
#attack = fb.attacks.BoundaryAttack()
#attack = fb.attacks.L2CarliniWagnerAttack()
#attack = fb.attacks.LinfDeepFoolAttack()
#attack = fb.attacks.NewtonFoolAttack()
#attack = fb.attacks.LinfinityBrendelBethgeAttack()

fmodel = fb.PyTorchModel(mua.eval(), bounds = (0, 1), preprocessing=preprocessing)

epsilons = [0.1]

#images, labels = ep.astensors(*fb.samples(fmodel, dataset="mnist", batchsize=16) )
startTime = time.time()
raw_advs, advs, success = attack(fmodel, images, labels, epsilons = epsilons)
endTime = time.time()
total_time = endTime - startTime
#%% evalutaion
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

label_names = ['airplane',
'automobile',
'bird',
'cat',
'deer',
'dog',
'frog',
'horse',
'ship',
'truck']

robust_accuracy = 1 - success.float32().mean(axis=-1)
print("Robust acc: ", robust_accuracy.raw)
print("attack success rate: ", 1 - robust_accuracy.raw)

raw_acc = fb.accuracy(fmodel, images, labels)
print("Acc on unmodified: ", raw_acc)
perturbation_sizes = (advs[0] - images).norms.l2(axis=(1, 2, 3)).numpy()
print("perturbation: ", np.mean(perturbation_sizes) / (32 * 32))
print('average time:', total_time / 100)

#%% plot attacks
ndisplay = 0
count= 0
for adv_img in raw_advs[0]:
    print("raw max: ", torch.max(adv_img.raw))
    fig = plt.figure(figsize=(10, 10))
    orig = torch.tensor(training_data[count])
    predicted = F.softmax(mua(images[count].raw.view(32, 32, 3).sub(mean_tensor).div(std_tensor).view(1, 3, 32, 32)))
    label = predicted.data.max(1, keepdim = True)[1][0].item()
    prob = predicted.data.max(1, keepdim = True)[0][0].item()
    title = label_names[label] + ": " + "{con:}".format(con = str(prob)[0:7])
    plt.imshow(orig.view(3, 32, 32).T)
    plt.title(title, fontsize = 50)
    plt.xticks([])
    plt.yticks([])
    fig.show()
    
    # plot attack
    temp_img = adv_img.raw
    print(torch.min(temp_img), torch.max(temp_img))
    adv_normlized = temp_img.view(32, 32, 3).sub(mean_tensor).div(std_tensor)
    pre = mua(adv_normlized.view(1, 3, 32, 32)).data.max(1, keepdim = True)[1][0].item()
    print("predicted ", pre)
    print("real label: ", labels[count])
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(temp_img.detach().cpu().numpy().T)
    predicted = F.softmax(mua(adv_img.raw.view(32, 32, 3).sub(mean_tensor).div(std_tensor).view(1, 3, 32, 32)))
    label = predicted.data.max(1, keepdim = True)[1][0].item()
    prob = predicted.data.max(1, keepdim = True)[0][0].item()
    title = label_names[label] + ": " + "{con:}".format(con = str(prob)[0:7])
    plt.xticks([])
    plt.yticks([])
    plt.title(title,  fontsize=50)
    
    fig.show()
    count += 1
    if count > ndisplay:
        break

