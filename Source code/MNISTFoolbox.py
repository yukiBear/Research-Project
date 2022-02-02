# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 15:37:29 2021

@author: LENOVO
"""
"""
Run this file to generate attacks with other methods.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision

import foolbox as fb
import eagerpy as ep
import time

train_data = torchvision.datasets.MNIST('../Datasets/', train=True, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor()
                                 ]))
# normlize data
training_data = train_data.data.float().div(255)
training_targets = train_data.train_labels
# load the model under attack
model_path = '../Models/MUA.pt'
nua = torch.load(model_path)
nua.eval()

#%%
# select an attack to use, some of them may take minutes to run

#attack = fb.attacks.LinfFastGradientAttack()
attack = fb.attacks.LinfBasicIterativeAttack()
#attack = fb.attacks.PGD()
#attack = fb.attacks.BoundaryAttack()
#attack = fb.attacks.L2CarliniWagnerAttack()
#attack = fb.attacks.LinfDeepFoolAttack()
#attack = fb.attacks.NewtonFoolAttack()
#attack = fb.attacks.LinfinityBrendelBethgeAttack()

fmodel = fb.PyTorchModel(nua.eval(), bounds=(0, 1))
images = training_data[0:6].view(-1, 1, 28, 28).to('cuda')
labels = training_targets[0:6].long().to('cuda')

images = ep.astensors(images)[0]
labels = ep.astensors(labels)[0]

epsilons = [0.25]

startTime = time.time()
raw_advs, advs, success = attack(fmodel, images, labels, epsilons = epsilons)
endTime = time.time()
total_time = endTime - startTime

#%% accuracy of nua
import matplotlib.pyplot as plt
import numpy as np
robust_accuracy = 1 - success.float32().mean(axis=-1)
print("Robust acc: ", robust_accuracy.raw)
print("attack success rate: ", 1 - robust_accuracy.raw)

raw_acc = fb.accuracy(fmodel, images, labels)
print("Acc on unmodified: ", raw_acc)

n_display = 1
count = 0
for adv_img in raw_advs[0]:
    fig = plt.figure(figsize=(6, 6))
    predicted = F.softmax(nua(adv_img.raw.view(1, 1, 28, 28)))
    label = predicted.data.max(1, keepdim = True)[1][0].item()
    prob = predicted.data.max(1, keepdim = True)[0][0].item()
    title = "Predicted as: " + str(label) + "\nConfidence: " + "{con:}".format(con = str(prob)[0:6])
    if (int(label) == int(labels.raw[count].item())):
        color = 'g'
    else:
        color = 'r'
    plt.imshow(adv_img.raw.detach().cpu().numpy().reshape(28, 28), cmap = 'gray')
    plt.title(title, fontsize = 30, color = color)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    count += 1
    if (count >= n_display):
        break
#plt.plot(epsilons, robust_accuracy.numpy())
#plt.show()

perturbation_sizes = (raw_advs[0] - images).norms.l2(axis=(1, 2, 3)).numpy()

print('mean l2 perturbation: ', np.mean(perturbation_sizes))
print('per pixel perturbation: ', np.mean(perturbation_sizes) / (28 * 28))
print('average running time:', total_time / len(images))