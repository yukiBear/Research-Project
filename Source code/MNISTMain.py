# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 20:25:09 2020

@author: LENOVO
"""
"""
Main file for MNIST experiments.
A GPU is required to run this.
"""
import numpy as np
import autoencoderMNIST

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision

import random
import math

batch_size = 50
hidden_nodes = 128

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('../Datasets/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ])),
  batch_size=batch_size, shuffle=False)


#%%
# init autoencoder
net = autoencoderMNIST.AutoEncoder().to('cuda')
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr = 0.001)
sampleBatch = []

#%%
# get training data and batches
from MNISTLoader import MNISTLoader
trainLoader = MNISTLoader()
# @params Get training data or test data: Boolean
batch_idxes, training_data, training_targets, validation_idx = trainLoader.GetBatchesAndData(True, batch_size)

#%% set up training
def addPanelties(net, loss, batch_input, batch_labels, panelty_rate1, panelty_rate2, use_panelty):
    l1_activations = torch.zeros(1).to('cuda')
    
    # add sparse penalty, this works slow, dropped
    if (use_panelty == 1 or use_panelty == 3):
        for sample in batch_input:
            sparse_loss = net.sparseLoss(sample)
            l1_activations.add_(sparse_loss)
        l1_activations.mul_(panelty_rate1)        
        loss.add_(l1_activations[0])
        
    # unused penalty, still under exploration, see additional work
    if (use_panelty > 1):
        batch_loss, batch_mean_loss = net.batchVarianceLoss(batch_input, batch_labels)
        
        batch_loss.pow_(-1)
        
        print(batch_loss.mul(panelty_rate2))
        print(batch_mean_loss.mul(0.000005))
        
        loss.add_(batch_loss.mul_(panelty_rate2))
        loss.add_(batch_mean_loss.mul_(0.000005)[0])
    
    return loss

# for training
def Train(net, optimizer, criterion, 
          n_epoch, penalty_rate1, penalty_rate2, use_penalty):
    net.train()
    epoch_loss = []
    for i in range(1, n_epoch + 1):  
        print("starting epoch: " , i)
        avg_loss = torch.zeros(1).cuda()

        # batch training
        j = 0
        current_batch_idx = 0
        new_batch_idxes = trainLoader.ShuffleBatches(batch_size)
        for batch_idx in new_batch_idxes:
            
            data = training_data[batch_idx]
            batch_input = data.view(batch_size, 1, 784).to('cuda')
            current_batch_idx = current_batch_idx + 1
            
            # get the output
            out, sparse_penalty = net(batch_input)
            net.zero_grad()
            
            # calculate the loss
            loss = criterion(out, batch_input)
            avg_loss.add_(loss)
            batch_labels = training_targets[batch_idx]
            # used in additional work
            if (use_penalty == 2 or use_penalty == 3):
                loss = addPanelties(net, loss, batch_input, batch_labels, 
                                   penalty_rate1, penalty_rate2, use_penalty)
            
            # add L-1 penalty
            loss.add_(sparse_penalty.mul(penalty_rate1))
            
            loss.backward()
            optimizer.step()
            j = j + batch_size
            if (j % 5000 == 0):
                print("Finished: ", j, "samples", "loss: ", loss)
        
        # record the loss
        print("Epoch loss: ", avg_loss.item())
        epoch_loss.append(avg_loss.item() / batch_size)
        
    return epoch_loss
#%%
# start training autoencoder, or use the trained one in next section
net.train()
n_epoch = 10
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr = 0.001)
epoch_loss = Train(net, optimizer, criterion, n_epoch, 0.00001, 5, 1)
net.eval()

#%% load the autoencoder
model_path = '../Models/autoencoder_MNIST.pt'
net = torch.load(model_path)
net.eval()

#%%
# get some training samples and explore
# batches are shuffled at each running
new_batch_idxes = trainLoader.ShuffleBatches(batch_size)
data = training_data[new_batch_idxes[0]]

# data and targets to use
target = training_targets[new_batch_idxes[0]]
inputData = data.view(batch_size,1, 784).to("cuda")

(unique, counts) = np.unique(target, return_counts=True)
print("Unique values and counts: ", unique, counts)

#%% plot the activations for a training sample
import matplotlib.pyplot as plt
def PlotActivations(net, inputs, targets, All = False, targeted = False, target_label = 0):
    o = np.zeros(hidden_nodes) 
    max_value = 0
    min_value = 1
    idx = -1
    for i in inputs:
        idx += 1
        if (targeted == True):
            if (int(targets[idx]) != target_label):
                continue
        
        outImg = net.decode(net.encode(i))
        act = net.encode(i).cpu().detach().numpy()
        if (max_value < np.max(act)):
            max_value = np.max(act)
        if (min_value > np.min(act)):
            min_value = np.min(act)
        o = o + act
        
    X = range(0, hidden_nodes)
    if (All == False):
        plt.bar(X, o)
    else:
        plt.bar(X, o[0])
    plt.title("Magnutide of latent features")
    plt.xlabel("index") 
    plt.xticks([])
    plt.yticks([])
    plt.ylabel("Magnitude of feature")
    plt.show()
    print("Range of latent features: ", min_value, ", ",max_value)
    return o

def PlotLoss(epoch_loss, n_epoch):
    plt.plot(range(0, n_epoch), epoch_loss)
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('BCE Loss')
    
# plot training loss and activation
#PlotLoss(epoch_loss, n_epoch)
#plt.show()

o = PlotActivations(net, inputData[6, :], target[1], False, False, 0)
#%% add naive noises and check autoencoder's behaviour, no noise in default
n_display = 1
offset = 2
net_to_use = net

another_feature = net_to_use.encode(inputData[0]).cpu().detach().numpy().tolist()
for j in range(0 + offset, n_display + offset):
    nFeatures = net_to_use.encode(inputData[j]).cpu().detach().numpy()
    
    plt.imshow(inputData[j].cpu().detach().numpy().reshape(28,28), cmap='gray')
    plt.title("Original")
    plt.xticks([])
    plt.yticks([])
    plt.show()
    n_modified = 0
    for i in range(0, len(nFeatures[0])):
        # add features from another sample
        nFeatures[0][i] += another_feature[0][i] * 0
        if (nFeatures[0][i] < 2):
            # add noise
            nFeatures[0][i] += 0
            n_modified += 0
    newSample = net_to_use.decode(torch.tensor(nFeatures).to('cuda'))
    
    plt.imshow(inputData[0].cpu().detach().numpy().reshape(28, 28), cmap='gray')
    plt.imshow(newSample.cpu().detach().numpy().reshape(28, 28), cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title("Reconstructed")
    plt.show()
#%% train the network under attacke, or use the trained model in next section
train_loader_attackModel = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('../Datasets/', train=True, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ])),
  batch_size=64, shuffle=True)
def TrainAttackedModel(netUA, learning_rate = 0.0001, nepoch = 20):
    optimizer = optim.Adam(netUA.parameters(), lr = learning_rate)
    train_losses = []
    test_losses = []
    for i in range(0, nepoch):
        print("Starting epoch: ", i)
        train_losses.append(0)
        count = 0
        for batch_idx, (data, target) in enumerate(train_loader_attackModel):    
            optimizer.zero_grad()
            output = netUA(data.to('cuda').reshape(-1, 1, 28, 28))
            loss = F.nll_loss(output, target.to('cuda'))
            train_losses[i] += loss 
            count += 1
            loss.backward()
            optimizer.step()
        train_losses[i] /= count
        
    return train_losses

import ModelUnderAttack
nua = ModelUnderAttack.SimpleMLP().to('cuda')
t_loss = TrainAttackedModel(nua, 0.001, nepoch = 25)
#%% plot the loss of nua, if loss are collected
plt.title('Training loss of MUA')
plt.plot(t_loss)
plt.xlabel('Epoch')
plt.ylabel('NLL Loss')

#%% load the model under attack
model_path = '../Models/MUA.pt'
nua = torch.load(model_path)
nua.eval()

#%% test the model under attck with testing data
nua.eval()
test_data_MUA = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../Datasets/', train=False, download=False,transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ])))
n_total_sample = 0
n_correct_predicted = 0
for testing_sample in test_data_MUA:
    image, label = testing_sample
    image = image.to('cuda').reshape(-1, 1, 28, 28)
    label = label.to('cuda')
    result = nua(image)
    result = result.data.max(1, keepdim=True)[1][0].item()
    
    if (result == label.cpu().item()):
        n_correct_predicted += 1
    n_total_sample += 1
    
print("Predicting acc on", n_total_sample, "testing samples: ", n_correct_predicted / n_total_sample)

#%% generate attack with seeds
from MNISTAtkTest import GenerateSamples
net.eval()
nua.eval()

# use random seeds from batch, run previous section to shuffle it, batch size is 50
#seed_idx = [21]
#seeds = inputData[seed_idx]
#real_labels = target[seed_idx]

# use seeds from training set in order
train_data = torchvision.datasets.MNIST('../Datasets/', train=True, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor()
                                 ]))
# use the first image as seed
seeds = train_data.data[0:1].float() / 255
seeds = seeds.view(-1, 1, 784).float().to('cuda')
real_labels = train_data.train_labels[0:1]
                        
# mode, important_threshold, targeted_fitness, target_seed
# are not used and to be explored in future
adverSamples = GenerateSamples(net, nua, seeds, real_labels, mode = 3,
                               important_threshold = 0.1,
                               max_generation = 150, 
                               targeted_fitness = 400,
                               crossover_rate = 1,
                               mutation_rate = 0.05, 
                               population_per_generation = 50,
                               nSample_to_keep = 15, step_size= 0.005,
                               initial_range = 0.02, beta = 1,
                               alpha = 0.4, is_targeted = False, targeted_label = 8,
                               target_seed = None
                               )
adverSamples = adverSamples[0]
#%% calculate average distortion and report attck success rate
distortion = []
succ_count = 0

for i in range(0, len(seeds)):
    adv_code = torch.tensor(adverSamples[i][0]).to('cuda').view(1, -1)
    seed_code = net.encode(seeds[i])
    adv_code.add_(seed_code.view(1, -1))
    
    img = net.decode(adv_code)
    img = torch.clamp(img, 0, 1)
    
    pre_label = nua(img.view(1, 1, 28, 28)).data.max(1, keepdim=True)[1][0].item()
    if (int(pre_label) != int(real_labels[i].item())):
        succ_count += 1
    
    dis = torch.cdist(seeds[i], img, p = float(2))
    distortion.append(dis.detach().cpu().item())

print('mean distortion: ', np.mean(distortion) / 784)
print("success rate: ", succ_count)    
#%% Generate attack with FGSM using same seeds
from FastGradientSignMethod import GenerateWithFGSM
nua.eval()
fgsm_imgs = []
fgsm_distances = []
fgsm_predlabel = []

label_idx = 0
# generate attacks
for seed in seeds:
    fgsm_img, pred_label, fgsm_dis = GenerateWithFGSM(seed, real_labels[label_idx], 0.25, nua)
    fgsm_imgs.append(fgsm_img)
    fgsm_distances.append(fgsm_dis)
    fgsm_predlabel.append(pred_label)
    label_idx += 1
    
# plot generated attacks
for fgimg in fgsm_imgs:
    plt.xticks([])
    plt.yticks([])
    plt.imshow(fgimg, cmap = 'gray')
    plt.show()

#%% predict modified samples and plot
nua.eval()
net.eval()

lines = len(seeds)
columns = 4
fitness = 0
nDisplay = 5
net_to_use = net

# display generated samples, figure is not used 
fig = plt.figure(figsize=(6 , 6))
#figure with subplots is disabled to make clear plots
for i in range(0, len(seeds)):
    
    adverSample = adverSamples[i][0]
    fitness = fitness + adverSamples[i][1]
    real_labeli = real_labels[i].cpu().item()
    subplot_id = i * 4 + 1
    #subplot_id = i + 1
    
    # plot original features and deviations
    adverTensor = torch.tensor(adverSample).to('cuda')
    seedTensor = net_to_use.encode(seeds[i]).to('cuda')
    #plt.subplot(lines, columns, subplot_id + 1)
    plt.bar(range(0, hidden_nodes), seedTensor.cpu().detach().numpy()[0])
    seedTensor.add_(adverTensor)
    plt.bar(range(0, hidden_nodes), adverTensor.cpu().numpy())
    plt.legend(["Original", "Deviation"])
    plt.show()

    # generate attack image, clamp it into [0, 1]
    adv_img = net_to_use.decode(seedTensor)
    adv_img = torch.clamp(adv_img, 0, 1)
    adv_img = adv_img.cpu().detach().numpy()
    adv_img = torch.tensor(adv_img).to('cuda')
    
    # get the confidence
    output_test = F.softmax(nua(adv_img.reshape(1, 1, 28, 28)))
    
    predicted_as = output_test.data.max(1, keepdim=True)[1][0].item()
    confidence = str(output_test.data.max(1, keepdim=True)[0][0].item())
    print("Real label: ", real_labeli)
    print("predicted as: ", predicted_as)
    
    
    # plot original img
    #plt.subplot(lines, columns, subplot_id)
    plt.title('Original: ' + str(int(real_labeli)))
    plt.imshow(seeds[i].cpu().detach().numpy().reshape(28, 28), cmap='gray')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.show()
    
    # calculate distortion
    #plt.subplot(lines, columns, subplot_id + 1)
    gimg = net.decode(net.encode(seeds[i]))
    distance_orig = torch.cdist(seeds[i], adv_img, p = float(2))
    print("cdist: ", distance_orig)
    
    # plot modified img
    #plt.subplot(lines, columns, subplot_id + 2)
    adv_img = adv_img.cpu().detach().numpy().reshape(28, 28)
    plt.xticks([])
    plt.yticks([])
    if (real_labeli != predicted_as):
        color = 'r'
    else:
        color = 'g'
    plt.title("Predicted as: " + str(predicted_as) + "\nConfidence: " + confidence[0:6], color = color, fontsize=20)
    plt.imshow(adv_img, cmap='gray')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.show()
    
    i = i + 1
    
plt.show()
print("Average fitness: ", fitness / len(seeds))



#%% help functions for parameter tuning
"""
Codes below were used to explore with parameters, no need to worry about them.
"""
# return predicted label given a seed and attack
def PredictWithModel(nua, net_to_use, adv_code, seed):
    adverTensor = torch.tensor(adv_code).to('cuda')
    seedTensor = net_to_use.encode(seed).to('cuda')
    seedTensor.add_(adverTensor)
    
    adv_img = net_to_use.decode(seedTensor)
    adv_img = torch.tensor(adv_img).to('cuda')
    adv_img = torch.clamp(adv_img, 0, 1)
    predicted = nua(adv_img.reshape(1, 1, 28, 28))
    
    return  int(predicted.data.max(1, keepdim=True)[1][0].item())
    
# return L2 distance between seed and attack
def GetDistanceBetweenImgs(seed, net_to_use, adv_img):
    seedTensor = net_to_use.encode(seed).to('cuda')
    adv_tensor = seedTensor.add(torch.tensor(adv_img).to('cuda'))
    adv_img = net_to_use.decode(adv_tensor)
    
    distance = torch.cdist(adv_img, seed, p = 2)
    
    return distance.item()
#%% explore with crossover and mutation rate, init and step size
from MNISTAtkTest import GenerateSamples

# init range
var1 = [0.0, 0.005, 0.01, 0.02]
# step size
var2 = [0.01, 0.02, 0.03]

# take 10 data and create 4 copy for each of them
# 50 data in total
data = training_data[0:10]
target = training_targets[0:10]

data = data.view(10, 1, 784).numpy()
target = target.view(10, 1).numpy()
orig_data = data
for j in range(0, 4):
    for i in range(0, 10):
        data = np.append(data, orig_data[i])
        target = np.append(target, target[i])
data = torch.tensor(data)
target = torch.tensor(target)

inputData = data.view(50, 1, 784).to("cuda")
seeds = inputData
real_labels = target

# start generation
succ_rate = []
distances = []
fitnesses = []
for i in range(len(var1)):
    succ_rate.append([])
    distances.append([])
    fitnesses.append([])
    for j in range(len(var2)):
        succ_rate[i].append(0)
        distances[i].append(0)
        fitnesses[i].append(0)
        print("Doing var1 = ", var1[i])
        # for each alpha, generate examples
        adverSamples = GenerateSamples(net, nua, seeds, real_labels, mode = 3,
                                   important_threshold = 0.1,
                                   max_generation = 100, 
                                   targeted_fitness = 400,
                                   crossover_rate = 1, 
                                   mutation_rate = 0.05, 
                                   population_per_generation = 50,
                                   nSample_to_keep = 15, step_size = var2[j],
                                   initial_range = var1[i], beta = 1,
                                   alpha = 0.2,
                                   is_targeted = False, targeted_label = 0
                                   )
        adverSamples = adverSamples[0]
        
        # calculate success rate and average distances, fitness
        for k in range(len(seeds)):
            predicted_label = PredictWithModel(nua, net, adverSamples[k][0], seeds[k])
            distance = GetDistanceBetweenImgs(seeds[k], net, adverSamples[k][0])
            if (predicted_label != int(real_labels[k].item())):
                succ_rate[i][j] += 1
                distances[i][j] += distance
                fitnesses[i][j] += adverSamples[k][1]
            print(int(real_labels[k].item()), " predicted as ", predicted_label)
            print("distace: ", distance)
            print('Fitness: ', adverSamples[k][1])
        # average them
        distances[i][j] /= succ_rate[i][j]
        fitnesses[i][j] /= succ_rate[i][j]
        succ_rate[i][j] /= len(seeds)
#%% plot fitness
linestyles =  ['dotted', 'dashdot', 'solid', 'dashed']
for i in range(0, len(fitnesses)):
    plt.plot(fitnesses[i], linestyle = linestyles[i])
plt.legend(['init: 0', 'init:0.005', 'init: 0.01', 'init: 0.02'])
plt.xticks([0, 1, 2], [0.005, 0.01, 0.02])
plt.xlabel('Step size')
plt.ylabel('Average fitness')
plt.title('Average fitness with different initialization and step size')
plt.show()
#%% plot distances
linestyles =  ['dotted', 'dashdot', 'solid', 'dashed']
for i in range(0, len(fitnesses)):
    plt.plot(distances[i], linestyle = linestyles[i])
plt.legend(['init: 0', 'init:0.005', 'init: 0.01', 'init: 0.02'])
plt.xticks([0, 1, 2], [0.005, 0.01, 0.02])
plt.xlabel('Step size')
plt.ylabel('L2 norm of perturbation')
plt.title('Perturbation with different initialization and step size')
plt.show()
#%% plot succ rate
linestyles =  ['dotted', 'dashdot', 'solid', 'dashed']
for i in range(0, len(fitnesses)):
    plt.plot(succ_rate[i], linestyle = linestyles[i])
plt.legend(['init: 0', 'init:0.005', 'init: 0.01', 'init: 0.02'])
plt.xticks([0, 1, 2], [0.005, 0.01, 0.02])
plt.xlabel('Step size')
plt.ylabel('Success rate')
plt.title('Attacck success rate with different initialization and step size')
plt.show()


#%%
# Explore with different alpha values and generation 
from MNISTAtkTest import GenerateSamples
net.eval()
nua.eval()

# variable to explore, use generation 
alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
#generations = [10, 30, 50, 100, 150]

# use data from a random batch, rerun the previous section to take another batch
data = inputData
target = target

# create 1 copy for each sample, so 100 samples in total
data = data.view(50, 1, 784).detach().cpu().numpy()
target = target.view(50, 1).detach().cpu().numpy()
orig_data = data
for j in range(0, 1):
    for i in range(0, 50):
        data = np.append(data, orig_data[i])
        target = np.append(target, target[i])
data = torch.tensor(data)
target = torch.tensor(target)

inputData = data.view(100, 1, 784).to("cuda")
seeds = inputData
real_labels = target

# start generation
i = -1
succ_rate = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
distances = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
fitnesses = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for alpha in alphas:    
    i += 1
    print("Doing alpha = ", alpha)
    # for each alpha, generate examples
    adverSamples = GenerateSamples(net, nua, seeds, real_labels, mode = 3,
                               important_threshold = 0.1,
                               max_generation = 100, 
                               targeted_fitness = 400,
                               crossover_rate = 1, 
                               mutation_rate = 0.05,
                               population_per_generation = 50,
                               nSample_to_keep = 15, step_size= 0.005,
                               initial_range = 0.02, beta = 1,
                               alpha = alpha, 
                               is_targeted = False, targeted_label = alpha
                               )
    adverSamples = adverSamples[0]
    
    # for each parameter, calculate success rate and average distance, fitness
    for j in range(len(seeds)):
        predicted_label = PredictWithModel(nua, net, adverSamples[j][0], seeds[j])
        distance = GetDistanceBetweenImgs(seeds[j], net, adverSamples[j][0])
        if (predicted_label != int(real_labels[j].item())):
            succ_rate[i] += 1
            distances[i] += distance
            fitnesses[i] += adverSamples[j][1]
        print(int(real_labels[j].item()), " predicted as ", predicted_label)
        print("distace: ", distance)
    # average them
    distances[i] /= succ_rate[i]
    fitnesses[i] /= succ_rate[i]
    succ_rate[i] /= len(seeds)
#%% plot for different alphas
xticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
plt.plot(xticks, succ_rate)
plt.xlabel("alpha")
plt.ylabel("attck success rate")
plt.title('Attack success rate with different alpha values')


plt.xticks(xticks, xticks)
plt.show()

plt.plot(xticks, distances)
plt.xlabel("alpha")
plt.ylabel("L2 norm of perturbation")
plt.xticks(xticks, xticks)
plt.title('Average perturbation with different alpha')
plt.show()

plt.plot(xticks, fitnesses)
plt.xlabel("alpha")
plt.ylabel("fitness")
plt.xticks(xticks, xticks)
plt.title('Average fitness with different alpha')
plt.show()
#%% explore with different autoencoder, additional work
# take testing data
from MNISTLoader import MNISTLoader
trainLoader = MNISTLoader()
batch_idxes, training_data, training_targets = trainLoader.GetBatchesAndData(False, batch_size)
#%% training an autoencoder with other penalties
net_2 = autoencoderMNIST.AutoEncoder().to('cuda')
net_2.train()
n_epoch = 40
criterion = nn.MSELoss()
optimizer = optim.Adam(net_2.parameters(), lr = 0.001)
epoch_loss = Train(net_2, optimizer, criterion, n_epoch, 0.000001, 0.5, 2)
net_2.eval()

#%% compare cluster performance using first 10000 data
from DistributionChecker import DistributionChecker
data = training_data[0:10000]
target = training_targets[0:10000]
#%% first autoencoder
net.eval()
dc = DistributionChecker(net)
cluster_error_net, var_net = dc.GetClusterError(data, target)
#%% second autoencoder
dc2 = DistributionChecker(net_2)
cluster_error_net_2, var_net_2 = dc2.GetClusterError(data, target)

#%% plot cluster error
xtick = [i for i in range(10)]
xtick2 = [i+0.2 for i in range(10)]

plt.bar(xtick, cluster_error_net, width = 0.3)
plt.bar(xtick2, cluster_error_net_2, width = 0.3)

plt.plot(xtick, cluster_error_net)
plt.plot(xtick2, cluster_error_net_2)

plt.xticks(xtick, xtick)
plt.xlabel('Class label')
plt.ylabel('Average Distance')
plt.legend(['with penalty', 'without penalty'])
plt.title("Average distance of samples to their own cluster's centroid")
plt.show()
#%% plot distances between classes
plt.plot(var_net)
plt.plot(var_net_2)
plt.xlabel('Class label')
plt.ylabel('Average Distance')

plt.xticks(xtick, xtick)

plt.legend(['with penalty', 'without penalty'])
plt.title('Average distance of each class\'s centroid to other classes\'')
