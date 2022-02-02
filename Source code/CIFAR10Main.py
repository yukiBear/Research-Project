# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 15:02:59 2021

@author: LENOVO
"""

"""
Main codes for CIFAR10 experiments
It has to be run on a GPU.
"""

import torch
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.nn.functional as F

# for normalization
mean = [0.4914, 0.4822, 0.4465]
std = [0.2471, 0.2435, 0.2616]
mean_tensor = torch.tensor(mean).to('cuda')
std_tensor = torch.tensor(std).to('cuda')
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

# load the data
train_data = CIFAR10('../Datasets/CIFAR10/', train = True,
                                  transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ]), 
                                  download = True)
test_data = CIFAR10('../Datasets/CIFAR10/', train = False,
                               transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(mean, std)]), 
                                  download = True)

# for normalizing images, VGG11 requires this
def normalize_img(img):
    img = img.view(32, 32, 3).sub(mean_tensor).div(std_tensor).view(1, 3, 32, 32)
    return img

#%% For training autoencoder
from torch.autograd import Variable

def TrainCIFAR10AE(data, net, criterion, optimizer, epoches, lr, batch_size):
    net.train()
    net.zero_grad()
    data_loader = torch.utils.data.DataLoader(data, 
                                              batch_size=batch_size,
                                              shuffle=True)
    epoch_losses = []
    for i in range(epoches):
        epoch_loss = 0
        n_sample = 0
        torch.cuda.empty_cache()
        print("Doing epoch: ", i)
        count = 0
        for idx, (img, label) in enumerate(data_loader, 0):
            torch.cuda.empty_cache()
            
            img = img.view(-1, 3, 32, 32).to('cuda')
            latent_code, output = net(img)
            output = output.view(-1, 3, 32, 32)
            net.zero_grad()
            # calculate the loss
            loss = criterion(output, img)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().cpu().item()
            count += 1
            
            # report loss
            n_sample += batch_size
            if (n_sample >= 5000):
                print("Batch loss: ", loss)
                n_sample = 0
                
        epoch_losses.append(epoch_loss / count)
        print(epoch_losses[i])
    return epoch_losses
                
#%% Define the autoencoder
from AutoencoderCIFAR10 import AutoencoderCIFAR10
torch.cuda.empty_cache()
net = AutoencoderCIFAR10().to('cuda')
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters())
#%% Train the autoencoder, or use the trained one in next sections
epoch_loss = TrainCIFAR10AE(train_data, net, criterion, optimizer, 30, 0.001, 16)
#%% plot loss
plt.plot(epoch_loss)
plt.title('Training loss')
plt.xlabel('Epoch')
plt.ylabel('BCE loss')
plt.show()

#%% load the autoencoder
model_path = '../Models/autoencoder_CIFAR10.pt'
net = torch.load(model_path)
net.eval()

#%% Display to check autoencoder
display_index = 0
# random split the training data
data = torch.utils.data.random_split(train_data, [10000, len(train_data) - 10000])

# decode and encoder
net.eval()
test_img = net.decode(net.encode(data[0][display_index][0].to('cuda').view(1, 3, 32, 32)))

# plot 
plt.imshow(data[0][display_index][0].cpu().detach().numpy().reshape(3, 32, 32).T)
plt.title('Original')
plt.show()
plt.imshow(test_img.cpu().detach().numpy().reshape(3, 32, 32).T)
plt.title('Reconstructed')
plt.show()
#%% Display latent space of autoencoder with TSNE, this runs very slow
from sklearn.manifold import TSNE
X = train_data.data
X_embedded = TSNE(n_components=2).fit_transform(X.reshape(-1, 32 * 32 * 3))

#%% Plot the latent space
y = train_data.targets
X_emx = X_embedded[:, 0]
X_emy = X_embedded[:, 1]
fig = plt.figure(figsize=(10,10))
sc = plt.scatter(X_emx, X_emy, c = y, s = 2, label=y, alpha=1)
plt.legend(*sc.legend_elements())
plt.title("Distribution of data on latent space (T-SNE)", fontsize = 20)
plt.xlabel('X embedding', fontsize = 20)
plt.ylabel('Y embedding', fontsize = 20)
plt.show()

#%% load pretrained VGG11 model
from PyTorch_CIFAR10_master.cifar10_models.vgg import vgg11_bn
mua = vgg11_bn(pretrained = True).to('cuda').eval()

#%% test the accuracy of VGG11 on testing data
mua.eval()
cifar10_test_loader = torch.utils.data.DataLoader(test_data)
correct_prediction = 0
total_sample = len(test_data)
for batch in cifar10_test_loader:
    img, label = batch
    img = img.view(-1, 3, 32, 32).to('cuda')
    result = mua(img)
    result = result.data.max(1, keepdim=True)[1][0].item()
    if (result == label.item()):
        correct_prediction += 1
print("Test accuracy is: ", correct_prediction / total_sample)
#%% Help functions for attack evaluation
def PredictWithModel(nua, net_to_use, adv_code, seed):
    # create adversarial attack with seed
    adverTensor = torch.tensor(adv_code).view(1, -1).to('cuda')
    seedTensor = net_to_use.encode(seed).view(1, -1).to('cuda')
    seedTensor.add_(adverTensor)
    adv_img = net_to_use.decode(seedTensor.view(1, 48, 4, 4))
    adv_img = torch.tensor(adv_img).to('cuda')
    # clamp it to [0, 1]
    adv_img = torch.clamp(adv_img, 0, 1)
    # normalize it and predict with target model
    predicted = nua(normalize_img(adv_img.reshape(1, 3, 32, 32)))
        
    return  int(predicted.data.max(1, keepdim=True)[1][0].item())
    
def GetDistanceBetweenImgs(seed, net_to_use, adv_img):
    seedTensor = net_to_use.encode(seed).to('cuda')
    adv_tensor = seedTensor.add(torch.tensor(adv_img).view(1, 48, 4, 4).to('cuda'))
    adv_img = net_to_use.decode(adv_tensor)
    adv_img = torch.clamp(adv_img, 0, 1)
    # calculate the L2 distance between attack and original seed
    distance = torch.cdist(adv_img.view(1, -1), seed.view(1, -1), p = 2)
    
    return distance.item()

#%% Take a seed with idx for exploration
idx = [0]
seeds = []
real_labels = []
for i in range(0, len(idx)):
    seeds.append(torch.tensor(train_data[idx[i]][0]).view(1, 3, 32, 32).to('cuda'))
    real_labels.append(train_data[idx[i]][1])
print(real_labels[0])
real_labels = torch.tensor(real_labels).view(-1, 1).to('cuda')
#%% Generate attacks with seeds
from CIFAR10AtkTest import GenerateSamples
# mode, important_threshold, max_generation, targeted_fitness, target_seed are dropped
# they have no effect
adverSamples = GenerateSamples(net, mua, seeds, real_labels, mode = 3,
                               important_threshold = 0.1,
                               max_generation = 150, 
                               targeted_fitness = 400,
                               crossover_rate = 1,
                               mutation_rate = 0.05, 
                               population_per_generation = 50,
                               nSample_to_keep = 15, step_size = 0.3,
                               initial_range = 1.5, beta = 0.01,
                               alpha = 0.1, is_targeted = False, targeted_label = 4,
                               target_seed = None
                               )
adverSamples = adverSamples[0]

#%% Calculate success rate, average distances, fitness
from CIFAR10AtkTest import GenerateSamples

succ_rate = 0
distances = 0
fitnesses = 0
for k in range(len(seeds)):
    predicted_label = PredictWithModel(mua, net, adverSamples[k][0], seeds[k])
    distance = GetDistanceBetweenImgs(seeds[k], net, adverSamples[k][0])
    if (predicted_label != int(real_labels[k].item())):
        succ_rate += 1
        distances += distance
        fitnesses += adverSamples[k][1]
        print(int(real_labels[k].item()), " predicted as ", predicted_label)

print("Distortation: ", (distances / len(seeds)) / (32 * 32))
print("Success rate: ", succ_rate / len(seeds))
#%% plot modified attack
import torch.nn.functional as F

idx = 0
# plot unmodified image
print("Real label is: ", real_labels[0].item())
predicted = F.softmax(mua(normalize_img(net(seeds[idx])[1])), dim=1)
print("Unmodified predicted as: ", predicted.data.max(1, keepdim=True)[1][0].item())
print("With probability: ", predicted.data.max(1, keepdim = True)[0][0].item())
label = label_names[predicted.data.max(1, keepdim = True)[1][0].item()]
prob = predicted.data.max(1, keepdim = True)[0][0].item()
title =  label + ": " + "{con:}".format(con = str(prob)[0:7])
fig = plt.figure(figsize = (10, 10))
plt.title(title, fontsize = 50)

# create attack with decoder
adver_gene = torch.tensor(adverSamples[idx][0]).to('cuda')
orig_gene = net.encode(seeds[idx])
orig_gene.add_(adver_gene.view(orig_gene.shape))
adv_img = net.decode(orig_gene)

# plot attack
plt.imshow(seeds[idx].detach().cpu().numpy().reshape(3, 32, 32).T)
plt.xticks([])
plt.yticks([])
plt.title("Original image", fontsize = 50)
fig.show()

# calculate l2 norm of noise
distance = torch.dist(seeds[idx].view(1, -1), adv_img.view(1, -1))
predicted = F.softmax(mua(normalize_img(adv_img.view(1, 3, 32, 32))), 1)
fig = plt.figure(figsize=(10, 10))
label = label_names[predicted.data.max(1, keepdim = True)[1][0].item()]
prob = predicted.data.max(1, keepdim = True)[0][0].item()
print("modified predicted as: ", label)
print("With probability: ", predicted.data.max(1, keepdim = True)[0][0].item())
title =  label + ": " + "{con:}".format(con = str(prob)[0:7])
plt.title(title, fontsize = 50)
plt.imshow(adv_img.detach().cpu().numpy().reshape(3, 32, 32).T)
plt.xticks([])
plt.yticks([])
fig.show()
print("L2 distance: ", distance)
print(label_names[real_labels[idx].item()] + " predicted as " + label_names[predicted.data.max(1, keepdim = True)[1][0].item()])






#%%
"""
Codes below were used for parameter tuning, no need to worry about them.
"""

#randomly take some data for exploration
data = torch.utils.data.random_split(train_data, [40000, len(train_data) - 40000])

#%% prepare seeds
temp_data = data[0]

seeds = []
real_labels = []
for i in range(0, 30):
    d = temp_data[i]
    img = d[0]
    label = d[1]
    seeds.append(img.view(1, 3, 32, 32).to('cuda'))
    real_labels.append(torch.tensor(label).to('cuda'))


#%% explore with different params
from CIFAR10AtkTest import GenerateSamples

succ_rate = []
distances = []
fitnesses = []

var1 = [1]
var2 = [0.05, 0.1, 0.3, 0.5]
for i in range(0, len(var1)):
    succ_rate.append([])
    distances.append([])
    fitnesses.append([])
    for j in range(0, len(var2)):
        succ_rate[i].append(0)
        distances[i].append(0)
        fitnesses[i].append(0)
        adverSamples = GenerateSamples(net, mua, seeds, real_labels, mode = 3,
                               important_threshold = 0.1,
                               max_generation = 150, 
                               targeted_fitness = 400,
                               crossover_rate = 1,
                               mutation_rate = 0.05, 
                               population_per_generation = 50,
                               nSample_to_keep = 15, step_size = 0.3,
                               initial_range = 1.5, beta = 0.01,
                               alpha = var2[j], is_targeted = False, targeted_label = 8,
                               target_seed = None
                               )
        adverSamples = adverSamples[0]
        
        # collect data
        for k in range(len(seeds)):
            predicted_label = PredictWithModel(mua, net, adverSamples[k][0], seeds[k])
            distance = GetDistanceBetweenImgs(seeds[k], net, adverSamples[k][0])
            if (predicted_label != int(real_labels[k].item())):
                succ_rate[i][j] += 1
                distances[i][j] += distance
                fitnesses[i][j] += adverSamples[k][1]
            print(int(real_labels[k].item()), " predicted as ", predicted_label)
            print("distace: ", distance)
            print('Fitness: ', adverSamples[k][1])
        if (succ_rate[i][j] != 0):
            distances[i][j] /= succ_rate[i][j]
            fitnesses[i][j] /= succ_rate[i][j]
        succ_rate[i][j] /= len(seeds)
#%% plot them
xticks = var2
linestyles = ['solid', 'dashed', 'dotted']
for i in range(0, 1):
    plt.plot(xticks,succ_rate[i], linestyle=linestyles[i])
plt.title('attack success rate with different alpha values')
plt.xlabel('alpha')
plt.ylabel('attack success rate')
plt.xticks(xticks)
plt.show()

for i in range(0, 1):
    plt.plot(xticks,distances[i],linestyle=linestyles[i])
plt.title('average perturbation with different alpha values')
plt.xticks(xticks)
plt.xlabel('alpha')
plt.ylabel('average perturbation')
plt.show()

for i in range(0, 1):
    plt.plot(xticks, fitnesses[i],linestyle=linestyles[i])
plt.title('average fitness with different alpha values')
plt.xlabel('alpha')
plt.ylabel('average fitness')
plt.xticks(xticks)
plt.show()