# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 12:03:54 2021

@author: LENOVO
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class FGSM:
    def __init__(self):
        return
    
    # calculate gradients
    def fgsm_attack(self, image, epsilon, data_grad):
        # calculate sign
        sign_data_grad = data_grad.sign()
        
        # create noises        
        perturbed_image = image + epsilon*sign_data_grad
        
        # clamp image into [0, 1]
        perturbed_image = torch.clamp(perturbed_image, 0, 1)        
        
        return perturbed_image
    
def GenerateWithFGSM(seed, target, p, nua):
    f = FGSM()    
    target_tensor = torch.tensor([target.item()]).to('cuda')
    seed_tensor = torch.tensor(seed, requires_grad = True)
    print(torch.max(seed), torch.min(seed))
    
    seed_tensor.requires_grad = True
    pred_output = nua(seed_tensor.view(1, 1, 28, 28))
    
    nua.zero_grad()
    loss = F.nll_loss(pred_output, target_tensor.long())
    loss.backward()
    
    # add noise
    datagrad = seed_tensor.grad.data
    perturbed_img = f.fgsm_attack(seed, p, datagrad)
    
    # predict the modified image
    predicted_as = nua(perturbed_img.view(1, 1, 28, 28)).data.max(1, keepdim=True)[1][0].item()
    print("Predicted label: ", predicted_as)
    # plot it
    plt.title(str(predicted_as))
    plt.show()
    
    return perturbed_img.view(28, 28).cpu().detach().numpy(), predicted_as, torch.cdist(perturbed_img, seed.view(1, 784), p = float('2'))
