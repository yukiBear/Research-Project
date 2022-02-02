# -*- coding: utf-8 -*-
"""
Created on Sat May 15 16:43:19 2021

@author: LENOVO
"""
"""
Sample generator for CIFAR10,
"""
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
import torchvision


# most of the code are copied from samplre generator for MNIST
# prediction function is different as VGG11 requires normlization
# mutation function is different as latent space are extracted by convolutional layer
# perturbation calculation is different 
hidden_nodes = 48 * 4 * 4

class SampleGenerator:
    def __init__(self, net, targetModel):
        self.net = net
        self.targetModel = targetModel
        self.currentGeneration = []
        
        self.newGeneration = []
        self.step = 0.5
        self.first_generation = True
        self.max_distance = 0
        # generate adversarial sample with a given seed
        # mode: 1: use only unimportant features
        # mode: 2: use only important features
        # mode: 3: use all features
    def GenerateSample(self, seed, real_label, is_targeted, target_label,
                       target_fitness, threshold, mode, maxGeneration,
                       mutation_rate = 0.1, crossover_rate = 0.6, 
                       population_per_generation = 30, nSample_to_next_generation = 10, 
                       step_size = 0.5, initial_range = 0.1, beta = 1, alpha = 0.1, target_seed = None):
        
        # initializa the first generation
        self.currentGeneration.append([np.array(np.random.normal(0, 0, hidden_nodes), dtype=np.float32),0])
        # set up variables
        max_generation = maxGeneration
        self.step = step_size
        n_generation = 0
        print("real_label is", real_label)
        o = self.net.encode(seed)
        self.o = o.detach().cpu().numpy().reshape(1, -1)[0]
        
        
        # dropped; use help seed if given
        strength = [0.05, 0.1, 0.15, 0.2, 0.5, 1]
        init_count = 1
        if (target_seed != None):
            print("Using target seed")
            target_o = self.net.encode(target_seed).detach().cpu().numpy()
            target_o -= o.detach().cpu().numpy()
            
            # plot deviation
            temp = self.net.decode(torch.tensor(target_o).view(1, 200).to('cuda'))
            plt.imshow(temp.cpu().detach().numpy().reshape(28, 28), cmap = 'gray')
            plt.show()
            for i in strength:
                self.currentGeneration.append([target_o[0] * i, 0])
                init_count += 1
        # random noise within 0 ~ init_range
        for i in range(init_count, int(population_per_generation / 2)):
            self.currentGeneration.append([np.array(np.random.normal(-initial_range, initial_range, hidden_nodes), dtype=np.float32),0])
        # random noise within 0 ~ init_range / 2
        for i in range(int(population_per_generation / 2), population_per_generation):
            self.currentGeneration.append([np.array(np.random.normal(-initial_range / 2, initial_range / 2, hidden_nodes), dtype=np.float32),0])
        
        
        # get the original img (generated, unmodified)
        self.original_img = self.net.decode(self.net.encode(seed)).cpu().detach().numpy()
        
        # calculate the important features, those are > mean(features)
        self.importantFeatures = self.GetImportantFeatureIdx(o.cpu().detach().numpy(), threshold, mode)
       
        # calculate fitness for first generation
        for i in range(0, len(self.currentGeneration)):
            self.currentGeneration[i][1] = self.Fitness(self.currentGeneration[i][0],
                                                         real_label, o, is_targeted, 
                                                         target_label, 
                                                         beta, alpha)
        
        # generate offsprings    
        average_fitness = []
        ans = 0
        foundAns = False
        fitness_not_increase = 0
        previous_max_fitness = 0.0
        while(True):       
            #sort current generation by fitness
            self.currentGeneration.sort(key=lambda k:k[1], reverse=True)
            #print(self.currentGeneration)
            # use dynamic step, decrease step when fitness stops increasing
            if (abs(self.currentGeneration[0][1] - previous_max_fitness) < 0.05):
                fitness_not_increase = fitness_not_increase + 1
            if (fitness_not_increase > 5 and self.step > 0.0001):
                self.step = self.step / 1.1
                fitness_not_increase = 0
                
            previous_max_fitness = self.currentGeneration[0][1]
            
            #keep the top samples
            self.previousGeneration = self.currentGeneration
            self.currentGeneration = self.currentGeneration[0:nSample_to_next_generation]
            self.newGeneration = []
            
            #generate offspring
            probabilities = []
            sum_fitness = 0
            # +100 to make fitness positive, just for calculating proportion during selection 
            for i in range(0, len(self.previousGeneration)):
                sum_fitness += self.previousGeneration[i][1] + 100
            
            # calculate the probability according to fitness
            for i in range(0, len(self.previousGeneration)):
                probabilities.append((self.previousGeneration[i][1]+100) / sum_fitness)
            
            for j in range(0, population_per_generation - nSample_to_next_generation):
                # select 2 parents
                p1, p2 = self.Selection(probabilities)
                newOffspring = self.GenerateOffspring(self.previousGeneration[p1][0],
                                                      self.previousGeneration[p2][0], crossover_rate)
                # mutate the new one
                newOffspring[0] = self.Mutate(np.array(newOffspring[0]), self.step, 
                                              mode, mutation_rate)                
                # calculate its fitness
                newOffspring[1] = self.Fitness(newOffspring[0],
                                               real_label, o, is_targeted,
                                               target_label, beta,
                                               alpha)
                self.newGeneration.append(newOffspring)
            
            # add new offsprings to current generation list
            for j in range(0, 20):
                self.currentGeneration.append(self.newGeneration[j])
            
            
            #test them 
            avg_fit = 0.0
            for i in range(0, len(self.currentGeneration)):
                #record the average fitness
                avg_fit += self.currentGeneration[i][1]
                
                # check if an answer is found
                if (self.PredictWithModel(o, self.currentGeneration[i][0]) != real_label):
                    ans = self.currentGeneration[i][0]
                    foundAns = True
            
            # report fitness
            if (n_generation % 50 == 0):
                print("average fitness of ", n_generation,"th generation: ", 
                      avg_fit / population_per_generation)
                print("Max fitness: ", self.currentGeneration[0][1])
                #self.PredictWithModel(o, self.currentGeneration[0][0], plot=True)
            
            
            average_fitness.append(avg_fit / population_per_generation)  
            
            # target fitness is not used
            if target_fitness <= self.currentGeneration[0][1]:
                print("Target fitness reached!")            
                break
            
            # stop when targget generation is reached
            if n_generation >= max_generation:
                print("Max generation reached!")
                break
            n_generation = n_generation + 1
            
        #print(average_fitness)
        
        print("Max fitness: ", self.currentGeneration[0][1])
        return self.currentGeneration
    
    # select two samples from current generation as parent
    # the probability that each of sample got selected is depended on its fitness
    # return: two int values, which are the idxes of parents
    def Selection(self, probabilities):
        # select 2 parents
        parents = np.random.choice(len(probabilities), 2, p = probabilities)
        return parents[0], parents[1]
    
    # mutation
    def Mutate(self, gene, amount, mode, mutation_rate):
        mutation_threshold = 1 - mutation_rate
        
        for i in range(0, hidden_nodes):
            coin = np.random.random()
            if (coin > mutation_threshold and coin <= (1 - mutation_rate / 2)):
                #mutate
                gene[i] = gene[i] + amount
            elif (coin > (1 - mutation_rate / 2)):
                gene[i] = gene[i] - amount

        return np.array(gene)
                    
    # generate an offsprint with given parents
    def GenerateOffspring(self, p1, p2, crossover_rate):
        newSample = []
        coin = np.random.random()
        
        # do crossover
        if (coin > (1 - crossover_rate)):
            point = random.randint(1, len(p1) - 2)
            newSample.extend(p1[0:point])
            newSample.extend(p2[point:len(p1)])
        else:
            coin = np.random.random()
            if (coin <= 0.5):
                newSample = p1
            else:
                newSample = p2
        newSample = np.array(newSample)
        
        
        return [newSample, 0]
    
    # Fitness = reconstruction error * alpha + semantic distance * beta + attack performance
    def Fitness(self, gene, real_label, seed, targeted, target_Label, beta, alpha):
        deviation = 0
        fitness = 0
        # calculate semantic distance
        for i in range(0, hidden_nodes):
            deviation = deviation + gene[i] ** 2
        deviation = deviation ** 0.5
        
        # predict with target model
        predict, generated_img, predict_prob = self.PredictWithModel(seed, gene)
        top_k = torch.topk(predict_prob, 10)
        predicted_topk = top_k[0].cpu().detach().numpy()
        predicted_topk_idx = top_k[1].cpu().detach().numpy()
        
        # Use probabilities as searching direction
        # for untargeted attack
        if (targeted == False):
            prob1 = predicted_topk[0][0]
            prob2 = predicted_topk[0][1]
        
            if(int(predicted_topk_idx[0][0]) == int(real_label)):
                fitness += prob2 - prob1
            else:
                fitness += prob1 - prob2
        else: # for targeted attack
            pidx = np.where(predicted_topk_idx[0] == int(target_Label))
            pidx = int(pidx[0])
            prob2 = np.exp(predicted_topk[0][pidx])
            if (pidx != 0):
                prob1 = np.exp(predicted_topk[0][0])
                fitness += prob2 - prob1
            else:
                prob1 = np.exp(predicted_topk[0][1])
                fitness += prob2 - prob1
            
        reconstruction_error = self.CalculateReconstructionError(generated_img, 2)
        
        if (int(predict) != int(real_label)):
            misclassified = 1
            if (targeted == False):
                deviation = -deviation
            else:
                if (int(predict) == int(targeted)):
                    deviation = -deviation
                else:
                    deviation = 0
        else:
            misclassified = 0
            deviation = 0
        
        fitness -= reconstruction_error * alpha
        fitness += deviation * beta
   
        return fitness
    
    # mode 1: l1 distance
    # mode 2: l2 distance
    # mode 3: l-inf distance
    # mode 4: sigmoid
    def CalculateReconstructionError(self, generated_img, mode = 2):       
        generated_img = generated_img.cpu().detach().numpy()
        #generated_img = generated_img / np.max(generated_img)
        orig_img = self.original_img
        
        result = 0
        temp = (1 / np.exp(5.8))
        if (mode == 1):
            result = np.linalg.norm(orig_img[0] - generated_img[0], ord = 1)
        if (mode == 2):
            result = np.linalg.norm(orig_img[0].reshape(1, -1) - generated_img[0].reshape(1, -1), ord = 2)
        if (mode == 3):
            result = np.linalg.norm(orig_img[0] - generated_img[0], ord = np.inf)
        if (mode == 4):
            dist = abs(orig_img[0] - generated_img[0])
            for d in dist:
                result += 1 / (1 + np.exp(- (d * 10) + 5.8)) - temp
        
        return result

    # predict and get probabilities     
    def PredictWithModel(self, seed_code, gene, plot=False):
        geneTensor = torch.FloatTensor(gene).to('cuda')
        advSample = torch.add(seed_code, geneTensor.view(1, 48, 4, 4))
        generated_img = self.net.decode(advSample)

        # create adversarial img, make sure its in [0, 1]
        final_adv = generated_img.clone()
        final_adv = final_adv.cuda()
        final_adv = torch.clamp(final_adv, 0, 1).view(32, 32, 3)
        
        # normalize the img
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2471, 0.2435, 0.2616]
        mean_tensor = torch.tensor(mean).to('cuda')
        std_tensor = torch.tensor(std).to('cuda')
        final_adv = final_adv.view(32, 32, 3).sub(mean_tensor).div(std_tensor)
        
        # predicte with normalized sample
        result = F.softmax(self.targetModel(final_adv.view(1, 3, 32, 32)))
        predict_result = result
        result = result.data.max(1, keepdim=True)[1][0].item()

        return result, generated_img, predict_result
    
    # not used
    def GetImportantFeatureIdx(self, o, threshold, mode):
        result = []
        if (mode == 3):
            return result
        summ = 0
        for i in range(len(o[0])):
            summ += abs(o[0][i])
        threshold = summ / len(o[0])    
        print("Importance threshold is: ", threshold)
        
        for i in range(len(o[0])):
            if (mode == 1):
                if (abs(o[0][i]) >= threshold):
                    result.append(i)
            if (mode == 2):
                if (abs(o[0][i]) <= threshold):
                    result.append(i)
        return result