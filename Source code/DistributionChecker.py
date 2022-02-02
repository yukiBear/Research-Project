# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 11:25:15 2021

@author: LENOVO
"""
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# this code is used in additional work
class DistributionChecker:
    def __init__(self, net):
        self.net = net
            
    # calculate cluster error 
    def GetClusterError(self, data, label):
        # group data by class
        label =  label.cpu().detach().numpy()
        
        classes = np.unique(label)
        data_by_class = []
        classes = np.sort(classes)
        print(len(data), len(label))
        
        # process 50 samples each time (no enough memory)
        for b in range(0, len(label), 50):
            batch_data = data[b : b + 50]
            batch_label = label[b : b + 50]
            batch_data = batch_data.view(50, 1, 784).to('cuda')
            j = -1
            # for all classes
            for c in classes:
                j += 1
                if (b == 0):
                    data_by_class.append([])
                for i in range(len(batch_label)):
                    if (int(batch_label[i]) == int(c)):
                        data_by_class[j].append(batch_data[i])
            batch_data.cpu()
            
        # get their hidden features
        hiddenFeatures_by_class = []
        max_value = 0
        for i in range(len(classes)):
            hiddenFeatures_by_class.append([])
            for j in range(len(data_by_class[i])):
                hiddenFeatures_by_class[i].append(self.net.encode(data_by_class[i][j])
                                                  .cpu().detach().numpy()[0])
                if (max_value < np.max(np.abs(hiddenFeatures_by_class[i]))):
                    max_value = np.max(np.abs(hiddenFeatures_by_class[i]))
        
        # normalize them
        hiddenFeatures_by_class = np.array(hiddenFeatures_by_class)
        for i in range(len(hiddenFeatures_by_class)):
            hiddenFeatures_by_class[i] /= max_value
        # display all data with tsne
        self.DisplayWithTSNE(hiddenFeatures_by_class)
        
        
        # calculate mean for each class
        cluster_error = []
        mean_distance = []
        for i in range(len(hiddenFeatures_by_class)):
            classMean = np.mean(hiddenFeatures_by_class[i], axis=0)
            cluster_error.append(0.0)
            mean_distance.append(0.0)
            # calculate average error
            for j in range(len(hiddenFeatures_by_class[i])):
                cluster_error[i] += np.linalg.norm(
                    hiddenFeatures_by_class[i][j] - classMean, ord = 2)
            # calculate cluster mean distance
            for j in range(len(hiddenFeatures_by_class)):
                if (j != i):
                    mean_i = np.mean(hiddenFeatures_by_class[j], axis = 0)
                    
                    mean_distance[i] += np.linalg.norm(
                    classMean - mean_i, ord = 2)
            
            
        self.hiddenFeatures_by_class = hiddenFeatures_by_class
        for i in range(len(cluster_error)):
            cluster_error[i] /= len(label)
            mean_distance[i] /= len(cluster_error)
        
        return cluster_error, mean_distance
    
    
    def DisplayWithTSNE(self, X_by_class):
        n_display = -1
        X = np.array(X_by_class[0])
        y = np.zeros(len(X_by_class[0]))
        for i in range(1, len(X_by_class)):
            X = np.append(X, X_by_class[i])
            y = np.append(y, np.zeros(len(X_by_class[i])) + i)
        X = X.reshape(-1, 128)
        X_embedded = TSNE(n_components=2).fit_transform(X)
        X_emx = X_embedded[:, 0]
        X_emy = X_embedded[:, 1]
        sc = plt.scatter(X_emx, X_emy, c = y, s = 2, label=y)
        plt.legend(*sc.legend_elements())
        
        plt.show()