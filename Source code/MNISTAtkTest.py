# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 15:12:31 2021

@author: LENOVO
"""
import SampleGenerator
import time
import matplotlib.pyplot as plt

# given seeds, generate 1 adversarial sample for each seed
def GenerateSamples(feature_extracter, model_under_attack, seeds, labels, mode,
                    important_threshold, is_targeted = False, targeted_label = 0,
                    max_generation = 1000, crossover_rate = 0.6, mutation_rate = 0.1,
                    population_per_generation = 30, nSample_to_keep = 10,
                    targeted_fitness = 4999.5, step_size = 0.5, initial_range = 0.1, 
                    beta = 1, alpha = 0.1, target_seed = None):
    # start counting the running time 
    startTime = time.time()

    # generate samples for seeds
    adverSamples = []
    for i in range(0, len(seeds)):
        attackGenerator = SampleGenerator.SampleGenerator(feature_extracter, model_under_attack)
        lastGeneration = attackGenerator.GenerateSample(seeds[i],
                   labels[i].cpu().item(), is_targeted, targeted_label, targeted_fitness, 
                   important_threshold,
                   mode, max_generation, crossover_rate = crossover_rate,
                   mutation_rate = mutation_rate,
                   population_per_generation = population_per_generation,
                   nSample_to_next_generation = nSample_to_keep, step_size = step_size,
                   initial_range = initial_range, beta = beta,
                   alpha = alpha, target_seed = target_seed)
        
        adverSamples.append(lastGeneration[0])
        i = i + 1

    endTime = time.time()
    
    total_time = endTime - startTime
    print("Time token for generation: ", endTime - startTime, " seconds")
    print("Average time for each sample: ", total_time / len(seeds), "seconds")
    
    return adverSamples, total_time

