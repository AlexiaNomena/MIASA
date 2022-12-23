#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 20:04:14 2022

@author: raharinirina
"""
import numpy as np


""" Classification Experiment on random samples from specific probability distributions """
def generate_data_dist(var_data = False, noise = False):
    """Generate Artificial data"""
    per_spl = 200 # Num of iid observation in each samples 
    data_dic = {}
    class_type1 = ["1a", "1b", "1c"] # Normal Dist
    #val1 = [(0, 1), (0, 5), (2, 1)]
    val1_mean = np.random.choice(5, size = len(class_type1)) # allowing repeating means
    val1_std = np.random.choice(5, size = len(class_type1), replace = False) # not allowing repeated variance
    val1 = [(val1_mean[k], val1_std[k]) for k in range(len(class_type1))]
    
    class_type2 = ["2a", "2b", "2c"] # Uniform Dist
    #val2 = [(0, 1), (0, 5), (2, 3)]
    val2_a = np.random.choice(5, size = len(class_type2)) # allowing repeating start
    val2_b = np.random.choice(5, size = len(class_type2), replace = False) # not allowing repeated end
    val2 = [(val2_a[k], val2_a[k] + val2_b[k]) for k in range(len(class_type2))]
    
    class_type3 = ["3a", "3b", "3c"] # Pareto Dist
    #val3 = [1, 2, 3]
    val3_shape = np.random.choice(np.arange(1, 5), size = len(class_type3), replace = False)
    val3_scale = np.random.choice(5, size = len(class_type3), replace = False)
    val3 = [(val3_shape[k], val3_scale[k]) for k in range(len(class_type3))]
    
    class_type4 = ["4a", "4b", "4c"] # Poisson Dist
    #val4 = [1, 3, 6]
    val4 = np.random.choice(5, size = len(class_type4), replace = False)
    
    
    num_clust = len(class_type1) + len(class_type2) + len(class_type3) + len(class_type4)
    labs = np.cumsum(np.ones(num_clust)) - 1
    
    # Number of samples per classes
    MaxNumVar = 15
    if var_data:
        num_var_list = np.random.choice(np.arange(2, MaxNumVar), size = len(labs), replace = False)
        num_var = {labs[k]: num_var_list[k] for k in range(len(labs))}
    else:
        num_var = {labs[k]:10 for k in range(len(labs))}
    
    
    class_dic = {}
    k = 0
    for i in range(3):
        lab = labs[k:k+4]
        for j in range(MaxNumVar + 1):
            if j <= num_var[lab[0]]:
                data_dic[class_type1[i]+"%d"%(j+1)] = np.random.normal(val1[i][0], val1[i][1], size = per_spl)
                class_dic[class_type1[i]+"%d"%(j+1)] = lab[0]
                
            if j <= num_var[lab[1]]:
                data_dic[class_type2[i]+"%d"%(j+1)] = np.random.uniform(val2[i][0], val2[i][1], size = per_spl)
                class_dic[class_type2[i]+"%d"%(j+1)] = lab[1]
              
            if j <= num_var[lab[2]]:
                data_dic[class_type3[i]+"%d"%(j+1)] = np.random.pareto(val3[i][0], size = per_spl)*val3[i][1]
                class_dic[class_type3[i]+"%d"%(j+1)] = lab[2]
                
            if j <= num_var[lab[3]]:
                data_dic[class_type4[i]+"%d"%(j+1)] = np.random.poisson(val4[i], size = per_spl)
                class_dic[class_type4[i]+"%d"%(j+1)] = lab[3]
        
        k += 4    
    
    
    dtp = ("<U4", "<U4") #checked from printing variables
    return data_dic, class_dic, num_clust, dtp