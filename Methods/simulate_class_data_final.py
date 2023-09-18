#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 11:05:47 2023

@author: raharinirina
"""

import numpy as np
import scipy.stats as stats
import joblib as jb
from functools import partial 
import pdb
import seaborn as sns


""" Classification Experiment on random samples from FAMILIES of probability distributions with two replicates X,Y for each samples"""
def generate_data_dist(var_data = False, noise = False, palette = "tab20", custom_palette = False, random_state = None):
    
    if random_state is not None:
        np.random.seed(random_state) # in case one needs a reproducible result
    
    """Generate Artificial dinstinct distributions data but group them by familly"""
    per_spl = 300 # Num of iid observation in each samples 
    data_dic = {}
    #class_type1 = ["1a", "1b", "1c", "1d", "1e", "1f", "1g", "1h", "1i", "1j"] # Normal Dist
    class_type1 = ["1a", "1b", "1c", "1d", "1e"] # Family of Normal Dist
    val1_mean = np.random.choice(15, size = len(class_type1), replace = False) # not allowing repeating means
    val1_std = np.random.uniform(1, 8, size = len(class_type1)) 
    val1 = [(val1_mean[k], val1_std[k]) for k in range(len(class_type1))]
    
    #class_type2 = ["2a", "2b", "2c", "2d", "2e", "2f", "2g", "2h", "2i", "2j"] # Uniform Dist
    class_type2 = ["2a", "2b", "2c", "2d", "2e"] # Family Uniform Dist

    val2_a = np.random.choice(10, size = len(class_type2)) # allowing repeating start
    val2_b = np.random.choice(15, size = len(class_type2), replace = False) # not allowing repeated end
    val2 = [(val2_a[k], val2_a[k] + val2_b[k]) for k in range(len(class_type2))]
    
    #class_type3 = ["3a", "3b", "3c", "3d", "3e", "3f", "3g", "3h", "3i", "3j"] # Pareto Dist
    class_type3 = ["3a", "3b", "3c", "3d", "3e"] # Family of Pareto Dist
    
    val3_shape = np.random.choice(np.arange(3, 15), size = len(class_type3), replace = False)
    val3_scale = np.random.choice(np.arange(1, 15), size = len(class_type3), replace = False)
    val3 = [(val3_shape[k], val3_scale[k]) for k in range(len(class_type3))]
    
    #class_type4 = ["4a", "4b", "4c", "4d", "4e", "4f", "4g", "4h", "4i", "4j"] # Poisson Dist
    class_type4 = ["4a", "4b", "4c", "4d", "4e"] # Family of Poisson Dist
    val4 = np.random.choice(np.arange(1, 15), size = len(class_type4), replace = False)
    
    
    num_dist = len(class_type1) + len(class_type2) + len(class_type3) + len(class_type4)

    labs = np.cumsum(np.ones(num_dist)) - 1
    
    colors_dic = {}
    
    if not custom_palette:
        colors = sns.color_palette(palette, num_dist)
    else:
        colors = palette
    
    num_clust = 4 # 4 familly to identify
    # Number of samples per classes
    MaxNumVar = 25
    if var_data:
        num_var_list = np.random.choice(np.arange(2, MaxNumVar), size = len(labs))
        num_var = {labs[k]: num_var_list[k] for k in range(len(labs))}
    else:
        num_var = {labs[k]:MaxNumVar for k in range(len(labs))}
    
    ## collecting separated duplicates of samples
    data_dic["X_vars"] = []
    data_dic["Y_vars"] = []
    
    class_dic = {}
    k = 0
    for i in range(5):
        lab = labs[k:k+4]
        for j in range(MaxNumVar + 1):
            if j < num_var[lab[0]]:
                Z = np.random.normal(val1[i][0], val1[i][1], size = (2, per_spl))
                
                data_dic[class_type1[i]+"%d_%d"%(j+1, 0)] = Z[0, :]
                class_dic[class_type1[i]+"%d_%d"%(j+1, 0)] = 0#lab[0] 
                colors_dic[class_type1[i]+"%d_%d"%(j+1, 0)] = colors[k]
                data_dic["X_vars"].append(class_type1[i]+"%d_%d"%(j+1, 0))
                
                data_dic[class_type1[i]+"%d_%d"%(j+1, 1)] = Z[1, :]
                class_dic[class_type1[i]+"%d_%d"%(j+1, 1)] = 0#lab[0] 
                colors_dic[class_type1[i]+"%d_%d"%(j+1, 1)] = colors[k]
                data_dic["Y_vars"].append(class_type1[i]+"%d_%d"%(j+1, 1))

            if j < num_var[lab[1]]:
                Z = np.random.uniform(val2[i][0], val2[i][1], size = (2, per_spl))
                
                data_dic[class_type2[i]+"%d_%d"%(j+1, 0)] = Z[0, :]
                class_dic[class_type2[i]+"%d_%d"%(j+1, 0)] = 1#lab[1] 
                colors_dic[class_type2[i] +"%d_%d"%(j+1, 0)] = colors[k+1]
                data_dic["X_vars"].append(class_type2[i]+"%d_%d"%(j+1, 0))
                
                data_dic[class_type2[i]+"%d_%d"%(j+1, 1)] = Z[1, :]
                class_dic[class_type2[i]+"%d_%d"%(j+1, 1)] = 1#lab[1] 
                colors_dic[class_type2[i]+"%d_%d"%(j+1, 1)] = colors[k+1]
                data_dic["Y_vars"].append(class_type2[i]+"%d_%d"%(j+1, 1))
                
            if j < num_var[lab[2]]:
                Z = (np.random.pareto(val3[i][0], size = (2, per_spl)) + 1)*val3[i][1] ### according to the doc
                
                data_dic[class_type3[i]+"%d_%d"%(j+1, 0)] = Z[0, :]
                class_dic[class_type3[i]+"%d_%d"%(j+1, 0)] = 2#lab[2] 
                colors_dic[class_type3[i]+"%d_%d"%(j+1, 0)] = colors[k+2]
                data_dic["X_vars"].append(class_type3[i]+"%d_%d"%(j+1, 0))
                
                data_dic[class_type3[i]+"%d_%d"%(j+1, 1)] = Z[1, :]
                class_dic[class_type3[i]+"%d_%d"%(j+1, 1)] = 2#lab[2] 
                colors_dic[class_type3[i]+"%d_%d"%(j+1, 1)] = colors[k+2]
                data_dic["Y_vars"].append(class_type3[i]+"%d_%d"%(j+1, 1))
            
            
            if j <= num_var[lab[3]]:
                Z = np.random.poisson(val4[i], size = (2, per_spl))
                
                data_dic[class_type4[i]+"%d_%d"%(j+1, 0)] = Z[0, :]
                class_dic[class_type4[i]+"%d_%d"%(j+1, 0)] = 3#lab[3] 
                colors_dic[class_type4[i]+"%d_%d"%(j+1, 0)] = colors[k+3]
                data_dic["X_vars"].append(class_type4[i]+"%d_%d"%(j+1, 0))
                
                data_dic[class_type4[i]+"%d_%d"%(j+1, 1)] = Z[1, :]
                class_dic[class_type4[i]+"%d_%d"%(j+1, 1)] = 3#lab[3] 
                colors_dic[class_type4[i]+"%d_%d"%(j+1, 1)] = colors[k+3]
                data_dic["Y_vars"].append(class_type4[i]+"%d_%d"%(j+1, 1))
                     
        k += 4   
    
    data_dic["true_colors"] = colors_dic
    dtp = (str, str)
    return data_dic, class_dic, num_clust, dtp

""" Classification Experiment on random samples from bivariate probability distributions """

def generate_data_correlated(var_data = False, noise = False, palette = "tab20", random_state = None):
    
    if random_state is not None:
        np.random.seed(random_state) # in case one needs a reproducible result
        
    """Generate Artificial data from bivariate distributions """
    per_spl = 300 # Num of iid observation in each samples 
    data_dic = {}
    class_type1 = ["1a", "1b", "1c", "1d", "1e", "1f", "1g", "1h", "1i", "1j"]  # bivariate Normal Dist
    
    mean_list = np.random.choice(30, size = (len(class_type1), 2), replace = False) # not allowing repeating means
    var1 =  np.random.uniform(5, 10, size = (len(class_type1), 2)) # 2*np.ones((len(class_type1), 2)) # fix variance for stability of stimulations
    cov1 = np.random.uniform(-3, 3, size = len(class_type1)) # always has to be less than the variance for a PSD covariance matrix (Gershgorin)
    
  
    num_clust = len(class_type1)
    labs = np.cumsum(np.ones(num_clust)) - 1
    
    colors_dic = {}
    colors = sns.color_palette(palette, num_clust)
    
    # Number of samples per classes
    MaxNumVar = 25
    if var_data:
        num_var_list = np.random.choice(np.arange(2, MaxNumVar), size = len(labs))
        num_var = {labs[k]: num_var_list[k] for k in range(len(labs))}
    else:
        num_var = {labs[k]:MaxNumVar for k in range(len(labs))}
    
    ## collecting separated samples
    data_dic["X_vars"] = []
    data_dic["Y_vars"] = []
    
    class_dic = {}
    k = 0
    for i in range(10):
        lab = labs[k:k+1]
        for j in range(MaxNumVar + 1):
            if j < num_var[lab[0]]:
                cov_i = np.array([[0, cov1[i]], [cov1[i], 0]]) + np.diag(var1[i, :])
                Z = np.random.multivariate_normal(mean_list[i, :], cov_i, size = per_spl)
                data_dic[class_type1[i]+"%d_%d"%(j+1, 0)] = Z[:, 0]
                class_dic[class_type1[i]+"%d_%d"%(j+1, 0)] = lab[0]
                colors_dic[class_type1[i]+"%d_%d"%(j+1, 0)] =  colors[k]
                data_dic["X_vars"].append(class_type1[i]+"%d_%d"%(j+1, 0))
                
                data_dic[class_type1[i]+"%d_%d"%(j+1, 1)] = Z[:, 1]
                class_dic[class_type1[i]+"%d_%d"%(j+1, 1)] = lab[0]
                colors_dic[class_type1[i]+"%d_%d"%(j+1, 1)] = colors[k]
                data_dic["Y_vars"].append(class_type1[i]+"%d_%d"%(j+1, 1))
   
        k +=1
    dtp = (str, str)  #This is the type of the labels checked from printing
    
    data_dic["true_colors"] = colors_dic
    
    return data_dic, class_dic, num_clust, dtp


""" Classification Experiment on random samples from SSA of two Gene Regulation Network """
""" GRN_Models folder containing all relevant modules Can be downloaded from https://github.com/vikramsunkara/ScRNAseqMoments"""

from .GRN_Models.SSA import SSA_Fixed_Width_Trajectory as ssa_func

from .GRN_Models.A_MRNA_No_UpRegulation import transitions as trans_NoI_G1 
from .GRN_Models.B_MRNA_No_UpRegulation import transitions as trans_NoI_G2 
from .GRN_Models.MRNA_Single_UpRegulation import transitions as trans_Up_G1
from .GRN_Models.MRNA_Double_UpRegulation import transitions as trans_Up_G1G2

from .GRN_Models.A_MRNA_No_UpRegulation import propensities as prop_NoI_G1 # extract M1
from .GRN_Models.B_MRNA_No_UpRegulation import propensities as prop_NoI_G2 # extract M2
from .GRN_Models.MRNA_Single_UpRegulation import propensities as prop_Up_G1 # extract M1, M2
from .GRN_Models.MRNA_Double_UpRegulation import propensities as prop_Up_G1G2 # extract M1, M2

    
def load_data_twoGRN(var_data = False, noise = False, palette = "tab20", random_state = None):
    
    #if random_state is not None:
    #    np.random.seed(random_state) # in case one needs a reproducible result
    
    loc_mRNA = np.array([False,False,False,False,True,True]) #np.array([4, 5]) # M1, M2 are the simulated mRNA counts
    
    data_dic = {}
    class_type1 = ["NoI_", "S_Up_", "D_Up_"] 
    files = ["Data/2mRNA_100000/two_MRNA_No_Up_data_100000.pck", "Data/2mRNA_100000/two_MRNA_Single_Up_data_100000.pck", "Data/2mRNA_100000/two_MRNA_Double_Up_data_100000.pck"]
    
    labs = np.cumsum(np.ones(len(class_type1))) - 1
    num_clust = len(class_type1) + 1 #if the 2 Genes in the NoI was assigned to different classes (does not make much difference)
    
    # Number of samples per classes
    MaxNumVar = 25#25
    if var_data:
        num_var_list = np.random.choice(np.arange(2, MaxNumVar), size = len(labs), replace = False)
        num_var = {labs[k]: num_var_list[k] for k in range(len(labs))}
    else:
        num_var = {labs[k]:MaxNumVar for k in range(len(labs))}
    
    ## collecting separated samples
    data_dic["X_vars"] = []
    data_dic["Y_vars"] = []    
    
    colors_dic = {}
    colors = sns.color_palette(palette, num_clust)
    
    class_dic = {}
    k = 0
    for i in range(len(class_type1)):
        lab = [labs[k]]
        ## Load relevant datafiles
        Z = GRN_load(num_var[lab[0]], filename = files[i], loc_species = loc_mRNA, random_state = random_state)
       
        ## place the runs into the data_dic 
        for j in range(MaxNumVar + 1):
            if j < num_var[lab[0]]:
                data_dic[class_type1[i]+"%d_%d"%(j+1, 0)] = Z[:, 0, j]
                data_dic[class_type1[i]+"%d_%d"%(j+1, 1)] = Z[:, 1, j]
                data_dic["X_vars"].append(class_type1[i]+"%d_%d"%(j+1, 0))
                data_dic["Y_vars"].append(class_type1[i]+"%d_%d"%(j+1, 1))
                if class_type1[i] != "NoI_":
                    class_dic[class_type1[i]+"%d_%d"%(j+1, 0)] = lab[0]
                    class_dic[class_type1[i]+"%d_%d"%(j+1, 1)] = lab[0]
                    
                    colors_dic[class_type1[i]+"%d_%d"%(j+1, 0)] = colors[i]
                    colors_dic[class_type1[i]+"%d_%d"%(j+1, 1)] = colors[i]
                else:
                    class_dic[class_type1[i]+"%d_%d"%(j+1, 0)] = lab[0]
                    class_dic[class_type1[i]+"%d_%d"%(j+1, 1)] = max(labs)+1+lab[0] # separate label for non-interacting species 
                    colors_dic[class_type1[i]+"%d_%d"%(j+1, 0)] = colors[i]
                    colors_dic[class_type1[i]+"%d_%d"%(j+1, 1)] = colors[i] ## keep the same color because the difference is already shown in the shapes of the points
        
        k += 1    
    data_dic["true_colors"] = colors_dic
    dtp = (str, str) #This is the type of the labels checked from printing
    return data_dic, class_dic, num_clust, dtp
 
import pickle 
import matplotlib.pyplot as plt

def GRN_load(sample_size, filename, loc_species, random_state = None):
    if random_state is not None:
        np.random.seed(random_state) # in case one needs a reproducible result   
 
    f = open(filename,'rb')
    input_dic = pickle.load(f)
    f.close()
    
    Sim_data = input_dic['Obs'] # -> numpy array           (#Time, #Dim, #Repeats)
    inds = np.arange(0,Sim_data.shape[-1],1).astype(int) # repeats indexes, i.e., different cells
    sampled_inds = np.random.choice(inds, size=(4000, sample_size), replace = False) ### No repeating cells
    
    for s in range(sample_size):
        Samples = Sim_data[:, loc_species][:, :, sampled_inds[:, s]]
        mean_timecourse = np.mean(Samples, axis = 2)
        variance_timecourse = np.var(Samples, axis = 2)
        skew_timecourse = stats.skew(Samples, axis = 2)
        ### Delete Nan informations
        mean_timecourse[np.isnan(mean_timecourse)] = 0
        variance_timecourse[np.isnan(skew_timecourse)] = 0
        skew_timecourse[np.isnan(skew_timecourse)] = 0
        ### min-max normalization to bring the central moments on the same scale
        Z_sub = np.row_stack((mean_timecourse, variance_timecourse, skew_timecourse))      
        if s == 0:
            Z = Z_sub[:, :, np.newaxis]
        else:
            Z = np.concatenate((Z, Z_sub[:, :, np.newaxis]), axis = 2)
    
    ### min-max normalization to bring all central moments on the same scale
    Z = (Z - np.min(Z))/(np.max(Z) - np.min(Z))
    
    """
    w1 = 1#(1e-20+ np.linalg.norm(skew_timecourse[20:, 0]))
    w2 = 1#(1e-20+ np.linalg.norm(skew_timecourse[20:, 1]))
    
    fig = plt.figure(figsize = (3*10, 10))
    ax = fig.add_subplot(int("%d%d%d"%(1, 3, 1)))
    plt.plot(mean_timecourse[:, 0]/w1, label = "mean A")
    plt.plot(mean_timecourse[:, 1]/w2, label = "mean B")
    #ax.set_aspect("equal")
    plt.legend()
    
    ax1 = fig.add_subplot(int("%d%d%d"%(1, 3, 2)))
    plt.plot(variance_timecourse[:, 0]/w1, label = "var A")
    plt.plot(variance_timecourse[:, 1]/w2, label = "var B")
    #ax1.set_aspect("equal")
    plt.legend()
    
    ax2 = fig.add_subplot(int("%d%d%d"%(1, 3, 3)))
    plt.plot(skew_timecourse[:, 0]/w1, label = "skew A")
    plt.plot(skew_timecourse[:, 1]/w2, label = "skew B")
    #ax2.set_aspect("equal")
    plt.legend()
    plt.show()
    print("-----------------------------------------------------------------")
    """
    return Z
