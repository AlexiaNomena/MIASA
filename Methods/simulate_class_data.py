#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 20:04:14 2022

@author: raharinirina
"""
import numpy as np
import scipy.stats as stats
import pdb


""" Classification Experiment on random samples from specific probability distributions """
def generate_data_dist(var_data = False, noise = False):
    """Generate Artificial dinstinct distributions data"""
    per_spl = 200 # Num of iid observation in each samples 
    data_dic = {}
    class_type1 = ["1a", "1b", "1c", "1d", "1e", "1f", "1g", "1h", "1i", "1j"] # Normal Dist
    val1_mean = np.random.choice(15, size = len(class_type1), replace = False) # not allowing repeating means
    val1_std = np.random.uniform(1, 8, size = len(class_type1)) 
    val1 = [(val1_mean[k], val1_std[k]) for k in range(len(class_type1))]
    
    class_type2 = ["2a", "2b", "2c", "2d", "2e", "2f", "2g", "2h", "2i", "2j"] # Uniform Dist
    val2_a = np.random.choice(10, size = len(class_type2)) # allowing repeating start
    val2_b = np.random.choice(15, size = len(class_type2), replace = False) # not allowing repeated end
    val2 = [(val2_a[k], val2_a[k] + val2_b[k]) for k in range(len(class_type2))]
    
    class_type3 = ["3a", "3b", "3c", "3d", "3e", "3f", "3g", "3h", "3i", "3j"] # Pareto Dist
    val3_shape = np.random.choice(np.arange(1, 15), size = len(class_type3), replace = False)
    val3_scale = np.random.choice(np.arange(1, 15), size = len(class_type3), replace = False)
    val3 = [(val3_shape[k], val3_scale[k]) for k in range(len(class_type3))]
    
    class_type4 = ["4a", "4b", "4c", "4d", "4e", "4f", "4g", "4h", "4i", "4j"] # Poisson Dist
    val4 = np.random.choice(np.arange(1, 15), size = len(class_type4), replace = False)
    
    
    num_clust = len(class_type1) + len(class_type2) + len(class_type3) + len(class_type4)
    labs = np.cumsum(np.ones(num_clust)) - 1
    
    # Number of samples per classes
    MaxNumVar = 25
    if var_data:
        num_var_list = np.random.choice(np.arange(2, MaxNumVar), size = len(labs))
        num_var = {labs[k]: num_var_list[k] for k in range(len(labs))}
    else:
        num_var = {labs[k]:MaxNumVar for k in range(len(labs))}
    
    
    class_dic = {}
    k = 0
    for i in range(10):
        lab = labs[k:k+4]
        for j in range(MaxNumVar + 1):
            if j < num_var[lab[0]]:
                data_dic[class_type1[i]+"%d"%(j+1)] = np.random.normal(val1[i][0], val1[i][1], size = per_spl)
                class_dic[class_type1[i]+"%d"%(j+1)] = lab[0]
                
            if j < num_var[lab[1]]:
                data_dic[class_type2[i]+"%d"%(j+1)] = np.random.uniform(val2[i][0], val2[i][1], size = per_spl)
                class_dic[class_type2[i]+"%d"%(j+1)] = lab[1]
              
            if j < num_var[lab[2]]:
                data_dic[class_type3[i]+"%d"%(j+1)] = np.random.pareto(val3[i][0], size = per_spl)*val3[i][1]
                class_dic[class_type3[i]+"%d"%(j+1)] = lab[2]
                
            if j <= num_var[lab[3]]:
                data_dic[class_type4[i]+"%d"%(j+1)] = np.random.poisson(val4[i], size = per_spl)
                class_dic[class_type4[i]+"%d"%(j+1)] = lab[3]
        
        k += 4   
    
    
    dtp = ("<U4", "<U4") #his is the type of the labels checked from printing
    return data_dic, class_dic, num_clust, dtp

""" Classification Experiment on random samples from bivariate probability distributions """

def generate_data_correlated(var_data = False, noise = False):
    """Generate Artificial data from bivariate distributions """
    per_spl = 200 # Num of iid observation in each samples 
    data_dic = {}
    class_type1 = ["1a", "1b", "1c"] # bivariate Normal Dist
    mean_list = np.random.choice(10, size = (len(class_type1), 2), replace = False) # not allowing repeating means
    var1 =  np.random.uniform(2, 5, size = (len(class_type1), 2)) # 2*np.ones((len(class_type1), 2)) # fix variance for stability of stimulations
    corr1 = np.random.uniform(-1, 1, size = len(class_type1)) # always has to be less than the variance for a PSD covariance matrix (Gershgorin)
    
    class_type2 = ["2a", "2b", "2c"] # bivariate t Dist
    #val1 = [(0, 1), (0, 5), (2, 1)]
    loc_list = np.random.choice(10, size = (len(class_type1), 2), replace = False) # not allowing repeating means
    var2 = np.random.uniform(2, 5, size = (len(class_type1), 2)) # 2*np.ones((len(class_type1), 2)) # fix variance for stability of stimulations
    corr2 = np.random.uniform(-1, 1, size = len(class_type2))
    
    
    num_clust = len(class_type1) + len(class_type2)
    labs = np.cumsum(np.ones(num_clust)) - 1
    
    # Number of samples per classes
    MaxNumVar = 25
    if var_data:
        num_var_list = np.random.choice(np.arange(2, MaxNumVar), size = len(labs))
        num_var = {labs[k]: num_var_list[k] for k in range(len(labs))}
    else:
        num_var = {labs[k]:MaxNumVar for k in range(len(labs))}
    
    
    class_dic = {}
    k = 0
    for i in range(3):
        lab = labs[k:k+2]
        for j in range(MaxNumVar + 1):
            if j < num_var[lab[0]]:
                cov_i = np.array([[0, corr1[i]], [corr1[i], 0]]) + np.diag(var1[i, :])
                Z = np.random.multivariate_normal(mean_list[i, :], cov_i, size = per_spl)
                data_dic[class_type1[i]+"%d_%d"%(j+1, 0)] = Z[:, 0]
                class_dic[class_type1[i]+"%d_%d"%(j+1, 0)] = lab[0]
                
                data_dic[class_type1[i]+"%d_%d"%(j+1, 1)] = Z[:, 1]
                class_dic[class_type1[i]+"%d_%d"%(j+1, 1)] = lab[0]
                
            if j < num_var[lab[1]]:
                cov_i = np.array([[0, corr2[i]], [corr2[i], 0]]) + np.diag(var2[i, :])
                frozen_t = stats.multivariate_t(loc_list[i, :], cov_i)
                Z = frozen_t.rvs(size = per_spl)
                data_dic[class_type2[i]+"%d_%d"%(j+1, 0)] = Z[:, 0]
                class_dic[class_type2[i]+"%d_%d"%(j+1, 0)] = lab[1]
                
                data_dic[class_type2[i]+"%d_%d"%(j+1, 1)] = Z[:, 1]
                class_dic[class_type2[i]+"%d_%d"%(j+1, 1)] = lab[1]
               
        k += 2    
    dtp = ("<U4", "<U4") #This is the type of the labels checked from printing
    return data_dic, class_dic, num_clust, dtp



""" Classification Experiment on random samples from SSA of two Gene Regulation Network """

from .GRN_Models.SSA import SSA_Fixed_Width_Trajectory as ssa_func

from .GRN_Models.A_MRNA_No_UpRegulation import transitions as trans_NoI_G1 
from .GRN_Models.B_MRNA_No_UpRegulation import transitions as trans_NoI_G2 
from .GRN_Models.MRNA_Single_UpRegulation import transitions as trans_Up_G1
from .GRN_Models.MRNA_Double_UpRegulation import transitions as trans_Up_G1G2

from .GRN_Models.A_MRNA_No_UpRegulation import propensities as prop_NoI_G1 # extract M1
from .GRN_Models.B_MRNA_No_UpRegulation import propensities as prop_NoI_G2 # extract M2
from .GRN_Models.MRNA_Single_UpRegulation import propensities as prop_Up_G1 # extract M1, M2
from .GRN_Models.MRNA_Double_UpRegulation import propensities as prop_Up_G1G2 # extract M1, M2


def generate_data_twoGRN(var_data = False, noise = False):
    """Generate Artificial data from SSA of two Gene Regulation Network """
    per_spl = 100 #200 # Num of iid observation in each samples 
    T = np.linspace(0.0, 60.0, per_spl)
    #T = np.linspace(0.0, 5, per_spl)
    initial_state = np.array([0,0,0,0,0,0]) # ssa functions return trajectories of species in the order ('G1','G2','P1','P2', 'M1','M2'), 
    loc_mRNA = np.array([4, 5]) # M1, M2 are the simulated mRNA counts
    
    data_dic = {}
    class_type1 = ["NoI", "Up_G1", "Up_G1G2"] # bivariate Normal Dist
    ssa_func_list = [ssa_func, ssa_func, ssa_func, ssa_func] # can exclude change of parameters in this form
    trans = [(trans_NoI_G1, trans_NoI_G2), (trans_Up_G1, ), (trans_Up_G1G2, )]
    propens = [(prop_NoI_G1, prop_NoI_G2), (prop_Up_G1, ), (prop_Up_G1G2, )]
    
    num_clust = len(class_type1)
    labs = np.cumsum(np.ones(num_clust)) - 1
    
    # Number of samples per classes
    MaxNumVar = 25
    if var_data:
        num_var_list = np.random.choice(np.arange(2, MaxNumVar), size = len(labs), replace = False)
        num_var = {labs[k]: num_var_list[k] for k in range(len(labs))}
    else:
        num_var = {labs[k]:MaxNumVar for k in range(len(labs))}
    
    
    class_dic = {}
    k = 0
    for i in range(len(class_type1)):
        lab = [labs[k]]
        for j in range(MaxNumVar + 1):
            if j < num_var[lab[0]]:
                if class_type1[i] != "NoI":
                    ssa_i = ssa_func_list[i](Stochiometry = trans[i][0], Propensities = propens[i][0], X_0 = initial_state, T_Obs_Points = T)
                    Z = ssa_i[loc_mRNA, :]
                    data_dic[class_type1[i]+"%d_%d"%(j+1, 0)] = Z[0, :]
                    data_dic[class_type1[i]+"%d_%d"%(j+1, 1)] = Z[1, :]
                    class_dic[class_type1[i]+"%d_%d"%(j+1, 0)] = lab[0]
                    class_dic[class_type1[i]+"%d_%d"%(j+1, 1)] = lab[0]
                    
                else:
                    ssa_i_0 = ssa_func_list[i](Stochiometry = trans[i][0], Propensities = propens[i][0], X_0 = initial_state, T_Obs_Points = T)
                    data_dic[class_type1[i]+"%d_%d"%(j+1, 0)] = ssa_i_0[loc_mRNA[0], :]
                    ssa_i_1 = ssa_func_list[i](Stochiometry = trans[i][1], Propensities = propens[i][1], X_0 = initial_state, T_Obs_Points = T)
                    data_dic[class_type1[i]+"%d_%d"%(j+1, 1)] = ssa_i_1[loc_mRNA[1], :]
                    class_dic[class_type1[i]+"%d_%d"%(j+1, 0)] = lab[0]
                    class_dic[class_type1[i]+"%d_%d"%(j+1, 1)] = len(labs) # separate label for non-interacting species
        k += 1    
     
    dtp = ("<U4", "<U4") #This is the type of the labels checked from printing
    return data_dic, class_dic, num_clust, dtp
