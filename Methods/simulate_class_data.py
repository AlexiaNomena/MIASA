#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 20:04:14 2022

@author: raharinirina
"""
import numpy as np
import scipy.stats as stats
import joblib as jb
from functools import partial 
import pdb
import seaborn as sns


""" Classification Experiment on random samples from specific probability distributions """


def generate_data_dist(var_data = False, noise = False, palette = "tab20", custom_palette = False, random_state = None):
    
    if random_state is not None:
        np.random.seed(random_state) # in case one needs a reproducible result
    
    """Generate Artificial dinstinct distributions data"""
    per_spl = 300 # Num of iid observation in each samples 
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
    
    colors_dic = {}
    
    if not custom_palette:
        colors = sns.color_palette(palette, num_clust)
    else:
        colors = palette
    
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
                colors_dic[class_type1[i]+"%d"%(j+1)] = colors[k]
                
            if j < num_var[lab[1]]:
                data_dic[class_type2[i]+"%d"%(j+1)] = np.random.uniform(val2[i][0], val2[i][1], size = per_spl)
                class_dic[class_type2[i]+"%d"%(j+1)] = lab[1]
                colors_dic[class_type2[i] +"%d"%(j+1)] = colors[k+1]

              
            if j < num_var[lab[2]]:
                data_dic[class_type3[i]+"%d"%(j+1)] = np.random.pareto(val3[i][0], size = per_spl)*val3[i][1]
                class_dic[class_type3[i]+"%d"%(j+1)] = lab[2]
                colors_dic[class_type3[i]+"%d"%(j+1)] = colors[k+2]

                
            if j <= num_var[lab[3]]:
                data_dic[class_type4[i]+"%d"%(j+1)] = np.random.poisson(val4[i], size = per_spl)
                class_dic[class_type4[i]+"%d"%(j+1)] = lab[3]
                colors_dic[class_type4[i]+"%d"%(j+1)] = colors[k+3]

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
    #mean_list = np.random.choice(10, size = (len(class_type1), 2), replace = False) # not allowing repeating means
    #var1 =  np.random.uniform(2, 5, size = (len(class_type1), 2)) # 2*np.ones((len(class_type1), 2)) # fix variance for stability of stimulations
    #cov1 = np.random.uniform(-1, 1, size = len(class_type1)) # always has to be less than the variance for a PSD covariance matrix (Gershgorin)
    
    #mean_list = np.array([[0, 0], [10, 10], [0, 10], [10, 0], [20, 10], [10, 20]]) # not allowing repeating means
    #pdb.set_trace()
    #var1 = np.column_stack((2*(np.ones((len(class_type1)))), 2*(np.ones((len(class_type1))))))
    #cov1 = np.ones(len(class_type1))
    
    mean_list = np.random.choice(30, size = (len(class_type1), 2), replace = False) # not allowing repeating means
    #mean_list = np.zeros((len(class_type1), 2))
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
    num_clust = len(class_type1) # + 1 if the 2 Genes in the NoI was assigned to different classes (does not make much difference)
    
    # Number of samples per classes
    MaxNumVar = 25
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
                    class_dic[class_type1[i]+"%d_%d"%(j+1, 1)] = lab[0]#max(labs)+1+lab[0] # if separate label for non-interacting species (does not make much difference)
                    colors_dic[class_type1[i]+"%d_%d"%(j+1, 0)] = colors[i]
                    colors_dic[class_type1[i]+"%d_%d"%(j+1, 1)] = colors[i] #colors[-1] # if separate label for non-interacting species (does not make much difference)
        
        k += 1    
    data_dic["true_colors"] = colors_dic
    dtp = (str, str) #This is the type of the labels checked from printing
    return data_dic, class_dic, num_clust, dtp
 
import pickle 
import matplotlib.pyplot as plt

def GRN_load(sample_size, filename, loc_species, random_state = None):
    #if random_state is not None:
    #    np.random.seed(random_state) # in case one needs a reproducible result   
    
    for s in range(sample_size):
        f = open(filename,'rb')
        input_dic = pickle.load(f)
        f.close()
        
        Sim_data = input_dic['Obs'] # -> numpy array           (#Time, #Dim, #Repeats)
        inds = np.arange(0,Sim_data.shape[-1],1,dtype=np.int) # repeats indexes, i.e., different cells
        sampled_inds = np.random.choice(inds, size=10000, replace=False) # No repeating samples
        Samples = Sim_data[:, loc_species][:, :, sampled_inds]
        mean_timecourse = np.mean(Samples, axis = 2)
        variance_timecourse = np.var(Samples, axis = 2)
        skew_timecourse = stats.skew(Samples, axis = 2)
        Z_sub = np.row_stack((mean_timecourse, variance_timecourse, skew_timecourse))#, skew_timecourse, variance_timecourse))
        if s == 0:
            Z = Z_sub[:, :, np.newaxis]
        else:
            Z = np.concatenate((Z, Z_sub[:, :, np.newaxis]), axis = 2)
    
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

def GRN_load_raw(sample_size, filename, loc_species, random_state = None):
    if random_state is not None:
        np.random.seed(random_state) # in case one needs a reproducible result
        
    f = open(filename,'rb')
    input_dic = pickle.load(f)
    f.close()
    
    Sim_data = input_dic['Obs'] # -> numpy array           (#Time, #Dim, #Repeats)
    inds = np.arange(0,Sim_data.shape[-1],1,dtype=np.int) # repeats indexes, i.e., different cells
    sampled_inds = np.random.choice(inds, size=sample_size, replace=False) # No repeating samples
    Z = Sim_data[:, loc_species][:, :, sampled_inds]
    
    return Z

 
"""---------------------------- Prev functions --------------------------------------------------------"""
def generate_data_correlated_2(var_data = False, noise = False, palette = "tab20", random_state = None):
    
    if random_state is not None:
        np.random.seed(random_state) # in case one needs a reproducible result
        
    """Generate Artificial data from bivariate distributions """
    per_spl = 200 # Num of iid observation in each samples 
    data_dic = {}
    class_type1 = ["1a", "1b", "1c"] # bivariate Normal Dist
    mean_list1 = np.random.choice(10, size = (len(class_type1), 2), replace = False) # not allowing repeating means
    var1 =  np.random.uniform(2, 5, size = (len(class_type1), 2)) # 2*np.ones((len(class_type1), 2)) # fix variance for stability of stimulations
    corr1 = np.random.uniform(-1, 1, size = len(class_type1)) # always has to be less than the variance for a PSD covariance matrix (Gershgorin)
    
    class_type2 = ["2a", "2b", "2c"] # bivariate_Normal Dist
    mean_list2 = np.random.choice(15, size = (len(class_type2), 2), replace = False)
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
    
    ## collecting separated samples
    data_dic["X_vars"] = []
    data_dic["Y_vars"] = []
    
    class_dic = {}
    k = 0
    for i in range(3):
        lab = labs[k:k+2]
        for j in range(MaxNumVar + 1):
            if j < num_var[lab[0]]:
                cov_i = np.array([[0, corr1[i]], [corr1[i], 0]]) + np.diag(var1[i, :])
                Z = np.random.multivariate_normal(mean_list1[i, :], cov_i, size = per_spl)
                data_dic[class_type1[i]+"%d_%d"%(j+1, 0)] = Z[:, 0]
                class_dic[class_type1[i]+"%d_%d"%(j+1, 0)] = lab[0]
                data_dic["X_vars"].append(class_type1[i]+"%d_%d"%(j+1, 0))
                
                data_dic[class_type1[i]+"%d_%d"%(j+1, 1)] = Z[:, 1]
                class_dic[class_type1[i]+"%d_%d"%(j+1, 1)] = lab[0]
                data_dic["Y_vars"].append(class_type1[i]+"%d_%d"%(j+1, 1))

            if j < num_var[lab[1]]:
                cov_i = np.array([[0, corr2[i]], [corr2[i], 0]]) + np.diag(var2[i, :])
                Z = np.random.multivariate_normal(mean_list2[i, :], cov_i, size = per_spl)
                
                data_dic[class_type2[i]+"%d_%d"%(j+1, 0)] = Z[:, 0]
                class_dic[class_type2[i]+"%d_%d"%(j+1, 0)] = lab[1]
                data_dic["X_vars"].append(class_type2[i]+"%d_%d"%(j+1, 0))

                data_dic[class_type2[i]+"%d_%d"%(j+1, 1)] = Z[:, 1]
                class_dic[class_type2[i]+"%d_%d"%(j+1, 1)] = lab[1]
                data_dic["Y_vars"].append(class_type2[i]+"%d_%d"%(j+1, 1))

        k += 2    
    dtp = (str, str) 
    return data_dic, class_dic, num_clust, dtp

def generate_data_twoGRN(var_data = False, noise = False, palette = "tab20", random_state = None):
    if random_state is not None:
        np.random.seed(random_state) # in case one needs a reproducible result
    
    """Generate Artificial data from SSA of two Gene Regulation Network """
    per_spl = 100 # Num of observation in each samples 
    T = np.linspace(0.0, 60.0, per_spl)
    #T = np.linspace(0.0, 5, per_spl)
    initial_state = np.array([0,0,0,0,0,0]) # ssa functions return trajectories of species in the order ('G1','G2','P1','P2', 'M1','M2'), 
    loc_mRNA = np.array([4, 5]) # M1, M2 are the simulated mRNA counts
    
    data_dic = {}
    class_type1 = ["NoI", "S_Up", "D_Up"]
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
    
    
    ## collecting separated samples
    data_dic["X_vars"] = []
    data_dic["Y_vars"] = []
    
    class_dic = {}
    k = 0

    for i in range(len(class_type1)):
        lab = [labs[k]]
        ## Run SSA separately with Joblib because each of then is slow
        GRN_sub_p = partial(GRN_sub, i=i, num_var=num_var, lab=lab, 
                            class_type = class_type1, ssa_func_list=ssa_func_list, 
                            loc_mRNA = loc_mRNA, trans = trans, propens = propens, 
                            initial_state = initial_state, T = T)
        
        res_list = jb.Parallel(n_jobs = 8)(jb.delayed(GRN_sub_p)(j) for j in range(MaxNumVar + 1))
        ## place the runs into the data_dic 
        for j in range(MaxNumVar + 1):
            if j < num_var[lab[0]]:
                Z = res_list[j]
                data_dic["X_vars"].append(class_type1[i]+"%d_%d"%(j+1, 0))
                data_dic["Y_vars"].append(class_type1[i]+"%d_%d"%(j+1, 1))
                if class_type1[i] != "NoI":
                    data_dic[class_type1[i]+"%d_%d"%(j+1, 0)] = Z[0, :]
                    data_dic[class_type1[i]+"%d_%d"%(j+1, 1)] = Z[1, :]
                    class_dic[class_type1[i]+"%d_%d"%(j+1, 0)] = lab[0]
                    class_dic[class_type1[i]+"%d_%d"%(j+1, 1)] = lab[0]
                else:
                    data_dic[class_type1[i]+"%d_%d"%(j+1, 0)] = Z[0]
                    data_dic[class_type1[i]+"%d_%d"%(j+1, 1)] = Z[1]
                    class_dic[class_type1[i]+"%d_%d"%(j+1, 0)] = lab[0]
                    class_dic[class_type1[i]+"%d_%d"%(j+1, 1)] = max(labs)+1+lab[0] # separate label for non-interacting species
    
    num_clust = num_clust + 1 # the 2 Genes in the NoI was assigned to different classes
    dtp = (str, str) #This is the type of the labels checked from printing
    return data_dic, class_dic, num_clust, dtp


def GRN_sub(j, i, num_var, lab, class_type, ssa_func_list, loc_mRNA, trans, propens, initial_state, T):
    if j < num_var[lab[0]]:

        if class_type[i] != "NoI":
            ssa_i = ssa_func_list[i](Stochiometry = trans[i][0], Propensities = propens[i][0], X_0 = initial_state, T_Obs_Points = T)
            Z = ssa_i[loc_mRNA, :]
            
        else:
            ssa_i_0 = ssa_func_list[i](Stochiometry = trans[i][0], Propensities = propens[i][0], X_0 = initial_state, T_Obs_Points = T)
            ssa_i_1 = ssa_func_list[i](Stochiometry = trans[i][1], Propensities = propens[i][1], X_0 = initial_state, T_Obs_Points = T)
            Z = (ssa_i_0[loc_mRNA[0], :], ssa_i_1[loc_mRNA[1], :])
          
            
        return Z

def generate_data_correlated_ver2(var_data = False, noise = False, palette = "tab20", random_state = None):
    
    if random_state is not None:
        np.random.seed(random_state) # in case one needs a reproducible result
        
    """Generate Artificial data from bivariate distributions """
    per_spl = 300 # Num of iid observation in each samples 
    data_dic = {}
    class_type1 = ["1a", "1b", "1c", "1d", "1e", "1f", "1g", "1h", "1i", "1j"] # bivariate Normal Dist
    
    #mean_list = np.array([[0, 0], [10, 10], [0, 10], [10, 0], [20, 10], [10, 20]]) # not allowing repeating means
    #pdb.set_trace()
    #var1 = np.column_stack((2*(np.ones((len(class_type1)))), 2*(np.ones((len(class_type1))))))
    #cov1 = np.ones(len(class_type1))
    
    mean_list = np.random.choice(50, size = (len(class_type1), 2), replace = False) # not allowing repeating means
    var1 =  np.random.uniform(5, 10, size = (len(class_type1), 2)) # 2*np.ones((len(class_type1), 2)) # fix variance for stability of stimulations
    cov1 = np.random.uniform(-3, 3, size = len(class_type1)) # always has to be less than the variance for a PSD covariance matrix (Gershgorin)
    
    class_type2 = ["2a", "2b", "2c", "2d", "2e", "2f", "2g", "2h", "2i", "2j"] # bivariate t Dist
    nu = 3 ##
    loc_list = 2*np.random.choice(50, size = (len(class_type1), 2), replace = False) # not allowing repeating means
    var2 = np.random.uniform(5, 10, size = (len(class_type1), 2)) # 2*np.ones((len(class_type1), 2)) # fix variance for stability of stimulations
    sigma2 = np.random.uniform(-3, 3, size = len(class_type2))

    
    num_clust = len(class_type1) + len(class_type2)
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
        lab = labs[k:k+2]
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
                

               
            if j < num_var[lab[1]]:
                SIGMA = ((nu-2)/nu)*np.array([[0, sigma2[i]], [sigma2[i], 0]]) + np.diag(var2[i, :]) 
                frozen_t = stats.multivariate_t(loc_list[i, :], SIGMA, df = nu) # the covariance is given by (nu/(nu - 2))SIGMA for nu>3 (https://en.wikipedia.org/wiki/Multivariate_t-distribution)
                Z = frozen_t.rvs(size = per_spl)
                data_dic[class_type2[i]+"%d_%d"%(j+1, 0)] = Z[:, 0]
                class_dic[class_type2[i]+"%d_%d"%(j+1, 0)] = lab[1]
                colors_dic[class_type2[i]+"%d_%d"%(j+1, 0)] =  colors[k+1]
                data_dic["X_vars"].append(class_type2[i]+"%d_%d"%(j+1, 0))

                data_dic[class_type2[i]+"%d_%d"%(j+1, 1)] = Z[:, 1]
                class_dic[class_type2[i]+"%d_%d"%(j+1, 1)] = lab[1]
                colors_dic[class_type2[i]+"%d_%d"%(j+1, 1)] = colors[k+1]
                data_dic["Y_vars"].append(class_type2[i]+"%d_%d"%(j+1, 1))
               
        k += 2   
    dtp = (str, str)  #This is the type of the labels checked from printing
    
    data_dic["true_colors"] = colors_dic
    
    return data_dic, class_dic, num_clust, dtp

