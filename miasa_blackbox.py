#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 18:49:50 2023

@author: raharinirina
"""

""" Necessary modules """
from matplotlib.backends.backend_pdf import PdfPages
from Methods.classify import split_data, plotClass
from Methods.miasa_class import Miasa_Class

""" Name and origin of dataset """
DataName = "Distribution_data"
from Methods.simulate_class_data import generate_data_dist

""" Euclidean embedding pameters only used in MIASA (includes a finite number of auto adjustements)"""
# type c_dic = {"c1":float, "c2":float, "c3":float} 
c_dic = "default" 

""" Load or Generate data: 
    
    Required:
    X and Y are the required separated datasets with M, N samples respectively and K realizations on the rows
    X.shape = (M, K) 
    Y.shape = (N, K)
    num_clust = number of clusters
    dtp : tuple (datatype X_vars , datatype Y_vars) is needed for visualization only
    

    Not Required:
    X_vars, Y_vars = Labels of X and Y samples
    Class_True = True cluster labels of samples
    
"""
data_dic_orig, class_dic, num_clust, dtp = generate_data_dist(var_data = False)
X, Y, Class_True, X_vars, Y_vars = split_data(data_dic_orig, class_dic)


"""
Parameters of MIASA
"""

metric_method = ("eCDF", "KS-stat") # (Similarity, Association) distance models
clust_method = "Kmeans" # clustering aglorithm to use
palette = "tab20" # seaborn color palette
dist_origin = (True,False) # for datasets (X, Y) decide if the distance to the origin of the axes is interpretable as the norm of the feature representations of the samples
in_threads = False # True to avoid broken runs when using parallel jobs (relevant only for class_experiments)

"""
If desired custom similarity feature representation (defining Euclidean similarity distance) 
and association measures then 

1) set 
    metric_method = "precomputed"
2) give Feature_dic as parameter
    Feature_dic["Feature_X"] : array similarity features of dataset X: M samples and L features, X.shape = (M, L)
    Feature_dic["Feature_Y"] : array similarity features of dataset Y: M samples and S features, Y.shape = (N, S)
    Feature_dic["Asssociation_function"] : a function of argument tuple full datasets (X, Y) or pair of samples (X_i, Y_j) computing the pairwise association between the samples of X and Y
    
    Feature_dic["assoc_func_type"] : type of the association function as options
                                    option 1: str vectorized     : argument full datasets (X, Y) => return directly the Asscociation distance matrix of shape (M, N) 
                                    option 2: str not_vectorized : argument samples (X_i, Y_j)   => return a scalar = Associaiton distance between sample X_i and sample Y_j

        
example:
    
from Methods.Generate_Features import eCDF, get_assoc_func
metric_method = "precomputed"
Feature_dic = {} 
Feature_dic["Feature_X"], Feature_dic["Feature_Y"] = eCDF(X,Y)
Feature_dic["Asssociation_function"], Feature_dic["assoc_func_type"] = get_assoc_func("KS-stat")
"""
Feature_dic = None

""" 
Perform MIASA Classification of samples

"""
Id_Class = Miasa_Class(X, Y, num_clust, 
                       dist_origin = dist_origin, 
                       metric_method = metric_method, 
                       clust_method = clust_method, 
                       c_dic = c_dic, Feature_dic = Feature_dic,
                       in_threads = in_threads,
                       palette = palette)


"""Lower Dimensional visualization of clusters (UMAP visualization)"""
pdf= PdfPages("Figures/"+DataName+".pdf")
plotClass(Id_Class, X_vars, Y_vars, pdf, dtp, run_num = 1, n_neighbors = 5, min_dist = 0.99)
pdf.close()
import matplotlib.pyplot as plt
plt.show()





