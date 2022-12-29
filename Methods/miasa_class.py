#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 14:03:16 2022

@author: raharinirina
"""

from .Core.Generate_Distances import Similarity_Distance, Association_Distance
from .Generate_Features import eCDF, eCDF_KS_stat, eCDF_KS_p1, eCDF_KS_p2, Sub_Eucl, covariance, covariance_moms, moms_covariance, corrcoeff, corrcoeff_moms, moms_corrcoeff, moms, moms_OR
from .Core.Lower_dim import get_clusters
from .Core.qEmbedding import Euclidean_Embedding

import sys
import numpy as np
import pdb


def Miasa_Class(X, Y, num_clust, c_dic = None, dist_origin = True, metric_method = "eCDF-KS-stat", clust_method = "Kmeans", palette = "tab20", Feature_dic = None, in_threads = True):
    """Compute features"""
    
    if metric_method == "eCDF-KS-stat":
       Feature_X, Feature_Y, func, ftype = eCDF_KS_stat(X,Y)
       
    elif metric_method == "eCDF-KS-p1":
        Feature_X, Feature_Y, func, ftype = eCDF_KS_p1(X,Y)
        
    elif metric_method == "eCDF-KS-p2":
        Feature_X, Feature_Y, func, ftype = eCDF_KS_p2(X,Y)
        
    elif metric_method == "eCDF":
        Feature_X, Feature_Y, func, ftype = eCDF(X,Y)
        
    elif metric_method == "Cov":
        Feature_X, Feature_Y, func, ftype = covariance(X, Y)
    
    elif metric_method == "Cov_Moms":
        Feature_X, Feature_Y, func, ftype = covariance_moms(X, Y)
        
    elif metric_method == "Moms_Cov":
        Feature_X, Feature_Y, func, ftype = moms_covariance(X, Y)
    
    elif metric_method == "Corr":
        Feature_X, Feature_Y, func, ftype = corrcoeff(X, Y)
    
    elif metric_method == "Corr_Moms":
        Feature_X, Feature_Y, func, ftype = corrcoeff_moms(X, Y)
    
    elif metric_method == "Moms_Corr":
        Feature_X, Feature_Y, func, ftype = moms_corrcoeff(X, Y)
        
    elif metric_method == "Moms":
        Feature_X, Feature_Y, func, ftype = moms(X, Y)
        
    elif metric_method == "Moms_OR":
        Feature_X, Feature_Y, func, ftype = moms_OR(X, Y)
    
    elif metric_method == "Sub_Eucl":
       Feature_X, Feature_Y, func, ftype = Sub_Eucl(X, Y)
       
    else:
        try:
            Feature_X, Feature_Y, func, ftype = Feature_dic["Feature_X"], Feature_dic["Feature_Y"], Feature_dic["Asssociation_function"], Feature_dic["assoc_func_type"]
        except:
            sys.exit("Check implemented metric_methods or give a parameter Feature_dic must be given: keys Feature_X (ndarray), Feature_Y (ndarray), Association_function (func), assoc_func_type(str vectorized or str not_vectorized)")

    Result = get_class(X, Y, Feature_X, Feature_Y, func, ftype, metric_method, c_dic, dist_origin, num_clust, clust_method, palette, in_threads = in_threads)

    return Result
    

def get_class(X, Y, Feature_X, Feature_Y, func, ftype, metric_method, c_dic, dist_origin = False, num_clust=None, clust_method = "Kmeans", palette = "tab20", in_threads = True):
    """ Similarity metric """
    DX = Similarity_Distance(Feature_X, method = "Euclidean")
    DY = Similarity_Distance(Feature_Y, method = "Euclidean")
    
    """Association metric"""
   
    Z = (X, Y)
    
    D_assoc = Association_Distance(Z, func, ftype)
    """Distane to origin Optional but must be set to None if not used"""
    if dist_origin:
        Orows = np.linalg.norm(Feature_X, axis = 1)
        Ocols = np.linalg.norm(Feature_Y, axis = 1)
    else:
        Orows = None
        Ocols = None
    
   
    """ Get joint Euclidean embedding """
    M = Feature_X.shape[0]
    N = Feature_Y.shape[0]
    if c_dic is None or c_dic == "default":
        c1, c2 = 1/2, 2
        a = 1. - 1./(M+N)
        b = 2.*c2/(M+N)
        c3 =  min(((2.*c1 + c2) - b)/a, 2*c2+2)
        c_dic = {"c1":c1, "c2":c2, "c3":c3}
    else:
        c_dic = c_dic
    
    alpha = np.max(D_assoc) # adding a constant to the Euclidean distances to statisfy one of the conditions for embedding
    Coords, vareps = Euclidean_Embedding(DX+alpha, DY+alpha, Orows+alpha, Ocols+alpha, D_assoc, c_dic, in_threads = in_threads)
    
    if Coords is not None:
        if clust_method == "Kmeans":
            if num_clust == None:
                sys.exit("Kmeans requires number of clusters parameter: num_clust")
            else:
                clust_labels, color_clustered = get_clusters(Coords, num_clust, palette, method = clust_method)
        
        elif clust_method == "Kmedoids":
            if num_clust == None:
                sys.exit("Kmedoids requires number of clusters parameter: num_clust")
            else:
                clust_labels, color_clustered = get_clusters(Coords, num_clust, palette, method = clust_method)
                
        elif clust_method[:13] == "Agglomerative": 
            clust_labels, color_clustered = get_clusters(Coords, num_clust, palette, method = clust_method)
            
            
        else:
            sys.exit("A metric-distance based clustering method is better for MIASA \n Available here is Kmeans")
        
        if dist_origin:
            Coords = Coords - Coords[M, :][np.newaxis, :]
            Class_pred = np.concatenate((clust_labels[:M], clust_labels[M+1:]), axis = 0)
            was_orig = True
        else:
            Class_pred = clust_labels
            was_orig = False
        
        Result = {"Coords": Coords, "shape":(M, N), "was_orig":was_orig, "vareps":vareps, "Class_pred":Class_pred, "clust_labels":clust_labels, "color_clustered":color_clustered}
    
    else:
        
        Result = None   
        
    return Result
    
    