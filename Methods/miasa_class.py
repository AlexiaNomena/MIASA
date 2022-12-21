#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 14:03:16 2022

@author: raharinirina
"""

from .Core.Generate_Distances import Similarity_Distance, Association_Distance
from .Generate_Features import KS_v1, KS_v2, Sub_Eucl, covariance, corrcoeff
from .Core.Lower_dim import get_clusters
from .Core.qEmbedding import Euclidean_Embedding

import sys
import numpy as np
import pdb


def Miasa_Class(X, Y, num_clust, c_dic = None, dist_origin = True, metric_method = "KS-statistic", clust_method = "Kmeans", palette = "tab20"):
    """Compute features"""
    if metric_method == "KS-statistic":
       Feature_X, Feature_Y, func, ftype = KS_v1(X,Y)
       
    elif metric_method == "KS-p_value":
        Feature_X, Feature_Y, func, ftype = KS_v2(X,Y)
        
    elif metric_method == "Covariance":
        Fearture_X, Feature_Y, func, ftype = covariance(X, Y)
    
    elif metric_method == "CorrCoeff":
        Fearture_X, Feature_Y, func, ftype = corrcoeff(X, Y)
        
    else:
       Feature_X, Feature_Y, func, ftype = Sub_Eucl(X, Y)

    Result = get_class(X, Y, Feature_X, Feature_Y, func, ftype, metric_method, c_dic, dist_origin, num_clust, clust_method, palette)

    return Result
    

def get_class(X, Y, Feature_X, Feature_Y, func, ftype, metric_method, c_dic, dist_origin = False, num_clust=None, clust_method = "Kmeans", palette = "tab20"):
    """ Similarity metric """
    DX = Similarity_Distance(Feature_X, method = "Euclidean")
    DY = Similarity_Distance(Feature_Y, method = "Euclidean")
    
    """Association metric"""
    if metric_method in ("KS-statistic", "Covariance", "CorrCoeff"):
        Z = (X, Y)
    else:
        Z = (Feature_X, Feature_Y)
    
    D_assoc = Association_Distance(Z, func, ftype)
    """Distane to origin Optional but must be set to None if not used"""
    if dist_origin:
        Orows = np.linalg.norm(Feature_X, axis = 1)
        Ocols = np.linalg.norm(Feature_Y, axis = 1)
    else:
        Orows = None
        Ocols = None
    
    M = Feature_X.shape[0]
    N = Feature_Y.shape[0]
    if c_dic is None:
        c1, c2 = 1/2, 2
        a = 1. - 1./(M+N)
        b = 2.*c2/(M+N)
        c3 =  min(((2.*c1 + c2) - b)/a, 2*c2+2)
        c_dic = {"c1":c1, "c2":c2, "c3":c3}     
    else:
        c_dic = c_dic
    
    alpha = np.max(D_assoc) # adding a constant to the Euclidean distances to statisfy one of the conditions for embedding
    Coords, vareps = Euclidean_Embedding(DX+alpha, DY+alpha, Orows+alpha, Ocols+alpha, D_assoc, c_dic)
    
    if clust_method == "Kmeans":
        if num_clust == None:
            sys.exit("Kmeans requires number of clusters parameter: num_clust")
        else:
            clust_labels, color_clustered = get_clusters(Coords, num_clust, palette, method = "Kmeans")
    elif clust_method == "Kmedoids":
        if num_clust == None:
            sys.exit("Kmedoids requires number of clusters parameter: num_clust")
        else:
            clust_labels, color_clustered = get_clusters(Coords, num_clust, palette, method = "Kmedoids")
        
    else:
        sys.exit("A metric-distance based clustering method is better for MIASA \n Available here is Kmeans")
    
    if dist_origin:
        Coords = Coords - Coords[M, :][np.newaxis, :]
        Class_pred = np.concatenate((clust_labels[:M], clust_labels[M+1:]), axis = 0)
        was_orig = True
    else:
        Class_pred = clust_labels
        was_orig = False
        
    return {"Coords": Coords, "shape":(M, N), "was_orig":was_orig, "vareps":vareps, "Class_pred":Class_pred, "clust_labels":clust_labels, "color_clustered":color_clustered}
    
    