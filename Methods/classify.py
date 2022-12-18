#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 14:03:16 2022

@author: raharinirina
"""

from .Core.Generate_Distances import Similarity_Metric, Association_Metric
from .Generate_Features import EmpCDF
from .Core.Lower_dim import get_clusters
from .Core.qEmbedding import Euclidean_Embedding

import sys
import numpy as np

def Class_Identification(X, Y, num_clust, emb_params = None, dist_origin = True, method = "Kolmogorov-Smirnov", clust_method = "Kmeans", palette = "tab20"):
    """Compute features"""
    if method == "Kolmogorov-Smirnov":
       Feature_X, Feature_Y, func = KS(X,Y)
    else:
       Feature_X, Feature_Y, func = Sub_Eucl(X, Y)

    Result = get_class(Feature_X, Feature_Y, func, emb_params, dist_origin, num_clust, clust_method, palette)

    return Result


def KS(X,Y):
    lbd = min(np.min(X), np.min(Y))
    ubd = max(np.max(X), np.max(Y))
    
    interval = np.linspace(lbd, ubd, 500)
    Feature_X = EmpCDF(X, interval)
    Feature_Y = EmpCDF(Y, interval)
    func = lambda Features: np.max(np.abs(Features[0] - Features[1]))
    return Feature_X, Feature_Y, func


def Sub_Eucl(X, Y):
    Feature_X = X.copy()
    Feature_Y = Y.copy()
    func = lambda Features: np.max(np.abs(Features[0][:, np.newaxis] - Features[1][np.newaxis, :]))
    return Feature_X, Feature_Y, func
    

def get_class(Feature_X, Feature_Y, func, emb_params, dist_origin = False, num_clust=None, clust_method = "Kmeans", palette = "tab20"):
    """ Similarity metric """
    DX = Similarity_Metric(Feature_X, method = "Euclidean")
    DY = Similarity_Metric(Feature_Y, method = "Euclidean")
    
    """Association metric"""
    Features = (Feature_X, Feature_Y)
    D_assoc = Association_Metric(Features, func)
    
    """Distane to origin Optional but must be set to None if not used"""
    if dist_origin:
        Orow = np.linalg.norm(Feature_X, axis = 1)
        Ocols = np.linalg.norm(Feature_Y, axis = 1)
    else:
        Orow = None
        Ocols = None
    
    if emb_params is None:
        M = Feature_X.shape[0]
        N = Feature_Y.shape[0]
        c1, c2 = 1/2, 2
        a = 1. - 1./(M+N)
        b = 2.*c2/(M+N)
        c3 =  min(((2.*c1 + c2) - b)/a, 2*c2+2)
        c = {"c1":c1, "c2":c2, "c3":c3} 
    else:
        c = emb_params
        
    Coords, vareps = Euclidean_Embedding(DX, DY, Orow, Ocols, D_assoc, c)
    if clust_method == "Kmeans":
        if num_clust == None:
            sys.exit("Kmeans requires number of clusters parameter: num_clust")
        else:
            clust_labels, color_clustered = get_clusters(Coords, num_clust, palette, method = "Kmeans")
    else:
        sys.exit("clust_method is not available")
    
    if dist_origin:
        Coords = Coords - Coords[M, :][np.newaxis, :]
        Class_pred = np.concatenate((clust_labels[:M], clust_labels[M+1:]), axis = 0)
    else:
        Class_pred = clust_labels
        
    
    return {"Coords": Coords, "vareps":vareps, "Class_pred":Class_pred, "clust_labels":clust_labels, "color_clustered":color_clustered}
    
    