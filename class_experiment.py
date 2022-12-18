#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 13:21:27 2022

@author: raharinirina
"""
from Methods.Core.Generate_Distances import Similarity_Metric, Association_Metric
from Methods.Generate_Features import EmpCDF
import numpy as np

def Class_Identification(X, Y, method = "Kolmogorov-Smirnov"):
    # Compute features
    if method == "Kolmogorov-Smirnov":
        Feature_X = EmpCDF(X)
        Feature_Y = EmpCDF(Y)
        func = lambda Features: np.max(np.abs(Features[0][:, np.newaxis] - Features[1][np.newaxis, :]))
    else:
        Feature_X = X.copy()
        Feature_Y = Y.copy()
        func = lambda Features: np.max(np.abs(Features[0][:, np.newaxis] - Features[1][np.newaxis, :]))
    
    # Similarity metric
    DX = Similarity_Metric(Feature_X, method = "Euclidean")
    DY = Similarity_Metric(Feature_Y, method = "Euclidean")
    
    # Association metric
    D_assoc = Association_Metric(Feature_X, Feature_Y, func)
    
    