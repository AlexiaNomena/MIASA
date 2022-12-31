#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 10:27:56 2022

@author: raharinirina
"""
import numpy as np
from .Generate_Features import eCDF, eCDF_KS_stat, eCDF_KS_p1, eCDF_KS_p2, Sub_Eucl, covariance, covariance_moms 
from .Generate_Features import moms_covariance, corrcoeff, corrcoeff_moms, moms_corrcoeff, moms, OR, moms_OR, OR_moms
from .Core.Generate_Distances import Similarity_Distance, Association_Distance, KS_Distance, KS_Distance_Mixed

from .Core.Lower_dim import get_clusters
from .Core.CosLM import Prox_Mat
import sys
import pdb

def NonMetric_Class(X, Y, num_clust, dist_origin = True, metric_method = "eCDF-KS-stat", clust_method = "Kmeans", palette = "tab20", Feature_dic = None, in_threads = True):
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
    
    elif metric_method == "OR":
        Feature_X, Feature_Y, func, ftype = OR(X, Y)
        
    elif metric_method == "Moms_OR":
        Feature_X, Feature_Y, func, ftype = moms_OR(X, Y)
    
    elif metric_method == "OR_Moms":
        Feature_X, Feature_Y, func, ftype = OR_moms(X, Y)
    
    elif metric_method == "Sub_Eucl":
       Feature_X, Feature_Y, func, ftype = Sub_Eucl(X, Y)
    
    elif metric_method in ("KS-stat-stat", "KS-p1-p1", "KS-p1-stat", "KS-stat-p1"):
       Feature_X, Feature_Y, func, ftype = X, Y, None, None
       
    else:
        try:
            Feature_X, Feature_Y, func, ftype = Feature_dic["Feature_X"], Feature_dic["Feature_Y"], Feature_dic["Asssociation_function"], Feature_dic["assoc_func_type"]
        except:
            sys.exit("Check implemented metric_methods or give a parameter Feature_dic must be given: keys Feature_X (ndarray), Feature_Y (ndarray), Association_function (func), assoc_func_type(str vectorized or str not_vectorized)")
            
            
    Result = get_NMDclass(X, Y, Feature_X, Feature_Y, func, ftype, metric_method, dist_origin, num_clust, clust_method, palette)
    return Result

    
def get_NMDclass(X, Y, Feature_X, Feature_Y, func, ftype, metric_method, dist_origin = False, num_clust=None, clust_method = "Kmeans", palette = "tab20"):
    
    M = Feature_X.shape[0]
    N = Feature_Y.shape[0]
    
    if metric_method not in ("KS-stat-stat", "KS-p1-p1", "KS-p1-stat", "KS-stat-p1"):
        """ Similarity metric """
        DX = Similarity_Distance(Feature_X, method = "Euclidean")
        DY = Similarity_Distance(Feature_Y, method = "Euclidean")
        
        """Association metric"""
        Features = (X, Y)
        
        D_assoc = Association_Distance(Features, func, ftype)
        """Distane to origin Optional but must be set to None if not used"""
        if dist_origin:
            Orows = np.linalg.norm(Feature_X, axis = 1)
            Ocols = np.linalg.norm(Feature_Y, axis = 1)
        else:
            Orows = None
            Ocols = None
        
        DMat = Prox_Mat(DX, DY, UX = Orows, UY = Ocols, fXY = D_assoc)
    
    elif metric_method in ("KS-stat-stat", "KS-p1-p1"):
        Z = np.concatenate((X, Y), axis = 0)
        DMat = KS_Distance(Z, to_use = metric_method)
    
    else:
        Z = (X, Y)
        DMat = KS_Distance_Mixed(Z, to_use = metric_method)
    
    try:        
        if clust_method == "Kmedoids":
            if num_clust == None:
                sys.exit("Kmedoids requires number of clusters parameter: num_clust")
            else:
                clust_labels, color_clustered = get_clusters(DMat, num_clust, palette, method = clust_method, metric = "precomputed")
        
        elif clust_method[:13] == "Agglomerative":
            clust_labels, color_clustered = get_clusters(DMat, num_clust, palette, method = clust_method, metric = "precomputed")
        
        elif clust_method == "Spectral":
        	clust_labels, color_clustered = get_clusters(DMat, num_clust, palette, method = clust_method, metric = "precomputed")
            
        else:
            sys.exit("A non-metric distance clustering method is required for Non Metric Distance \n Available here is Kmedoids")
        
        if dist_origin and metric_method not in ("KS-stat-stat", "KS-p1-p1", "KS-p1-stat", "KS-stat-p1"):
            Class_pred = np.concatenate((clust_labels[:M], clust_labels[M+1:]), axis = 0)
            was_orig = True
        else:
            Class_pred = clust_labels
            was_orig = False
    except:
        pdb.set_trace()
    
    return {"shape":(M, N), "was_orig":was_orig, "Class_pred":Class_pred, "clust_labels":clust_labels, "color_clustered":color_clustered}
    