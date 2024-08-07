#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 10:27:56 2022

@author: raharinirina
"""
import numpy as np
from .Generate_Features import eCDF, Eucl, Null, covariance, get_assoc_func
from .Generate_Features import corrcoeff, moms, OR, Cond_proba, Granger_Cause, Histogram_feature
from .Core.Generate_Distances import Similarity_Distance, Association_Distance, KS_Distance, KS_Distance_Mixed

from .Core.Clustering import get_clusters
from .Core.CosLM import Prox_Mat
import sys
import pdb

def NonMetric_Class(X, Y, num_clust, DMat = None, dist_origin = (True, True), metric_method = ("eCDF", "KS-stat"), clust_method = "Kmeans", palette = "tab20", Feature_dic = None, in_threads = True, clust_orig = False):
    """Compute features"""
    if metric_method[0] == "eCDF":
       Feature_X, Feature_Y = eCDF(X,Y)
       func, ftype = get_assoc_func(assoc_type = metric_method[1], in_threads = in_threads)
    
    elif metric_method[0] == "Cov":
        Feature_X, Feature_Y = covariance(X, Y)
        func, ftype = get_assoc_func(assoc_type = metric_method[1], in_threads = in_threads)
    
    elif metric_method[0] == "Corr":
        Feature_X, Feature_Y= corrcoeff(X, Y)
        func, ftype = get_assoc_func(assoc_type = metric_method[1], in_threads = in_threads)
        
    elif metric_method[0] == "Moms":
        Feature_X, Feature_Y = moms(X, Y)
        func, ftype = get_assoc_func(assoc_type = metric_method[1], in_threads = in_threads)

    elif metric_method[0] == "OR":
        Feature_X, Feature_Y = OR(X, Y)
        func, ftype = get_assoc_func(assoc_type = metric_method[1], in_threads = in_threads)
    
    elif metric_method[0] == "Eucl":
       Feature_X, Feature_Y= Eucl(X, Y)
       func, ftype = get_assoc_func(assoc_type = metric_method[1], in_threads = in_threads)

    elif metric_method[0] == "Cond_proba":
       Feature_X, Feature_Y= Cond_proba(X, Y) 
       func, ftype = get_assoc_func(assoc_type = metric_method[1], in_threads = in_threads)
    
    elif metric_method[0] == "Null":
        Feature_X, Feature_Y = Null(X, Y) 
        func, ftype = get_assoc_func(assoc_type = metric_method[1], in_threads = in_threads)
    
    elif metric_method[0] == "Hist":
        Feature_X, Feature_Y = Histogram_feature(X), Histogram_feature(Y)
        func, ftype = get_assoc_func(assoc_type = metric_method[1], in_threads = in_threads)
        
    elif metric_method[0][:13] == "Granger-Cause":
       if metric_method[0][14:18] == "orig":
           diff = False
       else:
           diff = True
       Feature_X, Feature_Y= Granger_Cause(X, Y, diff = diff) 
       func, ftype = get_assoc_func(assoc_type = metric_method[1], in_threads = in_threads)
    
    elif metric_method in ("KS-stat-stat", "KS-p1-p1", "KS-p1-stat", "KS-stat-p1"):
       Feature_X, Feature_Y, func, ftype = X, Y, None, None
       
    else:
        try:
            Feature_X, Feature_Y, func, ftype, DMat, dist_origin = Feature_dic["Feature_X"], Feature_dic["Feature_Y"], Feature_dic["Asssociation_function"], Feature_dic["assoc_func_type"], Feature_dic["DMat"], Feature_dic["dist_origin"]
        except:
            sys.exit("Check implemented metric_methods or give a parameter Feature_dic must be given: keys Feature_X (ndarray), Feature_Y (ndarray), Association_function (func) with tuple argument (X, Y), assoc_func_type (str vectorized or str not_vectorized), DMat direclty given distance matrix, dist_origin bool tuple (orig X?, orig Y) ") 
            
    Result = get_NMDclass(X, Y, Feature_X, Feature_Y, func, ftype, metric_method, DMat, dist_origin, num_clust, clust_method, palette, in_threads, clust_orig = clust_orig)
    return Result

    
def get_NMDclass(X, Y, Feature_X, Feature_Y, func, ftype, metric_method, DMat = None, dist_origin = (True, True), num_clust=None, clust_method = "Kmeans", palette = "tab20", in_threads = False, clust_orig = False):
    
    M = Feature_X.shape[0]
    N = Feature_Y.shape[0]
    if (DMat is not None):
        if (DMat.shape == (M+N+1, M+N+1)):
            """ Similarity metric """
            DX = DMat[:M, :M]
            DY = DMat[M+1:, M+1:]
            
            """Association metric"""            
            D_assoc = DMat[:M, M+1:]
            
        elif (DMat.shape == (M+N, M+N)):
            """ Similarity metric """
            DX = DMat[:M, :M]
            DY = DMat[M:, M:]
            """Association metric"""            
            D_assoc = DMat[:M, M:]
        
        else:
            sys.exit("Wrong shape of distance matrix")
            
        if (dist_origin[0]) or (dist_origin[1]):
            """Distane to origin Optional but must be set to None if not used"""
            Orows = np.zeros(Feature_X.shape[0])
            Ocols = np.zeros(Feature_Y.shape[0])
            if dist_origin[0]:
                if (DMat.shape == (M+N+1, M+N+1)):
                    Orows = DMat[:M, M]
                else:
                    Orows = np.linalg.norm(Feature_X, axis = 1)
            if dist_origin[1]:
                if (DMat.shape == (M+N+1, M+N+1)):
                    Ocols = DMat[M+1:, M]
                else:
                    Ocols = np.linalg.norm(Feature_Y, axis = 1)
        else:
            Orows = None
            Ocols = None
                
        """Reconstruct distance matrix """
        DMat = Prox_Mat(DX, DY, UX = Orows, UY = Ocols, fXY = D_assoc)
        
    
    if metric_method not in ("KS-stat-stat", "KS-p1-p1", "KS-p1-stat", "KS-stat-p1"):
        if DMat is None:
            """ Similarity metric """
            DX = Similarity_Distance(Feature_X, method = "Euclidean")
            DY = Similarity_Distance(Feature_Y, method = "Euclidean")
            
            """Association metric"""
            Features = (X, Y)
            
            D_assoc = Association_Distance(Features, func, ftype)
            """Distane to origin Optional but must be set to None if not used"""
            if dist_origin[0] or dist_origin[1]:
                Orows = np.zeros(Feature_X.shape[0])
                Ocols = np.zeros(Feature_Y.shape[0])
                
                if dist_origin[0]:
                    Orows = np.linalg.norm(Feature_X, axis = 1)
                if dist_origin[1]:
                    Ocols = np.linalg.norm(Feature_Y, axis = 1)
            else:
                Orows = None
                Ocols = None
            
            """Reconstruct distance matrix """
            DMat = Prox_Mat(DX, DY, UX = Orows, UY = Ocols, fXY = D_assoc)
            
    elif metric_method in ("KS-stat-stat", "KS-p1-p1"):
        if DMat is None:
            Z = np.concatenate((X, Y), axis = 0)
            DMat = KS_Distance(Z, to_use = metric_method)
    
    elif metric_method in ("KS-p1-stat", "KS-stat-p1"):
        if DMat is None:
            Z = (X, Y)
            DMat = KS_Distance_Mixed(Z, to_use = metric_method)
    
    try: 
        if clust_orig:
            DMat_0 = DMat.copy()
        else:
            DMat_0 = Prox_Mat(DX, DY, UX = None, UY = None, fXY = D_assoc)
        
        if clust_method == "Kmedoids":
            if num_clust == None:
                sys.exit("Kmedoids requires number of clusters parameter: num_clust")
            else:
                clust_labels, color_clustered = get_clusters(DMat_0, num_clust, palette, method = clust_method, metric = "precomputed")
                
        elif clust_method[:13] == "Agglomerative":
            clust_labels, color_clustered = get_clusters(DMat_0, num_clust, palette, method = clust_method, metric = "precomputed")
            
        elif clust_method == "Spectral":
        	clust_labels, color_clustered = get_clusters(DMat_0, num_clust, palette, method = clust_method, metric = "precomputed")
            
        elif clust_method == "Simple_Min_Dist":
            clust_labels, color_clustered = get_clusters(DMat_0, num_clust, palette, method = clust_method, metric = "precomputed")  
        
        elif clust_method == "DBSCAN":
            clust_labels, color_clustered = get_clusters(DMat_0, num_clust, palette, method = clust_method, metric = "precomputed")
        
        elif clust_method[0] == "MLPClassifier":
            clust_labels, color_clustered = get_clusters(DMat_0, num_clust, palette, method = clust_method)
        
        elif clust_method[0] == "MLPRegressor":
            clust_labels, color_clustered = get_clusters(DMat_0, num_clust, palette, method = clust_method)
            
        elif clust_method[0] == "SVM_SVC":
            clust_labels, color_clustered = get_clusters(DMat_0, num_clust, palette, method = clust_method)
            
        elif clust_method[0] == "SOM": 
            """vareps_miasa parameter is here only used to pass a chosen learning rate as clust_method[1]"""
            clust_labels, color_clustered = get_clusters(DMat_0, num_clust, palette, method = clust_method, vareps_miasa="Non_MD_Case")
        
        elif clust_method[0] == "SOM_MIASA": 
            """vareps_miasa parameter is here only used to pass a chosen learning rate as clust_method[1]"""
            clust_labels, color_clustered = get_clusters(DMat_0, num_clust, palette, method = clust_method, vareps_miasa="Non_MD_Case")
        
        else:
            sys.exit("A non-metric distance clustering method is required for Non Metric Distance")
        
        if (dist_origin[0] or dist_origin[1]) & (metric_method not in ("KS-stat-stat", "KS-p1-p1", "KS-p1-stat", "KS-stat-p1")):
            Class_pred = np.concatenate((clust_labels[:M], clust_labels[-N:]), axis = 0)
            was_orig = True
        else:
            Class_pred = clust_labels
            was_orig = False
        Result = {"shape":(M, N), "was_orig":was_orig, "Class_pred":Class_pred, "clust_labels":clust_labels, "color_clustered":color_clustered, "DMat":DMat, "X":X, "Y":Y}
    
    except:
        print("Failed clustering, check that clustering method is implemented or suitable for a non-Euclidean distance, e.g.: Ward and Kmeans are not suitable")
        if not in_threads:
            pdb.set_trace()
        
        Result = None
    return Result
    