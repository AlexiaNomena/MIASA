#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 14:03:16 2022

@author: raharinirina
"""

from .Core.Generate_Distances import Similarity_Distance, Association_Distance
from .Generate_Features import eCDF, Eucl, Null, covariance, get_assoc_func
from .Generate_Features import corrcoeff, moms, OR, Cond_proba, Granger_Cause
from .Core.Clustering import get_clusters
from .Core.qEmbedding import Euclidean_Embedding
from .Core.CosLM import Prox_Mat

import sys
import numpy as np
import pdb


def Miasa_Class(X, Y, num_clust, DMat = None, c_dic = None, dist_origin = (True, True), metric_method = ("eCDF", "KS-stat"), clust_method = "Kmeans", palette = "tab20", Feature_dic = None, in_threads = True, clust_orig = False):
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
        Feature_X, Feature_Y = Eucl(X, Y)
        func, ftype = get_assoc_func(assoc_type = metric_method[1], in_threads = in_threads)

    elif metric_method[0] == "Cond_proba":
        Feature_X, Feature_Y = Cond_proba(X, Y) 
        func, ftype = get_assoc_func(assoc_type = metric_method[1], in_threads = in_threads)
    
    elif metric_method[0] == "Null":
        Feature_X, Feature_Y = Null(X, Y) 
        func, ftype = get_assoc_func(assoc_type = metric_method[1], in_threads = in_threads)
 
    elif metric_method[0][:13] == "Granger-Cause":
       if metric_method[0][14:18] == "orig":
           diff = False
           diff = False
       else:
           diff = True
       Feature_X, Feature_Y= Granger_Cause(X, Y, diff = diff) 
       func, ftype = get_assoc_func(assoc_type = metric_method[1], in_threads = in_threads)

    else:
        try:
            Feature_X, Feature_Y, func, ftype, DMat, dist_origin = Feature_dic["Feature_X"], Feature_dic["Feature_Y"], Feature_dic["Asssociation_function"], Feature_dic["assoc_func_type"], Feature_dic["DMat"], Feature_dic["dist_origin"]
        except:
            sys.exit("Check implemented metric_methods or give a parameter Feature_dic must be given: keys Feature_X (ndarray), Feature_Y (ndarray), Association_function (func) with tuple argument (X, Y), assoc_func_type (str vectorized or str not_vectorized), DMat direclty given distance matrix, dist_origin bool tuple (orig X?, orig Y) ")
            
    Result = get_class(X, Y, Feature_X, Feature_Y, func, ftype, metric_method, c_dic, DMat, dist_origin, num_clust, clust_method, palette, in_threads = in_threads, clust_orig = clust_orig)

    return Result
    

def get_class(X, Y, Feature_X, Feature_Y, func, ftype, metric_method, c_dic, DMat = None, dist_origin = (True, True), num_clust=None, clust_method = "Kmeans", palette = "tab20", in_threads = True, clust_orig = False):
    M = Feature_X.shape[0]
    N = Feature_Y.shape[0]
    
    if (DMat is not None):
        if (DMat.shape == (M+N+1, M+N+1)):
            ### remove the origin
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
    
    else:
        """ Similarity metric """
        DX = Similarity_Distance(Feature_X, method = "Euclidean")
        DY = Similarity_Distance(Feature_Y, method = "Euclidean")
        
        """Association metric"""
        Z = (X, Y)
        D_assoc = Association_Distance(Z, func, ftype)
            
        
    if (dist_origin[0]) or (dist_origin[1]):
        """Distane to origin Optional but must be set to None if not used"""
        Orows = np.zeros(Feature_X.shape[0])
        Ocols = np.zeros(Feature_Y.shape[0])
        if dist_origin[0]:
            if DMat is not None:
                if DMat.shape == (M+N+1, M+N+1):
                    Orows = DMat[:M, M+1]
                else:
                    Orows = np.linalg.norm(Feature_X, axis = 1)
            else:
                Orows = np.linalg.norm(Feature_X, axis = 1)
        if dist_origin[1]:
            if DMat is not None:
                if DMat.shape == (M+N+1, M+N+1):
                    Ocols = DMat[M+1:, M+1]
                else:
                    Ocols = np.linalg.norm(Feature_Y, axis = 1)
            else:
                Ocols = np.linalg.norm(Feature_Y, axis = 1)
    else:
        Orows = None
        Ocols = None
   
    """ Get joint Euclidean embedding """
    
    if c_dic is None or c_dic == "default":
        c1, c2 = 1/2, 2
        a = 1. - 1./(M+N)
        b = 2.*c2/(M+N)
        c3 =  min(((2.*c1 + c2) - b)/a, 2*c2+2)
        #c1, c2, c3 = np.random.uniform(0, 5, size = 3)
        c_dic = {"c1":c1, "c2":c2, "c3":c3}
    else:
        c_dic = c_dic
    
    
    alpha =  0 # np.max(D_assoc) adding a constant to the Euclidean distances to statisfy one of the conditions for embedding (obsolete)
    if (Orows is not None) or (Ocols is not None):
        Coords, vareps, num_it = Euclidean_Embedding(DX+alpha, DY+alpha, Orows+alpha, Ocols+alpha, D_assoc, c_dic, in_threads = in_threads, num_iterations = True)
    else:
        Coords, vareps, num_it = Euclidean_Embedding(DX+alpha, DY+alpha, None, None, D_assoc, c_dic, in_threads = in_threads, num_iterations = True)
    
    if Coords is not None:
        # Thre is no point that is close to the origin as a particular characteristic of the embedding, thus the origin must be removed otherwise it risk to be considered as one cluster
        if clust_orig:
            Coords_0 = Coords.copy()
        else:
            Coords_0 = np.row_stack((Coords[:M, :], Coords[-N:, :])) # works even if there was no origin considered
            
        if clust_method == "Kmeans":
            if num_clust == None:
                sys.exit("Kmeans requires number of clusters parameter: num_clust")
            else:
                clust_labels, color_clustered = get_clusters(Coords_0, num_clust, palette, method = clust_method)
        
        elif clust_method == "Kmedoids":
            if num_clust == None:
                sys.exit("Kmedoids requires number of clusters parameter: num_clust")
            else:
                clust_labels, color_clustered = get_clusters(Coords_0, num_clust, palette, method = clust_method)
                
        elif clust_method[:13] == "Agglomerative": 
            clust_labels, color_clustered = get_clusters(Coords_0, num_clust, palette, method = clust_method)
        
        elif clust_method == "Spectral":
        	clust_labels, color_clustered = get_clusters(Coords_0, num_clust, palette, method = clust_method) 
        
        elif clust_method == "Spectral_ver2":
        	clust_labels, color_clustered = get_clusters(Coords_0, num_clust, palette, method = clust_method)   
        
        elif clust_method == "Simple_Min_Dist":
            clust_labels, color_clustered = get_clusters(Coords_0, num_clust, palette, method = clust_method)   
                
        elif clust_method == "GMM":
            clust_labels, color_clustered = get_clusters(Coords_0, num_clust, palette, method = clust_method)
        
        elif clust_method == "BayesianGMM":
            clust_labels, color_clustered = get_clusters(Coords_0, num_clust, palette, method = clust_method)
         
        elif clust_method == "DBSCAN":
            clust_labels, color_clustered = get_clusters(Coords_0, num_clust, palette, method = clust_method, metric = "euclidean")
            
        elif clust_method == "BRW":
            clust_labels, color_clustered = get_clusters(Coords_0, num_clust, palette, method = clust_method)
            
        elif clust_method[0] == "MLPClassifier":
            clust_labels, color_clustered = get_clusters(Coords_0, num_clust, palette, method = clust_method)
            
        else:
            sys.exit("A metric-distance based clustering method is better for MIASA \n Available here is Kmeans")
        
        
        if dist_origin[0] or dist_origin[1]:
            Coords = Coords - Coords[M, :][np.newaxis, :]
            Class_pred = np.concatenate((clust_labels[:M], clust_labels[-N:]), axis = 0)
            was_orig = True
        else:
            Class_pred = clust_labels
            was_orig = False
       
        DMat = Prox_Mat(DX, DY, UX = Orows, UY = Ocols, fXY = D_assoc)
        Result = {"Coords": Coords, "shape":(M, N), "was_orig":was_orig, "vareps":vareps, "Class_pred":Class_pred, "clust_labels":clust_labels, "color_clustered":color_clustered, "DMat":DMat, "X":X, "Y":Y, "num_iterations":num_it}
    
    else:
        
        Result = None   
        
    return Result
    
    