#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 14:03:16 2022

@author: raharinirina
"""

from .Core.Clustering import get_clusters
from .Core.qEmbedding import Euclidean_Embedding
from .Core.CosLM import Prox_Mat

import sys
import numpy as np
import pandas as pd
import pdb


def Miasa_Class(X, Y, num_clust, DMat = None, c_dic = None, dist_origin = (True, True), metric_method = ("eCDF", "KS-stat"), clust_method = "Kmeans", palette = "tab20", Feature_dic = None, in_threads = True, clust_orig = False, similarity_method = ("Euclidean", "Euclidean")):
 
    try:
        DMat, dist_origin = Feature_dic["DMat"], Feature_dic["dist_origin"]
    except:
        sys.exit("Check implemented metric_methods or give a parameter Feature_dic must be given: keys Feature_X (ndarray), Feature_Y (ndarray), Association_function (func) with tuple argument (X, Y), assoc_func_type (str vectorized or str not_vectorized), DMat direclty given distance matrix, dist_origin bool tuple (orig X?, orig Y) ")
        
    Result = get_class(X, Y, c_dic, DMat, dist_origin, num_clust, clust_method, palette, in_threads = in_threads, clust_orig = clust_orig, similarity_method = similarity_method)

    return Result
    

def get_class(X, Y, Feature_X, c_dic, DMat, dist_origin = (True, True), num_clust=None, clust_method = "Kmeans", palette = "tab20", in_threads = True, clust_orig = False, similarity_method = ("Euclidean", "Euclidean")):
    M = Feature_X.shape[0]
    N = Feature_Y.shape[0]
    
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
        Coords, vareps, num_it = Euclidean_Embedding(DX+alpha, DY+alpha, Orows+alpha, Ocols+alpha, D_assoc, c_dic, in_threads = in_threads, num_iterations = True, similarity_method = similarity_method)
    else:
        Coords, vareps, num_it = Euclidean_Embedding(DX+alpha, DY+alpha, None, None, D_assoc, c_dic, in_threads = in_threads, num_iterations = True, similarity_method = similarity_method)
    
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
    


Data_X = pd.read_excel(sys.argv[1], engine='xlrd')
Data_Y = pd.read_excel(sys.argv[2], engine='xlrd')
sim_meth_X = str(sys.argv[3])
sim_meth_Y = str(sys.argv[4])
assoc_method = str(sys.argv[5])
eucl_X = str(sys.argv[6])
eucl_Y = str(sys.argv[7])
norm_X = str(sys.argv[8])
norm_Y = str(sys.argv[9])
clust_method = str(sys.argv[10])
num_clust = int(sys.argv[11])


X = Data_X.to_numpy()
Y = Data_Y.to_numpy()

if eucl_X == "TRUE":
    meth_X = "Euclidean"
else:
    meth_X = "precomputed"

if eucl_Y == "TRUE":
    meth_Y = "Euclidean"
else:
    meth_Y = "precomputed" 
similarity_method = (meth_X, meth_Y)


Feature_dic = {}
DMat = Prox_Mat(DX, DY, UX = Orows, UY = Ocols, fXY = D_assoc)
if sim_meth_X == Eu



Result = Miasa_Class(X, Y, Feature_dic, clust_method, similarity_method)
