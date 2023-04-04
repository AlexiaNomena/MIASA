#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 09:09:33 2022

@author: raharinirina
"""

import numpy as np
import sklearn.cluster as sklc
import sklearn_extra.cluster as sklEc

import sklearn.manifold as sklm
import sklearn.neighbors as sklNB

import pdb
import scipy as sp
from scipy.cluster.hierarchy import linkage
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from functools import partial
import umap
import seaborn as sns


def low_dim_coords(Coords, dim=2, method  = "umap", n_neighbors = 15, min_dist = None, scale = None, metric = "euclidean"):
    '''
    @ brief          : embeding of points onto a lower-dimensional manifold of using sklean.manifold
    @ param Coords   : Coords, dim, method
    @ Coords           : Q by K array with the Coordinates on the rows (Q points to embed)
    @ dim            : Dimensions of the lower-dimensional manifold
    @ method         : sklearn.manifold methods preserves the structure of the distances in the original data : 
                       OPTIONS "MDS" (respect well the distances), "Isomap" (preserves geodesic distances)
    '''
    if scale == "standard":
        scaled_coords = StandardScaler().fit_transform(Coords) 
    elif scale == "pca":
        scaled_coords = PCA(n_components = 5).fit_transform(Coords) 
    else:
        scaled_coords = Coords
    
    if method == "MDS":
        embedding = sklm.MDS(n_components = dim, metric = True, dissimilarity = metric)
        Emb_coords = embedding.fit_transform(scaled_coords)
    
    elif method == "Isomap":
        embedding = sklm.Isomap(n_components = dim, metric = metric) 
        Emb_coords = embedding.fit_transform(scaled_coords)
    
    elif method == "MDS_YH":
        Emb_coords = MDS_YH(scaled_coords, dim)
        
    elif method == "umap":
        Emb_coords = umap_reducer(scaled_coords, dim, n_neighbors, min_dist)
    
    elif method == "t-SNE":
        Emb_coords = tSNE_reducer(scaled_coords, dim, n_neighbors, metric = metric) 
       
    return Emb_coords

rand = 0 # fixed initialization for reproducibility of UMAP and Kmeans
def umap_reducer(Coords, dim, np, min_dist):
    if min_dist == None:
        reducer = umap.UMAP(n_neighbors = np, metric = "euclidean", n_components = dim, random_state= rand) # n_neighbor = 2 (local structure) --- 200 (global structure, truncated when larger than dataset size)
    else:
        reducer = umap.UMAP(n_neighbors = np, min_dist = min_dist, metric = "euclidean", n_components = dim, random_state= rand)
    Emb_coords = reducer.fit_transform(Coords)
    return Emb_coords


def tSNE_reducer(Coords, dim, np, metric = "euclidean"):
    reducer = sklm.TSNE(n_components = dim, perplexity = np, random_state=rand, metric = metric)
    Emb_coords = reducer.fit_transform(Coords)
    return Emb_coords

def dist_error(tXflat, D, dim):
    if dim>= 2:
        tD = sp.spatial.distance.pdist(tXflat.reshape((D.shape[0], dim)))
    else:
        tX = np.concatenate((tXflat[:, np.newaxis], np.zeros(len(tXflat))[:, np.newaxis]), axis = 1)
        tD = sp.spatial.distance.pdist(tX)
        
    tD = sp.spatial.distance.squareform(tD)
    T = tD - D
    return T.flatten()


def MDS_YH(Coords, dim, method = "LQ"): 
    '''
    @ brief          : embeding points onto a lower-dimensional Euclidean space based on least_square distane error minimization (Matrix Frobenius norm minimization)
    @ param Coords   : Coords, dim, method
    @ Coords         : Q by K array with the Coordinates on the rows (Q points to embed)
    @ dim            : Dimensions of the lower-dimensional manifold
    @ method         : minimization method 
                       OPTIONS "LQ" (scipy.optimize.least_squares)
    '''

    DistMat = sp.spatial.distance.squareform(sp.spatial.distance.pdist(Coords))
    arg_stack = (DistMat, dim)
    sol = sp.optimize.least_squares(dist_error, Coords[:, :dim].flatten(), bounds = (-np.inf, np.inf), args = arg_stack)
    Emb_coords = sol.x.reshape((Coords.shape[0], dim))
    
    return Emb_coords


def single_branching(progeny, anc_lab, n_clust = 3):
    embedding = sklc.KMeans(n_clusters = n_clust, random_state = 0).fit(progeny)  
    centroids = embedding.cluster_centers_
    labels = embedding.labels_

    if anc_lab is not None:
        ulabs = np.array(labels, dtype = str)
        prog_labels = np.array([anc_lab + ulabs[i] for i in range(len(ulabs))])
    else:
        prog_labels =  np.array(labels, dtype = str)

    return centroids, prog_labels


def tree_emb(Coords, graph_type = "Agglomerative", linkage_type = "centroid", n_iter = 10, distance_threshold = 0):
    if graph_type == "Divisive":
        Tree = {"graph_type":graph_type}
        centroid0, prog_labels = single_branching(Coords, anc_lab = None, n_clust = 1)
        Tree["Ancestors"] = centroid0
        Tree["Ancestors_Labels"] = np.unique(prog_labels)
        Tree["Progeny_Labels"] = prog_labels
        
        local_prog = np.unique(prog_labels)
        for j in range(n_iter):
            new_progeny = prog_labels.copy()
            for i in range(len(local_prog)):
                prog = Coords[prog_labels == local_prog[i], :]
                if prog.shape[0]>5:
                    centroid_i, prog_i = single_branching(prog, local_prog[i], n_clust = 2)
                    new_progeny[prog_labels == local_prog[i]] = prog_i
                    if j != n_iter - 1:
                        Tree["Ancestors"] = np.concatenate((Tree["Ancestors"], centroid_i), axis = 0)
                        Tree["Ancestors_Labels"] = np.concatenate((Tree["Ancestors_Labels"], np.unique(prog_i)))
            Tree["Progeny_Labels"] = new_progeny
            prog_labels = new_progeny
            local_prog = np.unique(prog_labels)
            figtitle = "Divisive Hierarchical Clustering (sklearn-Kmeans centroids)"
            
    elif graph_type == "Agglomerative":
        Tree = {"graph_type":graph_type}
        if linkage_type in ("ward", "average", "complete", "single"):
            clusters = sklc.AgglomerativeClustering(compute_full_tree = True, distance_threshold = distance_threshold, linkage = linkage_type, n_clusters = None) 
            clusters = clusters.fit(Coords)
            # create the counts of samples under each node
            counts = np.zeros(clusters.children_.shape[0])
            n_samples = len(clusters.labels_)
            for i, merge in enumerate(clusters.children_):
                current_count = 0
                for child_idx in merge:
                    if child_idx < n_samples:
                        current_count += 1  # leaf node
                    else:
                        current_count += counts[child_idx - n_samples]
                counts[i] = current_count   
            linkage_matrix = np.column_stack([clusters.children_, clusters.distances_, counts]).astype(float)
            Tree["linkage"] = linkage_matrix
            Tree["cluster_labels"] = clusters.labels_
            figtitle = "Agglomerative %s linkage"%linkage_type
        else:
            linkage_matrix = linkage(Coords, method = linkage_type) 
            Tree["linkage"] = linkage_matrix
            figtitle = "Agglomerative %s linkage"%linkage_type
    
    return Tree, figtitle





                          
                
                
                
                
                