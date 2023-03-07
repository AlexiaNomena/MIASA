#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 15:58:03 2023

@author: raharinirina
"""
import numpy as np
import sklearn.cluster as sklc
import sklearn_extra.cluster as sklEc
import seaborn as sns
import scipy as sp

rand  = 0
def get_clusters(Coords, num_clust, palette, method = "Kmeans", init = "k-means++", metric = None):
    if method == "Kmeans":
        clusters = sklc.KMeans(n_clusters = num_clust, init = init, random_state = rand).fit(Coords)
        labels = clusters.labels_
    elif method == "Kmedoids":
        if init == "k-means++" or init == "k-means++":
            init = "k-medoids++"
        
        if metric == "precomputed":
            clusters = sklEc.KMedoids(n_clusters = num_clust, metric = "precomputed" ,init = init, random_state = rand).fit(Coords) # in this case coords it the proximity matrix
        else:
            try:
                clusters = sklEc.KMedoids(n_clusters = num_clust, metric = metric, init = init, random_state = rand).fit(Coords)
            except:
                clusters = sklEc.KMedoids(n_clusters = num_clust, init = init, random_state = rand).fit(Coords)
        labels = clusters.labels_        
    elif method[:13] == "Agglomerative":
        if metric == "precomputed":
            clusters = sklc.AgglomerativeClustering(n_clusters = num_clust, affinity = metric, linkage = method[14:]).fit(Coords)
            # parameter affinity will be deprecated, replace with metric in future
            #clusters = sklc.AgglomerativeClustering(n_clusters = num_clust, metric = metric, linkage = method[14:]).fit(Coords)
        else:
            clusters = sklc.AgglomerativeClustering(n_clusters = num_clust, linkage = method[14:], distance_threshold = None).fit(Coords)
        labels = clusters.labels_
    elif method == "Spectral":
        if metric == "precomputed":
            clusters = sklc.SpectralClustering(n_clusters = num_clust, affinity = metric).fit(Coords)
        else:
            try:
                clusters = sklc.SpectralClustering(n_clusters = num_clust, affinity = metric).fit(Coords)
            except:
                clusters = sklc.SpectralClustering(n_clusters = num_clust).fit(Coords)
        labels = clusters.labels_
        
    elif method == "Simple_Min_Dist":
        labels = Simple_Min_Dist(Coords, metric, num_clust)
    
    elif method[0] == "MLPClassifier":
        labels = Neural_Net(Coords, params = method[1])
    
    col_labels = get_col_labs(labels, palette)
    return labels, col_labels


def Simple_Min_Dist(Coords, metric, num_clust = None):
    labels = np.zeros(Coords.shape[0])
    
    if num_clust is None:
        num_clust = 5
    
    if metric != "precomputed":
        spDist = sp.spatial.distance.pdist(Coords)
        spDist = sp.spatial.distance.squareform(spDist)
    else:
        spDist = Coords
    
    # choose centroids as random points within percentiles
    perc = np.linspace(5, 95, num_clust)
    #out = np.percentile(Norm, 95, interpolation = "midpoint")
    #ind_out = np.random.choice(range(len(Norm))[Norm > out])
    all_indx = np.arange(0, spDist.shape[0])
    ind_centre = []
    centre_labels = []
    for n in range(int(num_clust)):
        point_n = np.percentile(spDist[0, :], perc[n], interpolation = "midpoint")
        where_point = spDist[0, :] <= point_n
        ind_centre.append(np.random.choice(all_indx[where_point*1 != 0]))
        centre_labels.append(n+1)
    
    
    # Asign labels of nearest centre
    for i in range(len(labels)):
        dist_i = spDist[i, np.array(ind_centre)]
        nearest_centre = np.argmin(dist_i)
        labels[i] = centre_labels[nearest_centre]
        
    return labels

def get_col_labs(labels, palette):               
    unique_labs = np.unique(labels)
    colors = sns.color_palette(palette,  len(unique_labs))
    col_labs = np.zeros((len(labels), 3))
    for i in range(len(unique_labs)):
        """
        if np.all(np.array(colors[i])<=1):
            col_i = np.array(255*(np.array()), dtype = int)
        else:
            col_i = np.array(colors[i], dtype = int)
        col_labs[labels == unique_labs[i], :] = '#%02x%02x%02x'%tuple(col_i)
        """  
        col_labs[labels == unique_labs[i], :] = colors[i]
    
    return col_labs
                    

from sklearn.neural_network import MLPClassifier
def Neural_Net(Coords, params, metric = None):
    """Remember that Class_True does not contain the origin and the origin"""
    M, N, Class_True, perc_train = params
    
    #Coords = StandardScaler().fit_transform(Coords) ## scale
    Coords_data = np.row_stack((Coords[:M, :], Coords[-N:, :]))
    K = int((perc_train/100)*len(Class_True))
    
    Inds = np.arange(0, Coords_data.shape[0], 1, dtype = int)
    np.random.shuffle(Inds)
    
    # remember to set metric  == precomputed for Non-metric Classification
    #DMat= sp.spatial.distance.pdist(Coords)
    #DMat = sp.spatial.distance.squareform(DMat)
    #DMat = StandardScaler().fit_transform(DMat)
    #DMat_data = np.row_stack((DMat[:M, :], DMat[-N:, :]))
    
    Coords_train = Coords_data[Inds[:K], :]
    Class_train = Class_True[Inds[:K]]
    clf = MLPClassifier(random_state=1, max_iter = 500).fit(Coords_train, Class_train)
    
    labels_pred = clf.predict(Coords)
    
    return labels_pred