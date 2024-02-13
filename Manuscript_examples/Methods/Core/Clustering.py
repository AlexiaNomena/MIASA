#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 15:58:03 2023

@author: raharinirina
"""
import numpy as np
import sklearn.cluster as sklc
import sklearn.mixture as sklMixt
import sklearn_extra.cluster as sklEc
import seaborn as sns
from sklearn import svm
import scipy as sp
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
import sys

import pdb

from sklearn_som.som import SOM
from .som_miasa import SOM_MIASA

rand  = 0 ## reproducibility of clustering for each method

def get_clusters(Coords, num_clust, palette, method = "Kmeans", init = "k-means++", metric = None, vareps_miasa=None):
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
        
    elif method == "Spectral_ver2":
        labels = Spectral_clust(Coords, num_clust)
        
    elif method == "GMM":
        cluster = sklMixt.GaussianMixture(n_components = num_clust, random_state = rand, max_iter = 200, n_init = 5, tol=0.0001, reg_covar=1e-8).fit(Coords)
        labels = cluster.predict(Coords)
    
    elif method == "BayesianGMM":
        cluster = sklMixt.BayesianGaussianMixture(n_components = num_clust, random_state = rand, max_iter = 200, n_init = 5, tol=0.0001, reg_covar=1e-8).fit(Coords)
        labels = cluster.predict(Coords)
    
    elif method == "DBSCAN":
        #Coords = StandardScaler().fit_transform(Coords)
        cluster = sklc.DBSCAN(metric = metric).fit(Coords)
        labels = cluster.labels_
    elif method == "BRW":
        labels = BRW(Coords, num_clust)
    
    elif method == "Simple_Min_Dist":
        epochs = 3
        labels = Simple_Min_Dist(Coords, metric, num_clust)
    
    elif method[0] == "MLPClassifier": ### uses a log loss function = cross-entropy
        epochs = 3
        labels = Neural_Net(Coords, params = method[1], random_state=rand, max_iter = epochs*Coords.shape[0])
    
    elif method[0] == "MLPRegressor": ### uses a Square Error Loss (SEL) function (more related to the Euclidean distance)
        epochs = 3
        labels = Neural_Net_Regressor(Coords, params = method[1], random_state=rand, max_iter = epochs*Coords.shape[0])
    
    elif method[0] == "SVM_SVC":
        epochs = 3
        labels = SVM_SVC(Coords, params = method[1], random_state=rand, max_iter = epochs*Coords.shape[0])
        
    elif method[0] == "SOM":
        epochs = 3
        
        if (vareps_miasa != "Non_MD_Case") and (vareps_miasa != 0):
            type_lr = method[1]
            if type_lr == "1/sqrt(vareps)":
                labels = SOM(m=num_clust, n=1, sigma=1, lr=1/np.sqrt(vareps_miasa), dim = Coords.shape[1], random_state = rand, max_iter = epochs*Coords.shape[0]).fit_predict(Coords, epochs = epochs)
            elif type_lr == "sqrt(vareps)":
                labels = SOM(m=num_clust, n=1, sigma=1, lr=np.sqrt(vareps_miasa), dim = Coords.shape[1], random_state = rand, max_iter = epochs*Coords.shape[0]).fit_predict(Coords, epochs = epochs)
            else:
                try:
                    labels = SOM(m=num_clust, n=1, sigma=1, lr= type_lr * np.sqrt(vareps_miasa), dim = Coords.shape[1], random_state = rand, max_iter = epochs*Coords.shape[0]).fit_predict(Coords, epochs = epochs)
                except:
                    sys.exit("clust_method[1] only options for SOM in qEE-MIASA are: str(1/sqrt(vareps)) to give lr params, str(sqrt(vareps)) to give lr params, or float(prop) to make lr = float(prop)*sqrt(vareps)")
        else:
            try:
                labels = SOM(m=num_clust, n=1, sigma=1, lr=method[1], dim = Coords.shape[1], random_state = rand, max_iter = epochs*Coords.shape[0]).fit_predict(Coords, epochs = epochs)
            except:
                sys.exit("clust_method[1] only options for SOM Non_MD case MIASA is exactly equal to lr: float(lr)")
       

    elif method[0] == "SOM_MIASA":
        clusters = sklc.AgglomerativeClustering(n_clusters = num_clust, linkage = "ward", distance_threshold = None).fit(Coords)
        labels_0 = clusters.labels_
        centroids_0 = np.array([np.mean(Coords[labels_0 == labs, :], axis = 0) for labs in np.unique(labels_0)])
        epochs = 3 ### des
        
        """
        Learing rate is taylored for MIASA lr_miasa = lr*(sqrt(vareps_miasa)/lambda_miasa) 
        --- because in the qEE spac, for every points z in space, there exists at most one sample point z* such that
        dist(z*, z) => R, where R=sqrt(vareps_miasa)/lambda_miasa (less than or equal to)
        which is actually true for all cloud of points if we take R = min(of all interpoint distances),
        however, in the qEE space, vareps_miasa is quite large, therefore we use it to have a better control our step sized in the learning process 
        """
        if (vareps_miasa != "Non_MD_Case") and (vareps_miasa != 0):
            type_lr = method[1]
            if type_lr == "1/sqrt(vareps)":
                lambda_miasa = vareps_miasa
            elif type_lr == "sqrt(vareps)":
                lambda_miasa = 1
            else:
                lambda_miasa = (1/type_lr)*np.sqrt(vareps_miasa)
            try:
                labels = SOM_MIASA(initial_centroids = centroids_0, vareps_miasa = vareps_miasa, lambda_miasa = lambda_miasa, m=num_clust, n=1, sigma=1, lr=1, dim = Coords.shape[1], random_state = rand, max_iter = epochs*Coords.shape[0]).fit_predict(Coords, epochs=epochs)
            except:
                sys.exit("clust_method[1] only options for SOM_MIASA in qEE-MIASA are: str(1/sqrt(vareps)) to give lr params, str(sqrt(vareps)) to give lr params, or float(prop) to make lr = float(prop)*sqrt(vareps)")
        else:
            try:
                labels = SOM_MIASA(initial_centroids = centroids_0, vareps_miasa = 1, lambda_miasa = 1, m=num_clust, n=1, sigma=1, lr=method[1], dim = Coords.shape[1], random_state = rand, max_iter = epochs*Coords.shape[0]).fit_predict(Coords, epochs=epochs)
            except:
                sys.exit("clust_method[1] only options for SOM_MIAS Non_MD case MIASA is exactly equal to lr: float(lr)")
        
    col_labels = get_col_labs(labels, palette)
    return labels, col_labels

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


def Spectral_clust(Coords, num_clust):
    D2Mat = sp.spatial.distance.squareform(sp.spatial.distance.pdist(Coords))
   
    W = D2Mat.copy()
    D = np.diag(np.sum(D2Mat, axis=1))
    L = D - W
    E, U = np.linalg.eigh(L)
    E = np.real(E)  # remove tiny imaginary numbers
    U = np.real(U)  # remove tiny imaginary numbers
    E[np.isclose(E, np.zeros(len(E)))] = 0  # remove tiny numbers that are too close to zeros
    
    sort = np.argsort(E)[::-1] # descending order
    U = U[:, sort]
    
    clusters = sklc.KMeans(n_clusters = num_clust, init = "k-means++", random_state = rand).fit(U[:, :num_clust]) #kmeans2(U, k=)
    labels = clusters.labels_
    return labels 


from .Tensor_Laws import convert_spherical
def BRW(Coords, num_clust, n_iter = 50): # n_iter = number of time for changing direction
    # centroid initialization
    clusters = sklc.KMeans(n_clusters = num_clust, init = "k-means++", random_state = rand).fit(Coords)
    centroids = clusters.cluster_centers_
    
    NN_dic = {}
    Rad = np.max(sp.linalg.norm(Coords, axis = 1))
    
    Weight_list = [np.zeros(1) for k in range(len(centroids))]
    Spher_coords, Transf_prev_list = convert_spherical(centroids)
    
    for k in range(len(centroids)):
        stop = 0
        while stop < n_iter:
            center = centroids[k]
            # compute current angular direction
            Transf_prev = Transf_prev_list[k, :]
            
            bias = DRW_bias(Weight_list[k])
            change = np.random.choice(np.array([0, 1]), p = np.array([bias, 1-bias]))
            #print(k, bias)
            if change == 0:
                # don't change direction
                new_center, NN, Transf = move_center(center, Coords, num_clust, Transf_prev, boundary = Rad, change = False)
            else:
                # change direction
                new_center, NN, Transf = move_center(center, Coords, num_clust, Transf_prev, boundary = Rad, change = True)
                if k!=0:
                    limit = 0
                    while (len(list(set.intersection(set(NN_dic[k-1]), set(NN)))) > ((1/5)*(Coords.shape[0])//num_clust)) and limit<100: 
                        ### keep moving until theshold intersection is reached in the direction
                        new_center, NN, Transf = move_center(new_center, Coords, num_clust, Transf, boundary = Rad, change = False)
                        limit += 1
                        #print("changed")
            
        
            ### update nearest neighbor to centroids
            NN_dic[k] = NN.copy()
            
            # update previous direction transformation
            Transf_prev_list[k, :] = Transf.copy()
            
            # new direction weight penalising clusters with high variance to center
            S_center = Coords[NN, :]
            
            # update centroid
            centroids[k] = new_center.copy()
            
            penalise = np.sum(sp.linalg.norm(S_center - new_center, axis = 1)**2)
            
            weight = np.exp(- 1/(penalise))
            Weight_list[k] = np.append(Weight_list[k], weight)

            stop +=1        
        #print(k, "done", NN)

    labels = construct_labels(centroids, Coords, NN_dic)
    return labels


def construct_labels(centroids, Coords, NN_dic):
    labels = []
    for i in range(Coords.shape[0]):
        memb_i = []
        for j in range(centroids.shape[0]):
            if i in NN_dic[j]:
                memb_i.append(j)
          
        if len(memb_i) == 1:
            labels.append(memb_i[0])
        
        elif len(memb_i)>1:
            best = np.argsort(np.array([np.linalg.norm(Coords[i, :] - centroids[j, :]) for j in memb_i]))[0]
            labels.append(best)
        else:
            ### assign to closest centroid
            best = np.argsort(sp.linalg.norm(Coords[i, :][np.newaxis, :] - centroids, axis = 1))[0]
            labels.append(best)
        
    return labels

def Impulse(t):
    res = np.exp(-t)*(1 - 0.5*(t + 0.5*t**2))
    return res

def DRW_bias(weights):
    ### Bias compares previous weight to past weight following bacterial chemotaxis model (see Masters project)
    if len(weights) < 20:
        perception =  np.array([Impulse(20 - i) for i in range(len(weights))])
        bias_pw = weights * perception

    else:
        perception =  np.array([Impulse(20 - i) for i in range(20)])
        bias_pw = weights[:20] * perception

    bias = min(1, max(0, 0.5 + 80**np.sum(bias_pw)))
    return bias

def move_center(center, Coords, num_clust, Transf_prev, boundary, change = False):
    # Update around spherical coordinates #https://en.wikipedia.org/wiki/N-sphere
    powers = np.ones((len(center) - 1, len(center)))
    pow_sin = np.triu(powers, k = 1)
    pow_cos = powers - pow_sin - np.tril(powers, k = -1)

    # sample change in angular direction
    dtheta = np.random.normal(3*np.pi/2, size = powers.shape[0])
    
    # construct transformation matrix
    if change:
        Transf_vect = Transf_prev + np.prod(np.power(np.sin(dtheta)[:, np.newaxis], pow_sin) , axis = 0)*np.prod(np.power(np.cos(dtheta)[:, np.newaxis], pow_cos), axis = 0)
    else:
        Transf_vect = Transf_prev
    
    # radius
    num_points = Coords.shape[0]//num_clust ### assuming equal number of points per cluster
    NN = np.argsort(sp.linalg.norm(Coords - center, axis = 1))[:num_points]
    S_center = Coords[NN, :]
    radius = np.std(S_center)/4
    # new centroid
    new_center = center + radius*Transf_vect
    
    try:
        if sp.linalg.norm(new_center) > boundary:
            # shrink position inside boundary
            new_center = (boundary/2)*Transf_vect
    except:
        pdb.set_trace()
        
    NN_new = np.argsort(sp.linalg.norm(Coords - new_center, axis = 1))[:num_points]
    return new_center, NN_new, Transf_vect


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


def Neural_Net(Coords, params, metric = None, random_state=None, max_iter = 500):
    """Remember that Class_True does not contain the origin"""
    M, N, Class_True, perc_train = params
    
    """In the non-Euclidean case it is the distance matrix could be used, but we are not sure how legit that is"""

    #Coords = StandardScaler().fit_transform(Coords) ## scale
    Coords_data = np.row_stack((Coords[:M, :], Coords[-N:, :]))
    K = int((perc_train/100)*len(Class_True))
    
    Inds = np.arange(0, Coords_data.shape[0], 1, dtype = int)
    if random_state is not None:
        np.random.seed(random_state) # in case one needs a reproducible result
    
    np.random.shuffle(Inds)
    Coords_train = Coords_data[Inds[:K], :]
    Class_train = Class_True[Inds[:K]]
    clf = MLPClassifier(random_state=rand, max_iter = max_iter).fit(Coords_train, Class_train)
    
    labels_pred = clf.predict(Coords)
    
    return labels_pred

def Neural_Net_Regressor(Coords, params, metric = None, random_state=None, max_iter = 500):
    """Remember that Class_True does not contain the origin"""
    M, N, Class_True, perc_train = params
    
    """In the non-Euclidean case it is the distance matrix could be used, but we are not sure how legit that is"""

    #Coords = StandardScaler().fit_transform(Coords) ## scale
    Coords_data = np.row_stack((Coords[:M, :], Coords[-N:, :]))
    K = int((perc_train/100)*len(Class_True))
    
    Inds = np.arange(0, Coords_data.shape[0], 1, dtype = int)
    if random_state is not None:
        np.random.seed(random_state) # in case one needs a reproducible result
    
    np.random.shuffle(Inds)
    Coords_train = Coords_data[Inds[:K], :]
    Class_train = Class_True[Inds[:K]]
    clf = MLPRegressor(random_state=rand, max_iter = max_iter).fit(Coords_train, Class_train)
    
    labels_pred = clf.predict(Coords)
    
    return labels_pred

def SVM_SVC(Coords, params, metric = None, random_state=None, max_iter = 500, vareps_miasa=0):
    """Remember that Class_True does not contain the origin"""
    M, N, Class_True, perc_train = params
    
    """In the non-Euclidean case it is the distance matrix could be used, but we are not sure how legit that is"""

    #Coords = StandardScaler().fit_transform(Coords) ## scale
    Coords_data = np.row_stack((Coords[:M, :], Coords[-N:, :]))
    K = int((perc_train/100)*len(Class_True))
    
    Inds = np.arange(0, Coords_data.shape[0], 1, dtype = int)
    if random_state is not None:
        np.random.seed(random_state) # in case one needs a reproducible result
    
    np.random.shuffle(Inds)
    Coords_train = Coords_data[Inds[:K], :]
    Class_train = Class_True[Inds[:K]]
    clf = svm.SVC(random_state=rand, max_iter = max_iter).fit(Coords_train, Class_train)
    #clf = svm.SVC(random_state=rand, max_iter = max_iter, gamma = 1, degree=1, coef0 = np.sqrt(vareps_miasa), kernel = "poly").fit(Coords_train, Class_train)
    labels_pred = clf.predict(Coords)
    
    return labels_pred
