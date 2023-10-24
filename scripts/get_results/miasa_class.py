#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 14:03:16 2022

@author: raharinirina
"""

import numpy as np
import sklearn.cluster as sklc
import sklearn.mixture as sklMixt
import sklearn_extra.cluster as sklEc
import scipy as sp
import scipy.spatial as spsp
from sklearn.preprocessing import StandardScaler
import sys
import numpy as np
import pandas as pd
import pdb

def Miasa_Class(X, Y, num_clust, DMat = None, c_dic = None, dist_origin = (True, True), metric_method = ("eCDF", "KS-stat"), clust_method = "Kmeans", 
                palette = "tab20", Feature_dic = None, in_threads = True, clust_orig = False, similarity_method = ("Euclidean", "Euclidean"),
                get_score = False, num_clust_range= None):
 
    try:
        DMat, dist_origin = Feature_dic["DMat"], Feature_dic["dist_origin"]
    except:
        sys.exit("Check implemented metric_methods or give a parameter Feature_dic must be given: keys Feature_X (ndarray), Feature_Y (ndarray), Association_function (func) with tuple argument (X, Y), assoc_func_type (str vectorized or str not_vectorized), DMat direclty given distance matrix, dist_origin bool tuple (orig X?, orig Y) ")
        
    Result = get_class(X, Y, c_dic, DMat, dist_origin, num_clust, clust_method, in_threads = in_threads, clust_orig = clust_orig, similarity_method = similarity_method, get_score = get_score, num_clust_range= num_clust_range)
    return Result

def Euclidean_Embedding(DX, DY, UX, UY, fXY, c_dic=None, in_threads = False, num_iterations = False, similarity_method = ("Euclidean", "Euclidean")):
    """
    @brief Joint Embedding of two disjoint sets of points (see Paper: Qualitative Euclidean Embedding)
    Parameters
    ----------
    DX : shape (M, M), Distance set or Proximity Set associated to a set of points X
    DY : shape (N, N), Distance set or Proximity Set associated to a set of points Y
    UX : shape (M,)  , Distance to theoretical origin point for the set of points X
    UY : shape (N,)    ,Distance to theoretical origin point for the set of points Y   
    fXY  : shape (M, N), Proximity set matrix between the points of X and Y, Compatible with the positions of the points in DX and DY
    c_dic  : Dictionary of parameters with keys "c1", "c2", "c3"
    
    Returns
    -------
    Coords : np.array shap (M+N, M+N+2)
            Coordinates of points X and Y on the rows
    vareps: > 0 scalar defining the Embedding (see Paper)
    """
    if c_dic is None or c_dic == "default":
        M = DX.shape[0]
        N = DY.shape[0]
        c1, c2 = 1/2, 2
        a = 1. - 1./(M+N)
        b = 2.*c2/(M+N)
        c3 =  min(((2.*c1 + c2) - b)/a, 2*c2+2)
        #c1, c2, c3 = np.random.uniform(0, 5, size = 3)
        c_dic = {"c1":c1, "c2":c2, "c3":c3}
        
   
    COS_MAT, c1, c2, c3, zeta_f = CosLM(DX, DY, UX, UY, fXY, c_dic, similarity_method = similarity_method) 
    sigma, U = sp.linalg.eigh(COS_MAT)
    sigma = np.real(sigma) # COS_MAT is symmetric, thus imaginary numbers are supposed to be zero or numerical zeros
    sigma[np.isclose(sigma, np.zeros(len(sigma)))] = 0
    
    test = np.sum(sigma<0)
    
    stop = 100
    sc = 0
    c0 = c1
    while test != 0 and sc<stop:
        c1 = c2
        c2 = 2*c1
        c3 = 2 + c2 + c1
        c_dic = {"c1":c1, "c2":c2, "c3":c3}
        COS_MAT, c1, c2, c3, zeta_f = CosLM(DX, DY, UX, UY, fXY, c_dic, similarity_method = similarity_method)
        sigma, U = sp.linalg.eigh(COS_MAT)
        sigma = np.real(sigma) # COS_MAT is symmetric, thus imaginary numbers are supposed to be zero or numerical zeros
        sigma[np.isclose(sigma, np.zeros(len(sigma)))] = 0
        test = np.sum(sigma<0)
        sc += 1
    sort = np.argsort(sigma)[::-1] # descending order
    sigma = sigma[sort]
    U = U[:, sort]
    
    if test == 0:
        if not in_threads:
            print("Replacement matrix is PSD: success Euclidean embedding")
        SS = np.sqrt(np.diag(sigma)) 
        Coords0 = np.real(U.dot(SS))
        
        """ Then remove the connecting point (see Paper: Qualitative Euclidean Embedding) """
        Coords = Coords0[1:, :]
    
    else:
        print("failed Euclidean embedding")
        sys.exit("fXY non-negative and not zero everywhere is needed \n fXY : Proximity set matrix between the points of X and Y compatible with the positions of the points in DX and DY")
        Coords = None
        c3 = 0
        zeta_f = 0
        
    vareps = c3*zeta_f
    if num_iterations:
        return Coords, vareps, sc
    else:
        return Coords, vareps    

from sklearn.metrics import silhouette_score
def compute_scores(Coords, list_num, lab_lists, M, N):
    silhouette = np.zeros(len(list_num))
    elbow = np.zeros(len(list_num))
    distortion = np.zeros(len(list_num))
    for i in range(len(list_num)):
        Coords_clustered = np.row_stack((Coords[:M, :], Coords[-N:, :])) # remove embedded origin because it is not used in the clustering
        
        Class_pred = lab_lists[i]
        
        silhouette[i] = silhouette_score(Coords_clustered, Class_pred)
        
        mean_all = np.mean(Coords_clustered, axis = 0)
        unique_id = np.unique(Class_pred)
        var_between = np.zeros(len(unique_id))
        var_within = np.zeros(len(unique_id))
        
        distortion_center = np.zeros(len(unique_id))
        for j in range(len(unique_id)):
            n_j = np.sum(Class_pred == unique_id[j])
            mean_j = np.mean(Coords_clustered[Class_pred == unique_id[j], :], axis = 0)
            var_between[j] = n_j * np.sum((mean_j - mean_all)**2)
            
            X_centred = Coords_clustered[Class_pred == unique_id[j], :] -  mean_j[np.newaxis, :]
            var_within[j] = np.sum(np.sum(X_centred**2, axis = 0))
            
            All_centred = Coords_clustered -  mean_j[np.newaxis, :]
            distortion_center[j] = np.sum(All_centred @ All_centred.T)/n_j ### average Mahalanobis distance
            
        explained = np.sum(var_between)/(len(unique_id) - 1)
        unexplained = np.sum(var_within)/(M+N - len(unique_id))
        
        elbow[i] = explained/unexplained # F_stat
        p = (M+N+2) # dimensions of embedded coordinates
        distortion[i] = np.min(distortion_center)/p
        
    return silhouette, elbow, distortion

def get_class(X, Y, c_dic, DMat, dist_origin = (True, True), num_clust=None, clust_method = "Kmeans", in_threads = True, clust_orig = False, similarity_method = ("Euclidean", "Euclidean"), get_score = False, num_clust_range= None):
    M = X.shape[0]
    N = Y.shape[0]
    
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
        Orows = np.zeros(X.shape[0])
        Ocols = np.zeros(Y.shape[0])
        if dist_origin[0]:
            if DMat is not None:
                if DMat.shape == (M+N+1, M+N+1):
                    Orows = DMat[:M, M+1]
                else:
                    Orows = np.linalg.norm(X, axis = 1)
            else:
                Orows = np.linalg.norm(X, axis = 1)
        if dist_origin[1]:
            if DMat is not None:
                if DMat.shape == (M+N+1, M+N+1):
                    Ocols = DMat[M+1:, M+1]
                else:
                    Ocols = np.linalg.norm(Y, axis = 1)
            else:
                Ocols = np.linalg.norm(Y, axis = 1)
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
                clust_labels = get_clusters(Coords_0, num_clust, method = clust_method)
                
        elif clust_method == "Kmedoids":
            if num_clust == None:
                sys.exit("Kmedoids requires number of clusters parameter: num_clust")
            else:
                clust_labels, color_clustered = get_clusters(Coords_0, num_clust, method = clust_method)
                
        elif clust_method[:13] == "Agglomerative": 
            clust_labels = get_clusters(Coords_0, num_clust, method = clust_method)
        
        elif clust_method == "Spectral":
        	clust_labels= get_clusters(Coords_0, num_clust, method = clust_method) 
        
        elif clust_method == "GMM":
            clust_labels = get_clusters(Coords_0, num_clust, method = clust_method)
        
        elif clust_method == "BayesianGMM":
            clust_labels= get_clusters(Coords_0, num_clust, method = clust_method)
         
        elif clust_method == "DBSCAN":
            clust_labels = get_clusters(Coords_0, num_clust, method = clust_method)
            
        else:
            sys.exit("A metric-distance based clustering method is better for MIASA \n Available here is Kmeans")
        
        if get_score:
            lab_lists = []
            list_num = np.arange(num_clust_range[0], num_clust_range[1]).astype(int)
            for i in range(len(list_num)):
                lab_lists.append(get_clusters(Coords_0, num_clust = list_num[i], method = clust_method))
            
            silhouette, elbow, distortion = compute_scores(Coords_0, list_num, lab_lists, M, N)
        else:
            silhouette, elbow, distortion,list_num = None, None, None, None
            
        if dist_origin[0] or dist_origin[1]:
            Coords = Coords - Coords[M, :][np.newaxis, :]
            Class_pred = np.concatenate((clust_labels[:M], clust_labels[-N:]), axis = 0)
            was_orig = True
        else:
            Class_pred = clust_labels
            was_orig = False
       
        DMat = Prox_Mat(DX, DY, UX = Orows, UY = Ocols, fXY = D_assoc)
        Result = {"Coords": Coords, "shape":(M, N), "was_orig":was_orig, "vareps":vareps, "Class_pred":Class_pred, "clust_labels":clust_labels, 
                  "DMat":DMat, "X":X, "Y":Y, "num_iterations":num_it, "silhouette":silhouette, "elbow":elbow, "distortion":distortion, "list_num":list_num}
    
    else:
        
        Result = None   
        
    return Result

rand = 0
def get_clusters(Coords, num_clust, method = "Kmeans", init = "k-means++", metric = "Euclidean"):
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
    
    return labels



def CosLM(DX, DY, UX = None, UY = None, fXY = None, c = None, similarity_method = ("Euclidean", "Euclidean")):
    """
    @brief Compute the cosine law matrix
    Parameters
    ----------
    DX : shape (M, M), Distance set or Proximity Set associated to a set of points X
    DY : shape (N, N), Distance set or Proximity Set associated to a set of points Y
    UX : shape (M,)  , Distance to theoretical origin point for the set of points X
    UY : shape (N,)    ,Distance to theoretical origin point for the set of points Y   
    fXY  : shape (M, N), Proximity set matrix between the points of X and Y, Compatible with the positions of the points in DX and DY
    c  : Dictionary of parameters with keys "c1", "c2", "c3"
    
    Returns
    -------
    CL_Mat : np.array
            Corresponding Cosine Law Matrix with reference at index 0
    """
    # compute f^0
    F0 = Prox_Mat(DX, DY, UX, UY, fXY)
    M = DX.shape[0]

    # compute cos Mat for W associated with f^0
    if (similarity_method[0] == "Euclidean")&(similarity_method[1] == "Euclidean"):
        a = 0
    elif (similarity_method[0] == "Euclidean"):
        a = 0
    elif (similarity_method[1] == "Euclidean"):
        if (UX is not None) or (UY is not None):
            a = M+1
        else:
            a = M
            
    else:
        sys.exit("similarity_method parameter: At least one similarity method has to be Euclidean")
    
    CL_Mat0 = (F0[a, :][np.newaxis, :]**2 + F0[:, a][:, np.newaxis]**2 - F0**2)/2
    # compute zeta_f
    CC = np.zeros(CL_Mat0.shape)
    if (UX is not None) or (UY is not None):
        # only pick the components involving f(X,Y) -- the antidiagonal blocks excluding positions of origin 
        CC[:M, M+1:] = CL_Mat0[:M, M+1:] 
        CC[M+1:,:M] = CL_Mat0[M+1: , :M]
    else:
    	# only pick the components involving f(X,Y) -- the antidiagonal blocks
        CC[:M, M:] = CL_Mat0[:M, M:] 
        CC[M:,:M] = CL_Mat0[M:,:M]
    
    #pdb.set_trace()
    Ri = np.sum(np.abs(CC), axis = 1)
    zeta_f = np.max(Ri)
    if zeta_f == 0:
        sys.exit("Distance/Proximity cannot be zero everywhere")
    # insert the proximity values for the theoretical point z and w_1 = x_1
    c1, c2, c3 = c["c1"], c["c2"], c["c3"]
    if c2 == "default":
        c2 = c1
    if c3 == "default":
        c3 = min(2*c2 + 2, c1 + c2 + 2)
    
    if c3<0:
        sys.exit("c3 is negative but c1, c2, c3 must be positive")
    
    
    D = np.zeros((F0.shape[0]+1, F0.shape[0]+1))
    D[0, 1] = np.sqrt(c1*zeta_f)
    D[0, 2:] =  np.sqrt(F0[0, 1:]**2 + c2*zeta_f)
    D[1:, 0] = D[0, 1:]  
    
    # compute f^vareps
    vareps =  c3*zeta_f
    F_vareps = np.sqrt(F0**2 + vareps) - np.diag(np.sqrt(vareps*np.ones(F0.shape[0]))) ### remove diagonal elements because there is should always be 0
    #pdb.set_trace()
    D[1:, 1:] = F_vareps

    # Compute cos Mat for V^z associated to f^vareps with referenc z at index 0       
    CL_Mat = (D[0, :][np.newaxis, :]**2 + D[:, 0][:, np.newaxis]**2 - D**2)/2
    
    return CL_Mat, c1, c2, c3, zeta_f


def Prox_Mat(DX, DY, UX = None, UY = None, fXY = None):
    M = DX.shape[0]
    N = DY.shape[0]
    if (fXY is None) or (np.all(np.isclose(fXY, np.zeros((M, N))))) or (np.any(fXY < 0)):
        sys.exit("fXY non-negative and not zero everywhere is needed \n fXY : Proximity set matrix between the points of X and Y compatible with the positions of the points in DX and DY")

    else:
        if UX is not None:
            # put the distance to the origin after all the points of X
            DX1 = np.concatenate((DX, UX[np.newaxis, :]), axis = 0)
            DX1 = np.concatenate((DX1,np.concatenate((UX.T, np.array([0])), axis = 0)[:, np.newaxis]), axis = 1)
    
        if UY is not None:
            # put the distance to the origin after all the points of X and before all the points of Y
            DY1 = np.concatenate((UY[np.newaxis, :], DY), axis = 0)
            DY1 = np.concatenate((np.concatenate((np.array([0]), UY.T), axis = 0)[:, np.newaxis], DY1), axis = 1)
            
        if (UX is not None) or (UY is not None):
            D = np.zeros((M+N+1, M+N+1))
            D[:M, M+1:] = fXY
            D[M+1:, :M] = D[:M, M+1:].T
            
            if (UX is not None)&(UY is None):
                D[:M+1, :M+1] = DX1
                D[M+1:, M+1:] = DY
        
            elif (UY is not None)&(UX is None):
                D[:M, :M] = DX
                D[M:, M:] = DY1
                
            else:
                D[:M+1, :M+1] = DX1
                D[M:, M:] = DY1
        
        else:
            D = np.zeros((M+N, M+N))
            D[ :M, M:] = fXY
            D[M:,  :M] = D[:M, M:].T
            
            D[:M, :M] = DX
            D[M:, M:] = DY
        
        return D
    

def Dist_Emb(D):
    Eucl_place_holder = spsp.distance.squareform(spsp.distance.pdist(D))
    ZuZ, vareps = Euclidean_Embedding(Eucl_place_holder, D, UX = None, UY = None, fXY = D, c_dic = "default", num_iterations = False) 
    Ztilde = ZuZ[D.shape[0]:, :]
    Dtilde = spsp.distance.pdist(Ztilde)
    Dtilde = spsp.distance.squareform(Dtilde)
    return Dtilde

def read_data(file, give_vars=False):
    rawData = pd.read_excel(file, engine='openpyxl')
    try:
        rawData.drop(columns = "Unnamed: 0", inplace = True)
    except:
        pass
    Data = rawData.drop(columns = "variable", inplace = False)
    if not give_vars:
        return Data.to_numpy().astype(float)
    else:
        return Data.to_numpy().astype(float), np.array(rawData["variable"]).astype(str)
    
X, X_vars = read_data(sys.argv[1], give_vars=True)
Y, Y_vars = read_data(sys.argv[2], give_vars=True)

sim_meth_X = str(sys.argv[3])
sim_meth_Y = str(sys.argv[4])

if sim_meth_X != "precomputed":
    DX = spsp.distance.pdist(X, metric = sim_meth_X)
    DX = spsp.distance.squareform(DX)
else:
    DX = read_data(sys.argv[5])
    DX = DX.to_numpy().astype(float)
    
if sim_meth_Y != "precomputed":
    DY = spsp.distance.pdist(Y, metric = sim_meth_Y)
    DY = spsp.distance.squareform(DY)
else:
    DY = read_data(sys.argv[6])
    DY = DY.to_numpy().astype(float)
    
D_assoc = read_data(sys.argv[7])

eucl_X = str(sys.argv[8])
eucl_Y = str(sys.argv[9])

if sim_meth_X != "Euclidean" and eucl_X == "TRUE":
   DX = Dist_Emb(DX)
   meth_X = "Euclidean" 
else:
   meth_X = "precomputed"
    
if sim_meth_Y != "Euclidean" and eucl_Y == "TRUE":
    DY = Dist_Emb(DY)
    meth_Y = "Euclidean" 
else:
    meth_Y = "precomputed"
    
if (sim_meth_X != "Euclidean" and sim_meth_Y != "Euclidean") and (meth_X == "precomputed" and meth_Y == "precomputed"):
    if sim_meth_X !="Euclidean":
        DX = Dist_Emb(DX) 
        meth_X = "Euclidean"
    if sim_meth_Y !="Euclidean":
        DY = Dist_Emb(DY)
        meth_Y = "Euclidean"
        
    
if eucl_X == "TRUE":
    meth_X = "Euclidean"
else:
    meth_X = "precomputed"

if eucl_Y == "TRUE":
    meth_Y = "Euclidean"
else:
    meth_Y = "precomputed" 

similarity_method = (meth_X, meth_Y)

norm_X = str(sys.argv[10])
norm_Y = str(sys.argv[11])

if norm_X == "TRUE":
    Orows = np.linalg.norm(X, axis = 1)
    Ox = True
else:
    Orows = None
    Ox = False
    
if norm_Y == "TRUE":
    Ocols = np.linalg.norm(Y, axis = 1)
    Oy = True
else:
    Oclos = None
    Oy = False
    
clust_method = str(sys.argv[12])
num_clust = sys.argv[13]
if str(num_clust) != "Score":
    get_score = False
    file_res = str(sys.argv[14])+"/miasa_results.pck"
    num_clust_range = None
    num_clust = int(num_clust)
else:
    num_clust = 2
    get_score = True
    file_res = str(sys.argv[14])+"/scored_miasa_results.pck"
    num_clust_range = (int(sys.argv[15]), int(sys.argv[16])+1)
    
    
Feature_dic = {}
Feature_dic["DMat"] = Prox_Mat(DX, DY, UX = Orows, UY = Ocols, fXY = D_assoc)
Feature_dic["dist_origin"] = (Ox, Oy)

Results = Miasa_Class(X, Y, num_clust, Feature_dic=Feature_dic, clust_method=clust_method, similarity_method= similarity_method, get_score = get_score, num_clust_range= num_clust_range)
Results["X_vars"]=X_vars
Results["Y_vars"]=Y_vars
import pickle
file = open(file_res, "wb")
pickle.dump(Results, file)
file.close()

import matplotlib.pyplot as plt
if get_score:
    list_num = Results["list_num"]
    elbow = Results["elbow"]
    distortion = Results["distortion"]
    silhouette = Results["silhouette"]
    
    fig = plt.figure(figsize = (21,5))
    ax1 = fig.add_subplot(131)
    elbow_norm = (elbow - min(elbow))/(max(elbow) - min(elbow))
    invert_elbow = 1 - elbow_norm
    plt.plot(list_num, invert_elbow, "o", label = "F-stat")
    plt.title("Inverted Elbow")
    plt.ylabel("1 - F-stat (normalized)", fontsize = 15)
    plt.plot(list_num, invert_elbow, "--")
    diff_norm = np.diff(invert_elbow)/max(np.diff(invert_elbow))
    diff_norm_full = np.zeros(len(elbow))
    diff_norm_full[1:] = diff_norm
    plt.plot(list_num, diff_norm_full, label = "curve")
    plt.legend()
    
    ax2 = fig.add_subplot(132)
    distortion_norm = (distortion - min(distortion))/(max(distortion) - min(distortion))
    plt.plot(list_num, distortion_norm, "^", label = "distortion")
    plt.plot(list_num, distortion_norm, "--")
    plt.title("Distortion")
    plt.ylabel("Score (normalized)", fontsize = 15)
    
    ax2 = fig.add_subplot(133)
    plt.plot(list_num, silhouette, "^", label = "silhouette")
    plt.plot(list_num, silhouette, "--")
    plt.title("Average Silhouette")
    plt.ylabel("score", fontsize = 15)
    
    from matplotlib.backends.backend_pdf import PdfPages
    pdf= PdfPages(str(sys.argv[14])+"/Cluster_scores.pdf")
    pdf.savefig(fig, bbox_inches = "tight")
    pdf.close()
    plt.savefig(str(sys.argv[14])+"/Cluster_scores.svg", bbox_inches='tight')
