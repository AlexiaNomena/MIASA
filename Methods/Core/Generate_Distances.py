#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 10:28:45 2022

@author: raharinirina
"""
import scipy.spatial as scSp
import numpy as np
import sys
from scipy.stats import ks_2samp
import pdb

def Similarity_Distance(Z, method = "Euclidean"):
    """@params: Coord is a datasets (num sample, num realization)
    """
    if method == "Euclidean":
        D = scSp.distance.pdist(Z)
        D = scSp.distance.squareform(D)
    else:
        sys.exit("Method is not suitable. Needed: Euclidean")   
    return D

def Association_Distance(Z, func, ftype = "vectorized"):
    """@params: Z is a tuple of two datasets, one dataset is an array (num samples, num realization)
                func is the function to be used to compute the association distance
                ftype should be set to str vectorized if func is array operation
    """
    if ftype == "vectorized":
        D = func(Z)
    
    else:
        M, N = Z[0].shape[0], Z[1].shape[0]
        D = np.zeros((M, N))
        for i in range(M):
            for j in range(N):
                Z_ij = (Z[0][i, :], Z[1][j, :])
                D[i, j] = func(Z_ij)
    return D


def KS_Distance(Z, to_use = "KS-stat-stat"):
    """@brief:  Kolmogorov-Snirnov statitics/pvalue to be used in Non-Metric clustering
       @params: Z is a is a datasets (num sample, num realization)
       @to_use: str KS-stat-stat: KS statistics, str KS-p1-p1: KS pvalue
    """
    D = np.zeros((Z.shape[0], Z.shape[0]))
    for i in range(Z.shape[0]):
        for j in range(Z.shape[0]):
            if to_use == "KS-stat-stat":
                D[i, j] = ks_2samp(Z[i, :], Z[j, :]).statistic
            else: #"KS-stat-stat"
                D[i, j] = 1e-5 + 1 - ks_2samp(Z[i, :], Z[j, :]).pvalue
    return D
            
def KS_Distance_Mixed(Z, to_use = "KS-p1-stat"):
    """@brief:  Mixing Kolmogorov-Snirnov statitics/pvalue to be used in Non-Metric clustering
       @params: Z is a tuple of two datasets, one dataset is an array (num samples, num realization)
       @to_use: str KS-p1-stat: KS statistics, str KS-stat-p1: KS pvalue
    """
    X, Y = Z
    M = X.shape[0]
    N = Y.shape[0]
    D = np.zeros((M+N, M+N))
    if to_use == "KS-p1-stat":
        D[:M, :M] = KS_Distance(X, to_use = "KS-p1-p1")
        D[M:, M:] = KS_Distance(Y, to_use = "KS-p1-p1")
        fXY = np.zeros((M, N))
        for i in range(M):
            for j in range(N):
                fXY[i, j] = ks_2samp(X[i, :], Y[j, :]).statistic
         
        D[ :M, M:] = fXY
        D[M:,  :M] = D[:M, M:].T
        
    else: #"KS-stat-p1":
        D[:M, :M] = KS_Distance(X, to_use = "KS-stat-stat")
        D[M:, M:] = KS_Distance(Y, to_use = "KS-stat-stat")
        fXY = np.zeros((M, N))
        for i in range(M):
            for j in range(N):
                fXY[i, j] = 1e-5 + 1 - ks_2samp(X[i, :], Y[j, :]).pvalue
         
        D[ :M, M:] = fXY
        D[M:,  :M] = D[:M, M:].T
    
    return D
    