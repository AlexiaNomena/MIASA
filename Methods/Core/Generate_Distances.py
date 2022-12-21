#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 10:28:45 2022

@author: raharinirina
"""
import scipy.spatial as scSp
import numpy as np
import sys


def Similarity_Metric(Coords, method = "Euclidean"):
    if method == "Euclidean":
        D = scSp.distance.pdist(Coords)
        D = scSp.distance.squareform(D)
    else:
        sys.exit("Method is not suitable. Needed: Euclidean")   
    return D


def Association_Metric(Coords, func, ftype = "vectorized"):
    if ftype == "vectorized":
        D = func(Coords)
    
    else:
        M, N = Coords[0].shape[0], Coords[1].shape[0]
        D = np.zeros((M, N))
        for i in range(M):
            for j in range(N):
                Coords_ij = (Coords[0][i, :], Coords[1][j, :])
                D[i, j] = func(Coords_ij)
    return D