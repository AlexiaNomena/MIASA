#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 10:28:45 2022

@author: raharinirina
"""
import scipy.spatial as scSp
import sys

def Similarity_Metric(Coords, method = "Euclidean"):
    if method == "Euclidean":
        D = scSp.distance.pdist(Coords)
        D = scSp.distance.squareform(D)
    else:
        sys.exit("Method is not suitable. Needed: Euclidean")   
    return D


def Association_Metric(Coords, func):
    D = func(Coords)
    return D