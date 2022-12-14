#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 10:28:45 2022

@author: araharin
"""
import scipy as sp
import sys

def Similarity_Metric(Coords, method = "Euclidean"):
    if method == "Euclidean":
        D = sp.spatial.distance.pdist(Coords)
        D = sp.spatial.distance.squareform(D)
    else:
        sys.exit("method is not implemented. Available: Euclidean")   
    return D


def Association_Metric(Coords_X, Coords_Y, func):
    Coords = (Coords_X, Coords_Y)
    D = func(Coords)
    return D