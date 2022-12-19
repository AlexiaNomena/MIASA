#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 10:04:28 2022

@author: araharin
"""
from .CosLM import *
import scipy as sp
import numpy as np

def Euclidean_Embedding(DX, DY, UX, UY, fXY, c):
    """
    @brief Joint Embedding of two disjoint sets of points (see Paper: Qualitative Euclidean Embedding)
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
    Coords : np.array shap (M+N, M+N+2)
            Coordinates of points X and Y on the rows
    vareps: > 0 scalar defining the Embedding (see Paper)
    """
    COS_MAT, c1, c2, c3, zeta_f = CosLM(DX, DY, UX, UY, fXY, c) 
    sigma, U = sp.linalg.eigh(COS_MAT)
    sigma = np.real(sigma) # COS_MAT is symmetric, thus imaginary numbers are supposed to be zero or numerical zeros
    sigma[np.isclose(sigma, np.zeros(len(sigma)))] = 0
    
    sort = np.argsort(sigma)[::-1] # descending order
    sigma = sigma[sort]
    U = U[:, sort]
    test = np.sum(sigma<0)
    if test == 0:
        print("Replacement matrix is PSD: success Euclidean embedding")
        SS = np.sqrt(np.diag(sigma)) 
        Coords0 = np.real(U.dot(SS))
        
        """ Then remove the connecting point (see Paper: Qualitative Euclidean Embedding) """
        Coords = Coords0[1:, :]
    
    else:
        sys.exit("Theorem was not satified there might be bugs in the code \n Check if c1, c2, c3 are chosen correctly")
    
    
    vareps = c3*zeta_f  
    return Coords, vareps