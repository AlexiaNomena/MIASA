#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 10:04:28 2022

@author: araharin
"""
from .CosLM import *
import scipy as sp
import numpy as np

def Euclidean_Embedding(DX, DY, UX, UY, fXY, c_dic, in_threads = False):
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
    try:
        COS_MAT, c1, c2, c3, zeta_f = CosLM(DX, DY, UX, UY, fXY, c_dic) 
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
            COS_MAT, c1, c2, c3, zeta_f = CosLM(DX, DY, UX, UY, fXY, c_dic)
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
            Coords = None
            if not in_threads:
                print("Wrong parameters in c_dic:", c_dic, "Auto adjusted %d/%d times"%(sc, stop))
                print("step 1: First progressively increase c2 and c3 together (try c3 = 2 + c2 + c1), then compute line 31 and 32")
                print("step 2a: if the value of the negative eigenvalues decreased then return to step 1 ")
                print("step 2b: if the value of the negative eigenvalues increased, then increase c1 (with c1 < c2 < c3) return to step 1")
                pdb.set_trace()
                sys.exit("Theorem was not satified \n Check if c1, c2, c3 are chosen correctly")
    
    except:
        print("failed Euclidean embedding")
        Coords = None
        c3 = 0
        zeta_f = 0
        
    vareps = c3*zeta_f
    return Coords, vareps