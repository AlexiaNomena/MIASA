#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 13:39:28 2022

@author: raharinirina
"""
import numpy as np
from scipy.stats import ks_2samp

import pdb

def EmpCDF(X, interval):
    cdf = np.sum(X[:, :, np.newaxis]<= interval, axis = 1)/X.shape[1]
    return cdf

def KS_v1(X,Y):
    lbd = min(np.min(X), np.min(Y))
    ubd = max(np.max(X), np.max(Y))
    
    interval = np.linspace(lbd, ubd, 500)
    Feature_X = EmpCDF(X, interval)
    Feature_Y = EmpCDF(Y, interval)
    func = lambda Z: 1e-3 + np.abs(ks_2samp(Z[0], Z[1]).statistic) # use the KS statistic  added a constant to avoid zero everywhere
    ftype = "not_vectorized"
    return Feature_X, Feature_Y, func, ftype

def KS_v2(X,Y):
    lbd = min(np.min(X), np.min(Y))
    ubd = max(np.max(X), np.max(Y))
    
    interval = np.linspace(lbd, ubd, 500)
    Feature_X = EmpCDF(X, interval)
    Feature_Y = EmpCDF(Y, interval)
    #func = lambda Z: np.exp(-ks_2samp(Z[0], Z[1]).pvalue) #1e-3 + 1 - ks_2samp(Z[0], Z[1]).pvalue # use the KS statistic  added a constant to avoid zero everywhere
    func = lambda Z: np.exp(-500*ks_2samp(Z[0], Z[1]).pvalue)
    ftype = "not_vectorized"
    return Feature_X, Feature_Y, func, ftype

def KS_v3(X,Y):
    lbd = min(np.min(X), np.min(Y))
    ubd = max(np.max(X), np.max(Y))
    
    interval = np.linspace(lbd, ubd, 500)
    Feature_X = EmpCDF(X, interval)
    Feature_Y = EmpCDF(Y, interval)
    func = lambda Z: 1e-3 + 1 - ks_2samp(Z[0], Z[1]).pvalue # use the KS statistic  added a constant to avoid zero everywhere
    ftype = "not_vectorized"
    return Feature_X, Feature_Y, func, ftype

def Sub_Eucl(X, Y):
    Feature_X = X.copy()
    Feature_Y = Y.copy()
    func = lambda Z: np.max(np.abs(Z[0][:, np.newaxis] - Z[1][np.newaxis, :]))
    ftype = "vectorized"
    return Feature_X, Feature_Y, func, ftype

def covariance(X, Y):
    Feature_X = np.cov(X)
    Feature_Y = np.cov(Y)
    func = lambda Z: np.exp(-(Z[0]- np.mean(Z[0], axis = 1)[np.newaxis, :]).dot((Z[1]- np.mean(Z[1], axis = 1)[np.newaxis, :]).T))
    ftype = "vectorized"      
    return Feature_X, Feature_Y, func, ftype
        
def corrcoeff(X, Y):
    Feature_X = np.cov(X)/np.var(X, axis = 1)
    Feature_Y = np.cov(Y)/np.var(Y, axis = 1)
    func = lambda Z: np.exp(-(Z[0]- np.mean(Z[0], axis = 1)[np.newaxis, :]).dot((Z[1]- np.mean(Z[1], axis = 1)[np.newaxis, :]).T)/np.std(Z[0], axis = 1)[:, np.newaxis]*np.std(Z[1], axis = 1)[np.newaxis, :])
    ftype = "vectorized"      
    return Feature_X, Feature_Y, func, ftype 
    
    