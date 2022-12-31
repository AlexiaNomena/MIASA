#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 13:39:28 2022

@author: raharinirina
"""
import numpy as np
from scipy.stats import ks_2samp, kurtosis, skew
import scipy

import pdb

def EmpCDF(X, interval):
    cdf = np.sum(X[:, :, np.newaxis]<= interval, axis = 1)/X.shape[1]
    return cdf

def eCDF(X,Y):
    """ Empirical CDF similarity and Komogorov-Smirnov statistics association version 2 (test)"""
    lbd = min(np.min(X), np.min(Y))
    ubd = max(np.max(X), np.max(Y))
    interval = np.linspace(lbd, ubd, 500)
    Feature_X = EmpCDF(X, interval)
    Feature_Y = EmpCDF(Y, interval)
    func = dCDF
    ftype = "vectorized"
    return Feature_X, Feature_Y, func, ftype

def dCDF(Z):
    X, Y = Z
    lbd = min(np.min(X), np.min(Y))
    ubd = max(np.max(X), np.max(Y))
    interval = np.linspace(lbd, ubd, 500)
    eX = EmpCDF(X, interval)
    eY = EmpCDF(Y, interval)
    
    D = 1e-5 + scipy.spatial.distance.cdist(eX, eY)
    return D
    
    

def eCDF_KS_stat(X,Y):
    """ Empirical CDF similarity and Komogorov-Smirnov statistics association version 2 (test)"""
    lbd = min(np.min(X), np.min(Y))
    ubd = max(np.max(X), np.max(Y))
    interval = np.linspace(lbd, ubd, 500)
    Feature_X = EmpCDF(X, interval)
    Feature_Y = EmpCDF(Y, interval)
    func = lambda Z: 1e-5 + np.abs(ks_2samp(Z[0], Z[1]).statistic) # use the KS statistic  added a constant to avoid zero everywhere
    ftype = "not_vectorized"
    return Feature_X, Feature_Y, func, ftype

def eCDF_KS_p1(X,Y):
    """ Empirical CDF similarity and p_value association version 1"""
    lbd = min(np.min(X), np.min(Y))
    ubd = max(np.max(X), np.max(Y))
    interval = np.linspace(lbd, ubd, 500)
    Feature_X = EmpCDF(X, interval)
    Feature_Y = EmpCDF(Y, interval)
    func = lambda Z: 1e-5 + 1 - ks_2samp(Z[0], Z[1]).pvalue # use the KS statistic  added a constant to avoid zero everywhere
    ftype = "not_vectorized"
    return Feature_X, Feature_Y, func, ftype

def eCDF_KS_p2(X,Y):
    """ Empirical CDF similarity and p_value association version 2 (test)"""
    lbd = min(np.min(X), np.min(Y))
    ubd = max(np.max(X), np.max(Y))
    interval = np.linspace(lbd, ubd, 500)
    Feature_X = EmpCDF(X, interval)
    Feature_Y = EmpCDF(Y, interval)
    func = lambda Z: np.exp(-500*ks_2samp(Z[0], Z[1]).pvalue)
    ftype = "not_vectorized"
    return Feature_X, Feature_Y, func, ftype


def Sub_Eucl(X, Y):
    """ Euclidean similarity and Association distance = Max of paiwise distances between all vector components"""
    Feature_X = X.copy()
    Feature_Y = Y.copy()
    func = lambda Z: np.max(np.abs(Z[0][:, np.newaxis] - Z[1][np.newaxis, :]))
    ftype = "vectorized"
    return Feature_X, Feature_Y, func, ftype

def covariance(X, Y):
    """Fully covariance feature"""
    Feature_X = np.cov(X)
    Feature_Y = np.cov(Y)
    func = dcov
    ftype = "vectorized"      
    return Feature_X, Feature_Y, func, ftype

def dcov(Z):
    """the empirical covariance formula bellow is only valid for equal number of observations in each samples it is equal to np.cov(X,Y)"""
    scov = (1/(Z[0].shape[1] - 1))*np.dot(Z[0] - np.mean(Z[0], axis = 1)[:, np.newaxis], (Z[1]- np.mean(Z[1], axis = 1)[:, np.newaxis]).T)
    
    U1 = np.abs(scov) - np.var(Z[0], axis = 1)
    U2 = np.abs(scov) - np.var(Z[1], axis = 1)
    
    U1 = U1/np.max(U1)
    U2 = U2/np.max(U2)
    
    res = np.exp(-(U1)**2)*np.exp(-(U2)**2)
    return 1e-3 + res # added a small constant to avoid identically zero
        
def corrcoeff(X, Y):
    """Fully corrcoef feature"""
    """np.cov computes sample covariance of equal number of observations per samples"""
    Feature_X = np.corrcoef(X) #np.cov(X)/np.var(X, axis = 1)
    Feature_Y = np.corrcoef(Y)#np.cov(Y)/np.var(Y, axis = 1)
    """the empirical corrcoeff formula bellow is only valid for equal number of observations in each samples it is equal to np.cov(X,Y)/std(Y)std(Y)"""
    func = dcorr
    ftype = "vectorized"      
    return Feature_X, Feature_Y, func, ftype 

def dcorr(Z):
    """the empirical corrcoeff formula bellow is only valid for equal number of observations in each samples it is equal to np.cov(X,Y)/std(Y)std(Y)"""
    scov = (1/(Z[0].shape[1] - 1))*np.dot(Z[0] - np.mean(Z[0], axis = 1)[:, np.newaxis], (Z[1]- np.mean(Z[1], axis = 1)[:, np.newaxis]).T)
    scorr = scov/np.std(Z[0], axis = 1)[:, np.newaxis]*np.std(Z[1], axis = 1)[np.newaxis, :]
    res = np.exp(- (np.abs(scorr) - 1)**2)
    return 1e-5 + res  # added a small constant to avoid identically zero

def covariance_moms(X, Y):
    """Covariance similarity and moments association"""

    Feature_X = np.cov(X)
    Feature_Y = np.cov(Y)
    func = dMoments
    ftype = "vectorized"      
    return Feature_X, Feature_Y, func, ftype

def corrcoeff_moms(X, Y):
    """Corrcoeff similarity and moments association"""
    
    """np.cov computes sample covariance of equal number of observations per samples"""
    Feature_X = np.corrcoef(X) #np.cov(X)/np.var(X, axis = 1)
    Feature_Y = np.corrcoef(Y)#np.cov(Y)/np.var(Y, axis = 1)
    func = dMoments
    ftype = "vectorized"      
    return Feature_X, Feature_Y, func, ftype 

def moms_covariance(X, Y):
    """Moments similarity and covariance associations"""
    Feature_X = Moments_feature(X)
    Feature_Y = Moments_feature(Y)
    func = dcov
    ftype = "vectorized"      
    return Feature_X, Feature_Y, func, ftype

def moms_corrcoeff(X, Y):
    """Moments similarity and corrcoeff association"""
    
    """np.cov computes sample covariance of equal number of observations per samples"""
    Feature_X = Moments_feature(X)
    Feature_Y = Moments_feature(Y)
    """the corrcoeff formula bellow is only valid for equal number of observation in at the samples it is equal to np.cov(X,Y)/std(Y)std(Y)"""
    func = dcorr
    ftype = "vectorized"      
    return Feature_X, Feature_Y, func, ftype 

def dMoments(Z):
    X, Y = Z
    dM1 = (np.mean(X, axis = 1)[:, np.newaxis] - np.mean(Y, axis = 1)[np.newaxis, :])
    dM2 = (np.var(X, axis = 1)[:, np.newaxis] - np.var(Y, axis = 1)[np.newaxis, :]) 
    dM3 = (skew(X, axis = 1)[:, np.newaxis] - skew(Y, axis = 1)[np.newaxis, :]) 
    dM4 = (kurtosis(X, axis = 1)[:, np.newaxis] - kurtosis(Y, axis = 1)[np.newaxis, :])
    
    dM = np.sqrt(dM1**2 + dM2**2 + dM3**2 + dM4**2)
    return dM  + 1e-5 # added a small constant to avoid identically zero

def Moments_feature(Z):
    M1 = np.mean(Z, axis = 1)
    M2 = np.var(Z, axis = 1)
    M3 = skew(Z, axis = 1)
    M4 = kurtosis(Z, axis = 1)
    res = np.column_stack((M1, M2, M3, M4))
    return res

def moms(X, Y):
    """ Fully Moments feature"""
    Feature_X = Moments_feature(X)
    Feature_Y = Moments_feature(Y)
    func = dMoments
    ftype = "vectorized"      
    return Feature_X, Feature_Y, func, ftype 

def moms_OR(X, Y):
    """ Moments similarity and dOR association"""
    Feature_X = Moments_feature(X)
    Feature_Y = Moments_feature(Y)
    
    Feature_X = Feature_X/np.max(Feature_X)
    Feature_Y = Feature_Y/np.max(Feature_Y)
    
    func = dOR
    ftype = "vectorized"      
    return Feature_X, Feature_Y, func, ftype 

def dOR(Z):
    X, Y = Z
    dX = X[:, 1:] - X[:, :-1]
    dY = Y[:, 1:] - Y[:, :-1]
    
    p_in_X = np.sum(dX > 0, axis = 1)/len(dX)
    p_out_X = 1 - p_in_X
    
    p_in_Y = np.sum(dY > 0, axis = 1)/len(dY)
    p_out_Y = 1 - p_in_Y
    
    OR_in_X = np.divide(p_in_X, p_out_X, out = 1e5*np.ones(len(p_in_X)), where = p_out_X != 0) # 1e5 replacing infinity
    OR_in_Y = np.divide(p_in_Y, p_out_Y, out = 1e5*np.ones(len(p_in_Y)), where = p_out_Y!=0)
    
    M = len(OR_in_X)
    N = len(OR_in_Y)
    
    OR_in_XY = np.divide(OR_in_X[:, np.newaxis], OR_in_Y[np.newaxis, :], out = 1e5*np.ones((M, N)), where = OR_in_Y != 0)
    OR_in_YX = np.divide(OR_in_Y[np.newaxis, :], OR_in_X[:, np.newaxis], out = 1e5*np.ones((M, N)), where = OR_in_X != 0)
    
    dOR = np.exp(- (OR_in_XY + OR_in_YX) )
    
    return dOR + 1e-35# added a small constant to avoid identically zero

def OR(X, Y):
    """ Fully Odd_ratio feature """
    dX = X[:, 1:] - X[:, :-1]
    dY = Y[:, 1:] - Y[:, :-1]
    
    p_in_X = np.sum(dX > 0, axis = 1)/len(dX)
    p_out_X = 1 - p_in_X
    
    p_in_Y = np.sum(dY > 0, axis = 1)/len(dY)
    p_out_Y = 1 - p_in_Y
    
    OR_in_X = np.divide(p_in_X, p_out_X, out = 1e5*np.ones(len(p_in_X)), where = p_out_X != 0) # 1e5 replacing infinity
    OR_in_Y = np.divide(p_in_Y, p_out_Y, out = 1e5*np.ones(len(p_in_Y)), where = p_out_Y!=0)
    
    M = len(OR_in_X)
    N = len(OR_in_Y)

    Feature_X = np.divide(OR_in_X[:, np.newaxis], OR_in_X[np.newaxis, :], out = 1e5*np.ones((M, M)), where = OR_in_X[np.newaxis, :] != 0)
    Feature_Y = np.divide(OR_in_Y[:, np.newaxis], OR_in_Y[np.newaxis, :], out = 1e5*np.ones((N, N)), where = OR_in_Y[np.newaxis, :] != 0)
    
    func = dOR
    ftype = "vectorized"
    return Feature_X, Feature_Y, func, ftype

def OR_moms(X, Y):
    """ Odd_ratio similarity and dMoments association """
    dX = X[:, 1:] - X[:, :-1]
    dY = Y[:, 1:] - Y[:, :-1]
    
    p_in_X = np.sum(dX > 0, axis = 1)/len(dX)
    p_out_X = 1 - p_in_X
    
    p_in_Y = np.sum(dY > 0, axis = 1)/len(dY)
    p_out_Y = 1 - p_in_Y
    
    OR_in_X = np.divide(p_in_X, p_out_X, out = 1e5*np.ones(len(p_in_X)), where = p_out_X != 0) # 1e5 replacing infinity
    OR_in_Y = np.divide(p_in_Y, p_out_Y, out = 1e5*np.ones(len(p_in_Y)), where = p_out_Y!=0)
    
    M = len(OR_in_X)
    N = len(OR_in_Y)
    
    Feature_X = np.divide(OR_in_X[:, np.newaxis], OR_in_X[np.newaxis, :], out = 1e5*np.ones((M, M)), where = OR_in_X[np.newaxis, :] != 0)
    Feature_Y = np.divide(OR_in_Y[:, np.newaxis], OR_in_Y[np.newaxis, :], out = 1e5*np.ones((N, N)), where = OR_in_Y[np.newaxis, :] != 0)
    
    func = dMoments
    ftype = "vectorized"
    return Feature_X, Feature_Y, func, ftype
    
    