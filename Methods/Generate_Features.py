#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 13:39:28 2022

@author: raharinirina
"""
import numpy as np
from scipy.stats import ks_2samp, kurtosis, skew
import scipy.spatial as scSp
from statsmodels.tsa.stattools import grangercausalitytests as GrCausTest

import pdb
import sys


def eCDF(X,Y):
    """ Empirical CDF similarity and Komogorov-Smirnov statistics association"""
    lbd = min(np.min(X), np.min(Y))
    ubd = max(np.max(X), np.max(Y))
    interval = np.linspace(lbd, ubd, 500)
    Feature_X = EmpCDF(X, interval)
    Feature_Y = EmpCDF(Y, interval)
    return Feature_X, Feature_Y  

def Eucl(X, Y):
    """ Euclidean similarity and Association distance = Max of paiwise distances between all vector components"""
    Feature_X = X.copy()
    Feature_Y = Y.copy()
    return Feature_X, Feature_Y

def covariance(X, Y):
    """Fully covariance feature"""
    """np.cov computes sample covariance of equal number of observations per samples"""
    Feature_X = np.cov(X)
    Feature_Y = np.cov(Y)     
    return Feature_X, Feature_Y

def corrcoeff(X, Y):
    """Corrcoef feature"""
    """np.corrcoef computes sample covariance of equal number of observations per samples"""
    Feature_X = np.corrcoef(X) #np.cov(X)/np.var(X, axis = 1)
    Feature_Y = np.corrcoef(Y)#np.cov(Y)/np.var(Y, axis = 1)
    return Feature_X, Feature_Y

def moms(X, Y):
    """Moments similarity and covariance associations"""
    Feature_X = Moments_feature(X)## same as np.diff
    Feature_Y = Moments_feature(Y)
    return Feature_X, Feature_Y

def OR(X, Y):
    """ Fully Odd_ratio feature using increments and decrements """
    dX = X[:, 1:] - X[:, :-1] # same as np.diff
    dY = Y[:, 1:] - Y[:, :-1]
    
    p_in_X = np.sum(dX > 0, axis = 1)/len(dX)
    p_out_X = 1 - p_in_X
    
    p_in_Y = np.sum(dY > 0, axis = 1)/len(dY)
    p_out_Y = 1 - p_in_Y
    
    # increments
    OR_in_X = np.divide(p_in_X, p_out_X, out = 1e5*np.ones(len(p_in_X)), where = p_out_X != 0) # 1e5 replacing infinity
    OR_in_Y = np.divide(p_in_Y, p_out_Y, out = 1e5*np.ones(len(p_in_Y)), where = p_out_Y != 0)
    
    M = len(OR_in_X)
    N = len(OR_in_Y)

    Feature_X_in = np.divide(OR_in_X[:, np.newaxis], OR_in_X[np.newaxis, :], out = 1e5*np.ones((M, M)), where = OR_in_X[np.newaxis, :] != 0)
    Feature_Y_in = np.divide(OR_in_Y[:, np.newaxis], OR_in_Y[np.newaxis, :], out = 1e5*np.ones((N, N)), where = OR_in_Y[np.newaxis, :] != 0)
    
    # decrements
    OR_out_X = np.divide(p_out_X, p_in_X, out = 1e5*np.ones(len(p_in_X)), where = p_in_X != 0) # 1e5 replacing infinity
    OR_out_Y = np.divide(p_out_Y, p_in_Y, out = 1e5*np.ones(len(p_in_Y)), where = p_in_Y != 0)
    
    M = len(OR_in_X)
    N = len(OR_in_Y)

    Feature_X_out = np.divide(OR_out_X[:, np.newaxis], OR_out_X[np.newaxis, :], out = 1e5*np.ones((M, M)), where = OR_out_X[np.newaxis, :] != 0)
    Feature_Y_out = np.divide(OR_out_Y[:, np.newaxis], OR_out_Y[np.newaxis, :], out = 1e5*np.ones((N, N)), where = OR_out_Y[np.newaxis, :] != 0)
    
    Feature_X = np.column_stack((Feature_X_in, Feature_X_out))
    Feature_Y = np.column_stack((Feature_Y_in, Feature_Y_out))
    
    return Feature_X, Feature_Y

def Cond_proba(X, Y):
    """ Conditional proba of increments and decrements"""
    dX = X[:, 1:] - X[:, :-1] ## same as np.diff
    dY = Y[:, 1:] - Y[:, :-1]

    ### All important interaction types in X
    cpX_in_in, cpX_in_in = getCond(dX > 0, dX > 0)
    cpX_out_out, cpX_out_out = getCond(dX < 0, dX < 0)
    cpX_in_out, cpX_in_out = getCond(dX > 0, dX < 0)
    cpX_out_in, cpX_out_in = getCond(dX < 0, dX > 0)
    cpX_No_No, cpX_No_No = getCond(dX == 0, dX == 0)
    cpX_in_No, cpX_in_No = getCond(dX > 0 , dX==0)
    cpX_out_No, cpX_out_No = getCond(dX < 0 , dX==0)
    
    Feature_X = np.column_stack((cpX_in_in, cpX_out_out, cpX_in_out, cpX_out_in, cpX_No_No, cpX_in_No, cpX_out_No))
    
    ### All important interaction types in Y
    cpY_in_in, cpY_in_in = getCond(dY > 0, dY > 0)
    cpY_out_out, cpY_out_out = getCond(dY < 0, dY < 0)
    cpY_in_out, cpY_in_out = getCond(dY > 0, dY < 0)
    cpY_out_in, cpY_out_in = getCond(dY < 0, dY > 0)
    cpY_No_No, cpY_No_No = getCond(dY == 0, dY == 0)
    cpY_in_No, cpY_in_No = getCond(dY > 0 , dY==0)
    cpY_out_No, cpY_out_No = getCond(dY < 0 , dY==0)
    
    Feature_Y = np.column_stack((cpY_in_in, cpY_out_out, cpY_in_out, cpY_out_in, cpY_No_No, cpY_in_No, cpY_out_No))
      
    return Feature_X, Feature_Y

def Granger_Cause(X, Y, diff = False):
    Feature_X = np.zeros((X.shape[0], X.shape[0]))
    Feature_Y = np.zeros((Y.shape[0], Y.shape[0]))
    for i in range(max(X.shape[0], Y.shape[0])):
        if i < X.shape[0] :
            for j in range(X.shape[0]):
                Feature_X[i, j] =  GrCaus_Test_p_val((X[i, :],X[j, :]), diff = diff)
        if i < Y.shape[0] :
            for j in range(Y.shape[0]):
                Feature_Y[i, j] = GrCaus_Test_p_val((Y[i, :],Y[j, :]), diff = diff)
    return Feature_X, Feature_Y
        

def get_assoc_func(assoc_type, in_threads = False):
    if assoc_type == "eCDF":
        func, ftype = dCDF, "vectorized"
        
    elif assoc_type == "KS-stat":
        func, ftype = lambda Z: 1e-5 + np.abs(ks_2samp(Z[0], Z[1]).statistic), "not_vectorized"
        
    elif assoc_type == "KS-p1":
        func, ftype = lambda Z: 1e-5 + 1 - ks_2samp(Z[0], Z[1]).pvalue, "not_vectorized" # H0: dist are equal, small p_value = reject the null, we do not want to reject the null by definition of association, thus we take 1 - p_val
        
    elif assoc_type == "KS-p2":
        func, ftype = lambda Z: np.exp(-500*ks_2samp(Z[0], Z[1]).pvalue), "not_vectorized"
    
    elif assoc_type == "Sub_Eucl":
        func, ftype = lambda Z: np.max(np.abs(Z[0][:, np.newaxis] - Z[1][np.newaxis, :])), "vectorized"
        
    elif assoc_type == "dCov":
        func, ftype = dcov, "vectorized"
        
    elif assoc_type == "dCorr":
        func, ftype = dcorr, "vectorized"
        
    elif assoc_type == "dCov_v2":
        func, ftype = dcov_v2, "vectorized"
            
    elif assoc_type == "dCorr_v2":
        func, ftype = dcorr_v2, "vectorized"
        
    elif assoc_type == "Moms":
        func, ftype = dMoments, "vectorized"
    
    elif assoc_type == "dOR":
        func, ftype = dOR, "vectorized"
        
    elif assoc_type == "dCond":
        func, ftype = dCond, "vectorized"
        
    elif assoc_type == "Granger-Cause-orig":
        func, ftype = lambda Z: 1e-5 + GrCaus_Test_p_val(Z, diff = False, test="params_ftest") , "not_vectorized" # H0: Z[1] does NOT granger cause Z[0] and vis versa,small p_value = reject the null, we want to reject the null by definition of association, thus we take p_val
    
    elif assoc_type == "Granger-Cause-diff":
        func, ftype = lambda Z: 1e-5 + GrCaus_Test_p_val(Z, diff = True, test="params_ftest"), "not_vectorized"
    
    else:
        if in_threads:
            print(assoc_type)
            sys.exit("Association type is not implemented")
    
    return func, ftype


def GrCaus_Test_p_val(Z, maxlag = 10, diff = False, test = "ssr_chi2test"):
    if diff:
        Z1 = np.diff(Z[0])
        Z2 = np.diff(Z[1])
    else:
        Z1 = Z[0]
        Z2 = Z[1]
    # test Z[1] does not Granger Cause Z[0]
    test_res_1 = GrCausTest(np.column_stack((Z1, Z2)), maxlag = maxlag, verbose = False)
    p_values_1 = [test_res_1[i+1][0][test][1] for i in range(maxlag)]
    p_mes_1 = np.max(np.array(p_values_1))
    
    # test Z[0] does not Granger Cause Z[1]
    test_res_2 = GrCausTest(np.column_stack((Z2, Z1)), maxlag = maxlag, verbose = False)
    p_values_2 = [test_res_2[i+1][0][test][1] for i in range(maxlag)]
    p_mes_2 = np.max(np.array(p_values_2))
    res = np.mean([p_mes_1, p_mes_2])
    return res

def dOR(Z):
    X, Y = Z
    dX = X[:, 1:] - X[:, :-1]
    dY = Y[:, 1:] - Y[:, :-1]
    
    p_in_X = np.sum(dX > 0, axis = 1)/len(dX)
    p_out_X = 1 - p_in_X
    
    p_in_Y = np.sum(dY > 0, axis = 1)/len(dY)
    p_out_Y = 1 - p_in_Y
    
    # increments
    OR_in_X = np.divide(p_in_X, p_out_X, out = 1e5*np.ones(len(p_in_X)), where = p_out_X != 0) # 1e5 replacing infinity
    OR_in_Y = np.divide(p_in_Y, p_out_Y, out = 1e5*np.ones(len(p_in_Y)), where = p_out_Y!=0)
    
    M = len(OR_in_X)
    N = len(OR_in_Y)
    
    OR_in_XY = np.divide(OR_in_X[:, np.newaxis], OR_in_Y[np.newaxis, :], out = 1e5*np.ones((M, N)), where = OR_in_Y[np.newaxis, :] != 0)
    OR_in_YX = np.divide(OR_in_Y[:, np.newaxis], OR_in_X[np.newaxis, :], out = 1e5*np.ones((N, M)), where = OR_in_X[np.newaxis, :] != 0)
    
    dOR_in_xy = np.exp(-(OR_in_XY - 1)**2) ### we are interested in both Y increases the odds of increments events in X (OR>1) and Y decreases the odds of increments events in X (OR<1)
    dOR_in_yx = np.exp(-(OR_in_YX.T - 1)**2) ### vis versa
    
    # decrements
    OR_out_X = np.divide(p_out_X, p_in_X, out = 1e5*np.ones(len(p_in_X)), where = p_in_X != 0) # 1e5 replacing infinity
    OR_out_Y = np.divide(p_out_Y, p_in_Y, out = 1e5*np.ones(len(p_in_Y)), where = p_in_Y!=0)
    
    M = len(OR_out_X)
    N = len(OR_out_Y)
    
    OR_out_XY = np.divide(OR_out_X[:, np.newaxis], OR_in_Y[np.newaxis, :], out = 1e5*np.ones((M, N)), where = OR_in_Y[np.newaxis, :] != 0)
    OR_out_YX = np.divide(OR_out_Y[:, np.newaxis], OR_in_X[np.newaxis, :], out = 1e5*np.ones((N, M)), where = OR_in_X[np.newaxis, :] != 0)
    
    dOR_out_xy = np.exp(-(OR_out_XY - 1)**2) ### we are interested in both Y increases the odds of increments events in X (OR>1) and Y decreases the odds of increments events in X (OR<1)
    dOR_out_yx = np.exp(-(OR_out_YX.T - 1)**2) ### vis versa
    
    return dOR_in_xy*dOR_in_yx*dOR_out_xy*dOR_out_yx + 1e-5# added a small constant to avoid identically zero



def getCond(boolX,boolY):
    pXY = np.dot(1*boolX, 1*boolY.T)/boolX.shape[1]

    mrj_X = np.sum(boolX, axis = 1)/boolX.shape[1]
    mrj_Y = np.sum(boolY, axis = 1)/boolY.shape[1]
    
    # Conditional proba features (Centred around independence model, e.g. = 1)
    if not np.all(pXY == 0.):
        cpX = np.divide(pXY, mrj_X[:, np.newaxis]*mrj_Y[np.newaxis, :], out = np.zeros(pXY.shape), where = mrj_X[:, np.newaxis]!= 0) - 1
    else:
        cpX = np.zeros(pXY.shape)
    cpY = cpX.T
    return cpX, cpY


def dCond(Z):
    """ Associatiom measure based on conditional proba of increments and decrements Version 2"""
    X, Y = Z
    dX = (X[:, 1:] - X[:, :-1])
    dY = (Y[:, 1:] - Y[:, :-1])
    
    ### All important interaction types
    cpXY_in_in, cpYX_in_in = getCond(dX > 0, dY > 0)
    cpXY_out_out, cpYX_out_out = getCond(dX < 0, dY < 0)
    cpXY_in_out, cpYX_in_out = getCond(dX > 0, dY < 0)
    cpXY_out_in, cpYX_out_in = getCond(dX < 0, dY > 0)
    cpXY_No_No, cpYX_No_No = getCond(dX == 0, dY == 0)
    
    cpXY_in_No, cpYX_in_No = getCond(dX > 0, dY == 0)
    cpXY_out_No, cpYX_out_No = getCond(dX < 0, dY == 0)
    cpXY_No_in, cpYX_No_in = getCond(dX == 0, dY > 0)
    cpXY_No_out, cpYX_No_out = getCond(dX == 0, dY < 0)

    # sum the effects of each interactions because they are all important
    total_interaction = cpXY_in_in + cpXY_out_out + cpXY_in_out + cpXY_out_in + cpXY_No_No + cpXY_in_No + cpXY_out_No + cpXY_No_in + cpXY_No_out
    
    return np.exp(-total_interaction)


def EmpCDF(X, interval):
    cdf = np.sum(X[:, :, np.newaxis]<= interval, axis = 1)/X.shape[1]
    return cdf


def dCDF(Z):
    "Association distance based on Empirical CDF"
    X, Y = Z
    lbd = min(np.min(X), np.min(Y))
    ubd = max(np.max(X), np.max(Y))
    interval = np.linspace(lbd, ubd, 500)
    eX = EmpCDF(X, interval)
    eY = EmpCDF(Y, interval)
    
    D = 1e-5 + scSp.distance.cdist(eX, eY)
    return D    

def dcov(Z):
    """the empirical covariance formula bellow is only valid for equal number of observations in each samples it is equal to np.cov(X,Y)"""
    scov = (1/(Z[0].shape[1] - 1))*np.dot(Z[0] - np.mean(Z[0], axis = 1)[:, np.newaxis], (Z[1]- np.mean(Z[1], axis = 1)[:, np.newaxis]).T)
    res = np.exp(-np.abs(scov)) 
    return 1e-5 + res # added a small constant to avoid identically zero

        
def dcorr(Z):
    """the empirical corrcoeff formula bellow is only valid for equal number of observations in each samples it is equal to np.cov(X,Y)/std(Y)std(Y)"""
    scov = (1/(Z[0].shape[1] - 1))*np.dot(Z[0] - np.mean(Z[0], axis = 1)[:, np.newaxis], (Z[1]- np.mean(Z[1], axis = 1)[:, np.newaxis]).T)
    scorr = scov/(np.std(Z[0], axis = 1)[:, np.newaxis]*np.std(Z[1], axis = 1)[np.newaxis, :])
    res = np.exp(-np.abs(scorr))
    return 1e-5 + res  # added a small constant to avoid identically zero

def dcov_v2(Z):
    """the empirical covariance formula bellow is only valid for equal number of observations in each samples it is equal to np.cov(X,Y)"""
    scov = (1/(Z[0].shape[1] - 1))*np.dot(Z[0] - np.mean(Z[0], axis = 1)[:, np.newaxis], (Z[1]- np.mean(Z[1], axis = 1)[:, np.newaxis]).T)
    U1 = scov - np.var(Z[0], axis = 1)[:, np.newaxis] ### if cov = var then the variables might belong to the same axis so they are not associated in the sense that they are not correlated variables
    U2 = scov - np.var(Z[1], axis = 1)[np.newaxis, :]
    res = np.exp(-50*(U1)**2)*np.exp(-50*(U2)**2)
    
    return 1e-5 + res # added a small constant to avoid identically zero

        
def dcorr_v2(Z):
    """the empirical corrcoeff formula bellow is only valid for equal number of observations in each samples it is equal to np.cov(X,Y)/std(Y)std(Y)"""
    scov = (1/(Z[0].shape[1] - 1))*np.dot(Z[0] - np.mean(Z[0], axis = 1)[:, np.newaxis], (Z[1]- np.mean(Z[1], axis = 1)[:, np.newaxis]).T)
    scorr = scov/(np.std(Z[0], axis = 1)[:, np.newaxis]*np.std(Z[1], axis = 1)[np.newaxis, :])
    res = np.exp(- 50*(scorr - 1)**2) ### if corr = 1 then the variables are not correlated thus they they are not associated in the sense that they are not correlated variables
    return 1e-5 + res  # added a small constant to avoid identically zero

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