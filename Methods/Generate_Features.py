#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 13:39:28 2022

@author: raharinirina
"""
import numpy as np
import pdb

def EmpCDF(X, interval):
    cdf = np.sum(X[:, :, np.newaxis]<= interval, axis = 1)/X.shape[1]
    return cdf
    