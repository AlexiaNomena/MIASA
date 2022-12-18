#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 13:39:28 2022

@author: raharinirina
"""
import numpy as np

def EmpCDF(X, interval):
    cdf = np.cumsum(X<=interval, axis = 1)/X.shape[0]
    return cdf
    