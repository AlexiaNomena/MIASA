#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 17:18:37 2022

@author: raharinirina
"""

def Classify_general(data_dic, class_dic, run_num, num_clust, pdf, plotfig = True):
    """Split data in two random groups of the same size"""
    samples = np.array(list(data_dic.keys()))
    
    np.random.shuffle(samples)
    X_vars = samples[:len(samples)//2]
    Y_vars = samples[len(X_vars):]
    M = len(X_vars)
    N = len(Y_vars)
    
    X = np.array([data_dic[X_vars[i]] for i in range(M)])
    Y = np.array([data_dic[Y_vars[i]] for i in range(N)])
   
    """ True Classes """
    Class_True = np.array([class_dic[samples[i]] for i in range(M+N)])
    
    """ Identify Class using MIASA framework """
    Orow, Ocols = True, True
    Id_Class = Miasa_Class(X, Y, num_clust, dist_origin = Orow*Ocols)
    Coords = Id_Class["Coords"]
    
    """Compute accuracy metric = rand_index metric"""
    Class_pred = Id_Class["Class_pred"]
    acc_metric = rand_score(Class_True, Class_pred)
    return Id_Class
    