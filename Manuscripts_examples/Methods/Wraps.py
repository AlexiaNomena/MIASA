#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 11:47:33 2023

@author: raharinirina
"""
import numpy as np
import scipy as sp
from scipy.spatial import ConvexHull
import pdb

def find_ellipse_params(DX):
    # Find the ellipse that best fit the variation points, i.e.
    
    eDX = np.mean(DX, axis = 0)
    
    u, svals, vt = np.linalg.svd((DX - eDX).T)   
    sigm = svals**2
      
    # eigenvectors on the columns of u are the direction of the principal axis of the ellipse that best fit the points on the columns of DX
    u = np.real(u)
    # https://math.stackexchange.com/questions/1447730/drawing-ellipse-from-eigenvalue-eigenvector
    # the equation of an ellipse is, for x in \bT{R}^2, xT A x (quadratic form), this gives an ellipse of horizontal radius = 1/sqrt(lambda[0]) and vertical radius = 1/sqrt(lambda[1]) 
    # where lambda are the the eigenvalues of the matrix  A 
    # The equation of the ellipse that best fit the data is  xT B x = crit_val where B = CovMat.inv (inverse of the covariance), this gives an ellipse of horizontal radius = sqrt(crit_val*sigm[0]) and vertical radius = sqrt(crit_val*sigm[0])
    # where sigm are the eigenvalues of CovMat because CovMat_inv = u diag(1/lambda) u.T 
    
    
    A = np.dot((DX - eDX).T, (DX - eDX))
    try:
        SX = ((DX - eDX)).dot(sp.linalg.inv(A).dot((DX - eDX).T)) ### xT A x/A^{-1}
        crit_val = np.percentile(SX, 100)

    except:
        crit_val = 1#1/np.percentile(sp.linalg.norm(DX - eDX, axis = 1), 100)
    
    
    if crit_val<0:
        pdb.set_trace()
        
    width = 2*np.sqrt(crit_val*sigm[0])
    height = 2*np.sqrt(crit_val*sigm[1])

    # arctan formula
    angle = np.arctan2(u[0, 0], u[1, 0])
    angle = angle*180/np.pi # convert in degree
    
    return height, width, angle, eDX

def convex_hull(points):
    hull = ConvexHull(points)
    return hull