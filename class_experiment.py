#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 13:21:27 2022

@author: raharinirina
"""
from Methods.classify import Class_Identification
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


"""Generate Artificial data"""
per_spl = 100 # # Num iid per samplea
num_spl = 10 # Num Samples

data_dic = {}

class1 = ["1a", "1b", "1c"]
val1 = [(0, 1), (0, 5), (2, 1)]

class2 = ["2a", "2b", "2c"]
val2 = [(0, 1), (0, 5), (2, 3)]

class3 = ["3a", "3b", "3c"]
val3 = [1, 2, 3]

class4 = ["4a", "4b", "4c"]
val4 = [1, 3, 6]

for i in range(3):
    for j in range(num_spl):
        data_dic[class1[i]+"%d"%(j+1)] = np.random.normal(val1[i][0], val1[i][1], size = per_spl)
        data_dic[class2[i]+"%d"%(j+1)] = np.random.uniform(val2[i][0], val2[i][1], size = per_spl)
        data_dic[class3[i]+"%d"%(j+1)] = np.random.pareto(val3[i], size = per_spl)
        data_dic[class4[i]+"%d"%(j+1)] = np.random.poisson(val4[i], size = per_spl)

"""Split data in two random groups of the same size"""
samples = np.random.shuffle(data_dic.keys())

X_vars = samples[:len(samples)//2]
Y_vars = samples[len(X_vars):]
M = len(X_vars)
N = len(Y_vars)

X = np.array([data_dic[X_vars[i]] for i in range(M)])
Y = np.array([data_dic[Y_vars[i]] for i in range(N)])


""" Identify Class using MIASA framework """
Orow, Ocols = True, True
Id_Class = Class_Identification(X, Y, dist_orig = Orow*Ocols)
Coords = Id_Class["Coords"]

"""Lower Dimensional visualization of clusters"""
from Methods.Core.Lower_dim import low_dim_coords
nb = 15 ### only for UMAP method for Egyptian texts data 
low_meth = "umap" # or sklearn.manifols methods: MDS, Isomap, 
Coords_manifold = low_dim_coords(Coords, dim=2, method  = low_meth, n_neighbors = nb) 
"""
Kmeans and UMAP are already parameterized for reproducibility (random_state = 0 for both).
However, slight changes could still happen due to the optimization procedure and versions of these packages.
"""

"""Coordinate system for regular projection on principal axes"""
if (Orow is not None)&(Ocols is not None):
    Rows_manifold = Coords_manifold[:M, :]
    Cols_manifold = Coords_manifold[M+1:, :]
    Origin_manifold = Coords_manifold[M, :] 
else:
    Emb_Rows_manifold = Coords_manifold[:M, :]
    Emb_Cols_manifold = Coords_manifold[M:, :]
    Origin_manifold = np.zeros(Emb_Rows.shape[1])

"""Plot and Save figure"""
from Methods.figure_settings import Display
Inertia = np.array([0, 1]) # not relevant for manifold
AllCols = ContDataFrame.columns
AllRows = ContDataFrame.index
col_rows = {rows_labels[ContDataFrame.index[i]]:color_clustered[i] for i in range(M)}
col_cols = {columns_labels[ContDataFrame.columns[i]]:color_clustered[i+M+1] for i in range(N)}
col_to_use = (col_rows, col_cols)
marker_to_use = [("o",20),("o",20)]
fig, xy_rows, xy_cols, gs, center = Display(Rows_manifold, 
                                             Cols_manifold, 
                                             Inertia, 
                                             ContDataFrame,
                                             center = Origin_manifold, 
                                             rows_to_Annot = AllRows,  # row items to annotate, if None then no annotation (None if none)
                                             cols_to_Annot = AllCols,  # column items to annotate (None if none)
                                             Label_rows = rows_labels, # dictionary of labels respectivelly corresponding to the row items (None if none)
                                             Label_cols = columns_labels,     # dictionary of labels respectivelly corresponding to the column items that (None if none)
                                             markers = marker_to_use,# pyplot markertypes, markersize: [(marker for the row items, size), (marker for the columb items, size)] 
                                             col = col_to_use,        # pyplot colortypes : [color for the row items, color for the column items] 
                                             figtitle = "Sklearn manifold embedding method = %s"%low_meth, 
                                             outliers = (True, True),
                                             dtp = dtp,
                                             chosenAxes = np.array([0,1]), 
                                             show_inertia = False, 
                                             model={"model":"stand"}, 
                                             ColName = ColName, 
                                             RowName = RowName,
                                             lims = False) # crop fig

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
pdf= PdfPages("Figures/fig%s_%s"%(col_val,row_val)+".pdf")
pdf.savefig(fig, bbox_inches = "tight")
pdf.close()
plt.show()





    
    