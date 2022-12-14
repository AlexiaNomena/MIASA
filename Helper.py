#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 09:32:55 2022

@author: araharin
"""
import numpy as np

from load_text_data import Coords_Rows, Coords_Cols, ContDataFrame, columns_labels, rows_labels, D, Dr, Dc, row_val, col_val, dtp, ColName, RowName

""" Compute similarity metrics """
from Core.Generate_Distances import Similarity_Metric, Association_Metric

D_row_var = Similarity_Metric(Coords_Rows, method = "Euclidean")
D_col_var = Similarity_Metric(Coords_Cols, method = "Euclidean")

func = lambda Coords: np.exp(-Dr.dot(D.dot(Dc))) ### Pearson ratio - 1  #### here independent of the coordinates
D_assoc = Association_Metric(Coords_Rows, Coords_Cols, func)

# Distance to origin
Orow = np.linalg.norm(Coords_Rows, axis = 1)
Ocols = np.linalg.norm(Coords_Cols, axis = 1)

"""joint Euclidean embedding of row and column variables"""
from Core.Embedding import Euclidean_Embedding
M = D_row_var.shape[0]
N = D_col_var.shape[0]

### Parameters needed for Euclidean Embedding (See Paper) ####
c1, c2 = 1/2, 1 # or c2 = 2 
a = 1. - 1./(M+N)
b = 2.*c2/(M+N)
c3 =  min(((2.*c1 + c2) - b)/a, 2*c2+2)
c = {"c1":c1, "c2":c2, "c3":c3} 
Coords, vareps = Euclidean_Embedding(D_row_var, D_col_var, Orow, Ocols, D_assoc) ### Orow, Ocols are optional, if not given, then distance to Origin is not interpretable

"""Coordinate system for regular projection on principal axes"""
if (Orow is not None)&(Ocols is not None):
    Emb_Rows = Coords[:M, :]
    Emb_Cols = Coords[M+1:, :]
    Origin = Coords[M, :] 
else:
    Emb_Rows = Coords[:M, :]
    Emb_Cols = Coords[M:, :]
    Origin = np.zeros(Emb_Rows.shape[1])


"""Clustering of Similarity and Association patterns"""
from Core.Lower_dim import get_clusters
palette = "tab20" ## seaborn color palette
num_clust = 6
clust_labels, colour_labels = get_clusters(Coords, num_clust, palette, method = "Kmeans")

"""Lower Dimensional visualization of clusters"""
from Core.Lower_dim import low_dim_coords
nb = 15 ### only for UMAP method for Egyptian texts data 
low_meth = "umap" # or sklearn.manifols methods: MDS, Isomap, 
Coords_manifold = low_dim_coords(Coords, dim=2, method  = low_meth, n_neighbors = nb)

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
from figure_setting import Display
Inertia = None # not relevant here
AllCols = ContDataFrame.columns
AllRows = ContDataFrame.index
fig, xy_rows, xy_cols, gs, center = Display(Rows_manifold, 
                                             Cols_manifold, 
                                             Inertia, 
                                             ContDataFrame,
                                             center = Origin_manifold, 
                                             rows_to_Annot = AllRows,  # row items to annotate, if None then no annotation (None if none)
                                             cols_to_Annot = AllCols,  # column items to annotate (None if none)
                                             Label_rows = rows_labels, # dictionary of labels respectivelly corresponding to the row items (None if none)
                                             Label_cols = columns_labels,     # dictionary of labels respectivelly corresponding to the column items that (None if none)
                                             markers =[(".",20),("+",50)],# pyplot markertypes, markersize: [(marker for the row items, size), (marker for the columb items, size)] 
                                             col = ["grey","red"],        # pyplot colortypes : [color for the row items, color for the column items]          
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
pdf= PdfPages("Figures/%s_%s"(col_val,row_val)+".pdf")
pdf.savefig(fig, bbox_inches = "tight")
plt.show()

