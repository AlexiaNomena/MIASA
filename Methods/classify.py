#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 17:18:37 2022

@author: raharinirina
"""
from .miasa_class import Miasa_Class


import numpy as np
from sklearn.metrics import rand_score


def Classify_general(data_dic, class_dic, num_clust):
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
    
    """Compute accuracy metric = rand_index metric"""
    Class_pred = Id_Class["Class_pred"]
    acc_metric = rand_score(Class_True, Class_pred)
    return Id_Class, X_vars, Y_vars, acc_metric
    


from .figure_settings import Display
from .Core.Lower_dim import low_dim_coords
import pandas as pd
def plotClass(Id_Class, X_vars, Y_vars, pdf, dtp, run_num):   
    """@brief Plot and Save class figures"""
    
    Coords = Id_Class["Coords"]
    """Lower Dimensional visualization of clusters"""
    nb = 2 ### 
    low_meth = "umap" # or sklearn.manifols methods: MDS, Isomap, 
    md = 0.99
    Coords_manifold = low_dim_coords(Coords, dim=2, method  = low_meth, n_neighbors = nb, min_dist = md) 
    """
    Kmeans and UMAP are already parameterized for reproducibility (random_state = 0 for both).
    However, slight changes could still happen due to the optimization procedure and versions of these packages.
    """
    
    """Coordinate system for regular projection on principal axes"""
    was_orig = Id_Class["was_orig"]
    M, N = Id_Class["shape"]
    if was_orig:
        Rows_manifold = Coords_manifold[:M, :]
        Cols_manifold = Coords_manifold[M+1:, :]
        Origin_manifold = Coords_manifold[M, :] 
    else:
        Rows_manifold = Coords_manifold[:M, :]
        Cols_manifold = Coords_manifold[M:, :]
        Origin_manifold = np.zeros(Coords_manifold.shape[1])
    
    Inertia = np.array([0, 1]) # not relevant for manifold
    
    ### Dummy dataframe
    DataFrame = pd.DataFrame({Y_vars[i]:np.zeros(M) for i in range(N)}, index = X_vars)
    rows_labels = {X_vars[i]:X_vars[i] for i in range(M)}
    columns_labels = {Y_vars[i]:Y_vars[i] for i in range(N)}
    
    AllCols = DataFrame.columns
    AllRows = DataFrame.index
    
    color_clustered = Id_Class["color_clustered"]
    ColName = None
    RowName = None
    
    col_rows = {rows_labels[DataFrame.index[i]]:color_clustered[i] for i in range(M)}
    col_cols = {columns_labels[DataFrame.columns[i]]:color_clustered[i+M+1] for i in range(N)}
    col_to_use = (col_rows, col_cols)
    marker_to_use = [("o",20),("o",20)]
    fig, xy_rows, xy_cols, gs, center = Display(Rows_manifold, 
                                                 Cols_manifold, 
                                                 Inertia, 
                                                 DataFrame,
                                                 center = Origin_manifold, 
                                                 rows_to_Annot = AllRows,  # row items to annotate, if None then no annotation (None if none)
                                                 cols_to_Annot = AllCols,  # column items to annotate (None if none)
                                                 Label_rows = rows_labels, # dictionary of labels respectivelly corresponding to the row items (None if none)
                                                 Label_cols = columns_labels,     # dictionary of labels respectivelly corresponding to the column items that (None if none)
                                                 markers = marker_to_use,# pyplot markertypes, markersize: [(marker for the row items, size), (marker for the columb items, size)] 
                                                 col = col_to_use,        # pyplot colortypes : [color for the row items, color for the column items] 
                                                 figtitle = "method = %s (%d)"%(low_meth, run_num), 
                                                 outliers = (True, True),
                                                 dtp = dtp, 
                                                 chosenAxes = np.array([0,1]), 
                                                 show_inertia = False, 
                                                 model={"model":"stand"}, 
                                                 ColName = ColName, 
                                                 RowName = RowName,
                                                 lims = False) # crop fig
    
    pdf.savefig(fig, bbox_inches = "tight")