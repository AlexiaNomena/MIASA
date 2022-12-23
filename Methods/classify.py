#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 17:18:37 2022

@author: raharinirina
"""
from .miasa_class import Miasa_Class
from .Generate_Features import KS_v1, KS_v2, Sub_Eucl
from .Core.Generate_Distances import Similarity_Distance, Association_Distance
from .Core.Lower_dim import get_clusters
from .Core.CosLM import Prox_Mat

import numpy as np
from sklearn.metrics import rand_score
import pdb
import sys

""" Classification general setting """
def one_classification(r, repeat, method_dic_list, var_data, generate_data):
    acc_res = np.zeros(len(method_dic_list))
    for i in range(len(method_dic_list)):    
        data_dic, class_dic, num_clust, dtp = generate_data(var_data)
        Id_Class, X_vars, Y_vars, acc_r = Classify_general(data_dic, class_dic, num_clust, method_dic = method_dic_list[i])
        print("method num %d/%d"%(i+1, len(method_dic_list)), "run %d/%d"%(r+1,repeat))
        acc_res[i] = acc_r
    return acc_res


import joblib as jb
from functools import partial
def repeated_classifications(repeat, method_dic_list, generate_data, var_data = False, n_jobs = 25):
    if repeat < 10:
        repeat = 10 + repeat
        
    acc_list = []
    ### plot and save the first 10 classification runs
    for r in range(10):
        sub_list = []
        for i in range(len(method_dic_list)):    
            data_dic, class_dic, num_clust, dtp = generate_data(var_data)
            Id_Class, X_vars, Y_vars, acc_r = Classify_general(data_dic, class_dic, num_clust, method_dic = method_dic_list[i])
            print("method num %d/%d"%(i+1, len(method_dic_list)), "run %d/%d"%(r+1,repeat))

            if method_dic_list[i]["class_method"] == "MIASA":
                pdf = method_dic_list[i]["fig"]
                plotClass(Id_Class, X_vars, Y_vars, pdf, dtp, r)
            sub_list.append(acc_r)
            
        acc_list.append(sub_list)
    
    if repeat > 0:
        pfunc = partial(one_classification, repeat = repeat, method_dic_list = method_dic_list, var_data = var_data, generate_data = generate_data)
        acc_list = acc_list + (jb.Parallel(n_jobs = n_jobs, prefer="threads")(jb.delayed(pfunc)(r) for r in range(10, repeat)))  
    
        """
        for r in range(repeat):
        acc_list.append(one_classification(r, method_dic_list))
        """
    
    acc_list = np.array(acc_list, dtype = float).T
    return acc_list



""" Identification of Classes """
def Classify_general(data_dic, class_dic, num_clust, method_dic):
    class_method = method_dic["class_method"]
    clust_method = method_dic["clust_method"]
    metric_method = method_dic["metric_method"]
    
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
    if class_method == "MIASA":
        Orows, Ocols = True, True
        Id_Class = Miasa_Class(X, Y, num_clust, dist_origin = Orows*Ocols, metric_method = metric_method, clust_method = clust_method)
    
    elif class_method == "non_MD":
        Orows, Ocols = True, True
        Id_Class = NonMetricDist_Class(X, Y, num_clust, dist_origin = Orows*Ocols, metric_method = metric_method, clust_method = clust_method)
    
    """Compute accuracy metric = rand_index metric"""
    Class_pred = Id_Class["Class_pred"]
    acc_metric = rand_score(Class_True, Class_pred)
    return Id_Class, X_vars, Y_vars, acc_metric
    

def NonMetricDist_Class(X, Y, num_clust, dist_origin = True, metric_method = "KS-statistic", clust_method = "Kmeans", palette = "tab20"):
    """Compute features"""
    if metric_method == "KS-statistic":
       Feature_X, Feature_Y, func, ftype = KS_v1(X,Y)
    elif metric_method == "KS-p_value":
        Feature_X, Feature_Y, func, ftype = KS_v2(X,Y)
    else:
       Feature_X, Feature_Y, func, ftype = Sub_Eucl(X, Y)

    Result = get_NMDclass(X, Y, Feature_X, Feature_Y, func, ftype, metric_method, dist_origin, num_clust, clust_method, palette)

    return Result

    
def get_NMDclass(X, Y, Feature_X, Feature_Y, func, ftype, metric_method, dist_origin = False, num_clust=None, clust_method = "Kmeans", palette = "tab20"):
    """ Similarity metric """
    DX = Similarity_Distance(Feature_X, method = "Euclidean")
    DY = Similarity_Distance(Feature_Y, method = "Euclidean")
    
    """Association metric"""
    if metric_method == "KS-statistic":
        Features = (X, Y)
    else:
        Features = (Feature_X, Feature_Y)
    
    D_assoc = Association_Distance(Features, func, ftype)
    """Distane to origin Optional but must be set to None if not used"""
    if dist_origin:
        Orows = np.linalg.norm(Feature_X, axis = 1)
        Ocols = np.linalg.norm(Feature_Y, axis = 1)
    else:
        Orows = None
        Ocols = None
    
    M = Feature_X.shape[0]
    N = Feature_Y.shape[0]
    
    DMat = Prox_Mat(DX, DY, UX = Orows, UY = Ocols, fXY = D_assoc)
    
    if clust_method == "Kmedoids":
        if num_clust == None:
            sys.exit("Kmedoids requires number of clusters parameter: num_clust")
        else:
            clust_labels, color_clustered = get_clusters(DMat, num_clust, palette, method = clust_method, metric = "precomputed")
    
    elif clust_method[:13] == "Agglomerative":
        clust_labels, color_clustered = get_clusters(DMat, num_clust, palette, method = clust_method, metric = "precomputed")
        
    else:
        sys.exit("A non-metric distance clustering method is required for Non Metric Distance \n Available here is Kmedoids")
    
    if dist_origin:
        Class_pred = np.concatenate((clust_labels[:M], clust_labels[M+1:]), axis = 0)
        was_orig = True
    else:
        Class_pred = clust_labels
        was_orig = False
        
    return {"shape":(M, N), "was_orig":was_orig, "Class_pred":Class_pred, "clust_labels":clust_labels, "color_clustered":color_clustered}
    


"""Visualization of classes"""
from .figure_settings import Display, PreFig
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
    
 
import matplotlib.pyplot as plt    
def BarPlotClass(data, method_name, pdf, stat_name = None):
    PreFig()
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)

    data_list = []
    for i in range(data.shape[0]):
        data_list.append(data[i, :])
        
    ax.boxplot(data_list, showfliers=False) # showfliers = False remove outliers
    plt.xticks(np.cumsum(np.ones(len(method_name))), method_name, rotation=75)
    xmin, xmax = ax.get_xlim()
    plt.ylabel(stat_name, fontsize = 20)
    pdf.savefig(fig, bbox_inches = "tight")
    
    