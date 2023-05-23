#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 17:18:37 2022

@author: raharinirina
"""
from .miasa_class import Miasa_Class
from .NonMD_class import NonMetric_Class

import numpy as np
from numpy.ma import masked_array
from sklearn.metrics import rand_score, adjusted_rand_score
import pdb
import sys
from copy import copy
import scipy.spatial as spsp

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon

from .Wraps import find_ellipse_params, convex_hull

""" Classification general setting """
def one_classification(r, repeat, method_dic_list, var_data, generate_data, c_dic = "default", in_threads = True, separation = False):
    data_dic, class_dic, num_clust, dtp = generate_data(var_data) # use the same dataset for all methods
    X, Y, Class_True, X_vars, Y_vars = split_data(data_dic, class_dic, separation)
    data_dic2 = {"X":X, "Y":Y, "Class_True":Class_True, "X_vars":X_vars, "Y_vars":Y_vars}
    acc_res = []
    DMat = None
    for i in range(len(method_dic_list)):
        Id_Class, X_vars, Y_vars, acc_r = Classify_general(data_dic2, class_dic, num_clust, method_dic = method_dic_list[i], DMat = DMat, c_dic = c_dic, in_threads = in_threads)
        print("Case %d -- method num %d/%d"%(var_data*1, i+1, len(method_dic_list)), "-- run %d/%d"%(r+1,repeat))
        
        # since this is the same dataset, if the metric_method is the same, then pass DMat directly to avoid recomputing it everytime for method with MIASA and Non_MD
        if (i>0)&(Id_Class is not None):
            if method_dic_list[i]["metric_method"] != method_dic_list[i-1]["metric_method"]:
                DMat = None
            else:
                DMat = Id_Class["DMat"]
        
        acc_res.append(acc_r)
    return acc_res


def split_data(data_dic, class_dic, separation = False):
    if separation:
        """Extract separated data"""
        X_vars = data_dic["X_vars"]
        Y_vars = data_dic["Y_vars"]
    else:
        """Split data in two random groups of the same size"""
        samples = np.array(list(data_dic.keys()))
        samples = samples[~(samples == "true_colors")]
        np.random.shuffle(samples)
        X_vars = samples[:len(samples)//2]
        Y_vars = samples[len(X_vars):]
        
    M = len(X_vars)
    N = len(Y_vars)
    X = np.array([data_dic[X_vars[i]] for i in range(M)])
    Y = np.array([data_dic[Y_vars[i]] for i in range(N)])
    """ True Classes """
    Vars = list(X_vars) + list(Y_vars)
    Class_True = np.array([class_dic[Vars[i]] for i in range(len(Vars))])
    return X, Y, Class_True, X_vars, Y_vars

import joblib as jb
from functools import partial
def repeated_classifications(repeat, method_dic_list, generate_data, var_data = False, n_jobs = -1, plot = True, c_dic = "default", in_threads = True, separation = False): 
    repeat = 10 + repeat
        
    acc_list = []
    ### plot and save the first 10 classification runs
    if plot:
        start = 10
        for r in range(10):
            sub_list = []
            data_dic, class_dic, num_clust, dtp = generate_data(var_data)
            X, Y, Class_True, X_vars, Y_vars = split_data(data_dic, class_dic, separation)
            data_dic2 = {"X":X, "Y":Y, "Class_True":Class_True, "X_vars":X_vars, "Y_vars":Y_vars}
            for i in range(len(method_dic_list)):    
                Id_Class, X_vars, Y_vars, acc_r = Classify_general(data_dic2, class_dic, num_clust, method_dic = method_dic_list[i], DMat = None, c_dic = c_dic, in_threads = in_threads)
                print("method num %d/%d"%(i+1, len(method_dic_list)), "run %d/%d"%(r+1,repeat))
    
                if method_dic_list[i]["class_method"] == "MIASA":
                    pdf = method_dic_list[i]["fig"]
                    plotClass(Id_Class, X_vars, Y_vars, pdf, dtp, run_num = r, n_neighbors = 15, method = "UMAP")
                    plotClass(Id_Class, X_vars, Y_vars, pdf, dtp, run_num = r, n_neighbors = 30, method = "t-SNE")
                sub_list.append(acc_r)
                
            acc_list.append(sub_list)
    else:
        start = 0
    
    if repeat-10 > 10:
        repeat = repeat - 10
        # n_jobs = -1 means use all CPUs
        pfunc = partial(one_classification, repeat = repeat, method_dic_list = method_dic_list, var_data = var_data, generate_data = generate_data, c_dic = c_dic, in_threads = in_threads, separation = separation)
        acc_list = acc_list + list(jb.Parallel(n_jobs = n_jobs, prefer = "threads")(jb.delayed(pfunc)(r) for r in range(start, repeat)))
    
    else:
        pfunc = partial(one_classification, repeat = repeat-10, method_dic_list = method_dic_list, var_data = var_data, generate_data = generate_data, c_dic = c_dic, in_threads = in_threads, separation = separation)
        for r in range(repeat-10):
            acc_list.append(pfunc(r))
    
    Acc = np.array(acc_list)
    all_acc_list = masked_array(Acc, mask = Acc == None)
    acc_list = all_acc_list[:, :, 0].T
    adjusted_acc_list = all_acc_list[:, :, 1].T
    return acc_list.astype(float), adjusted_acc_list.astype(float)


""" Identification of Classes """
def Classify_general(data_dic, class_dic, num_clust, method_dic, DMat = None, c_dic = "default", in_threads = True, Feature_dic = None):
    class_method = method_dic["class_method"]
    clust_method = method_dic["clust_method"]
    metric_method = method_dic["metric_method"]
    
    X, Y, Class_True, X_vars, Y_vars = data_dic["X"], data_dic["Y"], data_dic["Class_True"], data_dic["X_vars"], data_dic["Y_vars"] 
    
    """ Identify Class using MIASA framework """
    if class_method == "MIASA":
        Orows, Ocols = True, True
        Id_Class = Miasa_Class(X, Y, num_clust, DMat = DMat, dist_origin = (Orows,Ocols), metric_method = metric_method, clust_method = clust_method, c_dic = c_dic, Feature_dic = Feature_dic, in_threads = in_threads)
        
    elif class_method == "non_MD":
        Orows, Ocols = True, True
        Id_Class = NonMetric_Class(X, Y, num_clust, DMat = DMat, dist_origin = (Orows,Ocols), metric_method = metric_method, clust_method = clust_method, Feature_dic = Feature_dic, in_threads = in_threads)
        
    """Compute accuracy metric = rand_index metric"""
    if Id_Class is not None:
        Class_pred = Id_Class["Class_pred"]
        acc_metric = rand_score(Class_True, Class_pred),  adjusted_rand_score(Class_True, Class_pred)
    else:
        acc_metric = None, None
    return Id_Class, X_vars, Y_vars, acc_metric
    


"""Visualization of classes"""
from .figure_settings import Display, PreFig
from .Core.Lower_dim import low_dim_coords
import pandas as pd

def plotClass(Id_Class, X_vars, Y_vars, pdf, dtp, run_num = 0, n_neighbors = 2, min_dist = 0.99, method = "umap", 
                        scale = None, sub_fig_size = 7, cluster_colors = False, true_colors = None, markers = [("o",20),("o",20)],
                        show_labels = False, show_orig = True, metric = "euclidean", legend = True, place_holder = (0, 0), 
                        wrap_true = False, wrap_predicted = False, oultiers_markers = ("o", "^", 5),  wrap_type = "convexhull",
                        def_pred_outliers = (2, 0.95),show_pred_outliers = False):        
    """@brief Plot and Save class figures"""
    
    """Lower Dimensional visualization of clusters"""
    low_meth = method # methods: MDS, Isomap, TSNE
    if metric == "precomputed":
        DMat = Id_Class["DMat"]
        Coords_manifold = low_dim_coords(DMat, dim = 2, method = low_meth, scale = scale)
    else:
        Coords = Id_Class["Coords"]
        Coords_manifold = low_dim_coords(Coords, dim=2, method  = low_meth, n_neighbors = n_neighbors, min_dist = min_dist, scale = scale) 
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
    
    if not show_orig:
        Origin_manifold = None
        
    Inertia = np.array([0, 1]) # not relevant for manifold
    
    ### Dummy dataframe
    DataFrame = pd.DataFrame({Y_vars[i]:np.zeros(M) for i in range(N)}, index = X_vars)
    
    rows_labels = {X_vars[i]:X_vars[i][:2] for i in range(M)}
    columns_labels = {Y_vars[i]:Y_vars[i][:2] for i in range(N)}
   
    
    if show_labels:
        rows_to_Annot = np.array(DataFrame.index)
        cols_to_Annot = np.array(DataFrame.columns)
    else:
        rows_to_Annot = None
        cols_to_Annot = None
    
    color_clustered = Id_Class["color_clustered"]
    ColName = None
    RowName = None
    #pdb.set_trace()
    
    if cluster_colors:
        col_rows = {rows_labels[X_vars[i]]:color_clustered[i] for i in range(M)}
        col_cols = {columns_labels[Y_vars[i]]:color_clustered[-N+i] for i in range(N)}
    else:
        col_rows = {rows_labels[X_vars[i]]:true_colors[X_vars[i]] for i in range(M)}
        col_cols = {columns_labels[Y_vars[i]]:true_colors[Y_vars[i]] for i in range(N)}
    
    col_to_use = (col_rows, col_cols)
    marker_to_use = markers #[("o",20),("o",20)]
    
    fig, ax, xy_rows, xy_cols, gs, center = Display(Rows_manifold, 
                                                     Cols_manifold, 
                                                     Inertia, 
                                                     DataFrame,
                                                     center = Origin_manifold, 
                                                     rows_to_Annot = rows_to_Annot,  # row items to annotate, if None then no annotation (None if none)
                                                     cols_to_Annot = cols_to_Annot,  # column items to annotate (None if none)
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
                                                     lims = False,
                                                     give_ax = True) # crop fig
        
        
         
    if legend & (not cluster_colors) & (not wrap_true):
         col_done = []
         for i in range(len(X_vars)):
             if true_colors[X_vars[i]] not in col_done:
                 ax.scatter(np.zeros(1), np.zeros(1), marker = marker_to_use[0][0], s =  marker_to_use[0][1], color = true_colors[X_vars[i]], label = rows_labels[X_vars[i]][:2])
                 col_done.append(true_colors[X_vars[i]])
         
         ax.annotate("place_holder", xy=(0,0), 
                  xytext= (5, 5), textcoords='offset points', ha='center', va='bottom',
                  bbox=dict(boxstyle='circle', fc = "white"),
                  arrowprops=dict(arrowstyle='->', color = "black"),  #connectionstyle='arc3,rad=0.5'),
                  color= "black",
                  fontsize = 6
                   )
         
         plt.legend()
        
    
         
    if wrap_true:
        pred_class = np.unique(Id_Class["Class_pred"])
        lab_point = False
        for i in range(len(pred_class)):
            
            class_row = Id_Class["Class_pred"][:M] == pred_class[i]
            class_col =  Id_Class["Class_pred"][M:] == pred_class[i]

            coords_row = Rows_manifold[class_row, :]
            coords_col = Cols_manifold[class_col, :]
        
            X_var_sub = [X_vars[class_row][i] for i in range(coords_row.shape[0])] # the first two letters are always the true class labels
            Y_var_sub = [Y_vars[class_col][i] for i in range(coords_col.shape[0])]
            
            X_var_sub2 = [X_vars[class_row][i][:2] for i in range(coords_row.shape[0])] # the first two letters are always the true class labels
            Y_var_sub2 = [Y_vars[class_col][i][:2] for i in range(coords_col.shape[0])]
            
            class_labs = np.unique(X_var_sub2 + Y_var_sub2)
            
            X_var_sub = np.array(X_var_sub)
            Y_var_sub = np.array(Y_var_sub)
            
            X_var_sub2 = np.array(X_var_sub2)
            Y_var_sub2 = np.array(Y_var_sub2)
            
            done = []
            done2 = []
            for cl in class_labs:
                points = np.row_stack((coords_row[X_var_sub2 == cl, :], coords_col[Y_var_sub2 == cl, :]))
                cl_var = list(X_var_sub[X_var_sub2 == cl]) + list(Y_var_sub[Y_var_sub2 == cl])

                if wrap_type == "ellipse":
                    if points.shape[0] >= 3:
                        
                        height, width, angle, center = find_ellipse_params(points)
                        ellipse = Ellipse(xy = (center[0], center[1]), width = width, height = height, angle = angle, fill = True, edgecolor = true_colors[cl_var[0]], alpha = 0.25, lw = 2)
                        ellcopy = copy(ellipse)
                        ax.add_patch(ellcopy)
                    
                    else:
                        if not lab_point:
                            plt.plot(points[:, 0], points[:, 1], marker = "o", markersize =  marker_to_use[0][1], color = "grey", fillstyle = "full", linestyle = "")
                            lab_point = True
                        else:
                            plt.plot(points[:, 0], points[:, 1], marker = "o", markersize =  marker_to_use[0][1], color = true_colors[cl_var[0]], fillstyle = "none", linestyle = "")

                elif wrap_type == "convexhull":
               
                    cl_var = list(X_var_sub[X_var_sub2 == cl]) + list(Y_var_sub[Y_var_sub2 == cl])
                    if points.shape[0] >= 3:
                        hull = convex_hull(points)
                        #plt.plot(points[hull.vertices, 0], points[hull.vertices,1], "-", linewidth = 1, color = true_colors[cl_var[0]])
                        #plt.plot([points[hull.vertices, :][0, 0], points[hull.vertices, :][-1, 0]], [points[hull.vertices, :][0, 1], points[hull.vertices, :][-1, 1]], "-", linewidth = 1, color = true_colors[cl_var[0]])
                        
                        Vertices = points[hull.vertices, :]
                        if cl_var[0][:2] not in done:
                            try:
                                Poly = Polygon(Vertices, edgecolor = true_colors[cl_var[0]], facecolor = true_colors[cl_var[0]], fill = True, label = rows_labels[cl_var[0]][:2], alpha = 0.25)
                            except:
                                Poly = Polygon(Vertices, edgecolor = true_colors[cl_var[0]], facecolor = true_colors[cl_var[0]], fill = True, label = columns_labels[cl_var[0]][:2], alpha = 0.25)
                        else:
                            Poly = Polygon(Vertices, edgecolor = true_colors[cl_var[0]], facecolor = true_colors[cl_var[0]], fill = True)
                        
                        ax.add_patch(copy(Poly))
                        
                        done.append(cl_var[0][:2])
                        
                        

                    else:
                        if cl_var[0][:2] not in done2:
                            """
                            try:
                                plt.plot(points[:, 0], points[:, 1], marker = "o", markersize =  marker_to_use[0][1], color = true_colors[cl_var[0]], fillstyle = "full",  label = rows_labels[cl_var[0]][:2], linestyle = "")
                            except:
                                if not lab_point:
                                    plt.plot(points[:, 0], points[:, 1], marker = "o" , markersize =  marker_to_use[1][1], color = true_colors[cl_var[0]], fillstyle = "full", label = columns_labels[cl_var[0]][:2], linestyle = "")
                                    lab_point = True
                            """
                            if not lab_point:
                                plt.plot(points[:, 0], points[:, 1], marker = oultiers_markers[0], markersize =  oultiers_markers[2], color = "grey", fillstyle = "full",  label = "Outliers", linestyle = "")
                                plt.plot(points[:, 0], points[:, 1], marker = oultiers_markers[0], markersize =  oultiers_markers[2], color = true_colors[cl_var[0]], fillstyle = "full", linestyle = "")

                                lab_point = True
                            else:
                                plt.plot(points[:, 0], points[:, 1], marker = oultiers_markers[0], markersize =  oultiers_markers[2], color = true_colors[cl_var[0]], fillstyle = "full", linestyle = "")

                        else:
                            plt.plot(points[:, 0], points[:, 1], marker = oultiers_markers[0], markersize =  oultiers_markers[2], color = true_colors[cl_var[0]], fillstyle = "full", linestyle = "")
                        
                        done2.append(cl_var[0][:2])
    
    
    if wrap_predicted:
        pred_class = np.unique(Id_Class["Class_pred"])
        lab_point = False

        for i in range(len(pred_class)):
            class_row = Id_Class["Class_pred"][:M] == pred_class[i]
            class_col =  Id_Class["Class_pred"][M:] == pred_class[i]
    
            coords_row = Rows_manifold[class_row, :]
            coords_col = Cols_manifold[class_col, :]
            points = np.row_stack((coords_row, coords_col))
            
            dp = spsp.distance.pdist(points)
            dp = spsp.distance.squareform(dp)
            
            a, b = def_pred_outliers
            limit = np.std(dp.flatten())
            
            remove = np.sum(dp > a*limit, axis = 1) > (b)*points.shape[0]
            outliers = points[remove, :]
            points = points[~remove,  :]
            
            
            col_class = color_clustered[Id_Class["Class_pred"] == pred_class[i]][0]
            if wrap_type == "ellipse":
                if points.shape[0] >= 3:
                    
                    height, width, angle, center = find_ellipse_params(points)
                    if i == 0:
                        ellipse = Ellipse(xy = (center[0], center[1]), width = width, height = height, angle = angle, linestyle = "-", fill = False, edgecolor = "grey", lw = 1, label = "predicted")
                        ellipse = Ellipse(xy = (center[0], center[1]), width = width, height = height, angle = angle, linestyle = "-", fill = False, edgecolor = col_class, lw = 1)
    
                    else:
                        ellipse = Ellipse(xy = (center[0], center[1]), width = width, height = height, angle = angle, linestyle = "-", fill = False, edgecolor = col_class, lw = 1)
    
                    ellcopy = copy(ellipse)
                    ax.add_patch(ellcopy)
                else:
                    if not lab_point:
                        plt.plot(points[:, 0], points[:, 1], marker = oultiers_markers[1], markersize =  marker_to_use[0][1], color = "grey", fillstyle = "full", linestyle = "", label = "predicted (outliers)")
                    else:
                        plt.plot(points[:, 0], points[:, 1], marker = oultiers_markers[1], markersize =  marker_to_use[0][1], color = col_class, fillstyle = "full", linestyle = "", label = "predicted (outliers)")

            elif wrap_type == "convexhull":
                if points.shape[0] >= 3:
                    hull = convex_hull(points)
                    Vertices = points[hull.vertices, :]
                    if i == 0:
                        #Poly = Polygon(Vertices, edgecolor = "grey", fill = False, label = "predicted", linestyle = "-", linewidth = 1)
                        Poly = Polygon(Vertices, edgecolor = col_class, fill = False, label = "predicted %s"%(i+1), linestyle = "-", linewidth = 1)

                    else:
                        #Poly = Polygon(Vertices, edgecolor = col_class, fill = False, linestyle = "-", linewidth = 1)
                        Poly = Polygon(Vertices, edgecolor = col_class, fill = False, label = "predicted %s"%(i+1), linestyle = "-", linewidth = 1)

                    ax.add_patch(copy(Poly))
                
                else:
                    if show_pred_outliers:
                        if not lab_point:
                            plt.plot(points[:, 0], points[:, 1], marker = oultiers_markers[1], markersize =  oultiers_markers[2], color = "grey", fillstyle = "full", linestyle = "", label = "predicted (outliers)")
                            lab_point = True
                        else:
                            plt.plot(points[:, 0], points[:, 1], marker = oultiers_markers[1], markersize =  oultiers_markers[2], color = col_class, fillstyle = "full", linestyle = "")
                    
            if show_pred_outliers:
                if np.sum(remove) > 1:   
                    if not lab_point:
                        plt.plot(outliers[:, 0], outliers[:, 1], marker = oultiers_markers[1], markersize =  oultiers_markers[2], color = "grey", fillstyle = "full", linestyle = "", label = "predicted (outliers)")
                        lab_point = True
    
                    else:
                        plt.plot(outliers[:, 0], outliers[:, 1], marker = oultiers_markers[1], markersize =  oultiers_markers[2], color = col_class, fillstyle = "full", linestyle = "")
                
    if legend & (not cluster_colors):
        plt.legend(ncol = 3)
                
    """
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)      
    """
    pdf.savefig(fig, bbox_inches = "tight")
    
         

def plotClass_separated(Id_Class, X_vars, Y_vars, pdf, dtp, run_num = 0, n_neighbors = 2, min_dist = 0.99, method = "umap", 
                        scale = None, sub_fig_size = 7, num_row_col = None, cluster_colors = False, true_colors = None, markers = [("o",10),("o",10)], 
                        show_labels = False, show_orig = False, show_separation = False, legend = True, shorten_annots = True):   
    """@brief Plot and Save class figures"""
    
    Coords = Id_Class["Coords"]
    """Lower Dimensional visualization of clusters"""
    low_meth = method # methods: MDS, Isomap, TSNE
    Coords_manifold = low_dim_coords(Coords, dim=2, method  = low_meth, n_neighbors = n_neighbors, min_dist = min_dist, scale = scale) 
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
    
    if not show_orig:
        Origin_manifold = None
        
    Inertia = np.array([0, 1]) # not relevant for manifold
    
    ### Dummy dataframe
    DataFrame = pd.DataFrame({Y_vars[i]:np.zeros(M) for i in range(N)}, index = X_vars)
    if shorten_annots:
        rows_labels = {X_vars[i]:X_vars[i][:2] for i in range(M)}
        columns_labels = {Y_vars[i]:Y_vars[i][:2] for i in range(N)}
    else:
        rows_labels = {X_vars[i]:X_vars[i] for i in range(M)}
        columns_labels = {Y_vars[i]:Y_vars[i] for i in range(N)}
    
    if show_labels:
        AllRows = np.array(DataFrame.index)
        AllCols = np.array(DataFrame.columns)
    else:
        rows_to_Annot = None
        cols_to_Annot = None
       
    color_clustered = Id_Class["color_clustered"]
    ColName = None
    RowName = None
    #pdb.set_trace()
    if cluster_colors:
        col_rows = {rows_labels[X_vars[i]]:color_clustered[i] for i in range(M)}
        col_cols = {columns_labels[Y_vars[i]]:color_clustered[-N+i] for i in range(N)}
    else:
        col_rows = {rows_labels[X_vars[i]]:true_colors[X_vars[i]] for i in range(M)}
        col_cols = {columns_labels[Y_vars[i]]:true_colors[Y_vars[i]] for i in range(N)}
    
    col_to_use = (col_rows, col_cols)
    marker_to_use = markers
    unique_classe = np.unique(Id_Class["Class_pred"])
    
    if len(unique_classe)%2 == 0:
        F = int(len(unique_classe)//2)
    else:
        F = int(len(unique_classe)//2) + 1
    
    if num_row_col is None:
        fig = plt.figure(figsize = (sub_fig_size*F, sub_fig_size*F))
    else:
        fig = plt.figure(figsize = (sub_fig_size*num_row_col[1], sub_fig_size*num_row_col[0]))
    
    for i in range(len(unique_classe)):
        if num_row_col is None:
            ax = fig.add_subplot(F, F, i+1)
        else:
            ax = fig.add_subplot(num_row_col[0], num_row_col[1], i+1)
            
        class_row = Id_Class["Class_pred"][:M] == unique_classe[i]
        class_col =  Id_Class["Class_pred"][M:] == unique_classe[i]
        
        coords_row = Rows_manifold[class_row, :]
        coords_col = Cols_manifold[class_col, :]
        
        if show_labels:
            rows_to_Annot = AllRows[class_row]
            cols_to_Annot = AllCols[class_col]
        
        Data = DataFrame.copy()
        Data.drop(list(Data.columns[~class_col]), axis = 1, inplace = True)
        Data.drop(list(Data.index[~class_row]), axis = 0, inplace = True)
        fig, xy_rows, xy_cols, gs, center = Display(coords_row, 
                                                     coords_col, 
                                                     Inertia, 
                                                     Data,
                                                     center = Origin_manifold, 
                                                     rows_to_Annot = rows_to_Annot,#AllRows[class_row],  # row items to annotate, if None then no annotation (None if none)
                                                     cols_to_Annot = cols_to_Annot,#AllCols[class_col],  # column items to annotate (None if none)
                                                     fig = fig,# give fig
                                                     ax = ax, # give ax
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
        if show_separation:
            ax.axis("on")
        else:
            ax.axis("off")
    pdf.savefig(fig, bbox_inches = "tight")

def BarPlotClass(data, method_name, pdf, stat_name = None):
    PreFig()
    data_list = []
    colors = []
    for i in range(data.shape[0]):
        try:
            data_list.append(data[i, :].compressed()) # sometimes ax.boxplot fails to properly identify the masked values
        except:
            data_list.append(data[i, :])
            
        if method_name[i][:5] == "MIASA":
            colors.append("orange")
        else:
            colors.append("blue")
    vert = True
    if vert:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        bplot = ax.boxplot(data_list, notch = False, vert=vert, patch_artist = True, widths = .5, showfliers=False) # showfliers = False remove outliers
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            
        plt.xticks(np.cumsum(np.ones(len(method_name))), method_name, rotation = 90)
        plt.ylabel(stat_name, fontsize = 20)
    else:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        bplot = ax.boxplot(data_list, notch = False, vert=vert, patch_artist = True, widths = .5, showfliers=False) # showfliers = False remove outliers
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            
        plt.yticks(np.cumsum(np.ones(len(method_name))), method_name)
        plt.xlabel(stat_name, fontsize = 20)
    
    pdf.savefig(fig, bbox_inches = "tight")
    return fig
   
import seaborn as sns    
def BarPlotClass_sns(data, method_name, pdf, stat_name = None):
    PreFig()
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)

    data_dic = {}
    colors = []
    for i in range(data.shape[0]):
        data_dic[method_name[i]] = data[i, :].compressed()
        if method_name[i][:5] == "MIASA":
            colors.append("orange")
        else:
            colors.append("blue")
    
    df = pd.DataFrame(data_dic)
    bplot = sns.boxplot(df, ax = ax, orient= "v") # showfliers = False remove outliers
    
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)    
    
    plt.ylabel(stat_name, fontsize = 20)
    pdf.savefig(fig, bbox_inches = "tight")
    