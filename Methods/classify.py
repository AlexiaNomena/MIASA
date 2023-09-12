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
    acc_res_v0 = []
    acc_res_v1 = []
    num_it_list = []
    DMat = None
    for i in range(len(method_dic_list)):
        if method_dic_list[i]["class_method"] == "MIASA":
            Id_Class, X_vars, Y_vars, acc_r_v0, acc_r_v1, num_it = Classify_general(data_dic2, class_dic, num_clust, method_dic = method_dic_list[i], DMat = DMat, c_dic = c_dic, in_threads = in_threads)
            
        else:
            Id_Class, X_vars, Y_vars, acc_r_v0, acc_r_v1 = Classify_general(data_dic2, class_dic, num_clust, method_dic = method_dic_list[i], DMat = DMat, c_dic = c_dic, in_threads = in_threads)
            num_it = None, None
        print("Case %d -- method num %d/%d"%(var_data*1, i+1, len(method_dic_list)), "-- run %d/%d"%(r+1,repeat))
        # since this is the same dataset, if the metric_method is the same, then pass DMat directly to avoid recomputing it everytime for method with MIASA and Non_MD
        if (i>0)&(Id_Class is not None):
            if method_dic_list[i]["metric_method"] != method_dic_list[i-1]["metric_method"]:
                DMat = None
            else:
                DMat = Id_Class["DMat"]
        
        acc_res_v0.append(acc_r_v0)
        acc_res_v1.append(acc_r_v1)
        num_it_list.append(num_it)
        
    return acc_res_v0, acc_res_v1, num_it_list


def split_data(data_dic, class_dic, separation = False):
    if separation:
        """Extract separated data"""
        X_vars = data_dic["X_vars"]
        Y_vars = data_dic["Y_vars"]
    else:
        """Split data in two random groups of the same size"""
        samples = np.array(list(data_dic.keys()))
        samples = samples[~((samples == "true_colors")|(samples == "X_vars")|(samples == "Y_vars"))]
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
    acc_v0_list = []
    acc_v1_list = []
    num_it_list = []
    ### plot and save the first 10 classification runs
    if plot:
        start = 3
        for r in range(start):
            sub_list_v0 = []
            sub_list_v1 = []
            sub_list_it = []
            data_dic, class_dic, num_clust, dtp = generate_data(var_data)
            X, Y, Class_True, X_vars, Y_vars = split_data(data_dic, class_dic, separation)
            data_dic2 = {"X":X, "Y":Y, "Class_True":Class_True, "X_vars":X_vars, "Y_vars":Y_vars}
            for i in range(len(method_dic_list)): 
                print("method num %d/%d"%(i+1, len(method_dic_list)), "run %d/%d"%(r+1,repeat))
                if method_dic_list[i]["class_method"] == "MIASA":
                    Id_Class, X_vars, Y_vars, acc_r_v0, acc_r_v1, num_it = Classify_general(data_dic2, class_dic, num_clust, method_dic = method_dic_list[i], DMat = None, c_dic = c_dic, in_threads = in_threads)
                elif method_dic_list[i]["class_method"] == "non_MD":
                    Id_Class, X_vars, Y_vars, acc_r_v0, acc_r_v1 = Classify_general(data_dic2, class_dic, num_clust, method_dic = method_dic_list[i], DMat = None, c_dic = c_dic, in_threads = in_threads)
                    num_it = None, None
                if method_dic_list[i]["class_method"] == "MIASA":
                    pdf = method_dic_list[i]["fig"]
                    fig, ax = plotClass_separated_ver0(Id_Class, X_vars, Y_vars, pdf, dtp, run_num = r, n_neighbors = 30, method = "umap", true_colors=data_dic["true_colors"], 
                                                               scale = False, # scale = "pca", "standard", anything esle is taken as no scaling 
                                                               cluster_colors = False, # chosed_color: if False, true_colors bellow must be given 
                                                               markers = [("o",500),("^",500)], # optional markers list and their size for X and Y
                                                               sub_fig_size = 10, # optional sub figure size (as a square)
                                                               show_labels = False, # optional show the labels of X and Y
                                                               show_orig = False, # optional show the the axis lines going through origin 
                                                               show_separation = True, # optional separate all subfigs
                                                               )
                    #plotClass(Id_Class, X_vars, Y_vars, pdf, dtp, run_num = r, n_neighbors = 30, method = "t-SNE", true_colors=data_dic["true_colors"])
                    pdf.savefig(fig, bbox_inches = "tight")
                sub_list_v0.append(acc_r_v0)
                sub_list_v1.append(acc_r_v1)
                sub_list_it.append(num_it)
                
            acc_v0_list.append(sub_list_v0)
            acc_v1_list.append(sub_list_v1)
            num_it_list.append(sub_list_it)
            
    else:
        start = 0
    
    repeat = repeat - start
    if repeat > 10:
        # n_jobs = -1 means use all CPUs
        
        pfunc = partial(one_classification, repeat = repeat, method_dic_list = method_dic_list, var_data = var_data, generate_data = generate_data, c_dic = c_dic, in_threads = in_threads, separation = separation)
        try:
            res = np.array(jb.Parallel(n_jobs = n_jobs, prefer = "threads")(jb.delayed(pfunc)(r) for r in range(repeat)))
        except:
            res = np.array(jb.Parallel(n_jobs = n_jobs)(jb.delayed(pfunc)(r) for r in range(repeat)))
        
        if len(acc_v0_list) != 0:
            acc_v0_list = np.row_stack((np.array(acc_v0_list), res[:, 0, :, :]))
        else:
            acc_v0_list = res[:, 0, :, :]
        
        if len(acc_v1_list) != 0:
            acc_v1_list = np.row_stack((np.array(acc_v1_list), res[:, 1, :, :]))
        else:
            acc_v1_list = res[:, 1, :, :]
            
        if len(num_it_list) != 0:
            num_it_list = np.row_stack((np.array(num_it_list), res[:, 2, :, :]))
        else:
            num_it_list = res[:, 2, :, :] 

    elif repeat>0:
        pfunc = partial(one_classification, repeat = repeat, method_dic_list = method_dic_list, var_data = var_data, generate_data = generate_data, c_dic = c_dic, in_threads = in_threads, separation = separation)
        for r in range(repeat):
            res = np.array(pfunc(r)) 
            acc_v0_list.append(res[0, :, :])
            acc_v1_list.append(res[1, :, :])
            num_it_list.append(res[2, :, :])
        
    Acc_v0 = np.array(acc_v0_list)
    all_acc_list_v0 = masked_array(Acc_v0, mask = Acc_v0 == None)
    acc_list_v0 = all_acc_list_v0[:, :, 0].T
    adjusted_acc_list_v0 = all_acc_list_v0[:, :, 1].T
    
    Acc_v1 = np.array(acc_v1_list)
    all_acc_list_v1 = masked_array(Acc_v1, mask = Acc_v1 == None)
    acc_list_v1 = all_acc_list_v1[:, :, 0].T
    adjusted_acc_list_v1 = all_acc_list_v1[:, :, 1].T
    
    num_it_list_final = np.array(num_it_list)[:, :, 0]
    
    return acc_list_v0.astype(float), acc_list_v1.astype(float), adjusted_acc_list_v0.astype(float), adjusted_acc_list_v1.astype(float), np.array(num_it_list_final).T


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
        Class_Pred = Id_Class["Class_pred"]
        # normalize ARI between 0 and 1
        acc_metric_v0 = rand_score(Class_True, Class_Pred),  adjusted_rand_score(Class_True, Class_Pred)
        acc_metric_v1 = miasa_accuracy(Class_True, Class_Pred, X.shape[0], Y.shape[0])
        
        
    else:
        acc_metric_v0 = None, None
        acc_metric_v1 = None, None
        
    if class_method == "MIASA":
        return Id_Class, X_vars, Y_vars, acc_metric_v0, acc_metric_v1, (Id_Class["num_iterations"], 0) # 0 is just a placeholder
    else:
        return Id_Class, X_vars, Y_vars, acc_metric_v0, acc_metric_v1


def miasa_accuracy(Class_True, Class_Pred, M, N, quiet = True):
    """
    # accuracy measure suitable for miasa 
    # still work in progress
    Parameters
    ----------
    Class_True : TYPE
        DESCRIPTION.
    Class_Pred : TYPE
        DESCRIPTION.
    M : TYPE
        DESCRIPTION.
    N : TYPE
        DESCRIPTION.

    Returns
    -------
    mean_RI : TYPE
        DESCRIPTION.
    mean_ARI : TYPE
        DESCRIPTION.

    """
    class_true_x = Class_True[:M]
    class_pred_x = Class_Pred[:M]
    
    class_true_y = Class_True[-N:]
    class_pred_y = Class_Pred[-N:]
    
    lab_sep_true = np.sum(np.unique(Class_True)) + 1 ### a number that will not equal to any of the labels already present
    lab_sep_pred = np.sum(np.unique(Class_Pred)) + 1
    
    class_true_xy = []
    class_pred_xy = []
    for i in range(M):
        for j in range(N):
            if class_true_x[i] == class_true_y[j]:
                class_true_xy.append(class_true_x[i])
            else:
                class_true_xy.append(lab_sep_true)
            
            if class_pred_x[i] == class_pred_y[j]:
                class_pred_xy.append(class_pred_x[i])
            else:
                class_pred_xy.append(lab_sep_pred)
    
    class_true_xy = np.array(class_true_xy)    
    class_pred_xy = np.array(class_pred_xy)
    
    
    RI_x = rand_score(class_true_x, class_pred_x) 
    RI_y = rand_score(class_true_y, class_pred_y) 
    RI_xy_together = rand_score(class_true_xy, class_pred_xy) 
    
    mean_RI = np.mean([RI_x, RI_y, RI_xy_together]) 
    
    ARI_x = ARI_HA(class_true_x, class_pred_x)
    ARI_y = ARI_HA(class_true_y, class_pred_y)
    ARI_xy_together = ARI_HA(class_true_xy, class_pred_xy) 
    mean_ARI = np.mean([ARI_x, ARI_y, ARI_xy_together])
    
    if not quiet:
        print("ARI_HAx, ARI_HAy, ARI_HAxy", ARI_x, ARI_y, ARI_xy_together)
    return mean_RI, mean_ARI

import scipy.special
def ARI_HA(class_true, class_pred):
    ### compute contigency table
    ### See Hubert & Arabie 1985 and wikipedia https://en.wikipedia.org/wiki/Rand_index
    inds = np.arange(len(class_true)).astype(int)
    
    u_class_true = np.unique(class_true)
    u_class_pred = np.unique(class_pred)
    
    C = np.zeros((len(u_class_true), len(u_class_pred))).astype(int)
    bins = np.zeros(C.shape)
    bins_cols = np.zeros(len(u_class_true))
    for i in range(len(u_class_true)):
        for j in range(len(u_class_pred)):
            C[i, j] = len(inds[(class_true == u_class_true[i])&(class_pred == u_class_pred[j])])
            bins[i, j] = scipy.special.binom(C[i,j], 2)
        bins_cols[i] = scipy.special.binom(np.sum(C[i, :]), 2)
    
    bins_rows = np.zeros(len(u_class_pred))
    for j in range(len(u_class_pred)):
        bins_rows[j] = scipy.special.binom(np.sum(C[:, j]), 2)
        
    Index = np.sum(bins)
    mean_Index = (np.sum(bins_cols) + np.sum(bins_rows))/scipy.special.binom(np.sum(C), 2)
    max_Index = 0.5*(np.sum(bins_cols) + np.sum(bins_rows))
    
    return (Index - mean_Index)/(max_Index - mean_Index)

"""Visualization of classes"""
from .figure_settings import Display, PreFig
from .Core.Lower_dim import low_dim_coords
import pandas as pd

def plotClass(Id_Class, X_vars, Y_vars, pdf, dtp, run_num = 0, n_neighbors = 2, min_dist = 0.99, method = "umap", 
                        scale = None, sub_fig_size = 7, cluster_colors = False, true_colors = None, markers = [("o",20),("o",20)],
                        show_labels = False, show_orig = True, metric = "euclidean", legend = True, place_holder = (0, 0), 
                        wrap_true = False, wrap_predicted = False, wrap_pred_params = (None, 1), oultiers_markers = ("o", "^", 5),  wrap_type = "convexhull",
                        def_pred_outliers = (2, 0.95),show_pred_outliers = False, group_annot_size = 15, dataname = None, points_hull = 5, group_color = None, alpha = 0.25):        
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
    
    X_vars = np.array(X_vars)
    Y_vars = np.array(Y_vars)
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
        
        
         
    if legend & (not cluster_colors) & (not wrap_true) & (not wrap_predicted) :
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
                        ellipse = Ellipse(xy = (center[0], center[1]), width = width, height = height, angle = angle, fill = True, edgecolor = true_colors[cl_var[0]], alpha = alpha, lw = 2)
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
                    if points.shape[0] >= points_hull:
                        hull = convex_hull(points)
                        #plt.plot(points[hull.vertices, 0], points[hull.vertices,1], "-", linewidth = 1, color = true_colors[cl_var[0]])
                        #plt.plot([points[hull.vertices, :][0, 0], points[hull.vertices, :][-1, 0]], [points[hull.vertices, :][0, 1], points[hull.vertices, :][-1, 1]], "-", linewidth = 1, color = true_colors[cl_var[0]])
                        Vertices = points[hull.vertices, :]
                        
                        mark = rename_labels(cl, dataname)
                        
                        if group_color is not None:
                            col_center = group_color
                        else:
                            col_center = true_colors[cl_var[0]]
                            
                        plt.plot([Vertices[:, 0].mean()], [Vertices[:, 1].mean()], marker = "%s"%mark, markersize = group_annot_size, color = col_center)
                        
                        if cl_var[0][:2] not in done:
                            try:
                                Poly = Polygon(Vertices, edgecolor = true_colors[cl_var[0]], facecolor = true_colors[cl_var[0]], fill = True, label = rows_labels[cl_var[0]][:2], alpha = alpha)
                            except:
                                Poly = Polygon(Vertices, edgecolor = true_colors[cl_var[0]], facecolor = true_colors[cl_var[0]], fill = True, label = columns_labels[cl_var[0]][:2], alpha = alpha)
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
                                #plt.plot(points[:, 0], points[:, 1], marker = oultiers_markers[0], markersize =  oultiers_markers[2], color = true_colors[cl_var[0]], fillstyle = "full", linestyle = "")
                                lab_point = True
                            else:
                                plt.plot(points[:, 0], points[:, 1], marker = oultiers_markers[0], markersize =  oultiers_markers[2], color = "grey", fillstyle = "full", linestyle = "")
                                #plt.plot(points[:, 0], points[:, 1], marker = oultiers_markers[0], markersize =  oultiers_markers[2], color = true_colors[cl_var[0]], fillstyle = "full", linestyle = "")

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
                if points.shape[0] >= points_hull:
                    hull = convex_hull(points)
                    Vertices = points[hull.vertices, :]
                    if wrap_pred_params[0] is not None:
                        col_class = wrap_pred_params[0]
                        
                    if i == 0:
                        #Poly = Polygon(Vertices, edgecolor = "grey", fill = False, label = "predicted", linestyle = "-", linewidth = 1)
                        Poly = Polygon(Vertices, edgecolor = col_class, fill = False, label = "predicted %s"%(i+1), linestyle = "-", linewidth = wrap_pred_params[1])

                    else:
                        #Poly = Polygon(Vertices, edgecolor = col_class, fill = False, linestyle = "-", linewidth = 1)
                        Poly = Polygon(Vertices, edgecolor = col_class, fill = False, label = "predicted %s"%(i+1), linestyle = "-", linewidth = wrap_pred_params[1])

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
    return fig, ax
    
         
import re
def rename_labels(cl, dataname):
    if dataname in ("Dist", "Corr"):
        if cl[0] == "1":
            mark = "$N_%s$"%cl[1]
        elif cl[0] == "2":
            mark = "$U_%s$"%cl[1]
        elif cl[0] == "3":
            mark = "$Pa_%s$"%cl[1]
        elif cl[0] == "4":
            mark = "$Poi_%s$"%cl[1]
        else:
            mark = cl
    elif dataname == "GRN":
        if cl[0] == "D":
            mark = "$Bi$"
        elif cl[0] == "S":
            mark = "$Mo$"
        elif cl[0] == "N":
            mark  = "$No$"
        else:
            mark = cl
    else:
        mark = " ".join(re.split("[^a-zA-Z]*", cl))
        mark = mark.replace(" ", "")
        mark = "%s"%mark
    return mark
                         
    
def plotClass_separated(Id_Class, X_vars, Y_vars, pdf, dtp, run_num = 0, n_neighbors = 2, min_dist = 0.99, method = "umap", 
                        scale = None, sub_fig_size = 7, cluster_colors = False, true_colors = None, markers = [("o",20),("o",20)], markers_color = None,
                        show_labels = False, show_orig = True, metric = "euclidean", legend = True, place_holder = (0, 0), 
                        wrap_true = False, wrap_predicted = False, wrap_pred_params = (None, 1), oultiers_markers = ("o", "^", 5),  wrap_type = "convexhull",
                        def_pred_outliers = (2, 0.95),show_pred_outliers = False, group_annot_size = 15, dataname = None,
                        num_row_col = None, show_separation = False, points_hull = 5, group_color = None, alpha = 0.25):        
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
    
    X_vars = np.array(X_vars)
    Y_vars = np.array(Y_vars)
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
        
    if markers_color is None:
        col_to_use = (col_rows, col_cols)
    else:
        col_to_use = markers_color
        
    marker_to_use = markers #[("o",20),("o",20)]
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
            
        class_row_sub = Id_Class["Class_pred"][:M] == unique_classe[i]
        class_col_sub =  Id_Class["Class_pred"][M:] == unique_classe[i]
        
        coords_row_sub = Rows_manifold[class_row_sub, :]
        coords_col_sub = Cols_manifold[class_col_sub, :]
        
        X_vars_sub = np.array(X_vars)[class_row_sub]
        Y_vars_sub = np.array(Y_vars)[class_col_sub]
        if show_labels:
            rows_to_Annot_sub = rows_to_Annot[class_row_sub]
            cols_to_Annot_sub = cols_to_Annot[class_col_sub]
        else:
            rows_to_Annot_sub = None
            cols_to_Annot_sub = None
        
        Data = DataFrame.copy()
        Data.drop(list(Data.columns[~class_col_sub]), axis = 1, inplace = True)
        Data.drop(list(Data.index[~class_row_sub]), axis = 0, inplace = True)
        fig, xy_rows, xy_cols, gs, center = Display(coords_row_sub, 
                                                     coords_col_sub, 
                                                     Inertia, 
                                                     Data,
                                                     center = Origin_manifold, 
                                                     rows_to_Annot = rows_to_Annot_sub, # row items to annotate, if None then no annotation (None if none)
                                                     cols_to_Annot = cols_to_Annot_sub, # column items to annotate (None if none)
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
        
        
        if legend & (not cluster_colors) & (not wrap_true) & (not wrap_predicted) :
             col_done = []
             for i in range(len(X_vars_sub)):
                 if true_colors[X_vars_sub[i]] not in col_done:
                     ax.scatter(np.zeros(1), np.zeros(1), marker = marker_to_use[0][0], s =  marker_to_use[0][1], color = true_colors[X_vars_sub[i]], label = rows_labels[X_vars_sub[i]][:2])
                     col_done.append(true_colors[X_vars_sub[i]])
             
             ax.annotate("place_holder", xy=(0,0), 
                      xytext= (5, 5), textcoords='offset points', ha='center', va='bottom',
                      bbox=dict(boxstyle='circle', fc = "white"),
                      arrowprops=dict(arrowstyle='->', color = "black"),  #connectionstyle='arc3,rad=0.5'),
                      color= "black",
                      fontsize = 6
                       )
             
             plt.legend()
        
    
         
        if wrap_true:
            Id_class_pred_sub = Id_Class["Class_pred"][np.concatenate((class_row_sub, class_col_sub))]
            pred_class = np.unique(Id_class_pred_sub)
            lab_point = False
            for i in range(len(pred_class)):
                
                class_row = (Id_class_pred_sub == pred_class[i])[:np.sum(class_row_sub)]
                class_col = (Id_class_pred_sub == pred_class[i])[np.sum(class_row_sub):]
                
                coords_row = coords_row_sub[class_row, :]
                coords_col = coords_col_sub[class_col, :]
                
                X_var_sub = [X_vars_sub[class_row][i] for i in range(coords_row.shape[0])] # the first two letters are always the true class labels
                Y_var_sub = [Y_vars_sub[class_col][i] for i in range(coords_col.shape[0])]
                
                X_var_sub2 = [X_vars_sub[class_row][i][:2] for i in range(coords_row.shape[0])] # the first two letters are always the true class labels
                Y_var_sub2 = [Y_vars_sub[class_col][i][:2] for i in range(coords_col.shape[0])]
                
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
                            ellipse = Ellipse(xy = (center[0], center[1]), width = width, height = height, angle = angle, fill = True, edgecolor = true_colors[cl_var[0]], alpha = alpha, lw = 2)
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
                        if points.shape[0] >= points_hull:
                            hull = convex_hull(points)
                            #plt.plot(points[hull.vertices, 0], points[hull.vertices,1], "-", linewidth = 1, color = true_colors[cl_var[0]])
                            #plt.plot([points[hull.vertices, :][0, 0], points[hull.vertices, :][-1, 0]], [points[hull.vertices, :][0, 1], points[hull.vertices, :][-1, 1]], "-", linewidth = 1, color = true_colors[cl_var[0]])
                            Vertices = points[hull.vertices, :]
                            
                            mark = rename_labels(cl, dataname)
                            if group_color is not None:
                                col_center = group_color
                            else:
                                col_center = true_colors[cl_var[0]]
                            
                            plt.plot([Vertices[:, 0].mean()], [Vertices[:, 1].mean()], marker = "%s"%mark, markersize = group_annot_size, color = col_center)
                            
                                
                            if cl_var[0][:2] not in done:
                                try:
                                    Poly = Polygon(Vertices, edgecolor = true_colors[cl_var[0]], facecolor = true_colors[cl_var[0]], fill = True,label = rows_labels[cl_var[0]][:2], alpha = alpha)
                                except:
                                    Poly = Polygon(Vertices, edgecolor = true_colors[cl_var[0]], facecolor = true_colors[cl_var[0]], fill = True,label = columns_labels[cl_var[0]][:2], alpha = alpha)
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
                                    #plt.plot(points[:, 0], points[:, 1], marker = oultiers_markers[0], markersize =  oultiers_markers[2], color = true_colors[cl_var[0]], fillstyle = "full", linestyle = "")
    
                                    lab_point = True
                                else:
                                    plt.plot(points[:, 0], points[:, 1], marker = oultiers_markers[0], markersize =  oultiers_markers[2], color = "grey", fillstyle = "full", linestyle = "")
                                    #plt.plot(points[:, 0], points[:, 1], marker = oultiers_markers[0], markersize =  oultiers_markers[2], color = true_colors[cl_var[0]], fillstyle = "full", linestyle = "")
    
                            else:
                                plt.plot(points[:, 0], points[:, 1], marker = oultiers_markers[0], markersize =  oultiers_markers[2], color = true_colors[cl_var[0]], fillstyle = "full", linestyle = "")
                            
                            done2.append(cl_var[0][:2])
                        
        
        if wrap_predicted:
            Id_class_pred_sub = Id_Class["Class_pred"][np.concatenate((class_row_sub, class_col_sub))]
            pred_class = np.unique(Id_class_pred_sub)
            lab_point = False
            
            for i in range(len(pred_class)):
                class_row = (Id_class_pred_sub == pred_class[i])[:np.sum(class_row_sub)]
                class_col = (Id_class_pred_sub == pred_class[i])[np.sum(class_row_sub):]
                
                coords_row = coords_row_sub[class_row, :]
                coords_col = coords_col_sub[class_col, :]
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
                    if points.shape[0] >= points_hull:
                        hull = convex_hull(points)
                        Vertices = points[hull.vertices, :]
                        if wrap_pred_params[0] is not None:
                            col_class = wrap_pred_params[0]
                            
                        if i == 0:
                            #Poly = Polygon(Vertices, edgecolor = "grey", fill = False, label = "predicted", linestyle = "-", linewidth = 1)
                            Poly = Polygon(Vertices, edgecolor = col_class, fill = False, label = "predicted %s"%(i+1), linestyle = "-", linewidth = wrap_pred_params[1])
    
                        else:
                            #Poly = Polygon(Vertices, edgecolor = col_class, fill = False, linestyle = "-", linewidth = 1)
                            Poly = Polygon(Vertices, edgecolor = col_class, fill = False, label = "predicted %s"%(i+1), linestyle = "-", linewidth = wrap_pred_params[1])
    
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
    
        
        if show_separation:
            ax.axis("on")
            ax.margins(0.1, 0.1)

        else:
            ax.axis("off")
                
    """
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)      
    """
    return fig, ax

def plotClass_separated_ver0(Id_Class, X_vars, Y_vars, pdf, dtp, run_num = 0, n_neighbors = 2, min_dist = 0.99, method = "umap", 
                        scale = None, sub_fig_size = 7, num_row_col = None, cluster_colors = False, true_colors = None, markers = [("o",10),("o",10)], 
                        show_labels = False, show_orig = False, show_separation = False, legend = True, shorten_annots = True, dataname = None):   
    """@brief Plot and Save class figures"""
    
    Coords = Id_Class["Coords"]
    """Lower Dimensional visualization of clusters"""
    low_meth = method # methods: MDS, Isomap, TSNE
    Coords_manifold = low_dim_coords(Coords, dim=2, method  = low_meth, n_neighbors = n_neighbors, min_dist = min_dist, scale = scale) 
    """
    Kmeans and UMAP are already parameterized for reproducibility (random_state = 0 for both).
    However, slight changes could still happen due to the optimization procedure and versions of these packages.
    """
    
    X_vars = np.array(X_vars)
    Y_vars = np.array(Y_vars)
    
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
        rows_labels = {X_vars[i]:rename_labels(X_vars[i][:2], dataname) for i in range(M)}
        columns_labels = {Y_vars[i]:rename_labels(Y_vars[i][:2], dataname) for i in range(N)}
    else:
        rows_labels = {X_vars[i]:rename_labels(X_vars[i][:2], dataname)+X_vars[i][2:] for i in range(M)}
        columns_labels = {Y_vars[i]:rename_labels(Y_vars[i][:2], dataname)+Y_vars[i][2:] for i in range(N)}
    
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
            ax.margins(0.1, 0.1)
            
        else:
            ax.axis("off")
        
    return fig, ax

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
    