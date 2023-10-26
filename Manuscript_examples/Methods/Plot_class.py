#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 15:00:55 2023

@author: raharinirina
"""

"""Visualization of classes"""
from .figure_settings import Display, PreFig
from .Core.Lower_dim import low_dim_coords
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .Wraps import find_ellipse_params, convex_hull
import pdb
from matplotlib.patches import Ellipse, Polygon
from copy import copy
import scipy.spatial as spsp

def plotClass(Id_Class, X_vars, Y_vars, pdf, dtp, run_num = 0, n_neighbors = 2, min_dist = 0.99, method = "umap", 
                        scale = None, sub_fig_size = 7, cluster_colors = False, true_colors = None, markers = [("o",20),("o",20)],
                        show_labels = False, show_orig = True, metric = "euclidean", legend = True, place_holder = (0, 0), 
                        wrap_true = False, wrap_predicted = False, wrap_pred_params = (None, 1), oultiers_markers = ("o", "^", 5),  wrap_type = "convexhull",
                        def_pred_outliers = (2, 0.95),show_pred_outliers = False, group_annot_size = 15, dataname = None, points_hull = 5, group_color = None, alpha = 0.25,
                        shorten_annots = True, cut = (2, 2), connect_pred = False):        
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
    
    if shorten_annots:
        rows_labels = {X_vars[i]:X_vars[i][:cut[0]] for i in range(M)}
        columns_labels = {Y_vars[i]:Y_vars[i][:cut[1]] for i in range(N)}
    else:
        rows_labels = {X_vars[i]:X_vars[i] for i in range(M)}
        columns_labels = {Y_vars[i]:Y_vars[i] for i in range(N)}
   
    
    if show_labels:
        rows_to_Annot = np.array(DataFrame.index)
        cols_to_Annot = np.array(DataFrame.columns)
    else:
        rows_to_Annot = None
        cols_to_Annot = None
    
    color_clustered = Id_Class["color_clustered"]
    
    fig = plt.figure(figsize=(36+18,20+10))
    ax = fig.add_subplot(2,1,1)
    if connect_pred:
        pred_class = np.unique(Id_Class["Class_pred"])
        lab_point = False

        for i in range(len(pred_class)):
            class_row = Id_Class["Class_pred"][:M] == pred_class[i]
            class_col =  Id_Class["Class_pred"][M:] == pred_class[i]
    
            coords_row = Rows_manifold[class_row, :]
            coords_col = Cols_manifold[class_col, :]
            points = np.row_stack((coords_row, coords_col))
            
            center_0 = np.mean(points, axis = 0)
            Dists = np.sqrt(np.sum((points - center_0[np.newaxis, :])**2, axis = 1))
            center = points[np.argmin(Dists), :]
            
            col_class = color_clustered[Id_Class["Class_pred"] == pred_class[i]][0]
            for j in range(points.shape[0]):
                plt.plot([points[j, 0], center[0]], [points[j, 1], center[1]], linewidth = 2, alpha = 0.3, color = col_class)
                
   
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
                                                     fig = fig,
                                                     ax = ax,
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
        
    
    """  
    if legend & (not cluster_colors) & (not wrap_true) & (not wrap_predicted) :
         col_done = []
         for i in range(len(X_vars)):
             if true_colors[X_vars[i]] not in col_done:
                 ax.scatter(np.zeros(1), np.zeros(1), marker = marker_to_use[0][0], s =  marker_to_use[0][1], color = true_colors[X_vars[i]], label = rows_labels[X_vars[i]][:cut[0]])
                 col_done.append(true_colors[X_vars[i]])
         
         for i in range(len(X_vars)):
            if true_colors[Y_vars[i]] not in col_done:
                ax.scatter(np.zeros(1), np.zeros(1), marker = marker_to_use[0][1], s =  marker_to_use[0][1], color = true_colors[Y_vars[i]], label = columns_labels[Y_vars[i]][:cut[1]])
                col_done.append(true_colors[Y_vars[i]])
         
         ax.annotate("place_holder", xy=(0,0), 
                  xytext= (5, 5), textcoords='offset points', ha='center', va='bottom',
                  bbox=dict(boxstyle='circle', fc = "white"),
                  arrowprops=dict(arrowstyle='->', color = "black"),  #connectionstyle='arc3,rad=0.5'),
                  color= "black",
                  fontsize = 6
                   )
         
         plt.legend()
    """   
    
         
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
            
            #X_var_sub2 = [X_vars[class_row][i][:cut[0]] for i in range(coords_row.shape[0])] 
            #Y_var_sub2 = [Y_vars[class_col][i][:cut[1]] for i in range(coords_col.shape[0])]
            
            X_var_sub2 = [true_colors[X_vars[class_row][i]] for i in range(coords_row.shape[0])] 
            Y_var_sub2 = [true_colors[Y_vars[class_col][i]] for i in range(coords_col.shape[0])]
            
            #class_labs_x = np.unique(X_vars_sub2)
            #class_labs_y = np.unique(Y_vars_sub2)
            
            class_labs_x = np.unique(np.array(X_var_sub2), axis = 0)
            class_labs_y = np.unique(np.array(Y_var_sub2), axis = 0)
            
            X_var_sub = np.array(X_var_sub)
            Y_var_sub = np.array(Y_var_sub)
            
            X_var_sub2 = np.array(X_var_sub2)
            Y_var_sub2 = np.array(Y_var_sub2)
            
            done = []
            done2 = []
            for k in range(len(class_labs_x)):
                cl = class_labs_x[k, :]
                where_true_clust = np.all(X_var_sub2 == cl, axis = 1)
                points = coords_row[where_true_clust, :]
                cl_var = list(X_var_sub[where_true_clust])
                
                
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
               
                    if points.shape[0] >= points_hull:
                        hull = convex_hull(points)
                        Vertices = points[hull.vertices, :]
                        
                        mark = rename_labels(cl_var[0], dataname)
                        
                        if group_color is not None:
                            col_center = group_color
                        else:
                            col_center = true_colors[cl_var[0]]
                        
                        try:
                            plt.plot([Vertices[:, 0].mean()], [Vertices[:, 1].mean()], marker = "%s"%mark, markersize = group_annot_size, color = col_center)
                        except:
                            pdb.set_trace()
                        
                        if cl_var[0] not in done:
                            Poly = Polygon(Vertices, edgecolor = true_colors[cl_var[0]], facecolor = true_colors[cl_var[0]], fill = True, label = rows_labels[cl_var[0]][:cut[0]], alpha = alpha)
                        else:
                            Poly = Polygon(Vertices, edgecolor = true_colors[cl_var[0]], facecolor = true_colors[cl_var[0]], fill = True)
                        
                        ax.add_patch(copy(Poly))
                        
                        done.append(cl_var[0])
                        
                        

                    else:
                        if cl_var[0] not in done2:
                            if not lab_point:
                                plt.plot(points[:, 0], points[:, 1], marker = oultiers_markers[0], markersize =  oultiers_markers[2], color = "grey", fillstyle = "full",  label = "Outliers", linestyle = "")
                                #plt.plot(points[:, 0], points[:, 1], marker = oultiers_markers[0], markersize =  oultiers_markers[2], color = true_colors[cl_var[0]], fillstyle = "full", linestyle = "")
                                lab_point = True
                            else:
                                plt.plot(points[:, 0], points[:, 1], marker = oultiers_markers[0], markersize =  oultiers_markers[2], color = "grey", fillstyle = "full", linestyle = "")
                                #plt.plot(points[:, 0], points[:, 1], marker = oultiers_markers[0], markersize =  oultiers_markers[2], color = true_colors[cl_var[0]], fillstyle = "full", linestyle = "")

                        else:
                            plt.plot(points[:, 0], points[:, 1], marker = oultiers_markers[0], markersize =  oultiers_markers[2], color = true_colors[cl_var[0]], fillstyle = "full", linestyle = "")
                        
                        done2.append(cl_var[0])
            
            for k in range(len(class_labs_y)):
                 cl = class_labs_y[k, :]
                 where_true_clust = np.all(Y_var_sub2 == cl, axis = 1)
                 points = coords_col[where_true_clust, :]
                 cl_var = list(Y_var_sub[where_true_clust])
                 
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
                     
                     if points.shape[0] >= points_hull:
                         hull = convex_hull(points)
                         Vertices = points[hull.vertices, :]
                         
                         mark = rename_labels(cl_var[0], dataname)
                         
                         if group_color is not None:
                             col_center = group_color
                         else:
                             col_center = true_colors[cl_var[0]]
                         
                         try:
                             plt.plot([Vertices[:, 0].mean()], [Vertices[:, 1].mean()], marker = "%s"%mark, markersize = group_annot_size, color = col_center)
                         except:
                             pdb.set_trace()
                         
                         if cl_var[0] not in done:
                             Poly = Polygon(Vertices, edgecolor = true_colors[cl_var[0]], facecolor = true_colors[cl_var[0]], fill = True, label = columns_labels[cl_var[0]][:cut[1]], alpha = alpha)
                         else:
                             Poly = Polygon(Vertices, edgecolor = true_colors[cl_var[0]], facecolor = true_colors[cl_var[0]], fill = True)
                         
                         ax.add_patch(copy(Poly))
                         
                         done.append(cl_var[0])
                         
                         

                     else:
                         if cl_var[0] not in done2:
                             if not lab_point:
                                 plt.plot(points[:, 0], points[:, 1], marker = oultiers_markers[0], markersize =  oultiers_markers[2], color = "grey", fillstyle = "full",  label = "Outliers", linestyle = "")
                                 #plt.plot(points[:, 0], points[:, 1], marker = oultiers_markers[0], markersize =  oultiers_markers[2], color = true_colors[cl_var[0]], fillstyle = "full", linestyle = "")
                                 lab_point = True
                             else:
                                 plt.plot(points[:, 0], points[:, 1], marker = oultiers_markers[0], markersize =  oultiers_markers[2], color = "grey", fillstyle = "full", linestyle = "")
                                 #plt.plot(points[:, 0], points[:, 1], marker = oultiers_markers[0], markersize =  oultiers_markers[2], color = true_colors[cl_var[0]], fillstyle = "full", linestyle = "")

                         else:
                             #plt.plot(points[:, 0], points[:, 1], marker = oultiers_markers[0], markersize =  oultiers_markers[2], color = true_colors[cl_var[0]], fillstyle = "full", linestyle = "")
                             plt.plot(points[:, 0], points[:, 1], marker = oultiers_markers[0], markersize =  oultiers_markers[2], color = "grey", fillstyle = "full", linestyle = "")

                         done2.append(cl_var[0])       
             
            
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
        #mark = " ".join(re.split("[^a-zA-Z]*", cl))
        #mark = mark.replace(" ", "")
        #mark = "$%s$"%mark
        mark = "$%s$"%cl
        
    return mark
                         
    
def plotClass_separated(Id_Class, X_vars, Y_vars, pdf, dtp, run_num = 0, n_neighbors = 2, min_dist = 0.99, method = "umap", 
                        scale = None, sub_fig_size = 7, cluster_colors = False, true_colors = None, markers = [("o",20),("o",20)], markers_color = None,
                        show_labels = False, show_orig = True, metric = "euclidean", legend = True, place_holder = (0, 0), 
                        wrap_true = False, wrap_predicted = False, wrap_pred_params = (None, 1), oultiers_markers = ("o", "^", 5),  wrap_type = "convexhull",
                        def_pred_outliers = (2, 0.95),show_pred_outliers = False, group_annot_size = 15, dataname = None,
                        num_row_col = None, show_separation = False, points_hull = 5, group_color = None, alpha = 0.25, shorten_annots = True, cut = (2, 2)):        
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
    
    if shorten_annots:
        rows_labels = {X_vars[i]:X_vars[i][:cut[0]] for i in range(M)}
        columns_labels = {Y_vars[i]:Y_vars[i][:cut[1]] for i in range(N)}
    else:
        rows_labels = {X_vars[i]:X_vars[i] for i in range(M)}
        columns_labels = {Y_vars[i]:Y_vars[i] for i in range(N)}
   
    
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
        
        plt.title("Pred Class %d"%(i+1))
        
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
                     ax.scatter(np.zeros(1), np.zeros(1), marker = marker_to_use[0][0], s =  marker_to_use[0][1], color = true_colors[X_vars_sub[i]], label = rows_labels[X_vars_sub[i]][:cut[0]])
                     col_done.append(true_colors[X_vars_sub[i]])
             
             for i in range(len(X_vars)):
                if true_colors[Y_vars[i]] not in col_done:
                    ax.scatter(np.zeros(1), np.zeros(1), marker = marker_to_use[0][1], s =  marker_to_use[0][1], color = true_colors[Y_vars[i]], label = columns_labels[Y_vars[i]][:cut[1]])
                    col_done.append(true_colors[Y_vars[i]])
                    
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
                
                X_var_sub = [X_vars_sub[class_row][i] for i in range(coords_row.shape[0])] 
                Y_var_sub = [Y_vars_sub[class_col][i] for i in range(coords_col.shape[0])]
                
                #X_var_sub2 = [X_vars_sub[class_row][i][:cut[0]] for i in range(coords_row.shape[0])] 
                #Y_var_sub2 = [Y_vars_sub[class_col][i][:cut[1]] for i in range(coords_col.shape[0])]
                
                X_var_sub2 = [true_colors[X_vars_sub[class_row][i]] for i in range(coords_row.shape[0])] 
                Y_var_sub2 = [true_colors[Y_vars_sub[class_col][i]] for i in range(coords_col.shape[0])]
                
                #class_labs_x = np.unique(X_var_sub2)
                #class_labs_y = np.unique(Y_var_sub2)
                
                class_labs_x = np.unique(np.array(X_var_sub2), axis = 0)
                class_labs_y = np.unique(np.array(Y_var_sub2), axis = 0)
                
                X_var_sub = np.array(X_var_sub)
                Y_var_sub = np.array(Y_var_sub)
                
                X_var_sub2 = np.array(X_var_sub2)
                Y_var_sub2 = np.array(Y_var_sub2)
                
                done = []
                done2 = []
                for k in range(len(class_labs_x)):
                    cl = class_labs_x[k, :]
                    where_true_clust = np.all(X_var_sub2 == cl, axis = 1)
                    points = coords_row[where_true_clust, :]
                    cl_var = list(X_var_sub[where_true_clust])
                    
                    
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
                        if points.shape[0] >= points_hull:
                            hull = convex_hull(points)
                            Vertices = points[hull.vertices, :]
                            
                            mark = rename_labels(cl_var[0], dataname)
                            if group_color is not None:
                                col_center = group_color
                            else:
                                col_center = true_colors[cl_var[0]]
                            
                            plt.plot([Vertices[:, 0].mean()], [Vertices[:, 1].mean()], marker = "%s"%mark, markersize = group_annot_size, color = col_center)
                            
                                
                            if cl_var[0][:cut[0]] not in done:
                                Poly = Polygon(Vertices, edgecolor = true_colors[cl_var[0]], facecolor = true_colors[cl_var[0]], fill = True,label = rows_labels[cl_var[0]][:cut[1]], alpha = alpha)
                            else:
                                Poly = Polygon(Vertices, edgecolor = true_colors[cl_var[0]], facecolor = true_colors[cl_var[0]], fill = True)
                            
                            ax.add_patch(copy(Poly))
                            
                            done.append(cl_var[0][:cut[0]])
                            
                        else:
                            if cl_var[0] not in done2:
                                if not lab_point:
                                    plt.plot(points[:, 0], points[:, 1], marker = oultiers_markers[0], markersize =  oultiers_markers[2], color = "grey", fillstyle = "full",  label = "Outliers", linestyle = "")
                                    #plt.plot(points[:, 0], points[:, 1], marker = oultiers_markers[0], markersize =  oultiers_markers[2], color = true_colors[cl_var[0]], fillstyle = "full", linestyle = "")
    
                                    lab_point = True
                                else:
                                    plt.plot(points[:, 0], points[:, 1], marker = oultiers_markers[0], markersize =  oultiers_markers[2], color = "grey", fillstyle = "full", linestyle = "")
                                    #plt.plot(points[:, 0], points[:, 1], marker = oultiers_markers[0], markersize =  oultiers_markers[2], color = true_colors[cl_var[0]], fillstyle = "full", linestyle = "")
    
                            else:
                                #plt.plot(points[:, 0], points[:, 1], marker = oultiers_markers[0], markersize =  oultiers_markers[2], color = true_colors[cl_var[0]], fillstyle = "full", linestyle = "")
                                plt.plot(points[:, 0], points[:, 1], marker = oultiers_markers[0], markersize =  oultiers_markers[2], color = "grey", fillstyle = "full", linestyle = "")

                            done2.append(cl_var[0])
                
                for k in range(len(class_labs_y)):
                    cl = class_labs_y[k, :]
                    where_true_clust = np.all(Y_var_sub2 == cl, axis = 1)
                    points = coords_col[where_true_clust, :]
                    cl_var = list(Y_var_sub[where_true_clust])
                    
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
                        if points.shape[0] >= points_hull:
                            hull = convex_hull(points)
                            #plt.plot(points[hull.vertices, 0], points[hull.vertices,1], "-", linewidth = 1, color = true_colors[cl_var[0]])
                            #plt.plot([points[hull.vertices, :][0, 0], points[hull.vertices, :][-1, 0]], [points[hull.vertices, :][0, 1], points[hull.vertices, :][-1, 1]], "-", linewidth = 1, color = true_colors[cl_var[0]])
                            Vertices = points[hull.vertices, :]
                            
                            mark = rename_labels(cl_var[0], dataname)
                            if group_color is not None:
                                col_center = group_color
                            else:
                                col_center = true_colors[cl_var[0]]
                            
                            plt.plot([Vertices[:, 0].mean()], [Vertices[:, 1].mean()], marker = "%s"%mark, markersize = group_annot_size, color = col_center)
                            
                                
                            if cl_var[0][:cut[1]] not in done:
                                Poly = Polygon(Vertices, edgecolor = true_colors[cl_var[0]], facecolor = true_colors[cl_var[0]], fill = True,label = columns_labels[cl_var[0]][:cut[1]], alpha = alpha)
                            else:
                                Poly = Polygon(Vertices, edgecolor = true_colors[cl_var[0]], facecolor = true_colors[cl_var[0]], fill = True)
                            
                            ax.add_patch(copy(Poly))
                            
                            done.append(cl_var[0][:cut[1]])
    
                        else:
                            if cl_var[0] not in done2:
                                if not lab_point:
                                    plt.plot(points[:, 0], points[:, 1], marker = oultiers_markers[0], markersize =  oultiers_markers[2], color = "grey", fillstyle = "full",  label = "Outliers", linestyle = "")
                                    #plt.plot(points[:, 0], points[:, 1], marker = oultiers_markers[0], markersize =  oultiers_markers[2], color = true_colors[cl_var[0]], fillstyle = "full", linestyle = "")
                                    lab_point = True
                                else:
                                    plt.plot(points[:, 0], points[:, 1], marker = oultiers_markers[0], markersize =  oultiers_markers[2], color = "grey", fillstyle = "full", linestyle = "")
                                    #plt.plot(points[:, 0], points[:, 1], marker = oultiers_markers[0], markersize =  oultiers_markers[2], color = true_colors[cl_var[0]], fillstyle = "full", linestyle = "")
    
                            else:
                                #plt.plot(points[:, 0], points[:, 1], marker = oultiers_markers[0], markersize =  oultiers_markers[2], color = true_colors[cl_var[0]], fillstyle = "full", linestyle = "")
                                plt.plot(points[:, 0], points[:, 1], marker = oultiers_markers[0], markersize =  oultiers_markers[2], color = "grey", fillstyle = "full", linestyle = "")

                            done2.append(cl_var[0])    
        
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
                        show_labels = False, show_orig = False, show_separation = False, legend = True, shorten_annots = True, dataname = None, cut = (2, 2)):   
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
        rows_labels = {X_vars[i]:rename_labels(X_vars[i][:cut[0]], dataname) for i in range(M)}
        columns_labels = {Y_vars[i]:rename_labels(Y_vars[i][:cut[1]], dataname) for i in range(N)}
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
        
        plt.title("Pred Class %d"%(i+1))
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