#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 15:00:55 2023

@author: raharinirina
"""

"""Visualization of classes"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
pl = plt
import pdb
from matplotlib.patches import Ellipse, Polygon
from copy import copy
import scipy.spatial as spsp
import seaborn as sns
from scipy.spatial import ConvexHull as convex_hull
import sys

try:
    Connect_assoc = str(sys.argv[14])
    Connect_thres = float(sys.argv[15])
    Connect_center = str(sys.argv[16])
    Connect_assoc = True
except:
    try:
        try:
            Connect_thres = str(sys.argv[15])
            Connect_center = str(sys.argv[16]) 
            try:
                rawData = pd.read_excel(sys.argv[17], engine='openpyxl')
                Connect_thres = 0
            except:
                rawData = pd.read_csv(sys.argv[17])
                Connect_thres = 0
            
            try:
                rawData.drop(columns = "Unnamed: 0", inplace = True)
            except:
                pass
            
            rawData.drop(columns = "variable", inplace = True)
            Assoc_file = True
            Connect_assoc = True
        except:
            sys.exit("if connect_threshold is not a float, the connection_file must be given")
            Assoc_file = False
    except:
        Assoc_file = False
        Connect_assoc = False
     
def get_col_labs(labels, palette):   
	# make sure that colors are not cyclic:                 
    unique_labs = np.unique(labels)
    colors = sns.color_palette(palette,  len(unique_labs))
    n = 0
    col_labs = np.zeros((len(labels), 3))
    col_done = []
    for i in range(len(unique_labs)):
        """
        if np.all(np.array(colors[i])<=1):
            col_i = np.array(255*(np.array()), dtype = int)
        else:
            col_i = np.array(colors[i], dtype = int)
        col_labs[labels == unique_labs[i], :] = '#%02x%02x%02x'%tuple(col_i)
        """ 
        # make sure that colors are not cyclic: 
        colsub = colors[i]
        while colsub in col_done:
            np.random.seed(n)
            colsub = tuple(np.random.uniform(0, 1, size =3))
            n += 1
            
        col_labs[labels == unique_labs[i], :] = colsub
        col_done.append(colsub)
    
    return col_labs


#### Visualisation ###  
def PreFig(xsize = 12, ysize = 12):
    '''
    @brief: customize figure parameters
    '''
    matplotlib.rc('xtick', labelsize=xsize) 
    matplotlib.rc('ytick', labelsize=ysize)
    
def OneAnnotation(ax, lab, coords, col_val, xl=5, yl=5, arrow = False, fontsize = 12, alpha = 0.5):
    if arrow:
        ax.annotate("%s"%lab, xy=coords, 
                xytext= (xl, yl), textcoords='offset points', ha='center', va='bottom',
                #bbox=dict(boxstyle='round,pad=0.2', fc=col_val, alpha=alpha),
                bbox=dict(boxstyle='circle', fc=col_val, alpha=alpha),
                arrowprops=dict(arrowstyle='->', color = "black"),  #connectionstyle='arc3,rad=0.5'),
                color= "black",
                fontsize = fontsize # 6
                 )
    else:
         ax.annotate("%s"%lab, xy=coords, 
                xytext= (xl, yl), textcoords='offset points', ha='center', va='bottom',
                #bbox=dict(boxstyle='round,pad=0.2', fc=col_val, alpha=alpha),
                bbox=dict(boxstyle='circle', fc=col_val, alpha=alpha),
                color= "black",
                fontsize = fontsize # 6
                 )
    return ax


def Annotate(ax, rows_to_Annot, cols_to_Annot, Label_rows, Label_cols, xy_rows, xy_cols, col = ("green", "pink"), arrow = False):
    '''
    @brief : plot text annotations 
    @params: see function CA 
    '''
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    
    pdist_n = spsp.distance.pdist(np.concatenate((xy_rows,xy_cols), axis = 0))
    pdist_n[np.isnan(pdist_n)] = 10000000
    pdist_n[~np.isfinite(pdist_n)] = 10000000
    
    pdist = spsp.distance.squareform(pdist_n)
    
    
    if rows_to_Annot is not None: #and Label_rows is not None:
        #pdist_n = scp.spatial.distance.pdist(xy_rows)
        #pdist_n[np.isnan(pdist_n)] = 10000000
        #pdist_n[~np.isfinite(pdist_n)] = 10000000
        
        #pdist = scp.spatial.distance.squareform(pdist_n)
        
        l_special = []
        sup = 0
        f = 1.75
        for j in rows_to_Annot:
            if (np.sum(pdist[j, (j+1):] <= 0.15*np.mean(pdist_n)) != 0)*(j not in l_special)*(j != xy_rows.shape[0]-1):
                xl = 0#20*(f)*np.cos(sup) #0
                yl = -20 #20*(f)*np.sin(sup) #-10#
                l_special.append(j)
            else:
                xl = 0#20*(f)*np.cos(sup) #0
                yl = 20#20*(f)*np.sin(sup) #5#
            
            f += 0.05
            sup += 2*np.pi/len(rows_to_Annot)
            
            try:
                col_j = col[0][Label_rows[j]]
            except:
                try:
                    col_j = col[0]
                except:
                    print("Column categories are pink because you haven't entered an appropriate col parameter: it's either a tuple of 2 colors (one for all the row sand ones for all the cols) or a tuple of 2 dictionaries with labels as keys (one for all the rows and one for all the cols) ")
                    col_j = "green"
            
            try:
                ax = OneAnnotation(ax, int(Label_rows[j]), xy_rows[j, :], col_j, xl, yl, arrow = arrow)
            except:
                ax = OneAnnotation(ax, Label_rows[j], xy_rows[j, :], col_j, xl, yl, arrow = arrow)
        
    
    if cols_to_Annot is not None: #and Label_cols is not None:
        #pdist_n = scp.spatial.distance.pdist(xy_cols)
        #pdist_n[np.isnan(pdist_n)] = 10000000
        #pdist_n[~np.isfinite(pdist_n)] = 10000000
        
        #pdist = scp.spatial.distance.squareform(pdist_n)
        
        
        l_special = []
        for j in cols_to_Annot: 
            try:
                col_j = col[1][Label_cols[j]]
            except:
                try:
                    col_j = col[1]
                except:
                    print("Column categories are pink because you haven't entered an appropriate col parameter: it's either a tuple of 2 colors (one for all the row sand ones for all the cols) or a tuple of 2 dictionaries with labels as keys (one for all the rows and one for all the cols) ")
                    col_j = "pink"
    
            if (np.sum(pdist[j, (j+1):] <= 0.20*np.mean(pdist_n)) != 0)*(j not in l_special)*(j != xy_cols.shape[0]-1):
                xl =  0#0 + 20*(j+1) #0
                yl = -10#30 + 10*(j+1) #-10
                l_special.append(j)
                if Label_cols[j] in ["C2", "C4"]:
                    xl = 0#-40 #0
                    yl = 5#-40  #5
                ax = OneAnnotation(ax, Label_cols[j], xy_cols[j, :], col_j, xl, yl, arrow = False)

            else:
                xl =0 #-50#0
                yl =5 #50#5
                
                if Label_cols[j] in ["C8", "C10"]:
                    xl = 0 #0
                    yl = 5  #5
                ax = OneAnnotation(ax, Label_cols[j], xy_cols[j, :], col_j, xl, yl, arrow = False)
                  
            
    return ax
                
def Display(Coords_rows, Coords_cols, Inertia, Data, rows_to_Annot, cols_to_Annot, Label_rows, Label_cols, 
            markers, col, figtitle, outliers, dtp, chosenAxes = np.array([0, 1]), show_inertia = True, reverse_axis = False, 
            separate = False, center = None, model = None, ColName = None, RowName = None, lims = True, plotRows = True, plotCols = True,
            fig = None, ax = None, give_ax = False, with_ref = None, cut_dist = {"shift_orig":(False, False), "cut_all":False}, log_log = False):  
    """
    @brief: display results
    """                           

    # plot 2 components
    PreFig()    
    if len(chosenAxes) == 2:
        dim1, dim2 = Inertia[chosenAxes]
    xy_rows = Coords_rows[:, chosenAxes] 
    xy_cols = Coords_cols[:, chosenAxes]
    
    if center is not None:
        if separate:
            center = (center[0][chosenAxes], center[1][chosenAxes])
        else:
            if len(chosenAxes) == 2:
                center = center[chosenAxes] 
            else:
                center = [center[chosenAxes]]
    
    not_orig = np.arange(0, xy_rows.shape[0]+xy_cols.shape[0]+1, 1, dtype = int) != xy_rows.shape[0]
    if with_ref is not None:
        try:
            with_ref = with_ref[:, chosenAxes]
            with_ref_c = with_ref
            
        except:
            with_ref_c = None      
    
    # annotate points
    Rows_Labels = np.array([Label_rows[c] for c in Data.index], dtype = dtp[0])
    Cols_Labels = np.array([Label_cols[c] for c in Data.columns], dtype = dtp[1])
    
    if rows_to_Annot is not None:
        annot_rows = rows_to_Annot
        rows_to_Annot_index = []
        for s in range(len(annot_rows)):
            ind = np.where(Data.index == annot_rows[s])[0]
            if len(ind) >= 1: # should appear only one time
                rows_to_Annot_index = rows_to_Annot_index + list(ind)
    else:
        rows_to_Annot_index = None
                
    if cols_to_Annot is not None:
        annot_cols= cols_to_Annot
        cols_to_Annot_index = []
        for s in range(len(annot_cols)):
            ind = np.where(Data.columns == annot_cols[s])[0]
            if len(ind) >= 1: #  should appear only one time
                cols_to_Annot_index = cols_to_Annot_index + list(ind)
    else:
        cols_to_Annot_index = None
        
    if not separate:
        if fig == None:
            fig = pl.figure(figsize=(36+18,20+10))#pl.figure(figsize=(18,10))
        if ax == None:    
            ax = fig.add_subplot(2,1,1)
        
        if len(chosenAxes) == 1:
            xy_rows = np.concatenate((xy_rows, np.zeros(len(xy_rows))[:, np.newaxis]), axis = 1)
            xy_cols = np.concatenate((xy_cols, np.zeros(len(xy_cols))[:, np.newaxis]), axis = 1)
            rows_cols = np.concatenate((xy_rows[:, 0], np.array(center[0]), xy_cols[:, 0]))
            if with_ref is not None:
                with_ref_c = np.concatenate((rows_cols[:, np.newaxis], - np.abs(with_ref_c - center[0])), axis = 1) 
            
        if plotRows:
            try:
                # for coloring by cluster membership
                for j in range(xy_rows.shape[0]):
                    ax.scatter([xy_rows[j, 0]], [xy_rows[j, 1]], marker = markers[0][0], color = col[0][Rows_Labels[j]], s = markers[0][1])
            except:
                try:
                    col_cols = col[0]
                except:
                    print("Rows categories are green because you haven't entered an appropriate col parameter: it's either a tuple of 2 colors (one for all the row sand ones for all the cols) or a tuple of 2 dictionaries with labels as keys (one for all the rows and one for all the cols) ")
                    col_cols = "green"
                ax.scatter(xy_rows[:, 0], xy_rows[:, 1], marker = markers[0][0], color = col[0], s = markers[0][1], label= RowName)
            ax = Annotate(ax, rows_to_Annot_index, None, Rows_Labels, Cols_Labels, xy_rows, xy_cols, col, arrow = True)
            """
            if with_ref_c is not None:
                for rf in range(xy_rows.shape[0]):
                    pl.plot([xy_rows[rf, 0],with_ref_c[rf, 0]], [xy_rows[rf, 1], with_ref_c[rf, 1]], color = col[0], linewidth = 1)
            """
                
        
        if plotCols:
            try:
                # for coloring by cluster membership
                for j in range(xy_cols.shape[0]):
                    ax.scatter([xy_cols[j, 0]], [xy_cols[j, 1]], marker = markers[1][0], color = col[1][Cols_Labels[j]], s = markers[1][1])
            except:
                try:
                    col_cols = col[1]
                except:
                    print("Column categories are red because you haven't entered an appropriate col parameter: it's either a tuple of 2 colors (one for all the row sand ones for all the cols) or a tuple of 2 dictionaries with labels as keys (one for all the rows and one for all the cols) ")
                    col_cols = "red"
                
                ax.scatter(xy_cols[:, 0], xy_cols[:, 1], marker = markers[1][0], color = col_cols, s = markers[1][1], label= ColName)
            ax = Annotate(ax, None, cols_to_Annot_index, Rows_Labels, Cols_Labels, xy_rows, xy_cols, col, arrow = True)
            if with_ref is not None:
                for rc in range(xy_cols.shape[0]):
                    rf = xy_rows.shape[0] + 1 + rc
                    pl.plot([xy_cols[rc, 0],with_ref_c[rf, 0]], [xy_cols[rc, 1], with_ref_c[rf, 1]], color = col[1], linewidth = 1)
               
        
        if with_ref is not None:
            if plotRows and plotCols:
                ref_all = with_ref_c[not_orig, :]
                sort_all = np.argsort(ref_all[:, 0])
                ref_all[:, 0] = ref_all[sort_all, 0]
                ref_all[:, 1] = ref_all[sort_all, 1]
                pl.plot(ref_all[:, 0], ref_all[:, 1], color = "green", linewidth = 2)
            elif plotRows:
                ref_rows = with_ref_c[:xy_rows.shape[0], :]
                sortr = np.argsort(ref_rows[:, 0])
                ref_rows[:, 0] = ref_rows[sortr, 0]
                ref_rows[:, 1] = ref_rows[sortr, 1]
                pl.plot(ref_rows[:, 0], ref_rows[:, 1], color = col[0], linewidth = 2)
            elif plotCols:
                ref_cols = with_ref_c[xy_rows.shape[0]+1:, :]
                sortc = np.argsort(ref_cols[:, 0])
                ref_cols[:, 0] = ref_cols[sortc, 0]
                ref_cols[:, 1] = ref_cols[sortc, 1]
                pl.plot(ref_cols[:, 0], ref_cols[:, 1], color = col[1], linewidth = 2)
        
            if len(chosenAxes) == 1:
                not_orig = np.arange(0, with_ref_c.shape[0], 1, dtype = int) != xy_rows.shape[0]
                if plotRows and plotCols:
                    coords_1 = rows_cols[not_orig]
                elif plotRows:
                    coords = ref_rows
                elif plotCols:
                    coords_1 = ref_cols
                sort = np.argsort(coords_1)
                coords_1 = coords_1[sort]
                pl.plot(coords_1, -np.abs(coords_1 - center[0]), color = "orange", linewidth = 2)
            
        #ax.legend(loc= (1.05, 0))
        
         # label factor axis
        if show_inertia: # show percentage of inertia
            pl.xlabel("Dim %d (%.2f %%)"%(chosenAxes[0]+1, 100*dim1/np.sum(Inertia)), fontsize = 14)
            if len(chosenAxes) == 2:
                pl.ylabel("Dim %d (%.2f %%)"%(chosenAxes[1]+1, 100*dim2/np.sum(Inertia)), fontsize = 14)
            #pl.xlabel("Dim %d"%(chosenAxes[0]+1,), fontsize = 14)
            #pl.ylabel("Dim %d"%(chosenAxes[1]+1,), fontsize = 14)
        else:
            pl.xlabel("Dim %d"%(chosenAxes[0]+1,), fontsize = 14)
            if len(chosenAxes) == 2:
                pl.ylabel("Dim %d"%(chosenAxes[1]+1,), fontsize = 14)
            
        #ax = Separation_axis(ax, xy_rows, xy_cols, outliers, lims = True)
        if center is not None:
            #ax.plot([center[0]], [center[1]], "o", markersize = 10, color = "red", label = "I")
            
            if len(chosenAxes) == 2:
                ax.axvline(x = center[0], ls = "--", color = "black", linewidth =0.5)
                ax.axhline(y = center[1], ls = "--", color = "black", linewidth =0.5)
               
            if lims:
                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()
                
                # remove extreme outliers from the figure
                """
                if len(chosenAxes) == 2:
                    dist_xn = scp.spatial.distance.pdist(xy_rows)
                    dist_yn = scp.spatial.distance.pdist(xy_cols)
                    dist = np.concatenate((dist_xn, dist_yn))
                    iqr_dist = np.percentile(dist, 20, interpolation = "midpoint")
                    xmin, xmax = -iqr_dist, iqr_dist
                    ymin, ymax = -iqr_dist, iqr_dist
                elif len(chosenAxes) == 1:
                    xmin, xmax = -max(np.abs(rows_cols[rows_cols<0]))/2, max(np.abs(rows_cols[rows_cols<0]))/2
                    ymin, ymax = ymin, 1.5*ymax
                """
                pl.xlim((xmin, xmax))
                pl.ylim((ymin, ymax))
            
        #else:
        #    ax = Separation_axis(ax, xy_rows, xy_cols, outliers, lims = lims, col = "grey")
            
        # aspect ratio of axis
        #ax.set_aspect(1.0/(1.25*ax.get_data_ratio()), adjustable='box') # this can deforms the entire configuration, distances that are equal may look different and vice-versa
        
        if log_log:
            ax.set_yscale("symlog")
            ax.set_xscale("symlog")
        else:
            ax.set_aspect("equal")
        
        ax.set_xticks(())
        ax.set_yticks(())
        
        """
        if model is not None:
            if model["model"] == "x|y":
                ax.set_title("How present is %s ($X$) within variable %s ($Y$)? ($X|.$) \n How similar is the occurence of %s ($X$) among  %s ($Y$)? ($.|Y$) \n The distance between variables %s and %s \n shows the strengh of their relationships"%(ColName, RowName, ColName, RowName, ColName, RowName))
            elif model["model"] == "y|x":
                ax.set_title("How present is %s ($Y$) within variable %s ($X$) ? ($Y|.$) \n How similar is the occurence of %s ($Y$) among %s ($X$)? ($.|X$) \n The distance between variables %s and %s \n shows the strengh of their relationships"%(RowName, ColName, RowName, ColName, ColName, RowName))
            elif model["model"] == "ca_stand":
                ax.set_title("How similar is the occurence of %s ($Y$) among %s ($X$)? ($.|X$) \n How similar is the occurence of %s ($X$) among %s ($Y$)? ($.|Y$) \n The distance between variables %s and %s has no meaning?"%(RowName, ColName, ColName, RowName, ColName, RowName))
            elif model["model"] == "presence":
                ax.set_title("How present is %s ($Y$) within variable %s ($X$) ? ($Y|.$) \n How present is %s ($X$) within variables %s ($Y$)? ($X|.$) \n The distance between variables %s and %s \n shows the strengh of their relationships"%(RowName, ColName, ColName, RowName, ColName, RowName))
            elif model["model"] == "stand":
                ax.set_title("How similar is the occurence of %s ($Y$) among %s ($X$)? ($.|X$)"%(RowName, ColName)+"\n How similar is the occurence of %s ($X$) among  %s ($Y$)? ($.|Y$)"%(ColName, RowName)+ "\n The distance between variables %s and %s \n shows the strengh of their relationships"%(RowName, ColName))
                
            pl.suptitle(figtitle)
        else:
            pl.title(figtitle)
        """
        
        gs = None # just a placeholder
        
        ax.axis("off")
        
        pl.suptitle(figtitle)
        
    else:  # Separated
        fig = pl.figure(constrained_layout=True, figsize = (7, 9))
        gs = fig.add_gridspec(2, 2)
        pl.suptitle(figtitle)
        
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 0])
    
        ax1.scatter(xy_rows[:, 0], xy_rows[:, 1], marker = markers[0][0], color = col[0], s = markers[0][1])
        ax2.scatter(xy_cols[:, 0], xy_cols[:, 1], marker = markers[1][0], color = col[1], s = markers[1][1])
        
        xmin, xmax = 1.5*np.amin(xy_rows[:, 0]), 1.5*np.amax(xy_rows[:, 0])#ax1.get_xlim()
        ymin, ymax = 1.5*np.amin(xy_rows[:, 1]), 1.5*np.amax(xy_rows[:, 1]) #ax1.get_ylim()
        
        xmin2, xmax2 = 1.5*np.amin(xy_cols[:, 0]), 1.5*np.amax(xy_cols[:, 0]) #ax2.get_xlim()
        ymin2, ymax2 = 1.5*np.amin(xy_cols[:, 1]), 1.5*np.amax(xy_cols[:, 1])#ax2.get_ylim()
        
        try:
            ax1 = Annotate(ax1, rows_to_Annot_index, None, Rows_Labels, Cols_Labels, xy_rows, xy_cols, col, arrow = True)
            ax2 = Annotate(ax2, None, cols_to_Annot_index,  Rows_Labels, Cols_Labels, xy_rows, xy_cols, col, arrow = True)
        except:
            print("No annotations")

        #ax1.legend(loc= "best", fontsize = 10)
        #ax2.legend(loc= "best", fontsize = 10)
        if model is not None:
            if model["model"] == "x|y":
                ax1.set_title("Similarity within Y? ($.|Y$)") #How important is X for variable Y?
                ax2.set_title("$X|.$") #
            elif model["model"] == "y|x":
                ax1.set_title("$Y|.$")  #How important is Y for variable X?
                ax2.set_title("Similarity within X? ($.|X$)") 
            elif model["model"] == "presence":
                ax1.set_title("$Y|.$")  #How important is Y for variable X?
                ax2.set_title("$X|.$") # How similar is the occurence of Y within X? 
            else:
                 #ax1.set_title("Similarity within Y? ($.|Y$)") #How similar is the occurence of X within Y? 
                 #ax2.set_title("Similarity within X? ($.|X$)") #How similar is the occurence of Y within X?
                ax1.set_title("$.|Y$")  #How important is Y for variable X?
                ax2.set_title("$.|X$") 
        else:
            ax1.set_title(RowName)
            ax2.set_title(ColName)
        
        # label factor axis
        if show_inertia: # show percentage of inertia
            #pl.xlabel("Dim %d (%.2f %%)"%(chosenAxes[0]+1, 100*dim1/np.sum(Inertia)), fontsize = 14)
            #pl.ylabel("Dim %d (%.2f %%)"%(chosenAxes[1]+1, 100*dim2/np.sum(Inertia)), fontsize = 14)
            ax1.set_xlabel("Dim %d"%(chosenAxes[0]+1,), fontsize = 14)
            ax2.set_xlabel("Dim %d"%(chosenAxes[0]+1,), fontsize = 14)
            ax2.set_ylabel("Dim %d"%(chosenAxes[1]+1,), fontsize = 14)
        else:
            ax1.set_xlabel("Dim %d"%(chosenAxes[0]+1,), fontsize = 14)
            ax2.set_xlabel("Dim %d"%(chosenAxes[0]+1,), fontsize = 14)
            ax2.set_ylabel("Dim %d"%(chosenAxes[1]+1,), fontsize = 14)
            
        
        
        if center is not None: 
             ax1.axvline(x = center[0][0], ls = "--", color = "black", linewidth =0.5)
             ax1.axhline(y = center[0][1], ls = "--", color = "black", linewidth =0.5)
        
             ax2.axvline(x = center[1][0], ls = "--", color = "black", linewidth =0.5)
             ax2.axhline(y = center[1][1], ls = "--", color = "black", linewidth =0.5)
        else:
           ax1.axvline(x = 0, ls = "--", color = "black", linewidth =0.5)
           ax1.axhline(y = 0, ls = "--", color = "black", linewidth =0.5)
           
           ax2.axvline(x = 0, ls = "--", color = "black", linewidth =0.5)
           ax2.axhline(y = 0, ls = "--", color = "black", linewidth =0.5)
       
        # aspect ratio of axis
        #ax1.set_aspect(1.0/(1.25*ax1.get_data_ratio()), adjustable='box')
        #ax2.set_aspect(1.0/(1.25*ax2.get_data_ratio()), adjustable='box')
        
        ax1.set_aspect("equal")
        ax2.set_aspect("equal")
        
        ax1.set_xlim((xmin, xmax))
        ax1.set_ylim((ymin, ymax))
        
    
        ax2.set_xlim((xmin2, xmax2))
        ax2.set_ylim((ymin2, ymax2))
        
        ax1.set_xticks(())
        ax1.set_yticks(())
        
        ax2.set_xticks(())
        ax2.set_yticks(())
        
        ax1.axis("off")
        ax2.axis("off")
        pl.suptitle(figtitle)
      
    if give_ax == True:
        return fig, ax, xy_rows, xy_cols, gs, center
    else:         
        return fig, xy_rows, xy_cols, gs, center

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sklearn.manifold as sklm
import umap.umap_ as umap
import scipy as sp
rand = 0 # fixed initialization for reproducibility of UMAP and Kmeans
def umap_reducer(Coords, dim, np, min_dist):
    if min_dist == None:
        reducer = umap.UMAP(n_neighbors = np, metric = "euclidean", n_components = dim, random_state= rand) # n_neighbor = 2 (local structure) --- 200 (global structure, truncated when larger than dataset size)
    else:
        reducer = umap.UMAP(n_neighbors = np, min_dist = min_dist, metric = "euclidean", n_components = dim, random_state= rand)
    Emb_coords = reducer.fit_transform(Coords)
    return Emb_coords


def tSNE_reducer(Coords, dim, np, metric = "euclidean"):
    reducer = sklm.TSNE(n_components = dim, perplexity = np, random_state=rand, metric = "precomputed")
    DM = sp.spatial.distance.pdist(Coords, metric = metric)
    DM = sp.spatial.distance.squareform(DM)
    Emb_coords = reducer.fit_transform(DM)
    return Emb_coords

def dist_error(tXflat, D, dim):
    if dim>= 2:
        tD = sp.spatial.distance.pdist(tXflat.reshape((D.shape[0], dim)))
    else:
        tX = np.concatenate((tXflat[:, np.newaxis], np.zeros(len(tXflat))[:, np.newaxis]), axis = 1)
        tD = sp.spatial.distance.pdist(tX)
        
    tD = sp.spatial.distance.squareform(tD)
    T = tD - D
    return T.flatten()


def MDS_YH(Coords, dim, method = "LQ"): 
    '''
    @ brief          : embeding points onto a lower-dimensional Euclidean space based on least_square distane error minimization (Matrix Frobenius norm minimization)
    @ param Coords   : Coords, dim, method
    @ Coords         : Q by K array with the Coordinates on the rows (Q points to embed)
    @ dim            : Dimensions of the lower-dimensional manifold
    @ method         : minimization method 
                       OPTIONS "LQ" (scipy.optimize.least_squares)
    '''

    DistMat = sp.spatial.distance.squareform(sp.spatial.distance.pdist(Coords))
    arg_stack = (DistMat, dim)
    sol = sp.optimize.least_squares(dist_error, Coords[:, :dim].flatten(), bounds = (-np.inf, np.inf), args = arg_stack)
    Emb_coords = sol.x.reshape((Coords.shape[0], dim))
    
    return Emb_coords

def low_dim_coords(Coords, dim=2, method  = "umap", n_neighbors = 15, min_dist = None, scale = None, metric = "euclidean"):
    '''
    @ brief          : embeding of points onto a lower-dimensional manifold of using sklean.manifold
    @ param Coords   : Coords, dim, method
    @ Coords           : Q by K array with the Coordinates on the rows (Q points to embed)
    @ dim            : Dimensions of the lower-dimensional manifold
    @ method         : sklearn.manifold methods preserves the structure of the distances in the original data : 
                       OPTIONS "MDS" (respect well the distances), "Isomap" (preserves geodesic distances)
    '''
    if scale == "standard":
        scaled_coords = StandardScaler().fit_transform(Coords) 
    elif scale == "pca":
        scaled_coords = PCA(n_components = 5).fit_transform(Coords) 
    else:
        scaled_coords = Coords
    
    if method == "MDS":
        embedding = sklm.MDS(random_state = rand, n_components = dim, metric = True, dissimilarity = metric)
        Emb_coords = embedding.fit_transform(scaled_coords)
    
    elif method == "Isomap":
        DM = sp.spatial.distance.pdist(scaled_coords, metric = metric)
        DM = sp.spatial.distance.squareform(DM)
        embedding = sklm.Isomap(random_state = rand, n_components = dim, metric = "precomputed")#, metric = metric)
        Emb_coords = embedding.fit_transform(DM)
    
    elif method == "t-SNE":
        Emb_coords = tSNE_reducer(scaled_coords, dim, n_neighbors, metric = metric) 
        
    elif method == "umap":
        Emb_coords = umap_reducer(scaled_coords, dim, n_neighbors, min_dist)
    
    elif method == "PCA":
        Emb_coords = PCA(n_components = 5).fit_transform(Coords)
    else:
        Emb_coords = umap_reducer(scaled_coords, dim, n_neighbors, min_dist)
       
    return Emb_coords

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

def plotClass(Id_Class, X_vars, Y_vars, pdf, dtp, run_num = 0, n_neighbors = 2, min_dist = 0.99, method = "umap", 
                        scale = None, sub_fig_size = 7, cluster_colors = False, palette = "tab20", true_colors = None, markers = [("o",20),("o",20)],
                        show_labels = False, show_orig = True, metric = "euclidean", legend = True, place_holder = (0, 0), 
                        wrap_true = False, wrap_predicted = False, wrap_pred_params = (None, 1), oultiers_markers = ("o", "^", 5),  wrap_type = "convexhull",
                        def_pred_outliers = (2, 0.95),show_pred_outliers = False, group_annot_size = 15, dataname = None, hull_pred = False, points_hull = 5, group_color = None, alpha = 0.25,
                        shorten_annots = False, cut = (2, 2), connect_pred = False, true_labels_file = None):        
    """@brief Plot and Save class figures"""
    
    """Lower Dimensional visualization of clusters"""
    low_meth = method # methods: MDS, Isomap, TSNE
    if metric == "precomputed":
        DMat = Id_Class["DMat"]
        if method == "umap":
            sys.exit("umap, implemented here, is not designed to deal with metric = precomputed choose MDS, t-SNE, or Isomap ")
        Coords_manifold = low_dim_coords(DMat, dim = 2, method = low_meth, scale = scale, metric = metric)
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
        Cols_manifold = Coords_manifold[-N:, :]
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
    
    color_clustered = get_col_labs(Id_Class["Class_pred"], palette)
    if true_labels_file == None:
        true_colors = color_clustered
    else:
        try:
            true_labels = []
            if true_labels_file[-3:] == "csv":
                lab_file = pd.read_csv(true_labels_file)
            elif true_labels_file[-4:] == "xlsx":
                lab_file = pd.read_excel(true_labels_file, engine='openpyxl')
            else:
                lab_file = "None"
                
            varsXY = lab_file["variable"]
            vars_lab = lab_file["true labels"]
            for i1 in range(M):
                #true_labels.append(int(vars_lab[list(varsXY).index(X_vars[i1])]))
                true_labels.append(str(vars_lab[list(varsXY).index(X_vars[i1])]))
            for i1 in range(N):
                #true_labels.append(int(vars_lab[list(varsXY).index(Y_vars[i1])]))
                true_labels.append(str(vars_lab[list(varsXY).index(Y_vars[i1])]))
                    
            true_colors = get_col_labs(np.array(true_labels), palette)
            
        except:
            sys.exit("Please give the true label as a .csv or xlsx file with the categories of variable in one column (named: variable)  their true labels as integers in another column (named: true labels)")
    
    fig = plt.figure(figsize=(36+18,20+10))
    ax = fig.add_subplot(2,1,1)
    if connect_pred:
        pred_class = np.unique(Id_Class["Class_pred"])
        lab_point = False

        for i in range(len(pred_class)):
            class_row = Id_Class["Class_pred"][:M] == pred_class[i]
            class_col =  Id_Class["Class_pred"][-N:] == pred_class[i]
    
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
    
    if true_labels_file == None:
        col_rows = {rows_labels[X_vars[i]]:color_clustered[i] for i in range(M)}
        col_cols = {columns_labels[Y_vars[i]]:color_clustered[-N:][i] for i in range(N)}
    else:
        col_rows = {rows_labels[X_vars[i]]:true_colors[i] for i in range(M)}
        col_cols = {columns_labels[Y_vars[i]]:true_colors[-N:][i] for i in range(N)}
    
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
        
    if legend & (true_labels_file != None) :
         col_done = []
         for i in range(len(X_vars)):
             if str(true_colors[i]) not in col_done:
                 ax.scatter(Rows_manifold[i, 0], Rows_manifold[i, 1], marker = marker_to_use[0][0], s =  marker_to_use[0][1], color = true_colors[i], label = ""+ str(true_labels[i]))
                 col_done.append(str(true_colors[i]))
         col_done = []
         for i in range(len(Y_vars)):
            if (str(true_colors[-N:][i]) not in col_done):
                ax.scatter(Cols_manifold[i, 0], Cols_manifold[i, 1], marker = marker_to_use[1][0], s =  marker_to_use[1][1], color = true_colors[-N:][i], label = "" + str(true_labels[-N:][i]))
                col_done.append(str(true_colors[-N:][i]))
    
        
    if Connect_assoc in ("True", "TRUE", True):
        if not Assoc_file:
            if was_orig:
                Association = Id_Class["DMat"][:M, M+1:]
            else:
                Association = Id_Class["DMat"][:M, -N:]
        else:
            Association = rawData.to_numpy()

        X_0 = Rows_manifold
        Y_0 = Cols_manifold
        if Connect_center == "Y":
            for j in range(Association.shape[1]):
                locs = np.argsort(Association[:, j])[::-1]
                if not Assoc_file:
                    locs = locs[Association[locs, j]<Connect_thres]
                else:
                    locs = locs[Association[locs, j]>Connect_thres]
                # draw connecting lines
                for i in range(len(locs)):
                    #if Dist[locs[i], j]< Q1:
                    ax.plot([Y_0[j, 0], X_0[locs[i], 0]], [Y_0[j, 1],X_0[locs[i], 1]], color = col_cols[columns_labels[Y_vars[j]]], linewidth = 0.5)
        else:
            for j in range(Association.shape[0]):
                locs = np.argsort(Association[j, :])[::-1]
                if not Assoc_file:
                    locs = locs[Association[j, locs]<Connect_thres]
                else:
                    locs = locs[Association[j, locs]>Connect_thres]
                # draw connecting lines
                for i in range(len(locs)):
                    #if Dist[locs[i], j]< Q1:
                    ax.plot([X_0[j, 0], Y_0[locs[i], 0]], [X_0[j, 1],Y_0[locs[i], 1]], color = col_rows[rows_labels[X_vars[j]]], linewidth = 0.5)
      
    if wrap_true:
        pred_class = np.unique(Id_Class["Class_pred"])
        lab_point = False
        for i in range(len(pred_class)):
            
            class_row = Id_Class["Class_pred"][:M] == pred_class[i]
            class_col =  Id_Class["Class_pred"][-N:] == pred_class[i]

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
            class_col =  Id_Class["Class_pred"][-N:] == pred_class[i]
    
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
                        if hull_pred:
                            #Poly = Polygon(Vertices, edgecolor = "grey", fill = False, label = "predicted", linestyle = "-", linewidth = 1)
                            Poly = Polygon(Vertices, edgecolor = col_class, fill = False, label = "predicted %s"%(i+1), linestyle = "-", linewidth = wrap_pred_params[1])
                            Poly2 = Polygon(Vertices, facecolor = col_class, fill = True, label = "predicted %s"%(i+1), linestyle = "-", linewidth = wrap_pred_params[1], alpha=0.3)
                            ax.add_patch(copy(Poly))
                    else:
                        #Poly = Polygon(Vertices, edgecolor = col_class, fill = False, linestyle = "-", linewidth = 1)
                        if hull_pred:
                            Poly = Polygon(Vertices, edgecolor = col_class, fill = False, label = "predicted %s"%(i+1), linestyle = "-", linewidth = wrap_pred_params[1])
                            Poly2 = Polygon(Vertices, facecolor = col_class, fill = True, label = "predicted %s"%(i+1), linestyle = "-", linewidth = wrap_pred_params[1], alpha=0.3) 
                            ax.add_patch(copy(Poly))
                    if hull_pred:
                        ax.add_patch(copy(Poly2))
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
        
    
    if legend:
        plt.legend(loc = (0, 1.1), ncol = 2*num_clust % 10)     
                
    """
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)      
    """
    return fig, ax
    
         
#import re
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
                        scale = None, sub_fig_size = 7, cluster_colors = False, true_colors = None, palette = "tab20", markers = [("o",20),("o",20)], markers_color = None,
                        show_labels = False, show_orig = True, metric = "euclidean", legend = True, place_holder = (0, 0), 
                        wrap_true = False, wrap_predicted = False, wrap_pred_params = (None, 1), oultiers_markers = ("o", "^", 5),  wrap_type = "convexhull",
                        def_pred_outliers = (2, 0.95),show_pred_outliers = False, group_annot_size = 15, dataname = None,
                        num_row_col = None, show_separation = False, hull_pred = False, points_hull = 5, group_color = None, alpha = 0.25, shorten_annots = False, cut = (2, 2), true_labels_file = None):        
    """@brief Plot and Save class figures"""
    
    """Lower Dimensional visualization of clusters"""
    low_meth = method # methods: MDS, Isomap, TSNE
    if metric == "precomputed":
        DMat = Id_Class["DMat"]
        if method == "umap":
            sys.exit("umap, implemented here, is not designed to deal with metric = precomputed choose MDS, t-SNE, or Isomap ")
        Coords_manifold = low_dim_coords(DMat, dim = 2, method = low_meth, scale = scale, metric = metric)
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
        Cols_manifold = Coords_manifold[-N:, :]
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
    
    color_clustered = get_col_labs(Id_Class["Class_pred"], palette)
    if true_labels_file == None:
        true_colors = color_clustered
    else:
        try:
            true_labels = []
            if true_labels_file[-3:] == "csv":
                lab_file = pd.read_csv(true_labels_file)
            elif true_labels_file[-4:] == "xlsx":
                lab_file = pd.read_excel(true_labels_file, engine='openpyxl')
            else:
                lab_file = "None"
            
            varsXY = lab_file["variable"]
            vars_lab = lab_file["true labels"]
            for i1 in range(M):
                #true_labels.append(int(vars_lab[list(varsXY).index(X_vars[i1])]))
                true_labels.append(str(vars_lab[list(varsXY).index(X_vars[i1])]))
            for i1 in range(N):
                #true_labels.append(int(vars_lab[list(varsXY).index(Y_vars[i1])]))
                true_labels.append(str(vars_lab[list(varsXY).index(Y_vars[i1])]))
                    
            true_colors = get_col_labs(np.array(true_labels), palette)
        except:
            sys.exit("Please give the true label as a .csv or xlsx file with the categories of variable in one column (named: variable)  their true labels as integers in another column (named: true labels)")
    

    ColName = None
    RowName = None
    
    if true_labels_file == None:
        col_rows = {rows_labels[X_vars[i]]:color_clustered[i] for i in range(M)}
        col_cols = {columns_labels[Y_vars[i]]:color_clustered[-N:][i] for i in range(N)}
    else:
        col_rows = {rows_labels[X_vars[i]]:true_colors[i] for i in range(M)}
        col_cols = {columns_labels[Y_vars[i]]:true_colors[-N:][i] for i in range(N)}
        
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
        class_col_sub =  Id_Class["Class_pred"][-N:] == unique_classe[i]
        
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
        
        
        if legend & (true_labels_file != None) :
             col_done = []
             true_colors_x = np.array(true_colors)[:M][class_row_sub]
             true_labels_x = np.array(true_labels)[:M][class_row_sub]
             true_colors_y = np.array(true_colors)[-N:][class_col_sub]
             true_labels_y = np.array(true_labels)[-N:][class_col_sub]
             for i in range(len(X_vars_sub)):
                 if str(true_colors_x[i]) not in col_done:
                     ax.scatter(coords_row_sub[i, 0], coords_row_sub[i, 1], marker = marker_to_use[0][0], s =  marker_to_use[0][1], color = true_colors_x[i], label = ""+ str(true_labels_x[i]))
                     col_done.append(str(true_colors_x[i]))
    		
             for i in range(len(Y_vars_sub)):
                if str(true_colors_y[i]) not in col_done:
                    ax.scatter(coords_col_sub[i, 0], coords_col_sub[i, 1], marker = marker_to_use[1][0], s =  marker_to_use[1][1], color = true_colors_y[i], label = "" + str(true_labels_y[i]))
                    col_done.append(str(true_colors_y[i]))
         
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
                                ax.add_patch(copy(Poly))
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
                            if hull_pred:
                                Poly = Polygon(Vertices, edgecolor = col_class, fill = False, label = "predicted %s"%(i+1), linestyle = "-", linewidth = wrap_pred_params[1])
                                Poly2 = Polygon(Vertices, facecolor = col_class, fill = True, label = "predicted %s"%(i+1), linestyle = "-", linewidth = wrap_pred_params[1], alpha=0.3)
                                ax.add_patch(copy(Poly))
                        else:
                            #Poly = Polygon(Vertices, edgecolor = col_class, fill = False, linestyle = "-", linewidth = 1)
                            Poly = Polygon(Vertices, edgecolor = col_class, fill = False, label = "predicted %s"%(i+1), linestyle = "-", linewidth = wrap_pred_params[1])
                            if hull_pred:
                                Poly2 = Polygon(Vertices, facecolor = col_class, fill = True, label = "predicted %s"%(i+1), linestyle = "-", linewidth = wrap_pred_params[1], alpha=0.3) 
                                ax.add_patch(copy(Poly))
                        if hull_pred:
                            ax.add_patch(copy(Poly2))
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
                    
        if legend:
            plt.legend(loc = (0, 1.1), ncol = 2*num_clust % 10)     
    
        
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
                        scale = None, metric = "euclidean", sub_fig_size = 7, num_row_col = None, palette="tab20", cluster_colors = False, true_colors = None, markers = [("o",10),("o",10)], 
                        show_labels = False, show_orig = False, show_separation = False, legend = True, shorten_annots = False, dataname = None, cut = (2, 2), true_labels_file = None):   
    """@brief Plot and Save class figures"""
    
    Coords = Id_Class["Coords"]
    """Lower Dimensional visualization of clusters"""
    low_meth = method # methods: MDS, Isomap, TSNE
    if metric == "precomputed":
        DMat = Id_Class["DMat"]
        if method == "umap":
            sys.exit("umap, implemented here, is not designed to deal with metric = precomputed choose MDS, t-SNE, or Isomap ")
        Coords_manifold = low_dim_coords(DMat, dim = 2, method = low_meth, scale = scale, metric = metric)
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
        Cols_manifold = Coords_manifold[-N:, :]
        
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
       
    color_clustered = get_col_labs(Id_Class["Class_pred"], palette)
    if true_labels_file == None:
        true_colors = color_clustered
    else:
        try:
            true_labels = []
            if true_labels_file[-3:] == "csv":
                lab_file = pd.read_csv(true_labels_file)
            elif true_labels_file[-4:] == "xlsx":
                lab_file = pd.read_excel(true_labels_file, engine='openpyxl')
            else:
                lab_file = "None"
            
            varsXY = lab_file["variable"]
            vars_lab = lab_file["true labels"]
            for i1 in range(M):
                #true_labels.append(int(vars_lab[list(varsXY).index(X_vars[i1])]))
                true_labels.append(str(vars_lab[list(varsXY).index(X_vars[i1])]))
            for i1 in range(N):
                #true_labels.append(int(vars_lab[list(varsXY).index(Y_vars[i1])]))
                true_labels.append(str(vars_lab[list(varsXY).index(Y_vars[i1])]))
                    
            true_colors = get_col_labs(np.array(true_labels), palette)
        except:
            sys.exit("Please give the true label as a .csv or xlsx file with the categories of variable in one column (named: variable)  their true labels as integers in another column (named: true labels)")
    
    
    ColName = None
    RowName = None
    #pdb.set_trace()
    if true_labels_file == None:
        col_rows = {rows_labels[X_vars[i]]:color_clustered[i] for i in range(M)}
        col_cols = {columns_labels[Y_vars[i]]:color_clustered[-N:][i] for i in range(N)}
    else:
        col_rows = {rows_labels[X_vars[i]]:true_colors[i] for i in range(M)}
        col_cols = {columns_labels[Y_vars[i]]:true_colors[-N:][i] for i in range(N)}
    
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
        class_col =  Id_Class["Class_pred"][-N:] == unique_classe[i]
        
        coords_row = Rows_manifold[class_row, :]
        coords_col = Cols_manifold[class_col, :]
        
        X_vars_sub = np.array(X_vars)[class_row]
        Y_vars_sub = np.array(Y_vars)[class_col]
        
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
        
        if legend & (true_labels_file != None):
             col_done = []
             true_colors_x = np.array(true_colors)[:M][class_row]
             true_labels_x = np.array(true_labels)[:M][class_row]
             true_colors_y = np.array(true_colors)[-N:][class_col]
             true_labels_y = np.array(true_labels)[-N:][class_col]
             for i in range(len(X_vars_sub)):
                 if str(true_colors_x[i]) not in col_done:
                     ax.scatter(coords_row[i, 0], coords_row[i, 1], marker = marker_to_use[0][0], s =  marker_to_use[0][1], color = true_colors_x[i], label = ""+ str(true_labels_x[i]))
                     col_done.append(str(true_colors_x[i]))
    		
             for i in range(len(Y_vars_sub)):
                if str(true_colors_y[i]) not in col_done:
                    ax.scatter(coords_col[i, 0], coords_col[i, 1], marker = marker_to_use[1][0], s =  marker_to_use[1][1], color = true_colors_y[i], label = "" + str(true_labels_y[i]))
                    col_done.append(str(true_colors_y[i]))
                    
             plt.legend(loc = (0, 1.1), ncol = 2*num_clust % 10)           
        if show_separation:
            ax.axis("on")
            ax.margins(0.1, 0.1)
            
        else:
            ax.axis("off")
        
        
    return fig, ax



"""
2-Dimensional visualization of clusters (UMAP visualization) 
- all predicted classes 
- colored predicted classes or true classes 
"""
from matplotlib.backends.backend_pdf import PdfPages
import pickle
res_file=open(sys.argv[1], "rb")
Id_Class = pickle.load(res_file) 
res_file.close()
X_vars = Id_Class["X_vars"]
Y_vars = Id_Class["Y_vars"]
num_clust = len(np.unique(Id_Class["Class_pred"]))

fig_method = str(sys.argv[2])
n_neighbors = int(sys.argv[3])
palette = str(sys.argv[4])
min_dist = float(sys.argv[5])

if not Connect_assoc:
    to_save = "/"+ fig_method + "_One_Panel"
else:
    to_save = "/"+ fig_method + "_One_Panel_Connected"
    
pdf= PdfPages(str(sys.argv[6])+to_save+".pdf")

if str(sys.argv[7]) == "TRUE" or str(sys.argv[7]) == "True":
    show_labels = True
else:
    show_labels = False

if str(sys.argv[8]) == "TRUE" or str(sys.argv[8]) == "True":
    hull_pred = True
else:
    hull_pred = False

import os
true_labels_file = str(sys.argv[-1])
if not os.path.exists(true_labels_file):
    true_labels_file = None
    
markers = [(str(sys.argv[11]), int(sys.argv[9])), (str(sys.argv[12]), int(sys.argv[10]))]
num_col=int(sys.argv[13])    

dtp = (str, str)
fig, ax = plotClass(Id_Class, X_vars, Y_vars, pdf, dtp,
          run_num = 1, n_neighbors = n_neighbors, min_dist = min_dist, 
          method = fig_method, 
          scale = False, # scale = "pca", "standard", anything esle is taken as no scaling 
          palette = palette,
          cluster_colors = True, # chosed_color: if False, true_colors bellow must be given 
          true_colors = None,# give a true class colors as dictionary with X_vars and Y_vars as key
          hull_pred = hull_pred,
          markers = markers, # optional markers list and their size for X and Y
          show_labels = show_labels, # optional show the labels of X and Y
          show_orig = False, #optional show the the axis lines going through embedded origin 
          legend = True, # add legend only if true cluster are required
          wrap_true = False, # wrapp the members of a true cluster , in each indentified clusters
          group_annot_size = 15, ### size of the annotations in the center of polygones‚
          wrap_predicted = True, # full lines to wrap around the predicted cluster
          show_pred_outliers = False, #
          def_pred_outliers = (3, 0.95), # (a, b), greater than a*std of pairwise dist for more than b*100% of the points in the predicted class
          oultiers_markers = ("P", "^", 5), # (true, predicted, size)
          wrap_type = "convexhull", # convexhull or ellipse (ellipse does not look accurate)
          dataname = "Dist", # true cluster markers for this simulation
          true_labels_file = true_labels_file) ### true cluster labels file if given


pdf.savefig(fig, bbox_inches = "tight")
pdf.close()
plt.savefig(str(sys.argv[6])+to_save+".svg", bbox_inches='tight')


if not Connect_assoc:
    """2-Dimensional visualization visualization of clusters (UMAP visualization) 
    - separated predicted classes 
    - wrapped true classes
    """
    pdf2= PdfPages(str(sys.argv[6])+"/"+ fig_method + "_Separate_Panels.pdf")
    fig2, ax = plotClass_separated(Id_Class, X_vars, Y_vars, pdf, dtp,
              run_num = 1, n_neighbors = n_neighbors, min_dist = min_dist, 
              method = fig_method, 
              scale = False, # scale = "pca", "standard", anything esle is taken as no scaling 
              cluster_colors = True, # chosed_color: if False, true_colors bellow must be given 
              true_colors = None,
              palette = palette,
              markers = markers, # optional markers list and their size for X and Y
              show_labels = show_labels, # optional show the labels of X and Y
              show_orig = False, #optional show the the axis lines going through embedded origin 
              legend = True, # add legend only if true cluster are required
              wrap_true = False, # wrapp the members of a true cluster , in each indentified clusters
              hull_pred = hull_pred,
              group_annot_size = 35, ### size of the annotations in the center of polygones‚
              group_color = "black", ### color of cluster annotatations (if None then true colors)
              wrap_predicted = True, # full lines to wrap around the predicted cluster (excluding some outliers)
              #wrap_pred_params = ("black", 3), ### optional for pred wrap (color, linewidth)
              show_pred_outliers = False, 
              def_pred_outliers = (3.25, 0.75), # (a, b), greater than a*std of pairwise dist for more than b*100% of the points in the predicted class
              oultiers_markers = ("P", "^", 15), # (true, predicted, size)
              wrap_type = "convexhull", # convexhull or ellipse (ellipse does not look accurate)
              points_hull = 3, ## threshold for connecting points in convex hull
              dataname = "Dist",# true cluster markers for this simulation
              show_separation = True, ### show axis to clearly separate all predicted clusters
              num_row_col = (int(np.ceil(num_clust/num_col)), num_col),
              alpha = 0.25,
              true_labels_file = true_labels_file) ### true cluster labels file if given
    
    pdf2.savefig(fig2, bbox_inches = "tight")
    plt.savefig(str(sys.argv[6])+"/"+ fig_method + "_Separate_Panels_p1.svg", bbox_inches='tight')
    """2-Dimensional visualization visualization of clusters (UMAP visualization) 
    - separated predicted classes 
    - colored true classes
    """
    fig, ax = plotClass_separated_ver0(Id_Class, X_vars, Y_vars, pdf, dtp,
              run_num = 1, n_neighbors = n_neighbors, min_dist = min_dist, 
              method = fig_method,
              palette = palette,
              scale = False, # scale = "pca", "standard", anything esle is taken as no scaling 
              cluster_colors = True, # chosed_color: if False, true_colors bellow must be given 
              true_colors = None, # give a true class colors as dictionary with X_vars and Y_vars as key
              legend = True,
              markers = markers, # optional markers list and their size for X and Y
              sub_fig_size = 10, # optional sub figure size (as a square)
              show_labels = show_labels, # optional show the labels of X and Y
              show_orig = False, # optional show the the axis lines going through origin 
              show_separation = True, # optional separate all subfigs
              num_row_col = (int(np.ceil(num_clust/num_col)), num_col),  # number of subfigs in row and col
              dataname = "Dist",# true cluster markers for this simulation
              true_labels_file = true_labels_file) ### true cluster labels file if given
    
    pdf2.savefig(fig, bbox_inches = "tight")    
    pdf2.close()
    plt.savefig(str(sys.argv[6])+"/"+ fig_method + "_Separate_Panels_p2.svg", bbox_inches='tight')

#status_df=pd.DataFrame({"plot_one":"ok", "plot_separate":"ok"}, index=[1,2])
#status_df.to_csv(str(sys.argv[6])+"/status.csv")
