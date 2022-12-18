#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 13:21:27 2022

@author: raharinirina
"""
from Methods.classify import Class_Identification
from Methods.figure_settings import Display

from sklearn.metrics import rand_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb


def generate_data():
    """Generate Artificial data"""
    per_spl = 200 # # Num iid per samplea
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
    
    class_dic = {}
    
    labs = np.cumsum(np.ones(4*3)) - 1
    
    k = 0
    for i in range(3):
        lab = labs[k:k+4]
        for j in range(num_spl):
            data_dic[class1[i]+"%d"%(j+1)] = np.random.normal(val1[i][0], val1[i][1], size = per_spl)
            data_dic[class2[i]+"%d"%(j+1)] = np.random.uniform(val2[i][0], val2[i][1], size = per_spl)
            data_dic[class3[i]+"%d"%(j+1)] = np.random.pareto(val3[i], size = per_spl)
            data_dic[class4[i]+"%d"%(j+1)] = np.random.poisson(val4[i], size = per_spl)
            
            class_dic[class1[i]+"%d"%(j+1)] = lab[0]
            class_dic[class2[i]+"%d"%(j+1)] = lab[1]
            class_dic[class3[i]+"%d"%(j+1)] = lab[2]
            class_dic[class4[i]+"%d"%(j+1)] = lab[3]
        
        k += 4    
    
    return data_dic, class_dic

def Classify_test(data_dic, class_dic, run_num, pdf, plotfig = True):
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
    num_clust = 4*3
    Id_Class = Class_Identification(X, Y, num_clust, dist_origin = Orow*Ocols)
    Coords = Id_Class["Coords"]
    
    """Compute accuracy metric = rand_index metric"""
    Class_pred = Id_Class["Class_pred"]
    acc_metric = rand_score(Class_True, Class_pred)
    
    """Lower Dimensional visualization of clusters"""
    from Methods.Core.Lower_dim import low_dim_coords
    nb = 2 ### 
    low_meth = "umap" # or sklearn.manifols methods: MDS, Isomap, 
    md = 0.99
    Coords_manifold = low_dim_coords(Coords, dim=2, method  = low_meth, n_neighbors = nb, min_dist = md) 
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
        Rows_manifold = Coords_manifold[:M, :]
        Cols_manifold = Coords_manifold[M:, :]
        Origin_manifold = np.zeros(Coords_manifold.shape[1])
    
    """Plot and Save figure"""
    if plotfig:
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
                                                     dtp = ("<U4", "<U4"), # checked from printing variable samples line 56
                                                     chosenAxes = np.array([0,1]), 
                                                     show_inertia = False, 
                                                     model={"model":"stand"}, 
                                                     ColName = ColName, 
                                                     RowName = RowName,
                                                     lims = False) # crop fig
        
        pdf.savefig(fig, bbox_inches = "tight")


    return acc_metric

if __name__ == "__main__":
    from matplotlib.backends.backend_pdf import PdfPages
    pdf= PdfPages("Figures/fig_Class_test.pdf")
    repeat = 100
    acc_metric = []
    for r in range(repeat):
        data_dic, class_dic = generate_data()
        if r < 10:
            acc_metric.append(Classify_test(data_dic, class_dic, r, pdf, plotfig = True))
        else:
            acc_metric.append(Classify_test(data_dic, class_dic, r, pdf, plotfig = False))
    acc_metric = np.array(acc_metric)
    print("Accuracy: mean:%.2f, std:%.2f"%(np.mean(acc_metric), np.std(acc_metric)))    
    #print("Accuracy:%.2f %%"%(acc_metric[0]*100))
    pdf.close()
    #plt.show()





    
    