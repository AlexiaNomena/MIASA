#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 10:05:12 2023

@author: raharinirina
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import mannwhitneyu as MW_test
import pdb

#### Visualisation ###  
def PreFig(xsize = 12, ysize = 12):
    '''
    @brief: customize figure parameters
    '''
    matplotlib.rc('xtick', labelsize=xsize) 
    matplotlib.rc('ytick', labelsize=ysize)

def BarPlotClass(data, method_name, ax, fig, vert = True, labX = True, labY = True, stat_lim = None, stat_ticks = "Default", whis = 1.5, stat_name = None):
    data_list = []
    colors = []
    for i in range(len(data)):
        data_list.append(data[i][:])
            
        if method_name[i][:5] == "MIASA":
            colors.append((0.745, 0.498, 0.678))
        else:
            colors.append((0.314, 0.631, 0.384))

    bplot = ax.boxplot(data_list, notch = False, vert=vert, patch_artist = True, whis = whis, widths = .5, showfliers=False) # showfliers = False remove outliers
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    
    if vert:
        if labX:
            plt.xticks(np.cumsum(np.ones(len(method_name))), method_name, rotation = 90)
        if labY:
            plt.ylabel(stat_name, fontsize = 20)
        if stat_lim is not None:
            ax.set_ylim(stat_lim)

        if (stat_ticks is not None)&(stat_ticks != "Default"):
            ax.set_yticks(stat_ticks)
        elif stat_ticks is None:
            ax.set_yticks(())
            
            
    else:      
        if labY:
            plt.yticks(np.cumsum(np.ones(len(method_name))), method_name)
        if labX:
            plt.xlabel(stat_name, fontsize = 20)
        if stat_lim is not None:
            ax.set_xlim(stat_lim)
        
        if (stat_ticks is not None)&(stat_ticks !="Default"):
            ax.set_xticks(stat_ticks)
        elif stat_ticks is None:
            ax.set_xticks(())
        
    
    return fig

import pandas as pd
def pairwise_MW(X, Y, method_nameX, method_nameY, typeEs = "Kerby", snf_color = "yellow"):
    p_dic = {}
    U_dic = {}
    E_dic = {}
    Full_dic = {}
    index_list = []
    Col = np.tile(np.array(["white"]*(len(Y))), ((len(X)), 1))
    Col = np.array(Col,  dtype = "<U6")
    for i in range(len(X)):
        key = method_nameX[i][:5]+"_%d"%(i+1)
        p_dic[key] = []
        U_dic[key] = []
        E_dic[key] = []
        Full_dic[key] = []
        
        for j in range(len(Y)):
            if i == 0:
                index_list.append(method_nameY[j][:5]+"_%d"%(j+1))
                
            cond1 = method_nameX[i][:5] != method_nameY[j][:5]
            cond2 = method_nameX[i][:6] != method_nameY[j][:6]
            
            if (cond1 and cond2) or ((method_nameX[i][:5] == "MIASA") and (i!=j) and (j+1 in (1, 3))):
                U, pval = MW_test(X[i], Y[j], alternative = "greater")
    
                p_dic[key].append("%.3f"%pval)
                U_dic[key].append("%.3f"%U)
                
                n1, n2 = len(X[i]),len(Y[j])
                num_pairs = n1*n2
                
                if typeEs == "Wendt":
                    """ Wendt definition of MW Effect Size """
                    Uc, pval = MW_test(Y[j], X[i], alternative = "greater")
                    minU =  min(U, Uc) # it writen in Wendt publication that usually U for the MW test is the smallest between the U calculated for the first and U calculated second variable
                    mu = num_pairs/2
                    ES = (1 - minU/mu)
                elif typeEs == "Kerby":
                    """ Kerby MW Effect Size """
                    ties = np.sum(X[i][np.newaxis, :] == Y[j][:, np.newaxis])/num_pairs
                    f = (np.sum(X[i][np.newaxis, :] > Y[j][:, np.newaxis])/num_pairs) + 0.5*ties
                    u = (np.sum(X[i][np.newaxis, :] < Y[j][:, np.newaxis])/num_pairs) + 0.5*ties             
                    ES = (f - u)
                
                E_dic[key].append("%.3f"%ES) 
                Full_dic[key].append("p = %.3f, U = %d, r = %.3f"%(pval, U, ES))
                
                if pval < 0.01:
                    Col[j, i] = snf_color
              
            else:
                p_dic[key].append("--") 
                U_dic[key].append("--")
                E_dic[key].append("--")
                Full_dic[key].append("--,--,--")
            
    P = pd.DataFrame(p_dic, index = index_list)
    U = pd.DataFrame(U_dic, index = index_list)
    
    Eff_size = pd.DataFrame(E_dic, index = index_list)
    
    Full = pd.DataFrame(Full_dic, index = index_list)
    
    return P, U, Eff_size, Full, Col


if __name__ == "__main__": 
    var_data_list_labs = ["False", "True"]
    Fig_title = ("Fixed ssize", "Random ssize")
    
    """ Plot first method set """
    set_num_1 = 1
    save_at_1 = "Class_Data/meth_set_1/"
    repeat_1 = [1000, 300, 260, 240, 200]
    exclude_1 = ("MIASA-(eCDF, KS-stat)--Kmeans", "MIASA-(eCDF, KS-stat)--Kmedoids", "non_MD-(eCDF, KS-stat)--Kmedoids")
    exclude_1_b = ["MIASA-(eCDF, eCDF)--Kmeans", "MIASA-(eCDF, KS-p1)--Kmeans"]
    slim_1 = (0.5, 1.1) # range of statistic to show on final plot
    sticks_1 = np.arange(0.5, 1.1, 0.1)
    name_1 = "Dist."
    
    """ Plot second method set """
    set_num_2 = 2
    save_at_2 = "Class_Data/meth_set_2/"
    repeat_2 = [2005] 
    exclude_2 = ("MIASA-(Corr, Spearman_pval)--Kmeans", "MIASA-(Corr, Spearman_pval)--Kmedoids", "non_MD-(Corr, Spearman_pval)--Kmedoids")
    exclude_2_b = ["MIASA-(Corr, dCorr)--Kmeans", "MIASA-(Corr, Pearson_pval)--Kmeans"]
    slim_2 = (0.65, 0.74) # range of statistic to show on final plot
    sticks_2 = np.arange(0.65, 0.74+0.01, 0.01)
    name_2 = "Corr."
    
    """ Plot third method set"""
    set_num_3 = 3
    save_at_3 = "Class_Data/meth_set_3/"
    repeat_3 = [200, 201, 202, 203, 204, 205, 206, 207, 208, 209]
    exclude_3 = ("MIASA-(Corr, Granger-Cause-diff-params)--Kmeans", "MIASA-(Corr, Granger-Cause-diff-params)--Kmedoids", "non_MD-(Corr, Granger-Cause-diff-params)--Kmedoids")
    exclude_3_b = ["MIASA-(Corr, dCorr)--Kmeans", "MIASA-(Corr, Granger-Cause-diff-chi2)--Kmeans"]
    slim_3 = (0.52, 0.6)#0.64) # range of statistic to show on final plot
    sticks_3 = np.arange(0.52, 0.64+0.01, 0.02)
    name_3 = "GRN"
    
    ''' Separated & Together '''
    set_num_list = [set_num_1, set_num_2, set_num_3]
    save_at_list = [save_at_1, save_at_2, save_at_3]
    repeats = [repeat_1, repeat_2, repeat_3]
    exclude_list = [exclude_1, exclude_2, exclude_3]
    exclude_all = [exclude_1_b, exclude_2_b, exclude_3_b]
    stat_lim_list = [slim_1, slim_2, slim_3]
    sticks_list = [sticks_1, sticks_2, sticks_3]
    meth_list = [name_1, name_2, name_3]
    
    pdfb_all = PdfPages("Figures/Paper_Fig_RI.pdf")
    PreFig(xsize = 20, ysize = 20)
    fig_all = plt.figure(figsize = (20, 30))
    plt.subplots_adjust(bottom = 0.06, right = 0.95, left = 0.06, top = 0.90, wspace = 0.05, hspace = 0.05)
    
    pdfb_all_MW = PdfPages("Figures/Paper_MW_RI.pdf")
    PreFig(xsize = 20, ysize = 20)
    fig_all_MW_1 = plt.figure(figsize = (10, 15))
    
    k_all = 0
    for p in range(len(set_num_list)):
        set_num = set_num_list[p]
        save_at = save_at_list[p]
        repeat_list = repeats[p]
        exclude = exclude_list[p]
    
        PreFig(xsize = 16, ysize = 16)
        fig = plt.figure(figsize = (20, 10))
        k = 1
        for j in range(len(var_data_list_labs)):
            acc_dic = {}
            ax = fig.add_subplot(int("%d%d%d"%(1, 2, k)))
            ax.set_title("%s"%Fig_title[j])

            ax_all = fig_all.add_subplot(int("%d%d%d"%(3, 2, k_all+k)))
            ax_MW_1 = fig_all_MW_1.add_subplot(int("%d%d%d"%(6, 1, k_all+k)))

            for n in range(len(repeat_list)): 
                repeat = repeat_list[n]
                file = open(save_at + "Accuracy_set_%d_%d_varS%s.pck"%(set_num, repeat, var_data_list_labs[j]), "rb")
                AcData = pickle.load(file)
                file.close()
                
                acc_list_n, adjusted_acc_list_n, method_name_n = AcData["accuracy_list"], AcData["adjusted_accuracy_list"], AcData["method_name"]
                for i in range(len(method_name_n)):
                    meth = method_name_n[i]
                    if meth not in exclude:
                        try:
                            acc_dic[meth] = np.concatenate((acc_dic[meth], acc_list_n[i, :].compressed()))
                        except:
                            acc_dic[meth] = acc_list_n[i, :].compressed() ##  for mehods that is still not in dic ### normaly, by the way we run the simulations, this should only happen when n = 0
            
            method_name = list(acc_dic.keys())
            acc_list = [acc_dic[meth] for meth in method_name]
            labX, labY = True, True
            if k != 1:
                labY = False
            fig = BarPlotClass(acc_list, method_name, ax, fig, labX = labX, labY = labY, stat_name = "RI scores")

            
            acc_list_2 = []
            method_name_2 = []
            for i in range(len(method_name)):
                meth = method_name[i]
                if meth not in exclude_all[p]:
                    acc_list_2.append(acc_dic[meth])
                    method_name_2.append(meth)

            labX_all, labY_all = False, False
            
            vert = True
            if k_all+k in (1, 3, 5):
                fig_all = BarPlotClass(acc_list_2, method_name_2, ax_all, fig_all, vert, labX = labX_all, stat_lim = stat_lim_list[p], labY = labY_all, stat_ticks = sticks_list[p], whis = (20, 95), stat_name = "RI scores")
            else:
                fig_all = BarPlotClass(acc_list_2, method_name_2, ax_all, fig_all, vert, labX = labX_all, stat_lim = stat_lim_list[p], labY = labY_all, stat_ticks = None, whis = (20, 95), stat_name = "RI scores")
                if vert:
                    ax_all.set_yticks(sticks_list[p],[" " for s in range(len(sticks_list[p]))])
                else:
                    ax_all.set_xticks(sticks_list[p],[" " for s in range(len(sticks_list[p]))])
                    
            ticks = np.cumsum(np.ones(len(method_name_2)))
            if vert:
                ax_all.set_xticks(ticks, [" " for t in ticks])
            else:
                ax_all.set_yticks(ticks, [" " for t in ticks])
            
            P1, U1, Eff_size1, Full1, ColCell1 = pairwise_MW(acc_list_2, acc_list_2, method_name_2, method_name_2, typeEs = "Kerby", snf_color = "yellow")
            ax_MW_1.set_title("(%s) %s, H0: col = row. H1 col > row"%(meth_list[p], Fig_title[j]))
            pd.plotting.table(ax_MW_1, Full1, loc = "center", cellColours = ColCell1, colWidths = [1.5/len(method_name_2)]*len(method_name_2))            
            #pd.plotting.table(ax_MW_1, P1, loc = "upper center", colWidths = [0.75/len(method_name_2)]*len(method_name_2))
            #pd.plotting.table(ax_MW_1, U1, loc = "center", colWidths = [0.75/len(method_name_2)]*len(method_name_2))
            #pd.plotting.table(ax_MW_1, Eff_size1, loc = "lower center", colWidths = [0.75/len(method_name_2)]*len(method_name_2))
            ax_MW_1.axis("off")
                        
            k += 1
            
        pdfb = PdfPages("Figures/Paper_Fig_RI_%d_infos.pdf"%set_num)    
        pdfb.savefig(fig, bbox_inches = "tight")
        pdfb.close()
        
        k_all +=2
    
    pdfb_all.savefig(fig_all, bbox_inches = "tight")
    pdfb_all.close()
    
    pdfb_all_MW.savefig(fig_all_MW_1, bbox_inches = "tight")
    pdfb_all_MW.close()



"""Supposed to automatically draw significance information but it needs more work"""
#import matplotlib.lines as lines
"""
def test_infos(acc_list, step, col, ax, P, U, Eff_size):
    keys = list(P[0].keys()) 
    #arr_style = {'connectionstyle':'bar','arrowstyle':'-','shrinkA':20,'shrinkB':20,'linewidth':2, "color":"black"}
    y1list = [0]
    y2list = [0]
    for i in range(len(keys)):
        for j in range(len(keys)):
            for k in range(len(P)):
                if (i<j) and P[k][keys[i]][j]!= "--":
                    if float(P[k][keys[i]][j])<0.01:
                        if k == 0:
                            y = max(np.percentile(acc_list[i], 75, interpolation = "midpoint"), np.percentile(acc_list[j], 75, interpolation = "midpoint"))
                            s = 1
                            y1list.append(y)
                            y = max(y1list)
                        else:
                            y = min(np.percentile(acc_list[i], 25, interpolation = "midpoint"), np.percentile(acc_list[j], 25, interpolation = "midpoint"))
                            s = -1
                            y2list.append(y)
                            y = min(y2list[1:])
                            
                            
                        yloc = y+s*(step)
                        text = "p = %s, U = %d, r = %s"%(P[k][keys[i]][j], float(U[k][keys[i]][j]), Eff_size[k][keys[i]][j])
                        ax.annotate(text, xy=(j+1, yloc), bbox=dict(boxstyle='square', fc=col[k], alpha=0.25))
                        #ax.annotate("", xy=(i+1, yloc), xytext=((j+1), yloc), arrowprops=arr_style)
                        #ax.arrow(i+1, yloc, (j+1) - (i+1), 0, capstyle = 'round', linestyle = '-',linewidth=2, color ="black")
                        line = lines.Line2D([i+1, j+1], [yloc, yloc], lw = 2, color = "black", axes = ax)
                        ax.add_line(line)                                               
                
    return ax
"""    