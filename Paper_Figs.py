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
def pairwise_MW(acc_list, method_name):
    p1_dic = {}
    U1_dic = {}
    p2_dic = {}
    U2_dic = {}
    E1_dic = {}
    E2_dic = {}
    index_list = []
    for i in range(len(acc_list)):
        key = method_name[i][:5]+"_%d"%(i+1)
        p1_dic[key] = []
        U1_dic[key] = []
        p2_dic[key] = []
        U2_dic[key] = []
        E1_dic[key] = []
        E2_dic[key] = []
        
        for j in range(len(acc_list)):
            if i == 0:
                index_list.append(method_name[j][:5]+"_%d"%(j+1))
                
            cond1 = method_name[i][:5] != method_name[j][:5]
            cond2 = method_name[i][:6] != method_name[j][:6]
            
            if (cond1 and cond2) or ((method_name[i][:5] == "MIASA") and (i!=j) and (j+1 in (1, 3))):
                U1, pval1 = MW_test(acc_list[i], acc_list[j], alternative = "greater")
                U2, pval2 = MW_test(acc_list[j], acc_list[i], alternative = "greater")
                
                p1_dic[key].append("%.3f"%pval1)
                U1_dic[key].append("%.3f"%U1)
                
                p2_dic[key].append("%.3f"%pval2)
                U2_dic[key].append("%.3f"%U2)
                
                n1, n2 = len(acc_list[i]),len(acc_list[j])
                mu1 = (n1*n2)/2
                mu2 = (n1*n2)/2
                E1_dic[key].append("%.3f"%(1 - U1/mu1))
                E2_dic[key].append("%.3f"%(1 - U2/mu2))

            else:
                p1_dic[key].append("--")
                U1_dic[key].append("--")
                p2_dic[key].append("--")
                U2_dic[key].append("--")
                E1_dic[key].append("--")
                E2_dic[key].append("--")
            
    P1 = pd.DataFrame(p1_dic, index = index_list)
    U1 = pd.DataFrame(U1_dic, index = index_list)
    
    P2 = pd.DataFrame(p2_dic, index = index_list)
    U2 = pd.DataFrame(U2_dic, index = index_list)
    
    Eff_size1 = pd.DataFrame(E1_dic, index = index_list)
    Eff_size2 = pd.DataFrame(E2_dic, index = index_list)
    return P1, U1, P2, U2, Eff_size1, Eff_size2   
 

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
    repeat_3 = [200, 201, 202, 203, 204]
    exclude_3 = ("MIASA-(Corr, Granger-Cause-diff-params)--Kmeans", "MIASA-(Corr, Granger-Cause-diff-params)--Kmedoids", "non_MD-(Corr, Granger-Cause-diff-params)--Kmedoids")
    exclude_3_b = ["MIASA-(Corr, dCorr)--Kmeans", "MIASA-(Corr, Granger-Cause-diff-chi2)--Kmeans"]
    slim_3 = (0.52, 0.64) # range of statistic to show on final plot
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
    #plt.subplots_adjust(bottom = 0.06, right = 0.95, left = 0.06, top = 0.90, wspace = 0.05, hspace = 0.05)
    fig_all_MW_2 = plt.figure(figsize = (10, 15))
    #plt.subplots_adjust(bottom = 0.06, right = 0.95, left = 0.06, top = 0.90, wspace = 0.05, hspace = 0.05)
    
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
            ax_MW_1 = fig_all_MW_1.add_subplot(int("%d%d%d"%(3, 2, k_all+k)))
            ax_MW_2 = fig_all_MW_2.add_subplot(int("%d%d%d"%(3, 2, k_all+k)))

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
            
            P1, U1, P2, U2, Eff_size1, Eff_size2 = pairwise_MW(acc_list_2, method_name_2)
            ax_MW_1.set_title("(%s) %s, H0: col = row. H1 col > row"%(meth_list[p], Fig_title[j]))
            pd.plotting.table(ax_MW_1, P1, loc = "upper center", colWidths = [0.75/len(method_name_2)]*len(method_name_2), label = "p-value")
            pd.plotting.table(ax_MW_1, U1, loc = "center", colWidths = [0.75/len(method_name_2)]*len(method_name_2))
            pd.plotting.table(ax_MW_1, Eff_size1, loc = "lower center", colWidths = [0.75/len(method_name_2)]*len(method_name_2))
            ax_MW_1.axis("off")
            
            ax_MW_2.set_title("(%s) %s, H0: col = row. H1 col < row"%(meth_list[p], Fig_title[j]))
            pd.plotting.table(ax_MW_2, P2, loc = "upper center", colWidths = [0.75/len(method_name_2)]*len(method_name_2))
            pd.plotting.table(ax_MW_2, U2, loc = "center", colWidths = [0.75/len(method_name_2)]*len(method_name_2))
            pd.plotting.table(ax_MW_2, Eff_size2, loc = "lower center", colWidths = [0.75/len(method_name_2)]*len(method_name_2))
            
            ax_MW_2.axis("off")
            k += 1
            
        pdfb = PdfPages("Figures/Paper_Fig_RI_%d_infos.pdf"%set_num)    
        pdfb.savefig(fig, bbox_inches = "tight")
        pdfb.close()
        
        k_all +=2
    
    pdfb_all.savefig(fig_all, bbox_inches = "tight")
    pdfb_all.close()
    
    pdfb_all_MW.savefig(fig_all_MW_1, bbox_inches = "tight")
    pdfb_all_MW.savefig(fig_all_MW_2, bbox_inches = "tight")
    pdfb_all_MW.close()
        