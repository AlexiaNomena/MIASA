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
            #colors.append((0.745, 0.498, 0.678))
            #colors.append((0.804, 0.533, 0.686))
            colors.append((0.553, 0.812, 0.541))
        else:
            #colors.append((0.314, 0.631, 0.384))
            #colors.append((0.671, 0.867, 0.576))
            colors.append((1.00, 0.682, 0.667))
        
    bplot = ax.boxplot(data, notch = False, vert=vert, patch_artist = True, whis = whis, widths = .5, showfliers=False) # showfliers = False remove outliers
    
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    
    if vert:
        if labX:
            ax.set_xticks(np.cumsum(np.ones(len(method_name))), method_name, rotation = 90)
        if labY:
            plt.ylabel(stat_name, fontsize = 20)
        if stat_lim is not None:
            ax.set_ylim(stat_lim)

        if (stat_ticks is not None)&(stat_ticks != "Default"):
            ax.set_yticks(stat_ticks, fontsize = 40)
        elif stat_ticks is None:
            ax.set_yticks(())
            
            
    else:      
        if labY:
            ax.set_yticks(np.cumsum(np.ones(len(method_name))), method_name)
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



def Plot_ARI():
    var_data_list_labs = ["False"]
    Fig_title = ("Fixed sample size", "Random sample size")
    
    """ Plot first method set """
    set_num_1 = 1
    save_at_1 = "Class_Data/meth_set_1/"
    repeat_1 = [1000]
    select_1 = ("MIASA-(eCDF, KS-p1)--Agglomerative_ward", "MIASA-(eCDF, KS-p1)--Kmedoids", "non_MD-(eCDF, KS-p1)--Kmedoids")
    select_1_MW = select_1 ## included it MW test
    slim_1 = (-0.1, 0.8)# range of statistic to show on final plot
    sticks_1 = np.arange(slim_1[0], slim_1[1]+0.1, 0.1)
    name_1 = "Distribution"
    
    """ Plot second method set """
    set_num_2 = 2
    save_at_2 = "Class_Data/meth_set_2/"
    repeat_2 = [1000] 
    select_2 = ("MIASA-(eCDF, Pearson_pval)--Agglomerative_ward", "MIASA-(eCDF, Pearson_pval)--Kmedoids", "non_MD-(eCDF, Pearson_pval)--Kmedoids")
    select_2_MW = select_2
    slim_2 = (-0.1, 0.9) # (-0.01, 0.9) # range of statistic to show on final plot
    sticks_2 = np.arange(slim_2[0], slim_2[1]+0.1, 0.1)
    name_2 = "Correlation"
    
    """ Plot third method set"""
    set_num_3 = 3
    save_at_3 = "Class_Data/meth_set_3/"
    repeat_3 = [200, 201, 202, 203, 204]
    select_3 = ("MIASA-(Eucl, Granger-Cause-3diff-chi2)--Agglomerative_ward","MIASA-(Eucl, Granger-Cause-3diff-chi2)--Kmedoids", "non_MD-(Eucl, Granger-Cause-3diff-chi2)--Kmedoids")
    select_3_MW = select_3#["MIASA-(Corr, dCorr)--Kmeans", "MIASA-(Corr, Granger-Cause-diff-chi2)--Kmeans"]
    slim_3 = (-0.1, 0.7) #(-0.05, 0.7) # range of statistic to show on final plot
    sticks_3 = np.arange(slim_3[0], slim_3[1]+0.1, 0.1)
    name_3 = "GRN"
    
    ''' Pepare parameters for plotting Separated & Together '''
    set_num_list = [set_num_1, set_num_2, set_num_3]
    save_at_list = [save_at_1, save_at_2, save_at_3]
    repeats = [repeat_1, repeat_2, repeat_3]
    select_for_final_fig = [select_1, select_2, select_3]
    select_for_MW = [select_1_MW, select_2_MW, select_3_MW]
    stat_lim_list = [slim_1, slim_2, slim_3]
    sticks_list = [sticks_1, sticks_2, sticks_3]
    meth_list = [name_1, name_2, name_3]

    
    pdfb_all = PdfPages("Figures/Final/Paper_Fig_ARI.pdf")
    PreFig(xsize = 30, ysize = 30)
    #fig_all = plt.figure(figsize = (30, 20))
    fig_all = plt.figure(figsize = (15, 9))
    plt.subplots_adjust(bottom = 0.06, right = 0.95, left = 0.06, top = 0.90, wspace = 0.25, hspace = 0.1)
    
    pdfb_all_MW = PdfPages("Figures/Final/Paper_MW_ARI.pdf")
    PreFig(xsize = 30, ysize = 30)
    fig_all_MW_1 = plt.figure(figsize = (10, 35))
    
    k_all = 0
    for p in range(len(set_num_list)):
        set_num = set_num_list[p]
        save_at = save_at_list[p]
        repeat_list = repeats[p]
        select_list = select_for_final_fig[p]
    
        PreFig(xsize = 16, ysize = 16)
        fig = plt.figure(figsize = (7, 21))
        k = 1
        for j in range(len(var_data_list_labs)):
            acc_dic_all = {}
            ax = fig.add_subplot(int("%d%d%d"%(1, 2, j+1)))
            ax.set_title("%s"%Fig_title[j])

            ax_all = fig_all.add_subplot(int("%d%d%d"%(2, 3, k_all+k)))
            ax_MW_1 = fig_all_MW_1.add_subplot(int("%d%d%d"%(6, 1, k_all+k)))

            for n in range(len(repeat_list)): 
                repeat = repeat_list[n]
                file = open(save_at + "Accuracy_set_%d_%d_varS%s.pck"%(set_num, repeat, var_data_list_labs[j]), "rb")
                AcData = pickle.load(file)
                file.close()
                
                unused_acc_list_n, acc_list_n, method_name_n = AcData["accuracy_list"], AcData["adjusted_accuracy_list"], AcData["method_name"]
                
                for i in range(len(method_name_n)):
                    meth = method_name_n[i]
                    if n == 0:
                        acc_dic_all[meth] = acc_list_n[i, :].compressed()
                    else:
                        acc_dic_all[meth] = np.concatenate((acc_dic_all[meth], acc_list_n[i, :].compressed()))
                
            acc_list = []
            acc_list_2 = []
            method_name_all = list(acc_dic_all.keys())
            method_name = []
            method_name_2 = []
            
            for i in range(len(method_name_all)):
                meth = method_name_all[i]
                if len(select_list) != 0:
                    if meth in select_list:
                        acc_list.append(acc_dic_all[meth])
                        method_name.append(meth)
                else:
                    acc_list.append(acc_dic_all[meth])
                    method_name.append(meth)
                    
                if len(select_for_MW[p]) != 0:    
                    if meth in select_for_MW[p]:
                        acc_list_2.append(acc_dic_all[meth])
                        method_name_2.append(meth)
                else:
                    acc_list_2.append(acc_dic_all[meth])
                    method_name_2.append(meth)
            
            
            vert = True
            
            fig_all = BarPlotClass(acc_list, method_name, ax_all, fig_all, vert, labX = True, labY = True, stat_lim = stat_lim_list[p], stat_ticks = sticks_list[p], whis = (5, 95), stat_name = "ARI scores")
                
            """
            # if all methods have the same
            if vert:
                ax_all.set_yticks(sticks_list[p],[" " for s in range(len(sticks_list[p]))])
            else:
                ax_all.set_xticks(sticks_list[p],[" " for s in range(len(sticks_list[p]))])
            """
            
            ax_all.set_ylim(stat_lim_list[p])
            
            """
            # if all methods have the same y-x axis
            ax1 = ax_all.twinx()
            #if k == 3:
            ax1.set_xticks(sticks_list[p])
            ax1.set_ylim(stat_lim_list[p])
            #else:
            #ax1.set_xticks(())
            """
            
            
            P1, U1, Eff_size1, Full1, ColCell1 = pairwise_MW(acc_list_2, acc_list_2, method_name_2, method_name_2, typeEs = "Kerby", snf_color = "yellow")
            ax_MW_1.set_title("(%s) %s, H0: col = row. H1 col > row"%(meth_list[p], Fig_title[j]))
            pd.plotting.table(ax_MW_1, Full1, loc = "center", cellColours = ColCell1, colWidths = [1/len(method_name_2)]*len(method_name_2))            
            #pd.plotting.table(ax_MW_1, P1, loc = "upper center", colWidths = [0.75/len(method_name_2)]*len(method_name_2))
            #pd.plotting.table(ax_MW_1, U1, loc = "center", colWidths = [0.75/len(method_name_2)]*len(method_name_2))
            #pd.plotting.table(ax_MW_1, Eff_size1, loc = "lower center", colWidths = [0.75/len(method_name_2)]*len(method_name_2))
            ax_MW_1.axis("off")
                        
            """Plot all together"""
            fig = BarPlotClass(acc_list, method_name, ax, fig, vert, labX = True, labY = True, stat_lim = stat_lim_list[p], stat_ticks = sticks_list[p], whis = (5, 95), stat_name = "ARI scores")
            ax.axhline(y = 0, xmin = 0, xmax = len(method_name), ls = "--", linewidth = 1, color = "grey")
            
            ax_all.axhline(y = 0, xmin = 0, xmax = len(method_name), ls = "--", linewidth = 1, color = "grey")
            
            k += 3
            
        pdfb = PdfPages("Figures/Final/Paper_Fig_ARI_%d_infos.pdf"%set_num)    
        pdfb.savefig(fig, bbox_inches = "tight")
        pdfb.close()
        
        
        k_all +=1
    
    pdfb_all.savefig(fig_all, bbox_inches = "tight")
    pdfb_all.close()
    
    pdfb_all_MW.savefig(fig_all_MW_1, bbox_inches = "tight")
    pdfb_all_MW.close()


if __name__ == "__main__":
    Plot_ARI()

