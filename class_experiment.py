#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 13:21:27 2022

@author: raharinirina
"""
from Methods.classify import repeated_classifications, plotClass, BarPlotClass
from Methods.simulate_class_data import generate_data_dist
import pdb
from matplotlib.backends.backend_pdf import PdfPages


""" Classification experiments for different data types """
repeat = 10000 # Number of replicates of each experiments
var_data_list = [False, True]
var_data_list_labs = ["False", "True"]

""" first methods set: Saved/meth_set_1/"""
#set_num = 1
#classifiers = ["MIASA", "MIASA", "non_MD"]#,["MIASA"]#, non_MD = "Non_Metric_Distance"]
#clust_methods = ["Kmeans", "Kmedoids", "Kmedoids"] # MIASA uses preferably metric-based clust method (e.g. K-means) and "non_MD" uses only non-metric-distance clust method (e.g. K-medoids)

""" Secod methods set: Saved/meth_set_2/"""
set_num = 2
classifiers = ["MIASA"]*4 + ["non_MD"]*3 # "Agglomerative_ward" ward can only work with Euclidean distance
clust_methods = ["Agglomerative_ward", "Agglomerative_complete", "Agglomerative_average", "Agglomerative_single"]+["Agglomerative_complete", "Agglomerative_average", "Agglomerative_single"]
metric_method = ["KS-statistic", "KS-p_value"] 

""" Simulations """
for j in range(len(var_data_list)):
    method_dic_list = []
    method_name = []
    for k in range(len(metric_method)):
        for i in range(len(classifiers)):
            dic_meth = {"class_method":classifiers[i], "clust_method":clust_methods[i], "metric_method":metric_method[k], "fig": classifiers[i]+"-"+metric_method[k]}

            if dic_meth["class_method"] == "MIASA":
                dic_meth["fig"] = PdfPages("Figures/%s_var_%d.pdf"%(classifiers[i]+"-"+metric_method[k], var_data_list[j]*1))
                
            method_dic_list.append(dic_meth)
            method_name.append(classifiers[i]+"-"+metric_method[k])
            
    acc_list = repeated_classifications(repeat, method_dic_list, generate_data = generate_data_dist, var_data = var_data_list[j], n_jobs = 100)    
    for i in range(len(method_dic_list)):
        if method_dic_list[i]["class_method"] == "MIASA":
            method_dic_list[i]["fig"].close()
    
    import pickle
    file = open("Accuracy_set%d_%d_varS%s.pdf"%(set_num, repeat, var_data_list_labs[j]), "wb")
    pickle.dump({"method_name":method_name, "method_list":method_dic_list, "accuracy_list":acc_list}, file)
    file.close()
    
    pdfb= PdfPages("Figures/BP_set%d_%d_varS%s.pdf"%(set_num, repeat, var_data_list_labs[j]))    
    BarPlotClass(acc_list, method_name, pdfb, stat_name = "RI scores")
    pdfb.close()
    





    
    