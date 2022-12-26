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
repeat = 100 # Number of replicates of each experiments
var_data_list = [False, True]
var_data_list_labs = ["False", "True"]

""" Test method """
#set_num = 0
#classifiers = ["MIASA"]
#clust_methods = ["Agglomerative_single"] # Must be of the same length as classifiers and with a one-to-one mapping i.e. classifiers[i] uses clust_method[i]
#metric_methods = ["KS-p_value2"] # used by all couple (classifiers[i], clust_method[i])

""" First methods set"""
set_num = 1
save_at = "Class_Data/meth_set_1/"
classifiers = ["MIASA"]*6 + ["non_MD"]*4 # non_MD = Non_Metric_Distance
clust_methods = ["Kmeans", "Kmedoids", "Agglomerative_ward", "Agglomerative_complete", "Agglomerative_average", "Agglomerative_single"] # for MIASA
clust_methods = clust_methods + ["Kmedoids", "Agglomerative_complete", "Agglomerative_average", "Agglomerative_single"] # for non_MD
metric_methods = ["KS-statistic", "KS-p_value"] 

""" Secod methods set: Saved/meth_set_2/"""
#set_num = 2
#save_at = "Class_Data/meth_set_2/"
#classifiers = ["MIASA"]*6 + ["non_MD"]*4 # non_MD = Non_Metric_Distance
#clust_methods = ["Kmeans", "Kmedoids", "Agglomerative_ward", "Agglomerative_complete", "Agglomerative_average", "Agglomerative_single"] # for MIASA
#clust_methods = clust_methods + ["Kmedoids", "Agglomerative_complete", "Agglomerative_average", "Agglomerative_single"] # for non_MD
#metric_methods = ["Corr", "Corr_Moms"]  


""" Third methods set: Saved/meth_set_3/"""
#set_num = 3
#save_at = "Class_Data/meth_set_3/"
#classifiers = ["MIASA"]*6 + ["non_MD"]*4 # non_MD = Non_Metric_Distance
#clust_methods = ["Kmeans", "Kmedoids", "Agglomerative_ward", "Agglomerative_complete", "Agglomerative_average", "Agglomerative_single"] # for MIASA
#clust_methods = clust_methods + ["Kmedoids", "Agglomerative_complete", "Agglomerative_average", "Agglomerative_single"] # for non_MD
#metric_methods = ["OR", "OR_Moms"] 


""" Simulations """
for j in range(len(var_data_list)):
    method_dic_list = []
    method_name = []
    for k in range(len(metric_methods)):
        for i in range(len(classifiers)):
            dic_meth = {"class_method":classifiers[i], "clust_method":clust_methods[i], "metric_method":metric_methods[k], "fig": classifiers[i]+"-"+metric_methods[k]}

            if dic_meth["class_method"] == "MIASA":
                dic_meth["fig"] = PdfPages("Figures/miasa_set_%d_%s_var_%s.pdf"%(set_num, metric_methods[k], var_data_list_labs[j]))
                
            method_dic_list.append(dic_meth)
            method_name.append(classifiers[i]+"-"+metric_methods[k]+"-"+clust_methods[i])
            
    acc_list = repeated_classifications(repeat, method_dic_list, generate_data = generate_data_dist, var_data = var_data_list[j], n_jobs = 25, plot = False)    
    for i in range(len(method_dic_list)):
        if method_dic_list[i]["class_method"] == "MIASA":
            method_dic_list[i]["fig"].close()
    
    import pickle
    file = open(save_at + "Accuracy_set_%d_%d_varS%s.pck"%(set_num, repeat, var_data_list_labs[j]), "wb")
    pickle.dump({"method_name":method_name, "method_list":method_dic_list, "accuracy_list":acc_list}, file)
    file.close()
    
    pdfb= PdfPages("Figures/BP_set_%d_%d_varS%s.pdf"%(set_num, repeat, var_data_list_labs[j]))    
    BarPlotClass(acc_list, method_name, pdfb, stat_name = "RI scores")
    pdfb.close()
    





    
    