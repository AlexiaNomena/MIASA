#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 13:21:27 2022

@author: raharinirina
"""
from Methods.classify import repeated_classifications, plotClass, BarPlotClass
from Methods.simulate_class_data import generate_data_dist, generate_data_correlated, generate_data_twoGRN
import pdb
from matplotlib.backends.backend_pdf import PdfPages
import pickle
import time


""" Classification experiments for different data types """
repeat = 20 # Number of replicates of each experiments
var_data_list = [False, True]
var_data_list_labs = ["False", "True"]

""" Test method """
"""
set_num = 0
save_at = ""
classifiers = ["MIASA"] + ["non_MD"]
clust_methods = ["Spectral", "Spectral"] # Must be of the same length as classifiers and with a one-to-one mapping i.e. classifiers[i] uses clust_method[i]
metric_methods = ["eCDF-KS-stat", "eCDF-KS-p1"] # used by all couple (classifiers[i], clust_method[i])
generate_data = generate_data_dist
# Euclidean embedding pameters only usied in MIASA (includes a finite number of auto adjustements)
c_dic = "default" 
in_threads = True # avoid broken runs when using parallel jobs (repeat>10)
plotfew = True # plot 10 repeats (umap visualization) saved in Figures/
"""

""" First methods set"""
"""
set_num = 1
save_at = "Class_Data_Extended/meth_set_1/"
classifiers = ["MIASA"]*6 + ["non_MD"]*4 # non_MD = Non_Metric_Distance
clust_methods = ["Kmeans", "Kmedoids", "Agglomerative_ward", "Agglomerative_complete", "Agglomerative_average", "Agglomerative_single", "Spectral"] # for MIASA
clust_methods = clust_methods + ["Kmedoids", "Agglomerative_complete", "Agglomerative_average", "Agglomerative_single", "Spectral"] # for non_MD
metric_methods = ["eCDF", "eCDF-KS-stat", "eCDF-KS-p1"] 
generate_data = generate_data_dist
# Euclidean embedding pameters only usied in MIASA (includes a finite number of auto adjustements)
c_dic = "default" # seems no-auto adjustments was performed, default works well for this the datatype and distance measures
in_threads = True # avoid broken runs when using parallel jobs (repeat>10)
plotfew = False # plot 10 repeats (umap visualization) saved in Figures/
"""

""" First methods set bis"""
"""
set_num = 1
save_at = "Class_Data_Extended/meth_set_1bis/"
classifiers = ["non_MD"]*4 # non_MD = Non_Metric_Distance
clust_methods = ["Kmedoids", "Agglomerative_complete", "Agglomerative_average", "Agglomerative_single", "Spectral"] # for non_MD
metric_methods = ["KS-stat-stat", "KS-p1-p1", "KS-p1-stat", "KS-stat-p1"] # purely non-metric approach, not appropriate for MIASA because Similarity distance is not necessarily Euclidean
generate_data = generate_data_dist
# Euclidean embedding pameters only usied in MIASA (includes a finite number of auto adjustements)
c_dic = "default" # just passed but unused
in_threads = True # avoid broken runs when using parallel jobs (repeat>10)
plotfew = False # plot 10 repeats (umap visualization) saved in Figures/
"""

""" Secod methods set: Saved/meth_set_2/"""
"""
set_num = 2
save_at = "Class_Data_Extended/meth_set_2/"
classifiers = ["MIASA"]*6 + ["non_MD"]*4 # non_MD = Non_Metric_Distance
clust_methods = ["Kmeans", "Kmedoids", "Agglomerative_ward", "Agglomerative_complete", "Agglomerative_average", "Agglomerative_single", "Spectral"] # for MIASA
clust_methods = clust_methods + ["Kmedoids", "Agglomerative_complete", "Agglomerative_average", "Agglomerative_single", "Spectral"] # for non_MD
metric_methods = ["Cov", "Moms", "Cov_Moms", "Moms_Cov"]  
generate_data = generate_data_correlated
# Euclidean embedding pameters only usied in MIASA (includes a finite number of auto adjustements)
c_dic = "default" 
#c_dic = {"c1":35, "c2":1100, "c3":2+1100+35} ### This seems to give a good rate of success in Euclidean Embedding but c2, c3 are too large
in_threads = True # avoid broken runs when using parallel jobs (repeat>10)
plotfew = False # plot 10 repeats (umap visualization) saved in Figures/
"""

""" Third methods set"""
"""
set_num = 3
save_at = "Class_Data_Extended/meth_set_3/"
classifiers = ["MIASA"]*6 + ["non_MD"]*4 # non_MD = Non_Metric_Distance
clust_methods = ["Kmeans", "Kmedoids", "Agglomerative_ward", "Agglomerative_complete", "Agglomerative_average", "Agglomerative_single", "Spectral"] # for MIASA
clust_methods = clust_methods + ["Kmedoids", "Agglomerative_complete", "Agglomerative_average", "Agglomerative_single", "Spectral"] # for non_MD
metric_methods = ["Corr", "Moms", "Corr_Moms", "Moms_Corr"]  
generate_data = generate_data_correlated
# Euclidean embedding pameters only usied in MIASA (includes a finite number of auto adjustements)
c_dic = "default" 
#c_dic = {"c1":35, "c2":1100, "c3":2+1100+35} ### This seems to give a good rate of success in Euclidean Embedding but c2, c3 are too large
in_threads = True # avoid broken runs when using parallel jobs (repeat>10)
plotfew = False # plot 10 repeats (umap visualization) saved in Figures/
"""

""" Fourth methods set"""
"""
set_num = 4
save_at = "Class_Data_Extended/meth_set_4/"
classifiers = ["MIASA"]*6 + ["non_MD"]*4 # non_MD = Non_Metric_Distance
clust_methods = ["Kmeans", "Kmedoids", "Agglomerative_ward", "Agglomerative_complete", "Agglomerative_average", "Agglomerative_single", "Spectral"] # for MIASA
clust_methods = clust_methods + ["Kmedoids", "Agglomerative_complete", "Agglomerative_average", "Agglomerative_single", "Spectral"] # for non_MD
metric_methods = ["Moms", "OR", "Moms_OR", "OR_Moms"] 
generate_data = generate_data_twoGRN
# Euclidean embedding pameters only usied in MIASA (includes a finite number of auto adjustements)
c_dic = "default" 
in_threads = True # avoid broken runs when using parallel jobs (repeat>10)
plotfew = False # plot 10 repeats (umap visualization) saved in Figures/
"""

""" Fourth methods set bis"""
set_num = 4
save_at = "Class_Data_Extended/meth_set_4bis/"
classifiers = ["MIASA"]*6 + ["non_MD"]*4 # non_MD = Non_Metric_Distance
clust_methods = ["Kmeans", "Kmedoids", "Agglomerative_ward", "Agglomerative_complete", "Agglomerative_average", "Agglomerative_single", "Spectral"] # for MIASA
clust_methods = clust_methods + ["Kmedoids", "Agglomerative_complete", "Agglomerative_average", "Agglomerative_single", "Spectral"] # for non_MD
metric_methods = ["Cond_proba_v1", "Cond_proba_v2"]
generate_data = generate_data_twoGRN
# Euclidean embedding pameters only usied in MIASA (includes a finite number of auto adjustements)
c_dic = "default" 
in_threads = True # avoid broken runs when using parallel jobs (repeat>10)
plotfew = False # plot 10 repeats (umap visualization) saved in Figures/

""" Simulations """
t0 = time.time()
for j in range(len(var_data_list)):
    method_dic_list = []
    method_name = []
    for k in range(len(metric_methods)):
        for i in range(len(classifiers)):
            dic_meth = {"class_method":classifiers[i], "clust_method":clust_methods[i], "metric_method":metric_methods[k], "fig": classifiers[i]+"-"+metric_methods[k]}

            if dic_meth["class_method"] == "MIASA" and plotfew:
                dic_meth["fig"] = PdfPages("Figures/miasa_set_%d_%s_var_%s.pdf"%(set_num, metric_methods[k], var_data_list_labs[j]))
                
            method_dic_list.append(dic_meth)
            method_name.append(classifiers[i]+"-"+metric_methods[k]+"-"+clust_methods[i])
            
    acc_list, adjusted_acc_list = repeated_classifications(repeat, method_dic_list, generate_data = generate_data, c_dic = c_dic, var_data = var_data_list[j], n_jobs = 6, plot = plotfew, in_threads = in_threads)    
    
    if plotfew:
        for i in range(len(method_dic_list)):
            if method_dic_list[i]["class_method"] == "MIASA":
                method_dic_list[i]["fig"].close()
    
    file = open(save_at + "Accuracy_set_%d_%d_varS%s.pck"%(set_num, repeat, var_data_list_labs[j]), "wb")
    pickle.dump({"method_name":method_name, "method_list":method_dic_list, "accuracy_list":acc_list, "adjusted_accuracy_list": adjusted_acc_list}, file)
    file.close()
    
    file = open(save_at + "Accuracy_set_%d_%d_varS%s.pck"%(set_num, repeat, var_data_list_labs[j]), "rb")
    AcData = pickle.load(file)
    acc_list, adjusted_acc_list, method_name = AcData["accuracy_list"], AcData["adjusted_accuracy_list"], AcData["method_name"]
    file.close()
    
    pdfb= PdfPages("Figures/RI_set_%d_%d_varS%s.pdf"%(set_num, repeat, var_data_list_labs[j]))    
    BarPlotClass(acc_list, method_name, pdfb, stat_name = "RI scores")
    pdfb.close()
    
    pdfb= PdfPages("Figures/ARI_set_%d_%d_varS%s.pdf"%(set_num, repeat, var_data_list_labs[j]))    
    BarPlotClass(adjusted_acc_list, method_name, pdfb, stat_name = "ARI scores")
    pdfb.close()

t1 = time.time()
print("run time = ", t1 - t0, "s")





    
    