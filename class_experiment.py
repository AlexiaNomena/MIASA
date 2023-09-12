#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 13:21:27 2022

@author: raharinirina
"""
from Methods.classify import repeated_classifications, plotClass, BarPlotClass
from Methods.simulate_class_data_final import generate_data_dist, generate_data_correlated, load_data_twoGRN
import pdb
from matplotlib.backends.backend_pdf import PdfPages
import pickle
import time
import sys

type_list = ["Dist", "Corr", "GRN"]
repeat_list = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]

sim_list = []
### 1-30th 
for tp in type_list:
    for j in range(len(repeat_list)):
        sim_list.append((tp, j))

sim_list.append(("Test", 0)) ### len(type_list)*len(repeat_list) - th

#pdb.set_trace()
to_sim = sim_list[int(sys.argv[1]) - 1] 


""" Classification experiments for different data types """
repeat = repeat_list[to_sim[1]] # Number of replicates of each experiments used for the barplots
var_data_list = [False, True] # fixed: False , variable: True, number of points per true clusters
var_data_list_labs = ["False", "True"]

if to_sim[0] == "Dist":
    """ First methods set"""
    set_num = 1
    save_at = "Class_Data/meth_set_1/" #sample size = 300
    classifiers = ["MIASA"]*2 + ["non_MD"]*1 # non_MD = Non_Metric_Distance
    clust_methods = ["Agglomerative_ward", "Kmedoids"] # for MIASA
    clust_methods = clust_methods + ["Kmedoids"] # for non_MD
    metric_methods = [("Hist", "KS-p1")] #[("eCDF", "eCDF"), ("eCDF", "KS-stat"), ("eCDF", "KS-p1")] 
    # Already separated X, Y samples otherwise randomly separate the dataset into two equal number of sample sets
    sep_vars = True
    # data generating function
    generate_data = generate_data_dist
    # Euclidean embedding pameters only used in MIASA (includes a finite number of auto adjustements)
    c_dic = "default" # seems no-auto adjustments was performed, default works well for this the datatype and distance functions
    in_threads = True # avoid broken runs when using parallel jobs (repeat>10)
    plotfew = False # first run and plot 10 repeats (umap visualization) saved in Figures/

elif to_sim[0] == "Corr":
    """ Second methods set: Saved/meth_set_2/"""
    set_num = 2
    save_at = "Class_Data/meth_set_2/"
    classifiers = ["MIASA"]*2 + ["non_MD"]*1 # non_MD = Non_Metric_Distance = Non_MIASA
    clust_methods = ["Agglomerative_ward", "Kmedoids"] # for MIASA
    clust_methods = clust_methods + ["Kmedoids"] # for non_MD
    metric_methods = [("eCDF", "Pearson_pval"), ("eCDF", "Pearson_R"), ("eCDF", "Spearman_pval"), ("eCDF", "Spearman_R")] # Chosen runs, only normally distributed samples, samples size = 300 , already separated X, Y samples, i.e. , sep_vars = True
    
    # Already separated X, Y samples otherwise randomly separate the dataset into two equal number of sample sets
    sep_vars = True
    # data generating function
    generate_data = generate_data_correlated
    # Euclidean embedding pameters only used in MIASA (includes a finite number of auto adjustements)
    c_dic = "default" 
    in_threads = True # avoid broken runs when using parallel jobs (repeat>10)
    plotfew = False # first run and plot 10 repeats (umap visualization) saved in Figures/
    
elif to_sim[0] == "GRN":
    """ Third methods set"""
    set_num = 3
    save_at = "Class_Data/meth_set_3/"
    classifiers = ["MIASA"]*2 + ["non_MD"]*1 # non_MD = Non_Metric_Distance
    clust_methods = ["Agglomerative_ward", "Kmedoids"] # for MIASA
    clust_methods = clust_methods + ["Kmedoids"] # for non_MD
    metric_methods = [("Eucl", "Granger-Cause-3diff-params"), ("Eucl", "Granger-Cause-3diff-chi2")] # 
    
    # Already separated X, Y samples otherwise randomly separate the dataset into two equal number of sample sets
    sep_vars = True
    
    # data generating function
    generate_data = load_data_twoGRN
    # Euclidean embedding pameters only used in MIASA (includes a finite number of auto adjustements)
    c_dic = "default" 
    in_threads = True # avoid broken runs when using parallel jobs (repeat>10)
    plotfew = False # first run and plot 10 repeats (umap visualization) saved in Figures/
    
else:
    """ Test method """
    print("Run Test")
    repeat = 5
    set_num = 0
    save_at = ""
    classifiers = ["MIASA", "non_MD"]
    clust_methods = ["Agglomerative_ward", "Kmedoids"] # Must be of the same length as classifiers and with a one-to-one mapping i.e. classifiers[i] uses clust_method[i]
    metric_methods = [("Eucl", "Granger-Cause-3diff-params")] # (similarity, association) used by all couple (classifiers[i], clust_method[i])
    
    # Already separated X, Y samples otherwise randomly separate the dataset into two equal number of sample sets
    sep_vars = False
    # data generating function
    generate_data = load_data_twoGRN #generate_data_dist
    
    # Euclidean embedding pameters only usied in MIASA (includes a finite number of auto adjustements)
    c_dic = "default" 
    in_threads = True # avoid broken runs when using parallel jobs (repeat>10)
    plotfew = True # first run and plot 10 repeats (umap visualization) saved in Figures/
    
""" Simulations """
t0 = time.time()
for j in range(len(var_data_list)):
    
    method_dic_list = []
    method_name = []
    for k in range(len(metric_methods)):
        for i in range(len(classifiers)):
            #pdb.set_trace()
            dic_meth = {"class_method":classifiers[i], "clust_method":clust_methods[i], "metric_method":metric_methods[k], "fig": classifiers[i]+"-(%s, %s)-"%metric_methods[k]+"-"+clust_methods[i]}

            if dic_meth["class_method"] == "MIASA" and plotfew:
                dic_meth["fig"] = PdfPages("Figures/miasa_set_%d_%s_var_%s.pdf"%(set_num, metric_methods[k], var_data_list_labs[j]))
                
            method_dic_list.append(dic_meth)
            method_name.append(classifiers[i]+"-(%s, %s)-"%metric_methods[k]+"-"+clust_methods[i])
    
    acc_list_v0, adjusted_acc_list_v0, acc_list_v1, adjusted_acc_list_v1, num_it_list = repeated_classifications(repeat, method_dic_list, generate_data = generate_data, c_dic = c_dic, var_data = var_data_list[j], n_jobs = 10, plot = plotfew, in_threads = in_threads, separation = sep_vars)    

    if plotfew:
        for i in range(len(method_dic_list)):
            if method_dic_list[i]["class_method"] == "MIASA":
                method_dic_list[i]["fig"].close()
    
    if dic_meth["class_method"] == "MIASA":
        file = open(save_at + "Accuracy_set_%d_%d_varS%s.pck"%(set_num, repeat, var_data_list_labs[j]), "wb")
        pickle.dump({"method_name":method_name, "method_list":method_dic_list, "accuracy_list":acc_list_v0, "miasa_accuracy_list":acc_list_v1, "adjusted_accuracy_list": adjusted_acc_list_v0, "miasa_adjusted_accuracy_list": adjusted_acc_list_v1, "num_iterations": num_it_list}, file)
        file.close()
    else:
        file = open(save_at + "Accuracy_set_%d_%d_varS%s.pck"%(set_num, repeat, var_data_list_labs[j]), "wb")
        pickle.dump({"method_name":method_name, "method_list":method_dic_list, "accuracy_list":acc_list_v0, "miasa_accuracy_list":acc_list_v1, "adjusted_accuracy_list": adjusted_acc_list_v0, "miasa_adjusted_accuracy_list": adjusted_acc_list_v1}, file)
        file.close()
        
    file = open(save_at + "Accuracy_set_%d_%d_varS%s.pck"%(set_num, repeat, var_data_list_labs[j]), "rb")
    AcData = pickle.load(file)
    acc_list, adjusted_acc_list, method_name = AcData["accuracy_list"], AcData["miasa_adjusted_accuracy_list"], AcData["method_name"]
    file.close()
    pdfb= PdfPages("Figures/RI_set_%d_%d_varS%s.pdf"%(set_num, repeat, var_data_list_labs[j]))    
    BarPlotClass(acc_list, method_name, pdfb, stat_name = "RI scores")
    pdfb.close()
    
    pdfb= PdfPages("Figures/ARI_set_%d_%d_varS%s.pdf"%(set_num, repeat, var_data_list_labs[j]))    
    BarPlotClass(adjusted_acc_list, method_name, pdfb, stat_name = "ARI scores")
    pdfb.close()

t1 = time.time()
print("run time = ", t1 - t0, "s")





    
    
