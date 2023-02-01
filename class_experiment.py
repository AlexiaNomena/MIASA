#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 13:21:27 2022

@author: raharinirina
"""
from Methods.classify import repeated_classifications, plotClass, BarPlotClass
from Methods.simulate_class_data import generate_data_dist, generate_data_correlated, generate_data_correlated_2, load_data_twoGRN
import pdb
from matplotlib.backends.backend_pdf import PdfPages
import pickle
import time


""" Classification experiments for different data types """
repeat = 2 # Number of replicates of each experiments used for the barplots
var_data_list = [False, True]
var_data_list_labs = ["False", "True"]

""" Test method """
"""
set_num = 0
save_at = ""
classifiers = ["MIASA"]
clust_methods = ["Kmeans"] # Must be of the same length as classifiers and with a one-to-one mapping i.e. classifiers[i] uses clust_method[i]
metric_methods = [("eCDF", "KS-stat"), ("eCDF", "KS-stat")] # (similarity, association) used by all couple (classifiers[i], clust_method[i])
generate_data = generate_data_dist
# Euclidean embedding pameters only usied in MIASA (includes a finite number of auto adjustements)
c_dic = "default" 
in_threads = True # avoid broken runs when using parallel jobs (repeat>10)
plotfew = False # first run and plot 10 repeats (umap visualization) saved in Figures/
"""

""" First methods set"""
"""
set_num = 1
save_at = "Class_Data/meth_set_1/" #sample size = 200
classifiers = ["MIASA"]*2 + ["non_MD"]*1 # non_MD = Non_Metric_Distance
clust_methods = ["Kmeans", "Kmedoids"] # for MIASA
clust_methods = clust_methods + ["Kmedoids"] # for non_MD
metric_methods = [("eCDF", "eCDF"), ("eCDF", "KS-stat"), ("eCDF", "KS-p1")] 
generate_data = generate_data_dist
# Euclidean embedding pameters only used in MIASA (includes a finite number of auto adjustements)
c_dic = "default" # seems no-auto adjustments was performed, default works well for this the datatype and distance measures
in_threads = True # avoid broken runs when using parallel jobs (repeat>10)
plotfew = False # first run and plot 10 repeats (umap visualization) saved in Figures/
"""

""" Secod methods set: Saved/meth_set_2/"""

set_num = 2
save_at = "Class_Data/meth_set_2/"
classifiers = ["MIASA"]*2 + ["non_MD"]*1 # non_MD = Non_Metric_Distance = Non_MIASA
clust_methods = ["Kmeans", "Kmedoids"] # for MIASA
clust_methods = clust_methods + ["Kmedoids"] # for non_MD
#metric_methods = [("Corr", "dCorr"), ("Cov", "dCov"), ("Cov", "dCorr"), ("Corr", "dCov")] # sample size = 200, test repeats 1000
#metric_methods = [("Corr", "dCorr_v2"), ("Cov", "dCov_v2"), ("Cov", "dCorr_v2"), ("Corr", "dCov_2")] # sample size = 200, test repeats 1100
#metric_methods = [("OR", "dOR"), ("OR", "dCond"), ("Cond_proba", "dCond"), ("Cond_proba", "dOR")] # sample size = 200, test repeats 1200, proba are based on number of increments and decrements
#metric_methods = [("Corr", "dCorr"),  ("Cond_proba", "dCond"), ("Corr", "dCond"), ("Cond_proba", "dCorr")] # sample size = 200, test repeats 200
#metric_methods = [("Corr", "dCorr"), ("Corr", "dCond"), ("Cond_proba", "dCorr")] # sample size = 200, test repeats 2000, 2001, 2002, 2003, 2004 (differentiating filenames for a total of ~10000 repeats)
#metric_methods = [("Corr", "dCorr"), ("Corr", "Pearson_pval"), ("Corr", "Spearman_pval")] # test runs sample size = 500, repeats 2006 (only small improvements)
#metric_methods = [("Corr", "dCorr"), ("Corr", "Pearson_pval"), ("Corr", "Spearman_pval")] # test runs, only bivariate normal dist, sample size = 200, repeats 2007 (no difference between MIASA and non-MD for corr, dcorr, large difference for other)

metric_methods = [("Corr", "dCorr"), ("Corr", "Pearson_pval"), ("Corr", "Spearman_pval")] # chosen runs sample size = 200, repeats 2005

generate_data = generate_data_correlated_2
# Euclidean embedding pameters only used in MIASA (includes a finite number of auto adjustements)
c_dic = "default" 
in_threads = True # avoid broken runs when using parallel jobs (repeat>10)
plotfew = False # first run and plot 10 repeats (umap visualization) saved in Figures/


""" Third methods set"""
"""
set_num = 3
save_at = "Class_Data/meth_set_3/"
classifiers = ["MIASA"]*2 + ["non_MD"]*1 # non_MD = Non_Metric_Distance
clust_methods = ["Kmeans", "Kmedoids"] # for MIASA
clust_methods = clust_methods + ["Kmedoids"] # for non_MD
#metric_methods = [("Corr", "dCorr"), ("Corr", "Granger-Cause-orig-params"), ("Corr", "Granger-Cause-diff-params")] # tested but there was not much difference between Granger-Cause-orig and Granger-Cause-diff
metric_methods = [("Corr", "dCorr"), ("Corr", "Granger-Cause-diff-params"), ("Corr", "Granger-Cause-diff-chi2")] 
generate_data = load_data_twoGRN
# Euclidean embedding pameters only used in MIASA (includes a finite number of auto adjustements)
c_dic = "default" 
in_threads = True # avoid broken runs when using parallel jobs (repeat>10)
plotfew = False # first run and plot 10 repeats (umap visualization) saved in Figures/
"""

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
            
    acc_list, adjusted_acc_list = repeated_classifications(repeat, method_dic_list, generate_data = generate_data, c_dic = c_dic, var_data = var_data_list[j], n_jobs = 8, plot = plotfew, in_threads = in_threads)    
    
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





    
    