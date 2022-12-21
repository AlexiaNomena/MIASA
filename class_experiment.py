#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 13:21:27 2022

@author: raharinirina
"""
from Methods.classify import Classify_general, plotClass, BarPlotClass
import numpy as np
import pdb

def generate_data():
    """Generate Artificial data"""
    per_spl = 200 # # Num iid per samplea
    num_spl = 10 # Num Samples
    
    data_dic = {}
    class_type1 = ["1a", "1b", "1c"]
    val1 = [(0, 1), (0, 5), (2, 1)]
    
    class_type2 = ["2a", "2b", "2c"]
    val2 = [(0, 1), (0, 5), (2, 3)]
    
    class_type3 = ["3a", "3b", "3c"]
    val3 = [1, 2, 3]
    
    class_type4 = ["4a", "4b", "4c"]
    val4 = [1, 3, 6]
    
    class_dic = {}
    
    labs = np.cumsum(np.ones(4*3)) - 1
    
    k = 0
    for i in range(3):
        lab = labs[k:k+4]
        for j in range(num_spl):
            data_dic[class_type1[i]+"%d"%(j+1)] = np.random.normal(val1[i][0], val1[i][1], size = per_spl)
            data_dic[class_type2[i]+"%d"%(j+1)] = np.random.uniform(val2[i][0], val2[i][1], size = per_spl)
            data_dic[class_type3[i]+"%d"%(j+1)] = np.random.pareto(val3[i], size = per_spl)
            data_dic[class_type4[i]+"%d"%(j+1)] = np.random.poisson(val4[i], size = per_spl)
            
            class_dic[class_type1[i]+"%d"%(j+1)] = lab[0]
            class_dic[class_type2[i]+"%d"%(j+1)] = lab[1]
            class_dic[class_type3[i]+"%d"%(j+1)] = lab[2]
            class_dic[class_type4[i]+"%d"%(j+1)] = lab[3]
        
        k += 4    
    
    num_clust = len(labs)
    dtp = ("<U4", "<U4") #checked from printing variables
    return data_dic, class_dic, num_clust, dtp


def one_classification(r, method_dic_list):
    acc_res = np.zeros(len(method_dic_list))
    for i in range(len(method_dic_list)):    
        data_dic, class_dic, num_clust, dtp = generate_data()
        Id_Class, X_vars, Y_vars, acc_r = Classify_general(data_dic, class_dic, num_clust, method_dic = method_dic_list[i])
        print("method num %d/%d"%(i+1, len(method_dic_list)), "run %d/%d"%(r+1,repeat))
        acc_res[i] = acc_r
        
    return acc_res


import joblib as jb
from functools import partial

def repeated_classifications(repeat, method_dic_list, n_jobs = 25):
    if repeat < 10:
        repeat = 10 + repeat
        
    acc_list = []
    ### plot and save the first 10 classification runs
    for r in range(10):
        sub_list = []
        for i in range(len(method_dic_list)):    
            data_dic, class_dic, num_clust, dtp = generate_data()
            Id_Class, X_vars, Y_vars, acc_r = Classify_general(data_dic, class_dic, num_clust, method_dic = method_dic_list[i])
            print("method num %d/%d"%(i+1, len(method_dic_list)), "run %d/%d"%(r+1,repeat))

            if method_dic_list[i]["class_method"] == "MIASA":
                pdf = method_dic_list[i]["fig"]
                plotClass(Id_Class, X_vars, Y_vars, pdf, dtp, r)
            sub_list.append(acc_r)
            
        acc_list.append(sub_list)
    
    if repeat > 0:
        pfunc = partial(one_classification, method_dic_list = method_dic_list)
        acc_list = acc_list + (jb.Parallel(n_jobs = n_jobs, prefer="threads")(jb.delayed(pfunc)(r) for r in range(10, repeat)))  
    
        """
        for r in range(repeat):
        acc_list.append(one_classification(r, method_dic_list))
        """
    
    acc_list = np.array(acc_list, dtype = float).T
    return acc_list



if __name__ == "__main__":
    from matplotlib.backends.backend_pdf import PdfPages
    repeat = 1000
    
    classifiers = ["MIASA", "MIASA", "non_MD"]#,["MIASA"]#, non_MD = "Non_Metric_Distance"]
    clust_methods = ["Kmeans", "Kmedoids", "Kmedoids"] #MIASA uses preferably metric-based clust method (e.g. K-means) and "non_MD" uses only non-metric-distance clust method (e.g. K-medoids)
    metric_method = ["KS-statistic", "KS-p_value"]# "OR", "RR"]
    
    method_dic_list = []
    method_name = []
    for k in range(len(metric_method)):
        for i in range(len(classifiers)):
            dic_meth = {"class_method":classifiers[i], "clust_method":clust_methods[i], "metric_method":metric_method[k], "fig": classifiers[i]+"-"+metric_method[k]}

            if dic_meth["class_method"] == "MIASA":
                dic_meth["fig"] = (PdfPages("Figures/%s.pdf"%(classifiers[i]+"-"+metric_method[k]), ))
                
            method_dic_list.append(dic_meth)
            method_name.append(classifiers[i]+"-"+metric_method[k])
            
    acc_list = repeated_classifications(repeat, method_dic_list)    
    for i in range(len(method_dic_list)):
        if method_dic_list[i]["class_method"] == "MIASA":
            method_dic_list[i]["fig"].close()
    
    import pickle
    file = open("Accuracy_Data_%d_%d.pck"%(len(classifiers),repeat), "wb")
    pickle.dump({"method_name":method_name, "method_list":method_dic_list, "accuracy_list":acc_list}, file)
    file.close()
    
    pdfb= PdfPages("Figures/BP_Class_test_%d.pdf"%repeat)    
    BarPlotClass(acc_list, method_name, pdfb, stat_name = "RI scores")
    pdfb.close()

    #plt.show()





    
    