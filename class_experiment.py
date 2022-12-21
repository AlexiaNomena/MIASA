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



if __name__ == "__main__":
    from matplotlib.backends.backend_pdf import PdfPages
    repeat = 200
    
    classifiers = ["MIASA", "non_MD"]#,["MIASA"]#, non_MD = "Non_Metric_Distance"]
    clust_methods = ["Kmeans", "Kmedoids"] #MIASA uses only metric-based clust method (e.g. K-means) and "non_MD" uses non-metric-distance clust method (e.g. K-medoids)
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
            
    acc_list = np.zeros((len(method_dic_list), repeat))
    for r in range(repeat):
        for i in range(len(method_dic_list)):    
            data_dic, class_dic, num_clust, dtp = generate_data()
            Id_Class, X_vars, Y_vars, acc_r = Classify_general(data_dic, class_dic, num_clust, method_dic = method_dic_list[i])
            print("method num %d/%d"%(i+1, len(method_dic_list)), "run %d/%d"%(r+1,repeat))
            if r < 10:
                if method_dic_list[i]["class_method"] == "MIASA":
                    pdf = method_dic_list[i]["fig"]
                    plotClass(Id_Class, X_vars, Y_vars, pdf, dtp, r)
            
            acc_list[i, r] = np.array(acc_r)
        
    for i in range(len(method_dic_list)):
        if method_dic_list[i]["class_method"] == "MIASA":
            method_dic_list[i]["fig"].close()
    
    pdfb= PdfPages("Figures/BP_Class_test_%d.pdf"%repeat)    
    BarPlotClass(acc_list, method_name, pdfb, stat_name = "RI scores")
    pdfb.close()

    #plt.show()





    
    