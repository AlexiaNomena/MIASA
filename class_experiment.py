#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 13:21:27 2022

@author: raharinirina
"""
from Methods.classify import repeated_classifications, plotClass, BarPlotClass
import numpy as np
import pdb


""" Classification Experiment on random samples from specific probability distributions """
def generate_data_dist(var_data = False, noise = False):
    """Generate Artificial data"""
    per_spl = 200 # # Num iid per var / num sample per var
    data_dic = {}
    class_type1 = ["1a", "1b", "1c"] # Normal Dist
    #val1 = [(0, 1), (0, 5), (2, 1)]
    val1_mean = np.random.choice(5, size = len(class_type1)) # allowing repeating means
    val1_std = np.random.choice(5, size = len(class_type1), replace = False) # not allowing repeated variance
    val1 = [(val1_mean[k], val1_std[k]) for k in range(len(class_type1))]
    
    class_type2 = ["2a", "2b", "2c"] # Uniform Dist
    #val2 = [(0, 1), (0, 5), (2, 3)]
    val2_a = np.random.choice(5, size = len(class_type2)) # allowing repeating start
    val2_b = np.random.choice(5, size = len(class_type2), replace = False) # not allowing end
    val2 = [(val2_a[k], val2_a[k] + val2_b[k]) for k in range(len(class_type2))]
    
    class_type3 = ["3a", "3b", "3c"] # Pareto Dist
    #val3 = [1, 2, 3]
    val3_shape = np.random.choice(np.arange(1, 5), size = len(class_type3), replace = False)
    val3_scale = np.random.choice(5, size = len(class_type3), replace = False)
    val3 = [(val3_shape[k], val3_scale[k]) for k in range(len(class_type3))]
    
    class_type4 = ["4a", "4b", "4c"] # Poisson Dist
    #val4 = [1, 3, 6]
    val4 = np.random.choice(5, size = len(class_type4), replace = False)
    
    class_dic = {}
    
    num_clust = len(class_type1) + len(class_type2) + len(class_type3) + len(class_type4)
    labs = np.cumsum(np.ones(num_clust)) - 1
    
    MaxNumVar = 15
    if var_data:
        num_var_list = np.random.choice(np.arange(2, MaxNumVar), size = len(labs), replace = False)
        num_var = {labs[k]: num_var_list[k] for k in range(len(labs))}
    else:
        num_var = {labs[k]:10 for k in range(len(labs))}
    
    
    k = 0
    for i in range(3):
        lab = labs[k:k+4]
        for j in range(MaxNumVar + 1):
            if j <= num_var[lab[0]]:
                data_dic[class_type1[i]+"%d"%(j+1)] = np.random.normal(val1[i][0], val1[i][1], size = per_spl)
                class_dic[class_type1[i]+"%d"%(j+1)] = lab[0]
                
            if j <= num_var[lab[1]]:
                data_dic[class_type2[i]+"%d"%(j+1)] = np.random.uniform(val2[i][0], val2[i][1], size = per_spl)
                class_dic[class_type2[i]+"%d"%(j+1)] = lab[1]
              
            if j <= num_var[lab[2]]:
                data_dic[class_type3[i]+"%d"%(j+1)] = np.random.pareto(val3[i][0], size = per_spl)*val3[i][1]
                class_dic[class_type3[i]+"%d"%(j+1)] = lab[2]
                
            if j <= num_var[lab[3]]:
                data_dic[class_type4[i]+"%d"%(j+1)] = np.random.poisson(val4[i], size = per_spl)
                class_dic[class_type4[i]+"%d"%(j+1)] = lab[3]
        
        k += 4    
    
    
    dtp = ("<U4", "<U4") #checked from printing variables
    return data_dic, class_dic, num_clust, dtp





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
            
    acc_list = repeated_classifications(repeat, method_dic_list, var_data = True)    
    for i in range(len(method_dic_list)):
        if method_dic_list[i]["class_method"] == "MIASA":
            method_dic_list[i]["fig"].close()
    
    import pickle
    file = open("Accuracy_VariableData_%d_%d.pck"%(len(classifiers),repeat), "wb")
    pickle.dump({"method_name":method_name, "method_list":method_dic_list, "accuracy_list":acc_list}, file)
    file.close()
    
    pdfb= PdfPages("Figures/BP_Class_test_%d.pdf"%repeat)    
    BarPlotClass(acc_list, method_name, pdfb, stat_name = "RI scores")
    pdfb.close()

    #plt.show()





    
    