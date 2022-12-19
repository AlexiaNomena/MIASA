#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 13:21:27 2022

@author: raharinirina
"""
from Methods.classify import Classify_general, plotClass
import numpy as np
import pdb

def generate_data():
    """Generate Artificial data"""
    per_spl = 200 # # Num iid per samplea
    num_spl = 10 # Num Samples
    
    data_dic = {}
    class1 = ["1a", "1b", "1c"]
    val1 = [(0, 1), (0, 5), (2, 1)]
    
    class2 = ["2a", "2b", "2c"]
    val2 = [(0, 1), (0, 5), (2, 3)]
    
    class3 = ["3a", "3b", "3c"]
    val3 = [1, 2, 3]
    
    class4 = ["4a", "4b", "4c"]
    val4 = [1, 3, 6]
    
    class_dic = {}
    
    labs = np.cumsum(np.ones(4*3)) - 1
    
    k = 0
    for i in range(3):
        lab = labs[k:k+4]
        for j in range(num_spl):
            data_dic[class1[i]+"%d"%(j+1)] = np.random.normal(val1[i][0], val1[i][1], size = per_spl)
            data_dic[class2[i]+"%d"%(j+1)] = np.random.uniform(val2[i][0], val2[i][1], size = per_spl)
            data_dic[class3[i]+"%d"%(j+1)] = np.random.pareto(val3[i], size = per_spl)
            data_dic[class4[i]+"%d"%(j+1)] = np.random.poisson(val4[i], size = per_spl)
            
            class_dic[class1[i]+"%d"%(j+1)] = lab[0]
            class_dic[class2[i]+"%d"%(j+1)] = lab[1]
            class_dic[class3[i]+"%d"%(j+1)] = lab[2]
            class_dic[class4[i]+"%d"%(j+1)] = lab[3]
        
        k += 4    
    
    num_clust = len(labs)
    dtp = ("<U4", "<U4") #checked from printing variables
    return data_dic, class_dic, num_clust, dtp



if __name__ == "__main__":
    from matplotlib.backends.backend_pdf import PdfPages
    pdf= PdfPages("Figures/fig_Class_test.pdf")
    repeat = 100
    acc_metric = []
    for r in range(repeat):
        data_dic, class_dic, num_clust, dtp = generate_data()
        Id_Class, X_vars, Y_vars, acc_metric_r = Classify_general(data_dic, class_dic, num_clust)
        acc_metric.append(r)
        print("num run", r)
        if r < 10:
            plotClass(Id_Class, X_vars, Y_vars, pdf, dtp, r)
            
    acc_metric = np.array(acc_metric)
    print("Accuracy: mean:%.2f, std:%.2f"%(np.mean(acc_metric), np.std(acc_metric)))    
    #print("Accuracy:%.2f %%"%(acc_metric[0]*100))
    pdf.close()
    #plt.show()





    
    