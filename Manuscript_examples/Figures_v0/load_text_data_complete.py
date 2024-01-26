#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 14:50:35 2023

@author: raharinirina
"""

"""
Analysis : Text vs. Forms
For only one form to analyse
"""

#################### Enter your own data information as in the example here ################################################
RawData = "Data/Codings_Decrees_20_01_21.xlsx" # your data file name, including it's location
sheet_name_list = ["Synodal Decrees", "Comparative Corpus"] # the specific names of the excel sheets you would like to include in the analysis

sub = "" # indicate the subfolder for the result (might need to be created), use "" for saving in the main figure folder Figures/

##################### What are the name of the categories of column variable (here texts) you would like to analyse? Always between " " #####################
columns_list = ["Hermopolisstele", "Hungersnotstele", "Satrapenstele", "Taimuthesstele", "Gallusstele",
            "Alexandriadekret", "Kanopusdekret", "Philensis II Dekret", "Rosettanadekret", 
           "Philensis I Dekret"]
text_list = columns_list

#columns_list = ["Alexandriadekret", "Kanopusdekret", "Rosettanadekret", "Philensis II Dekret", "Philensis I Dekret"] # only the synodal texts

##################### Give the labels of your column category, format "Column_category":"Label" #####################
columns_labels = {"Hermopolisstele":"V1", "Satrapenstele":"V2", "Hungersnotstele":"V3", "Taimuthesstele":"V4", "Gallusstele":"V5", "Pharaon 6":"V6", "Mimosa":"Mim",
             "Alexandriadekret":"S1", "Kanopusdekret":"S2", "Rosettanadekret":"S3", "Philensis II Dekret":"S4", "Philensis I Dekret":"S5"}

#Chronological indexing
columns_labels = {"Hermopolisstele":"V1$^{1}$", "Satrapenstele":"V2$^{2}$", "Hungersnotstele":"V3$^{(?)}$", "Taimuthesstele":"V4$^{9}$", "Gallusstele":"V5$^{10}$", "Pharaon 6":"V6$", "Mimosa":"Mim",
             "Alexandriadekret":"S1$^{3}$", "Kanopusdekret":"S2$^{4}$", "Rosettanadekret":"S3$^{5}$", "Philensis II Dekret":"S4$^{6}$", "Philensis I Dekret":"S5$^{7}$"}

##################### give the dating, format "Colunm_category":"Dating" #####################
columns_dating = {"Hermopolisstele":"378 BCE", "Hungersnotstele":"200-100 BCE", "Taimuthesstele":"42 BCE", "Gallusstele":"29 BCE", 
               "Satrapenstele":"311 BCE", "Mimosa":"000", "Alexandriadekret":"243 BCE", "Kanopusdekret":"238 BCE", 
               "Philensis II Dekret":"186 BCE", "Philensis I Dekret":"185 BCE", "Rosettanadekret":"196 BCE"}

#Chronological indexing + dates
"""
for i in range(len(columns_labels.keys())):
    if list(columns_labels.keys())[i] in list(columns_dating.keys()):
       columns_labels[list(columns_labels.keys())[i]] = columns_labels[list(columns_labels.keys())[i]]+" (%s)"%columns_dating[list(columns_labels.keys())[i]]
"""
##################### List of possible row variables to analyse (here grammatical forms), format  "Accronym":"Description" ###########
rows_labels = {"Vb": "Verbal forms",  "Comp":"Complete Code", "SP":"Sentence particles", "PI":"Particles I", "PII":"Particles II", "P":"Particles", "Subj": "Subjects"}
rows_list = ["Vb", "Comp", "SP", "PI", "PII", "P", "Subj"]
form_list = ["Vb", "SP", "PI", "PII", "Subj"]
################## Build coordinates from all grammatical forms #################################
row_val = "AllForms" # appears in figure name
col_val = "_text_" # appears in figure name
ColName = "Texts" # appears in figure title
RowName = "AllForms"

#### Would you like to study a particular subgroup of categories of the row variables (here forms)? Yes = True, No = False ######
subset_rows = False 

#### datatype (row, col) ###
dtp = ("str", "str") 
from Methods.Read_Data import * # module for reading and cleaning the text dataset, Cleaned_Data is included in there
from Methods.CA_Models import Model_infos_2D
from Methods.Core.Generate_Distances import Association_Distance


rows_labels_all= {}
columns_labels_all = {}
data_dic = {}
Assoc_text_form_dic = {}
columns_labels_combined = {}

import seaborn as sns
colors_dic = {}
colors_rows = sns.color_palette("tab20", len(form_list))
colors_cols = sns.color_palette("Reds", len(text_list)*len(form_list))
colors_cols_dic = {}
p = 0
for k in range(len(form_list)):
    for j in range(len(text_list)):
        colors_cols_dic[(k,j)] = colors_cols[p]
        p +=1

for k in range(len(form_list)):
    rowName = form_list[k]

    Data = Cleaned_Data(RawData, sheet_name_list, columns_list, form = rowName, form_labels = rows_labels) 
    '''
    @what Cleaned_Data does:    - clean the whole dataset to produce a panda dataframe of qualitative variables needed: text vs form
                                - text in the columns and form in the rows
                                - if form is None, then the analysis is text vs complete code
    
    @params RawData:          excel file containing code data
    @params sheet_name_list:  list of sheet names or sheet number to include in the analysis  [string, string, ....] or [int, int, int]
    @params text_list:        list of texts to include in the analysis                        [string, string, ....]
    @params form:             qualitative variables in the rows, values: "SP", "PI", "PII", "Vb", "Subj", "Comp"
    @params form_labels:      dictionary for the labels of te forms:                 {form:string, ....}
    
    @Returns Returns the table of texts on the column and selected grammar on the row code (i.e. row_val)
    '''
    AllRows = Row_Vals(Data, nans_indices = "Remove Nans") # take all row elements of the dataset, remove the rows with nans
   
    # Extract Model_infos corresponding to the underlying question
    variables = [Row_Vals(Data, nans_indices = "Remove Nans"), columns_list]
    A, Num_Obs, ContDataFrame, degree_freedom, D, row_centroid, col_centroid, Dr, Dc, S_0, T_0  = Model_infos_2D(Data, variables, model={"model":"stand"}, isCont = False)
    
    Coords_Rows = (Dr).dot(D)
    Coords_Cols = (D.dot(Dc)).T ### bring coordinates on rows
 
    #D_row_var = Similarity_Distance(Coords_Rows, method = "Euclidean")
    #D_col_var = Similarity_Distance(Coords_Cols, method = "Euclidean")

    Coords = (Coords_Rows, Coords_Cols)
    func = lambda Coords: np.exp(-Dr.dot(D.dot(Dc))) ### Pearson ratio - 1  #### here independent of the coordinates
    D_assoc = Association_Distance(Coords, func)
    """Distane to origin Optional but must be set to None if not used"""
    Orows = np.linalg.norm(Coords_Rows, axis = 1)
    Ocols = np.linalg.norm(Coords_Cols, axis = 1)
    #"""If Distance to origin is set to None, then the corresponding norms of vectors pointing to the axes origin is not interpretable"""
    #Orow = None
    #Ocols = None
    
    for i in range(len(AllRows)):
        row = AllRows[i]
        rows_labels_all[rowName+"-"+str(row)] = rowName+"-"+str(row)
        data_dic[rowName+"-"+str(row)] = Coords_Rows[i, :]
        colors_dic[rowName+"-"+str(row)] = colors_rows[k]
        
    for j in range(len(ContDataFrame.columns)):
        cols = columns_labels[ContDataFrame.columns[j]]
        columns_labels_all[rowName+cols+"-"+str(j)] = rowName+cols+"-"+str(j)
        data_dic[rowName+cols+"-"+str(j)] = Coords_Cols[j, :]
        colors_dic[rowName+cols+"-"+str(j)] = colors_cols_dic[(k,j)]
        if k == 0:
            data_dic[cols] = np.array([])
            colors_dic[cols] = colors_cols[j]
            columns_labels_combined[cols] = cols
        else:
            data_dic[cols] = np.concatenate((data_dic[cols], Coords_Cols[j, :]))    
        
        
    Assoc_text_form_dic["Assoc-text-"+rowName] = D_assoc
    

data_dic["true_colors"] = colors_dic