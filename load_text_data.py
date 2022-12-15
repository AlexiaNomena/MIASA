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

################## Which specific row variable (here form) do you want to analyse? #################################
row_val = "Vb" # appears in figure name
col_val = "_text_" # appears in figure name
ColName = "Texts" # appears in figure title
RowName = "Vb" # appears in figure title

#### Would you like to study a particular subgroup of categories of the row variables (here forms)? Yes = True, No = False ######
subset_rows = False 

#### datatype (row, col) ###
dtp = ("int", "str") 

from Methods.Read_Data import * # module for reading and cleaning the text dataset, Cleaned_Data is included in there


Data = Cleaned_Data(RawData, sheet_name_list, columns_list, form = row_val, form_labels = rows_labels) 
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
rows_labels = {row:row for row in AllRows}

# Extract Model_infos corresponding to the underlying question
from Methods.CA_Models import Model_infos_2D
variables = variables = [Row_Vals(Data, nans_indices = "Remove Nans"), columns_list]
A, Num_Obs, ContDataFrame, degree_freedom, D, row_centroid, col_centroid, Dr, Dc, S_0, T_0  = Model_infos_2D(Data, variables, model={"model":"stand"}, isCont = False)

Filter = False ### taget a particular cosine threshold for each representations of the categories

if not Filter:
    Coords_Rows = (Dr).dot(D)
    Coords_Cols = (D.dot(Dc)).T ### bring coordinates on rows
else:
    from Methods.Core.Tensor_Factors import choose_Factors
    target_cos_thres = 100 # taget a particular cosine threshold for each representations
    Coords_infos = choose_Factors(D, Dr, Dc, model={"model":"stand"}, csq_thres = target_cos_thres)
    Coords_Rows = Coords_infos["Sp_Fact"].copy()
    Coords_Cols = Coords_infos["Sp_Twin_Fact"].copy()
    Coords_Cols = Coords_Cols.T ### bring coordinates on rows
    
       

