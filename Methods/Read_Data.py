# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pdb


#### Getting unique values for the categorical data in the row #####
def Row_Vals(Data, nans_indices = None):
    '''
    @brief:        give all the unique categorical data in the rows
    @params Data:  pandas Dataframe
    @params nans_indices: - if nans_indices is not none then remove the nans
                          - if row data types are numerical then no need to specify the nans_indices
                          - otherwise nans_indices must be a dictionary for all the indices to remove in the dataframe columns
    '''
    # column variables
    cols = Data.columns
    
    # row variables
    rows = []
    for col in cols:
        # remove the nans
        try:
             Ds = Data[col].to_numpy(dtype = float)
        except:
             Ds = Data[col].to_numpy()
         
        if nans_indices is not None:
            try:
                Ds = Ds[~np.isnan(Ds)] # nans indices should be extracted from numerical dataframe,cause Nans can only be a dtype float
            except:
                Ds = Ds[~nans_indices[col]]

        # adding getting unique elements to the row list
        rows = rows + list(np.unique(Ds))
    
    # getting unique elements over all columns
    rows = np.unique(np.array(rows))
    return rows

def decompose(Col, form):
    '''
   @brief:       Decompose the complete 10 digit code according to the qualitative variables needed
 
   @params Col:  ndarray of text codes 
   @params form: qualitative variables, values: "SP", "PI", "PII", "Vb", "Subj", "Comp", "P"
   '''
   
    if form == "SP": # Sentence particles 
        # take the first two digit of the 10 digit code
        Col = Col//10**8
        # remove code for destroyed
        destroyed = Col == 99
        Col = Col[~destroyed]
        
    elif form == "PI": # Particle I
        # take the 3th and 4th digits of the 10 digit code
        Col = Col//10**6 - (Col//10**8)*10**2
        # remove code for destroyed
        destroyed = Col == 99
        Col = Col[~destroyed]
        
    elif form == "PII": # Particle II
        # take the 5th and 6th digit of the 10 digit code
        Col = Col//10**4 - (Col//10**6)*10**2
        # remove code for destroyed
        destroyed = Col == 99
        Col = Col[~destroyed]
        
    elif form == "P": # the Whold particle code
        # take the code from the 1st to the 6th digit of the 10 digit code
        ColP = Col//10**4
        
        # remove code for destroyed (if any of the particle code is destroyed)
        Col1 = Col//10**8 # SP
        Col2 = Col//10**6 - (Col//10**8)*10**2 # PI
        Col3 = Col//10**4 - (Col//10**6)*10**2 # PII
        
        destroyed = (Col1 == 99) | (Col2 == 99) | (Col3 == 99)
        Col = ColP[~destroyed]
        
        
    elif form == "Vb": # Verbal forms
        # take the 7th 8th and 9th digits of the 10 digit code
        Col = Col//10 - (Col//10**4)*10**3
        # remove code for destroyed
        destroyed = Col == 999
        Col = Col[~destroyed]
        
        # Removing some verbal ambiguities in text dataset
        """
        Col[Col == 104] = 100
        Col[(Col == 127)|(Col == 128)] = 120
        Col[Col == 146] = 137
        Col[(Col == 156)|(Col == 163)] = 155
        Col[(Col == 267)|(Col == 268)] = 266
        Col[Col == 271] = 270
        Col[Col == 315] = 314
        """
            
    elif form == "Subj": # Subjects
        # take the 10th digits of the 10 digit code
        Col = Col - (10)*(Col//10)
        # remove code for destroyed
        destroyed = Col == 9
        Col = Col[~destroyed]
    
    elif form == "Comp": # Complete code
        ColC = Col
        
        # remove code for destroyed (if any of the form code is destroyed)
        Col1 = Col//10**8 # SP
        Col2 = Col//10**6 - (Col//10**8)*10**2 # PI
        Col3 = Col//10**4 - (Col//10**6)*10**2 # PII
        Col4 = Col//10 - (Col//10**4)*10**3 # Vb
        Col5 = Col - (10)*(Col//10) # Subj
        
        destroyed = (Col1 == 99) | (Col2 == 99) | (Col3 == 99) | (Col4 == 999) | (Col5 == 9)
        Col = ColC[~destroyed]
        
        # Removing some verbal ambiguities in text dataset
        """
        Col1 = Col//10**8 # SP
        Col2 = Col//10**6 - (Col//10**8)*10**2 # PI
        Col3 = Col//10**4 - (Col//10**6)*10**2 # PII
        Col4 = Col//10 - (Col//10**4)*10**3 # Vb
        Col5 = Col - (10)*(Col//10) # Subj
        Col4 = Col//10 - (Col//10**4)*10**3 # Vb
        
        Col = np.array(Col, dtype = str)
        Col[Col4 == 104] = np.array(list(map(lambda x:''.join([x[:6]+"100"+x[9]]), Col[Col4 == 104])))
        Col[(Col4 == 127)|(Col4 == 128)] = np.array(list(map(lambda x:''.join([x[:6]+"120"+x[9]]), Col[(Col4 == 127)|(Col4 == 128)])))
        Col[Col4 == 146] = np.array(list(map(lambda x:''.join([x[:6]+"137"+x[9]]), Col[Col4 == 146])))
        Col[(Col4 == 156)|(Col4 == 163)] = np.array(list(map(lambda x:''.join([x[:6]+"155"+x[9]]), Col[(Col4 == 156)|(Col4 == 163)])))
        Col[(Col4 == 267)|(Col4 == 268)] = np.array(list(map(lambda x:''.join([x[:6]+"266"+x[9]]), Col[(Col4 == 267)|(Col4 == 268)])))
        Col[Col4 == 271] = np.array(list(map(lambda x:''.join([x[:6]+"270"+x[9]]), Col[Col4 == 271])))
        Col[Col4 == 315] = np.array(list(map(lambda x:''.join([x[:6]+"314"+x[9]]), Col[Col4 == 315])))
        
        Col = np.array(Col, dtype = float)
        """
    else:
        print("Parameter form must be one of the : 'SP', 'PI', 'PII', 'Vb', 'Subj', 'Comp'")
    
    return Col[~np.isnan(Col)] # remove the NaNs   (operator ~ invert the boolean array np.isnan Test)


def extract_chapter_index(RawData, sheet_name, lenCol, Chapter_col, Chapter_name):
    Ds = pd.read_excel(RawData, sheet_name = sheet_name, engine='openpyxl')
    
    if Chapter_col in Ds.columns:
        Values = Ds[Chapter_col].to_numpy()
        onlyString = Ds[Chapter_col].apply(lambda x: isinstance(x, str))
        Chap_all = Values[onlyString]
        if Chapter_name in Values:
            chap_ind1 = np.where(Values == Chapter_name)[0][0]
            
            inter = np.where(Chap_all == Chapter_name)[0][0]
            
            if inter != len(Chap_all) - 1:
                chap_ind2 = np.where(Values == Chap_all[inter+1])[0][0]
            else:
                chap_ind2 = lenCol
        else:
            chap_ind1, chap_ind2 = 0, lenCol
     
    else:
        print("Chapter column not found in dataset, all data was considered")
        chap_ind1, chap_ind2 = 0, lenCol
    
    return chap_ind1, chap_ind2
        
            
def Cleaned_Data(RawData, sheet_name_list, text_list, form, form_labels, Chapter_dict = None, Chapter_name=None):
    '''
   @brief:                   - clean the whole dataset to produce a panda dataframe of qualitative variables needed: text vs form
                             - text in the columns and form in the rows
                             - if form is None, then the analysis is text vs complete code
 
   @params RawData:          excel file containing code data
   @params sheet_name_list:  list of sheet names or sheet number to include in the analysis  [string, string, ....] or [int, int, int]
   @params text_list:        list of texts to include in the analysis                        [string, string, ....]
   @params form:             qualitative variables in the rows, values: "SP", "PI", "PII", "Vb", "Subj", "Comp"
   @params form_labels:      dictionary for the labels of te forms:                 {form:string, ....}
   @params Chapter_dict:     dictionary of the Chapter columns for each text        {text:string, ....}
   @params Chapter_name:     Chapter to analyse                                      string

   '''
    
    if form is not None:
        print("The analysis was made on text vs. " + form_labels[form])
    else: 
        print("The analysis was made on text vs. the complete code")
        
    if (Chapter_dict is not None)*(Chapter_name is not None):
        print("The analysis is made for chapter" + Chapter_name)
    elif (Chapter_dict is not None)+(Chapter_name is not None):
        print("Please enter both parameters Chapter_list and Chapter_name")
        print("The analysis was made on the whole text.")
    else:
        print("The analysis was made on the whole text.")

    Data_sub = []
    AnalysedCol = []
    
    for sheet_name in sheet_name_list:
        # load data
        Ds = pd.read_excel(RawData, sheet_name = sheet_name, engine='openpyxl')
        
        Cols = Ds.columns

        for col in text_list:
            if col in Cols:
                AnalysedCol.append(col)

                NewDs = {}
                # remove the non numerical entries like "[zerstört]"
                notNum = Ds[col].apply(lambda x: isinstance(x, str))
                
                if (Chapter_dict is not None)*(Chapter_name is not None) and form is not None:
                    chap_ind1, chap_ind2 = extract_chapter_index(RawData, sheet_name = sheet_name, lenCol = len(Ds[col]), 
                                                                 Chapter_col = Chapter_dict[col], Chapter_name = Chapter_name)
                    
                    D1 = Ds[col][chap_ind1:chap_ind2]
                    notNum = D1.apply(lambda x: isinstance(x, str))
                    # extract form
                    if sum(notNum) != len(D1):
                        NewDs[col] = decompose(D1[~notNum].to_numpy(dtype = float), form)
                    else:
                        print("---------------------------------------------------")
                        print(col + " was removed because it contained no information about " + Chapter_name)
                        print("---------------------------------------------------")

                elif form is not None:
                    D = Ds[col][~notNum] # ~ invert notNum boolean
                    # extract the form
                    NewDs[col] = decompose(D.to_numpy(dtype = float), form)   # put the column in a python array and remove NANs  
                
                else:
                    D = Ds[col][~notNum] # ~ invert notNum boolean
                    NewDs[col] = D.to_numpy(dtype = float)
            
                
                Data_sub.append(pd.DataFrame(NewDs))
            
       
    
    for text in text_list:
        if text not in AnalysedCol:
            print("Text " + text + " is not included dataset")
    print("-----------------------------------------")    
    
    # concatenante columns of DataFrames      
    Data = pd.concat(Data_sub, axis = 1)
    return Data

    
    
    