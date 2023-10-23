import pandas as pd
import numpy as np
import pdb

######## Contingency function used for standard CA ###########
def contingency(Data, row_vals, col_vals, missing = (False, False)):
    '''
    @brief           : - Build contingency matrix formed of the frequency of elements 
                         bellonging to categories in row and categories in columns
                       - return (Contigentency table, total number of Observations)
                       -        (ndarray, int)
    @params Data     : panda dataframe 
    @params row_vals : categories to consider in the rows
    @params col_vals : categories to consider in the columns
    @params missing       : list of bool (Imputation, NaNs as a row variable) 
                            either considering a simple uniform data imputation for missing observations
                            or considering missing variables as independend row values, 
                            or none of those 
                            value cannot be (True, True), this params is ignorned
    '''
    
    # column variables
    cols = col_vals
    
    # row variables
    rows = row_vals
    
    ##### treat missing observations as an independent row variable #####
    if missing[1] and not missing[0]:
        rows = list(rows)+["NaN"]
    
    Cont = np.zeros((len(rows), len(cols)), dtype = int)
    row_notin = []
    for i in range(len(rows)):
        for j in range(len(cols)):
            data_col = np.array(Data[cols[j]], dtype = float)
            if rows[i] == "NaN" and missing[1] and not missing[0]:
                Cont[i, j] = np.sum(np.isnan(data_col))
            else:
                Cont[i, j] = np.sum(data_col == rows[i]) 
        if np.sum(Cont[i,:])==0:
            row_notin.append(i)
    
    ##### treat missing observations: uniform imputation of missing variables #####
    if missing[0] and not missing[1]:
        total = len(rows)
        numMissing = np.sum(np.isnan(Data.to_numpy()))
        Cont = Cont + (1/total)*(numMissing)
        # turn counts to integer
        Cont = np.array(Cont, dtype=int)
    
    # create the contingency table by including the row variables which where present in the dataset
    if len(row_notin) == 0:        
        ContDataFrame = pd.DataFrame(data = Cont, index = rows.astype(int), columns = col_vals)
    
    else:
        indexes = np.arange(0, len(rows), 1, dtype = int)
        remove = indexes == row_notin[0]
        for k in range(1, len(row_notin)):
            remove += indexes == row_notin[k]
        
        ContDataFrame = pd.DataFrame(data = Cont[~remove, :], index = row_vals[~remove].astype(int), columns = col_vals)
        Cont = Cont[~remove, :]
    return Cont, np.sum(Cont), ContDataFrame 

######## Contingency function used for Tensor CA ###########
    
def Remove_nans(Data, nans_indices=None):
    # needed because the data are not always of type float
    Ds = Data.to_numpy()

    if nans_indices is not None:
        Ds = Ds[~nans_indices]
    else:
        Ds = Ds[~np.isnan(Ds)]
        
    return Ds


def full_contingency(variables, Data_Arrays, pivot_index, data_type):
     # build combinatorial list of indices from the variables dimensions
    dimensions = tuple([len(variables[i]) for i in range(len(variables))])
    point_indices_full = np.indices(dimensions)
    temp = []
    for i in range(point_indices_full.shape[0]):
        temp0 = point_indices_full[i].flatten()
        temp.append(temp0)
        
    point_indices = np.vstack(temp).T
    
    col_vals = variables[pivot_index]
    if pivot_index == -1 or pivot_index == 1:
        row_vals = variables[0]
    elif pivot_index == 0:
        row_vals = variables[1]
    else:
        print("Only for two dimensions")
    # produce the multidimensional contingency table
    Cont = np.zeros(dimensions, dtype = int)

    for i in range(point_indices.shape[0]):
        ind = tuple(point_indices[i, :].flatten())
        joint_var = np.array([variables[j][ind[j]] for j in range(len(variables))], dtype = data_type)
        Cont[ind] = np.sum(np.all(Data_Arrays == joint_var, axis = -1)) # check all entries where joint_var exists on the last dimension
        
    return Cont, col_vals, row_vals

def tensor_contingency(Data, variables, nans_indices, pivot_index = -1, data_type = '<U32'):
    '''
    @brief            : - For dimension >= 2, Build multidimensional array formed of the frequency of elements 
                         bellongig jointly to each category variables
                       - return (Contigentency tensor, multi-indexes of each elements, indexes of the non-zero frequency, boolean array of non-zero frequency keep
                       -        (ndarray, ndarray, ndarray, ndarray)
    @params Data      : List of panda dataframe for all the variables, 
                        each columns in the dataset are the same and have the same length
                        and each variable categories at one position must have been extracted 
                        from the same obeservation at that position
    @params variables : list of all variables categories [(list, ), (list, ), ...]
    @params nans_indices: dictionaries of all the nans locations in each of the dataframes in Data 
                         (must be the same for each columns of the dataframe)
    @params pivot_index: the index of the variable categories on the columns of each dataframes in Data 
                            (which must be the same and each rows corresponding to the same observation/measurment)
                            better set it as the last index of variables (and at max must be len(variable)-1)
    '''
    # transform the panda dataframe into numpy arrays 
    for j in range(len(variables[pivot_index])):
        d = []
        pivot = variables[pivot_index][j]
        for i in range(len(Data)):
            if nans_indices is not None:
                d.append(Remove_nans(Data[i][pivot], nans_indices[pivot]))
            else:
                d.append(Remove_nans(Data[i][pivot], None))
        
        # put the joint variable as last dimension
        if j == 0:
            Data_Arrays = np.array(d).transpose()
        else:
            Data_Arrays = np.concatenate((Data_Arrays, np.array(d).transpose())) 
    
    Data_Arrays = np.array(Data_Arrays, dtype = data_type)
    
    # Contingency all variables
    Cont, col_vals, row_vals = full_contingency(variables, Data_Arrays, pivot_index, data_type)
   
    return Cont, col_vals, row_vals