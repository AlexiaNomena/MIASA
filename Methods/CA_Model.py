from .shared_func import tensor_contingency
import pandas as pd
#pd.options.display.float_format = "{:,.2f}".format ### Float format in console printing


def FilterCont_2D(Cont, col_vals, row_vals, pivot_index): 
    
    remove = np.sum(Cont, axis = pivot_index) ==  0
    Cont_Filt = Cont[~remove, :]
    ContDataFrame = pd.DataFrame(data = Cont_Filt, index = row_vals[~remove].astype(int), columns = col_vals)
    
    return np.sum(Cont_Filt), ContDataFrame



def Model_infos_2D(Data, variables, model={"model":"stand"}, isCont = False):
    # This will depend on the sub scientific question
    # example: what is the deviation from the strong independence model
    '''
    @brief           : - from the a 2 way contingency table return
                       - (Matrix deviation from independence, Contingency array) 
                       - (array, array)
    @params Data     : panda dataframe or contigency table (need to set isCont = True) 
    @model           : what model to study: str "y|x" or "x|y" or "stand"
                       if other model then you need to enter a function as value to "func" key
                       
    @isCont          : Boolean True if data is a contigency dataframe and False to compute the contingency dataframe 
                       (all row categories must be at least present with one of the columns categories)            
    '''
    # contigency
    if isCont:
        Cont, Num_Obs = Data.to_numpy(dtype=float), np.sum(Data.to_numpy(dtype=float)) # if Data is already a contigency dataframe
        ContDataFrame = Data
    else:
        d = Data
        unique_columns = d.columns
        Data_col = {}
        nans_indices = {}
        for col in unique_columns:
            nans_indices[col] = np.isnan(d[col].to_numpy(dtype=float))
            Data_col[col] = [col]*len(d[col]) # repeat text a required number of times
        
        Piv_Data = pd.DataFrame(Data_col)
        ContUnfiltered, col_vals, row_vals = tensor_contingency([Data, Piv_Data], variables, nans_indices=nans_indices, pivot_index = -1, data_type = '<U32') # compute the contingency
        
        Num_Obs, ContDataFrame = FilterCont_2D(ContUnfiltered, col_vals, row_vals, pivot_index = -1)
        Cont = ContDataFrame.to_numpy(dtype=float)
        
    # table of joint proba
    joint_proba = Cont/Num_Obs
    
    # marginals of the rows, vector composed of the marjinals of each row 
    marj_rows= np.sum(joint_proba, axis = 1)
    
    # marginals of the columns, vector composed of the marjinals of each column 
    marj_columns = np.sum(joint_proba, axis=0)
        
    # Compute deviation matrix Conditional proba minus the product of the marginals
    D = joint_proba - marj_rows[:, np.newaxis]*marj_columns
    
    # Standardise the deviation matrix
    Dr = np.diag(1/np.sqrt(marj_rows))
    
    Dc = np.diag(1/np.sqrt(marj_columns))
    
    # standardized residual
    D = Dr.dot(D.dot(Dc))
    
    row_centroid = marj_columns
    col_centroid = marj_rows
    
    # Compute regular row_column_profiles
    S_0 = (Dr**2).dot(Cont/Num_Obs)
    T_0 = (Cont/Num_Obs).dot(Dc**2)
    if model["model"] == "stand":
        # Standardized residuals
        A = D 
    else:
        import sys
        sys.exit("This code is only for the standard CA model using the standardized residual matrix")

    degree_freedom = (A.shape[0] - 1)*(A.shape[1] - 1)
    
    return A, Num_Obs, ContDataFrame, degree_freedom, D, row_centroid, col_centroid, Dr, Dc, S_0, T_0

    


    
    

