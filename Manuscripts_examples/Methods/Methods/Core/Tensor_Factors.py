from .Tensor_Laws import *
from operator import itemgetter

import pdb


def get_Factors(A, Dr, Dc, model={"model":"stand"}):
    '''
    @brief            : compute factor coordinates
                       
    @params A         : array or tensor
    
    '''
    A_pos = A.copy()
    A_pos[A<0] = 0
    
    if isinstance(A, np.ndarray):
        A = torch.tensor(A)
        A_pos = torch.tensor(A_pos)
    else:
        A = A.clone().detach()
        A_pos = A_pos.clone().detach()
        
    if len(A.shape) > 2:
        print("For Dim >= 3, method is not implemented")
        print("For Dim = 1, method is not appropriate")
      
    U, sigma, V = Tensor_SVD(A, some = True) # tensor SVD return V and not V.transpose as in numpy.linalg.svd
    
    # transpose U
    UT = transpose(U, mode = 2)
    VT = transpose(V, mode = 2)
    
   
    ### Get Factor Coordinates ######
    if model["model"] == "stand":
        Fact = mode_product(A, V, mode = 2)
        Fact = mode_product(Dr, Fact, mode = 2) # on the rows
        
        Stand_Fact = mode_product(Dr, U, mode = 2)
        
        Twin_Fact = mode_product(UT, A, mode = 2) # on the columns
        Twin_Fact = mode_product(Twin_Fact, Dc, mode = 2)
        
        Stand_Twin_Fact = mode_product(VT, Dc, mode = 2)
        
        
        return {"Fact":Fact, "Stand_Fact":Stand_Fact, "Twin_Fact":Twin_Fact, "Stand_Twin_Fact":Stand_Twin_Fact, 
            "Inertia":np.array(sigma)**2, "UT":UT, "V":V}
    
    else:
        print("This code is only for the standard CA model.")


def IndSorted(Perc, axis):  
    if axis == 1:
        sortF = np.argsort(Perc, axis = axis)[:, ::-1] # sort the columns in decreasing order 
        rowInds = np.cumsum(np.ones(Perc.shape, dtype = int), axis = 0) - 1 # generate indexes for each rows
        stkInd = np.stack((rowInds, sortF), axis = 1) # stack the rows and column indexes
        L = transpose(stkInd, mode = 3) # couple the indexes
    elif axis == 0:
        sortF = np.argsort(Perc, axis = axis)[::-1, :] # sort the rows in decreasing order
        colInds = np.cumsum(np.ones(Perc.shape, dtype = int), axis = 1) - 1 # generate indexes for each columns
        stkInd = np.stack((sortF, colInds), axis = 0) # stack the rows and column indexes
        
        L = nonlinear_transpose(stkInd, n = 0, m = 2) # couple the indexes
    else:
        print("Not implemented for num dimension > 2")
        
    L = Flatten(L, start=0, end=1) # extract the couple indexes 
    
    sortInds = list(map(tuple, np.array(L))) # transform each couple indexes into tuple
    
    return sortInds  


def SparseFact(F, Stand, Perc, Inds, csq_thres, axis):
    T = itemgetter(*Inds)(Perc)
    
    if axis == 1:
        T = np.array(T).reshape(Perc.shape)
    elif axis == 0:
        T = np.array(T).reshape((Perc.shape[1], Perc.shape[0])) # The function IndSorted which generateD Inds transposed the indexes 
        
    # Test 1 = Cosine square test
    # Test 2 = We need to keep the factor upperbound of csq_thres otherwise we might remove all the factor coordinates
    
    test_1 = np.cumsum(T, axis = 1) >= csq_thres 
    test_2 = np.cumsum(test_1, axis = 1) >= 2
    if np.sum(test_2) == 0:
        print("exact threshold was found")
        test = test_1
    else:
        print("take uperbound threshold")
        test = test_2
    
    remove = np.array(test.flatten(), dtype = bool)
    index = np.arange(0, len(Inds), dtype = int)
    
    toRemove = itemgetter(*index[remove])(Inds)


    # make a tuple of all the indexes to remove 
    #format ((all inds first dim), (corresponding inds second dim))
    toRemove = np.array(toRemove).transpose()
    if toRemove.shape != (2, ):
        toRemove = tuple(map(tuple, toRemove))
    else:
        toRemove = tuple(toRemove)
    
    # put zeros at indices of factors we discard
    spF = F.copy()
    spF[toRemove] = 0.
    
    Stand_new = Stand.copy()
    Stand_new[toRemove] = 0.
    
    return spF, Stand_new
    
       
def choose_Factors(A, Dr, Dc, model, csq_thres = 90): 
    '''
    @brief            : compute and choose factor coordinates
                       
    @params csq_thres : desired cosine square threshold in choosing factor coordinates
             
    '''
    
    print("Cosine square threshold is ", csq_thres)
    Fact_infos = get_Factors(A, Dr, Dc, model = model)
    
    Fact = Fact_infos["Fact"].copy()
    Twin_Fact = Fact_infos["Twin_Fact"].copy()
    StandFact = Fact_infos["Stand_Fact"].copy()
    StandTwFact = Fact_infos["Stand_Twin_Fact"].copy()
    
    ### Generate Percentage cosine square 
    N2A = (np.linalg.norm(Fact, axis = 1))**2
    N2Atw = (np.linalg.norm(Twin_Fact, axis = 0))**2
    
    Id = np.ones(Fact.shape)
    Idtw = np.ones(Twin_Fact.shape)
    
    Hyp = mode_product(np.diag(N2A), Id, mode = 2)
    Hyptw = mode_product(Idtw, np.diag(N2Atw), mode = 2)
    
    
    PercF = 100*(Fact**2/Hyp)
    PercFtw = 100*(Twin_Fact**2/Hyptw)
    
    ### Generate the tuple indexes of sorted factors (sorted in decreasing order)
    IndsFact = IndSorted(PercF, axis = 1)
    IndsFactTw = IndSorted(PercFtw, axis = 0)
    
    
    ### Generate the sparse factor matrix for which each profiles achieves the threshold cosine square
    sparse_Fact, Stand_Fact = SparseFact(Fact, StandFact, PercF, IndsFact, csq_thres, axis = 1)      # on the rows
    sparse_Twin_Fact, Stand_Twin_Fact = SparseFact(Twin_Fact, StandTwFact, PercFtw, IndsFactTw, csq_thres, axis = 0) # on the columns
    
    ### printing check ###
    PercF2 = 100*(sparse_Fact**2/Hyp)
    PercFtw2 = 100*(sparse_Twin_Fact**2/Hyptw)
    
    print("--- row cosine square ---")
    print(np.cumsum(PercF2, axis = 1)[:, -1])
    
    print("--- col cosine square ---")
    print(np.cumsum(PercFtw2, axis = 0)[-1, :])
    
    return {"Sp_Fact":sparse_Fact, "Coords_rows":Stand_Fact, "Sp_Twin_Fact":sparse_Twin_Fact, "Coords_columns":Stand_Twin_Fact.T, "Fact":Fact, "Twin_Fact":Twin_Fact, "Inertia":Fact_infos["Inertia"], "UT":Fact_infos["UT"], "V":Fact_infos["V"]}