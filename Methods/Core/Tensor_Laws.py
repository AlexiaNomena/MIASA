import numpy as np
import torch
import sys
import pdb

# the order of data tensor cannot exceed the lenght of alpha + ALPHA + 2, I don't have anything better for now
alpha = "abcdefghlnmopqrstuvwxyzABCDEFGHLNMOPQRSTUVWXYZ" # without i, j, k because they are used for special purposes

def mode_product(A, B, mode): # check compatibility    
    # positioning the axis to collapse, we call it here k
    order_A = len(A.shape) 
    n = mode
    
    string_A = alpha[:n-2] + "i" + "k" + alpha[n:order_A] # write other indices until the n-3rd dimension, i at the n-2nd dimension, k at the n-1st dimension, and write other indices on the rest
    #pdb.set_trace()
    order_B = len(B.shape) # must be the same as for A
    string_B = alpha[:n-2] + "k" + "j" + alpha[n:order_B] # write other indices until the n-3rd dimension, k at the n-2nd dimension, j at the n-1st dimension, and write other indices on the rest
    
    string_AB = alpha[:n-2] + "i" + "j" + alpha[n:order_B] # write other indices until the n-3rd dimension, i at the n-2nd dimension, j at the n-1st dimension, and write other indices on the rest
    
    res = np.einsum(string_A + "," + string_B + "->" + string_AB, A, B)
    
    #if len(res.shape) != len(A.shape):
    #    pdb.set_trace()
    
    return res 

def transpose(A, mode):
    # transpose the nth and (n-1)th dimensions
    order_A = len(A.shape) 
    n = mode
    
    string_A = alpha[:n-2] + "i" + "j" + alpha[n:order_A] # write other indices until the n-3rd dimension, i at the n-2nd dimension, j at the n-1st dimension, and write other indices on the rest
    transp_A = alpha[:n-2] + "j" + "i" + alpha[n:order_A] 
    
    res = np.einsum(string_A + "->" + transp_A, A)
    return res
 
def nonlinear_transpose(A, n, m): # m > n
    # positioning the axis to transpose at the mth and nth dimensions 
    order_A = len(A.shape)
    string_A = alpha[:n] + "i" + alpha[n+1:m] + "j" + alpha[m+1:order_A] # write other indices until the n-1 th dimension, i at the n-th dimension, j at the m-th dimension, and write other indices on the rest
    transp_A = alpha[:n] + "j" + alpha[n+1:m] + "i" + alpha[m+1:order_A]
    res = np.einsum(string_A + "->" + transp_A, A)
    return res

    
def Tensor_SVD(A, some = False):
    if isinstance(A, np.ndarray):
        A = torch.tensor(A)
    else:
        A = A.clone().detach()
    U, sigma, V = torch.svd(A, some = some)
    return U, sigma, V

def Tensor_inverse(A):
    if isinstance(A, np.ndarray):
        A = torch.tensor(A)
    else:
        A = A.clone().detach()
    
    Ainv = torch.inverse(A)
    return Ainv

def Tensor_pinverse(A):
    if isinstance(A, np.ndarray):
        A = A
    else:
        A = np.array(A)
    Ainv = np.linalg.pinv(A)
    return Ainv

def Tensor_diag(A, dim1, dim2):
    if isinstance(A, np.ndarray):
        A = torch.tensor(A)
    else:
        A = A.clone().detach()
    diagA = torch.diagonal(A, dim1 = dim1, dim2 = dim2)
    return diagA

def Tensor_det(A):
    if isinstance(A, np.ndarray):
        A = torch.tensor(A)
    else:
        A = A.clone().detach()
    det = torch.det(A)
    return det

def Flatten(A, start = 0, end = -1):
    if isinstance(A, np.ndarray):
        A = torch.tensor(A)
    else:
        A = A.clone().detach()
    A1 = torch.flatten(A, start_dim = start, end_dim = end)
    
    return A1
        

def convert_spherical(A, scale_rads = 1, cut = None, how_cut = None):
    A = np.array(A)
    # https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
    try:
    	radius = np.linalg.norm(A, axis = 1)
    	Abis = np.zeros((A.shape[0], A.shape[1]-1))
    	Abis[:, 1:] = A[:, :A.shape[1]-2]
    	s_radius = np.sqrt(radius[:, np.newaxis]**2 - np.cumsum(Abis**2, axis = 1))
    
    	B = A[:, :A.shape[1]-1]
    	cosvals = np.divide(B, s_radius, out = np.zeros(B.shape), where = (s_radius !=0))
    	angles = np.arccos(cosvals)
    	# special case
    	angles[:, -1] = (2*np.pi - np.arccos(A[:, -2]/s_radius[:, -1]))*(A[:, -1]<0) + angles[:, -1]*(A[:, -1]>=0)
    	sinA = np.sin(angles)
    	cosA = np.cos(angles)
    
    	powers = np.ones((len(angles), A.shape[1]))    
    	pow_sinA = np.triu(powers, k = 1)
    	pow_cosA = powers - pow_sinA - np.tril(powers, k = -1)

    	Transf_Mat = np.stack([np.prod(np.power(sinA[j, :][:, np.newaxis], pow_sinA) , axis = 0)*np.prod(np.power(cosA[j, :][:, np.newaxis], pow_cosA), axis = 0) for j in range(A.shape[0])])
    	if cut is not None:
            try:
                if how_cut == "stat_ind_point":
                    radius = radius - np.sqrt(radius**2 - cut)  ### used for finding the coordinates of the point for statistical independence: if cut = c3*Eps (YH.YH_Epsilon_Exact) then radius = radius - dist_to_stat_ind
                else:
                    radius = radius - cut
            except:
                sys.exit("Computing spherical coordinates:Paramater cut must be less that square norm of vector")
        
    	spherical_CoordMat =  np.diag(scale_rads*radius).dot(Transf_Mat)
    except:
    	sys.exit("Computing spherical coordinates: try first by removing the axes corresponding to zero eigenvalues \n Or remove the points with zeros in all coordinates")
    
    return spherical_CoordMat, Transf_Mat
