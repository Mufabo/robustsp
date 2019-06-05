import numpy as np
from robustsp import propform
'''
 elemfits compute the Nx(N-1)/2 elemental fits, i.e., intercepts b_{0,ij}
 and slopes b_{1,ij}, that define a line y = b_0+b_1 x that passes through 
 the data points (x_i,y_i) and (x_j,y_j), i<j, where i, j in {1, ..., N}. 
 and the respective weights | x_i - x_j | 
 INPUT: 
    y : (numeric) 1darray of real-valued outputs (response vector)
    x : (numeric) 1darray vector of inputs (feature vector) 
 OUTPUT:
    beta: (numeric) N*(N-1)/2 matrix of elemental fits 
    w: (numeric) N*(N-1)/2 matrix of weights
    
 Note: Numpy uses C memory order whereas Matlab uses Fortran ordering, thus the 
     indices of the solution elements are different
'''
def elemfits(y,x):

    y = np.asarray(y)
    x = np.asarray(x)

    N = len(x)
    B = np.repeat(np.arange(1,N+1)[:,np.newaxis],N,axis=1)
    A = B.T
    a = A[A<B] # order of elements different from matlab
    b = B[A<B] #
    w = x[a-1] - x[b-1]
    beta = np.zeros([2,len(a)])
    beta[1,:] = (y[a-1]-y[b-1])/w
    beta[0,:] = (x[a-1]*y[b-1] - x[b-1]*y[a-1])/w
    w= np.abs(w)

    return beta,w 