from robustsp import *
import numpy as np

'''
[b1, iter] = ranklasso(y,X,lambd,...)
ladreg computes the rank (LAD-)regression estimate 
INPUT:
      yx  : numeric data vector of size N  (output, respones)
      Xx  : numeric data matrix of size N x p (input, features)
  lambd : penalty parameter (>= 0) 
     b0  : numeric optional initial start (regression vector) of 
          iterations. If not given, we use LSE. 
printitn : print iteration number (default = 0, no printing) and
           other details 
OUTPUT:
  b1     : numeric the regression coefficient vector
  iter   : (numeric) # of iterations (given when IRWLS algorithm is used)
'''

def ranklasso(yx,Xx,lambd,b0=None,printitn=0):
    
    # ensure y and X are np.arrays and not lists. Copy to avoid reference mutations
    y = np.copy(np.asarray(yx))
    X = np.copy(np.asarray(Xx))
    
    n,p = X.shape
    intcpt = False
    
    if b0 is None:
        b0 = np.linalg.lstsq(\
                    np.hstack((np.ones((n,1)),X))[range(len(y)),:]\
                            ,y,rcond=None)[0]
        b0 = b0[1:]
        
    B = np.repeat(np.arange(1,n+1)[:,None],n,axis=1)
    A = np.copy(B.T)
    a = A[A<B] -1
    b = B[A<B] -1
    Xtilde = X[a,:]-X[b,:]
    ytilde = y[a]-y[b]
    
    return ladlasso(ytilde,Xtilde,lambd,intcpt,b0,printitn)
    
    