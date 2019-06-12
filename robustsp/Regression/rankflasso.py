'''
Computes the rank fused-Lasso regression estimates for given fused
penalty value lambda_2 and for a range of lambda_1 values

INPUT: 
  y       : numeric response N vector (real/complex)
  X       : numeric feature  N x p matrix (real/complex)
  lambda1 : positive penalty parameter for the Lasso penalty term
  lambda2 : positive penalty parameter for the fused Lasso penalty term
  b0      : numeric optional initial start (regression vector) of 
          iterations. If not given, we use LSE (when p>1).
printitn : print iteration number (default = 0, no printing)
OUTPUT:
  b      : numeric regression coefficient vector
  iter   : positive integer, the number of iterations of IRWLS algorithm

'''
import numpy as np

def rankflasso(yx,Xx,lambda1,lambda2,b0=None,printitn=0):
    y = np.copy(np.asarray(yx))
    X = np.copy(np.asarray(Xx))
    
    n,p = X.shape
    
    intcpt = False
    
    if b0 is None:
        b0 = np.linalg.lstsq(np.hstack((np.ones(n),X)),y).flatten()
        b0 = b0[1:]
    
    B = np.repeat(np.arange(1,n+1)[:,None],n,axis=1)
    A = np.copy(B.T)
    a = A[A<B]
    b = B[A<B]
    
    D = -1*np.eye(p-1)