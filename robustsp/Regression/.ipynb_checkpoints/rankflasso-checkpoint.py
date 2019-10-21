'''
Computes the rank fused-Lasso regression estimates for given fused
penalty value lambda_2 and for a range of lambda_1 values

INPUT: 
  y       : 1darray, dtype either real or complex, numeric response N vector (real/complex)
  X       : 2darray, numeric feature  N x p matrix (real/complex)
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
import robustsp as rsp

def rankflasso(yx,Xx,lambda1,lambda2,b0=None,printitn=0):

    y = np.array(yx)
    X = np.array(Xx)
    n,p = X.shape

    intcpt = False

    if b0 is None:
        # Note that underdetermined systems have indefinite solutions, numpy handles it differently
        # than matlab
        b0 = np.linalg.lstsq(np.hstack((np.ones((n,1)),X)),y[:,None])[0] 
        b0 = b0[1:]

    B = np.repeat(np.arange(1,n+1)[:,None],n,axis=1)
    A = np.array(B.T)
    a = A[A<B]
    b = B[A<B]

    D = -1*np.eye(p-1) + np.diag(np.ones(p-2),k=1)
    
    onev = np.append(np.zeros(p-2),1)
    D = np.hstack((D,onev[:,np.newaxis]))

    ytilde = np.append(y[a-1]-y[b-1],
                        np.zeros(p-1)
                       )
    
    Xtilde = np.vstack((X[a-1,:]-X[b-1,:],lambda2*D))

    if printitn > 0:
        print("rankflasso: starting iterations\n")

    b, iter = rsp.ladlasso(ytilde,Xtilde,lambda1,b0,intcpt,printitn)

    return b, iter
    
    