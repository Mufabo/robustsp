'''
[B, B0, stats] = ranklassopath(y, X,...)
ranklassopath computes the rank LAD-Lasso regularization path (over grid 
of penalty parameter values). Uses IRWLS algorithm.  
INPUT: 
       yx: Numeric data vector of size N or Nx1(output, respones)
       Xx: Numeric data matrix of size N x p. Each row represents one 
          observation, and each column represents one predictor (feature). 
       L: Positive integer, the number of lambda values on the grid to be  
          used. The default is L=120. 
     eps: Positive scalar, the ratio of the smallest to the 
          largest Lambda value in the grid. Default is eps = 10^-3. 
  reltol: Convergence threshold for IRWLS. Terminate when successive 
         estimates differ in L2 norm by a rel. amount less than reltol.
 printitn: print iteration number (default = 0, no printing)
OUTPUT:
       B: Fitted RLAD-Lasso regression coefficients, a p-by-(L+1) matrix, 
          where p is the number of predictors (columns) in X, and L is 
          the  number of Lambda values.
      B0: estimates values of intercepts
   stats: dictionar with following fields: 
          Lambda = lambda parameters in ascending order
          GMeAD = Mean Absolute Deviation (MeAD) of the residuals
          gBIC = generalized Bayesian information criterion (gBIC) value  
               for each lambda parameter on the grid. 
               
'''
import numpy as np
from robustsp import *

def ranklassopath(yx,Xx,L=120,eps=10**-3,reltol=1e-7,printitn=0):

    if hasattr(eps, "__iter__") or eps < 0 or not np.isfinite(eps) or \
    not np.isrealobj(eps):
        raise ValueError('eps should be a real, positive, scalar')
    if not np.isrealobj(L) or L < 0 or not np.isfinite(L) or \
    hasattr(L, "__iter__"):
        raise ValueError('L should be a real, positive, scalar')
    
    y = np.copy(np.asarray(yx))
    X = np.copy(np.asarray(Xx))
    y = y if not len(y.shape)==2 else y.flatten() # ensure that y is Nx1 and not just N

    X = np.asarray(Xx)
    
    intcpt = False
    n,p = X.shape
    
    B = np.repeat(np.arange(1,n+1)[:,None],n,axis=1)
    
    A = np.copy(B.T)
    a = A[A<B] -1
    b = B[A<B] -1

    Xtilde = X[a,:] \
            - X[b,:]
    ytilde = y[a] - y[b]
    lam0 = np.max(np.abs(Xtilde.T @ np.sign(ytilde)))
    
    lamgrid = eps**(np.arange(0,L+1)/L)*lam0
    B = np.zeros((p,L+1))
    B0 = np.zeros(L+1)
    binit = np.zeros(p)
    
    for jj in range(L+1):
        B[:,jj] ,_= ladlasso(ytilde,Xtilde,lamgrid[jj],binit,intcpt,reltol,printitn)       
        binit = B[:,jj]
        r = y-X@binit[:,None]
        B0[jj] = np.median((r[a]+r[b])/2) if np.isrealobj(y) else spatmed((r[a]+r[b])/2)
        
    B[np.abs(B)<1e-7] = 0
    stats = {}
    
    stats['DF'] = np.sum(np.abs(B),axis=0)
    Rmat = np.repeat(ytilde[:,None],L+1,axis=1)-Xtilde @ B # Matrix of residuals

    N=n*(n-1)/2
    stats['GMeAD'] = (np.sqrt(np.pi)/2)*np.mean(np.abs(Rmat),axis=0) # Gini's dispersion
    stats['GMeAD'] *= np.sqrt(n/(n-stats['DF']-1))
    stats['gBIC'] = 2*n*np.log(stats['GMeAD']) + stats['DF'] * np.log(n)
    stats['Lambda'] = lamgrid
    
    return B,B0,stats