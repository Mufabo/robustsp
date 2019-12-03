'''
[B, B0, stats] = hublassopath(y, X,...)
hublassopath computes the M-Lasso regularization path (over grid 
of penalty parameter values) using Huber's loss function  
INPUT: 
       yx: Numeric data vector of size N or N x 1 (output, respones)
       Xx: Numeric data matrix of size N x p. Each row represents one 
          observation, and each column represents one predictor (feature)
          columns are standardized to unit length.
       c: Threshold constant of Huber's loss function (optional;
           otherwise use defaul falue)
  intcpt: Logical (true/false) flag to indicate if intercept is in the 
          regression mode. Default is true.
     eps: Positive scalar, the ratio of the smallest to the 
          largest Lambda value in the grid. Default is eps = 10^-3. 
      L : Positive integer, the number of lambda values EN/Lasso uses.  
          Default is L=120. 
  reltol : Convergence threshold for IRWLS. Terminate when successive 
          estimates differ in L2 norm by a rel. amount less than reltol.
printitn: print iteration number (default = 0, no printing)
OUTPUT:
  B    : Fitted M-Lasso regression coefficients, a p-by-(L+1) matrix, 
         where p is the number of predictors (columns) in X, and L is 
         the  number of Lambda values.
    B0 : estimates values of intercepts
 stats  : dictionary with following fields: 
          Lambda = lambda parameters in ascending order
          sigma = estimates of the scale (a (L+1) x 1 vector)
          gBIC = generalized Bayesian information criterion (gBIC) value  
                for each lambda parameter on the grid. 
'''
import numpy as np
import robustsp as rsp

def hublassopath(yx,Xx,c=None,intcpt=True,eps=10**-3,L=120,reltol=1e-5,printitn=0):
    # ensure that y is Nx1 and not just N and proper formats
    y = np.array(yx) #np.copy(np.asarray(yx))
    X = np.array(Xx) #np.copy(np.asarray(Xx))
    y = y if not len(y.shape)==2 else y.flatten()
    
    n,p = X.shape
    
    realdata = np.isrealobj(y)
    
    if c is None:
        c = 1.3415 if realdata else 1.215
        
    locy, sig0,_ = rsp.hubreg(y,np.ones((n,1)),c)
                            
    if intcpt: 
        # center data
        ync = np.copy(y)
        Xnc = np.copy(X)
        meanX = np.mean(X,axis=0)
        X -= meanX
        y -= locy
    
    # standardize the predictors to unit norm columns
    sdX= np.sqrt(np.sum(X*np.conj(X),axis=0))
    X /= sdX
                            
    # compute the smallest penalty value yielding a zero solution
    yc = rsp.psihub(y/sig0,c)*sig0
    lam0 = np.max(np.abs(X.T@yc)) 
    
    lamgrid = eps**(np.arange(L+1)/L)*lam0
    B = np.zeros((p,L+1))
    sig = np.zeros(L+1)
    sig[0] = sig0
                            
    for jj in range(L):
        B[:,jj+1], sig[jj+1] = rsp.hublasso(y,X,lamgrid[jj+1],B[:,jj],sig[jj],c,reltol,printitn)
    
    B[np.abs(B)<5e-8]=0
    DF = np.sum(np.abs(B)!=0,axis=0)
    con = np.sqrt((n/(n-DF-1)))
                            
    stats = {}
    stats['gBIC'] = 2*n*np.log(sig*con) + DF *np.log(n) if n>p else None
    
    B = B / sdX[:,None]
                            
    B0 = locy-meanX@B if intcpt else None
    stats['sigma'] = sig
    stats['Lambda'] = lamgrid
    
    return B,B0,stats