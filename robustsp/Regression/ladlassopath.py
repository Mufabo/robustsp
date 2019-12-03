from robustsp import *
import numpy as np
'''
 [B, stats] = ladlassopath(y, X,...)
 ladlassopath computes the LAD-Lasso regularization path (over grid 
 of penalty parameter values). Uses IRWLS algorithm.  
 INPUT: 
       yx : Numeric data vector of size N or Nx1 (output, respones)
       Xx : Numeric data matrix of size N x p. Each row represents one 
           observation, and each column represents one predictor (feature). 
   intcpt: Logical (true/false) flag to indicate if intercept is in the 
           regression model
      eps: Positive scalar, the ratio of the smallest to the 
           largest Lambda value in the grid. Default is eps = 10^-3. 
       L : Positive integer, the number of lambda values EN/Lasso uses.  
           Default is L=120. 
   reltol : Convergence threshold for IRWLS. Terminate when successive 
           estimates differ in L2 norm by a rel. amount less than reltol.
 printitn: print iteration number (default = 0, no printing)
 OUTPUT:
   B    : Fitted LAD-Lasso regression coefficients, a p-by-(L+1) matrix, 
          where p is the number of predictors (columns) in X, and L is 
          the  number of Lambda values. If intercept is in the model, then
          B is (p+1)-by-(L+1) matrix, with first element the intercept.
  stats  : structure with following fields: 
           Lambda = lambda parameters in ascending order
           MeAD = Mean Absolute Deviation (MeAD) of the residuals
           gBIC = generalized Bayesian information criterion (gBIC) value  
                 for each lambda parameter on the grid. 
'''
def ladlassopath(yx,Xx,intcpt=True,eps=10**-3, L= 120,reltol=1e-6,printitn=0):
    if type(intcpt) != bool:
        raise TypeError('intcpt should be a boolean instead of ' + str(intcpt))
    if hasattr(eps, "__iter__") or eps < 0 or not np.isfinite(eps) or \
    not np.isrealobj(eps):
        raise ValueError('eps should be a real, positive, scalar')
    if not np.isrealobj(L) or L < 0 or not np.isfinite(L) or \
    hasattr(L, "__iter__"):
        raise ValueError('L should be a real, positive, scalar')

    y = np.array(y) #np.copy(np.asarray(yx))
    y = y if not len(y.shape)==2 else y.flatten() # ensure that y is Nx1 and not just N

    X = np.array(Xx) #np.copy(np.asarray(Xx))
    n,p = X.shape

    if intcpt:
        p = p+1
        medy = np.median(y) if np.isrealobj(y) else spatmed(y)
        yc = y - medy    
        lam0 = np.max(X.T @ np.sign(yc)) # max of a column vector
    else: np.max(X.T @ np.sign(yc))

    lamgrid = eps**(np.arange(0,L+1,1)/L) * lam0 # grid of penalty values
    B = np.zeros([p,L+1])

    # initial regression vector
    binit = np.concatenate((np.asarray([medy]),np.zeros(p-1))) if intcpt else np.zeros(p)


    for jj in range(L+1):
        B[:,jj] = ladlasso(y,X,lamgrid[jj],binit,intcpt,reltol,printitn)[0].flatten()
        binit = B[:,jj]

    stats = {}

    # slightly different than Matlab
    if intcpt:
        B[np.vstack((np.zeros((1,L+1),dtype=bool),np.abs(B[1:,:])<1e-7))] = 0
        stats['DF'] = np.sum(np.abs(B[1:,:])!=0,axis=0) 
        stats['MeAD'] = np.sqrt(np.pi/2) * \
                        np.mean( \
                                    np.abs(np.repeat(y[:,np.newaxis],L+1,axis=1) 
                                    -np.hstack((np.ones((n,1)),X))
                                               @B),axis=0)
        const = np.sqrt(n/(n-stats['DF']-1))
    else:
        B[np.abs(B)<1e-7]=0
        stats['DF'] = np.sum(np.abs(B)!=0,axis=0)
        stats['MeAD'] = np.sqrt(np.pi/2) * \
                        np.mean( \
                                np.abs(np.repeat(y[:,np.newaxis],L+1,axis=1)
                                       -X@B),axis=0)
        const = np.sqrt(n/(n-stats['DF']))

    stats['MeAD'] = stats['MeAD']*const
    stats['gBIC'] = 2*n*np.log(stats['MeAD']) + stats['DF'] * np.log(n) # BIC values
    stats['Lambda'] = lamgrid   

    return B, stats