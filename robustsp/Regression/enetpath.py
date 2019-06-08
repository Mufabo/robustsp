'''
 enethpath computes the elastic net (EN) regularization path (over grid 
 of penalty parameter values). Uses pathwise CCD algorithm. 
 INPUT: 
       y : Numeric 1darray of size N (output, respones)
       X : Nnumeric 2darray of size N x p. Each row represents one 
           observation, and each column represents one predictor (feature). 
   intcpt: Logical flag to indicate if intercept is in the model
  alpha  : Numeric scalar, elastic net tuning parameter in the range [0,1].
           If not given then use alpha = 1 (Lasso)
      eps: Positive scalar, the ratio of the smallest to the 
           largest Lambda value in the grid. Default is eps = 10^-4. 
       L : Positive integer, the number of lambda values EN/Lasso uses.  
           Default is L=100. 
 printitn: print iteration number (default = 0, no printing)
 OUTPUT:
   B    : Fitted EN/Lasso regression coefficients, a p-by-(L+1) matrix, 
          where p is the number of predictors (columns) in X, and L is 
          the  number of Lambda values. If intercept is in the model, then
          B is (p+1)-by-(L+1) matrix, with first element the intercept.
  stats  : Dictionary with following fields: 
           Lambda = lambda parameters in ascending order
           MSE = Mean squared error (MSE)
           BIC = Bayesian information criterion values 
'''
import numpy as np
from robustsp.Regression.enet import enet

def enetpath(yx,Xx,alpha=1,L=120,eps=10**-3,intcpt=True,printitn=0):

    # ensure inputs are ndarrays
    Xc = np.copy(np.asarray(Xx))
    y = np.copy(np.asarray(yx))
    if len(y.shape) == 2: y = y.flatten()
    n,p = Xc.shape

    # if intercept is in the model, center the data
    if intcpt:
        meanX = np.mean(Xc,axis=0)
        meany = np.mean(y)
        Xc -= meanX
        y -= meany
        
        
    if printitn > 0:
        print('enetpath: using alpha = %.1f \n' % alpha)

    sdX = np.sqrt(np.sum(Xc*np.conj(Xc),axis=0)) 
    Xc /= sdX
    
    lam0 = np.linalg.norm(Xc.T @ y,np.inf)/alpha # smallest penalty value giving zero solution
    
    lamgrid = eps**(np.arange(0,L+1,1)/L) * lam0 # grid of penalty values

    B = np.zeros([p,L+1])

    for jj in range(L):
        B[:,jj+1], b = enet(y,Xc,B[:,jj], lamgrid[jj+1], alpha, printitn)

    B[np.abs(B) < 5e-8] = 0

    DF = np.sum([np.abs(B)!=0],axis=1) # non-zero values in each column

    if n > p:
        MSE = np.sum(np.abs(np.repeat(y[:,np.newaxis],L+1,axis=1)
                            -Xc@B)**2,axis=0) *(1/(n-DF-1))
        BIC = n * np.log(MSE) + DF * np.log(n)
    else:
        MSE = []
        BIC = []

    B = B / sdX[:,None]
    if intcpt:
        B = np.vstack([meany - meanX @ B, B])

    stats = {'MSE':MSE,'BIC':BIC,'Lambda':lamgrid} 


    return B, stats