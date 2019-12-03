'''
[b1, sig1, iter] = hubreg(y,X,...)
hubreg computes the joint M-estimates of regression and scale using 
Huber's criterion. Function works for both real- and complex-valued data.
%
INPUT: 
       yx: Numeric data vector of size N (output, respones)
       Xx: Numeric data matrix of size N x p. Each row represents one 
          observation, and each column represents one predictor (feature). 
          If the model has an intercept, then first column needs to be a  
          vector of ones. 
        c: numeric threshold constant of Huber's function
     sig0: (numeric) initial estimator of scale [default: SQRT(1/(n-p)*RSS)]
       b0: initial estimator of regression (default: LSE)  
 printitn: print iteration number (default = 0, no printing)
 ITERMAX:  default = 2000, maximum number of iterations
 ERRORTOL: default = 1e-5, ERROR TOLERANCE FOR HALTING CRITERION
OUTPUT:
      b1: the regression coefficient vector estimate 
    sig1: the estimate of scale 
    iter: the # of iterations 
'''

import numpy as np
from scipy.stats import chi2
import robustsp as rsp

def hubreg(yx,Xx,c=None,sig0=None,b0=None,printitn=0,ITERMAX = 2000,ERRORTOL = 1e-5):

    # ensure that y is Nx1 and not just N and proper formats
    y = np.copy(np.asarray(yx))
    X = np.copy(np.asarray(Xx))
    y = y if not len(y.shape)==2 else y.flatten()
    
    n,p = X.shape
    
    realdata = np.isrealobj(y)
    
    if c is None:
        c = 1.3415 if realdata else 1.215
        # Default: approx 95 efficiency for Gaussian errors
    
    if b0 is None:
        b0 = np.linalg.lstsq(X[range(len(y)),:],y,rcond=None)[0]
        
    if sig0 is None:
        sig0 = np.linalg.norm(y-X@b0)/np.sqrt(n-p)
        
    csq = c**2
    
    if realdata:
        qn = chi2.cdf(csq,1)
        alpha = chi2.cdf(csq,3)+csq*(1-qn) # consistency factor for scale
    else:
        qn = chi2.cdf(2*csq,2)
        alpha = chi2.cdf(2*csq,4)+csq*(1-qn) # consistency factor for scale
        
    Z = np.linalg.pinv(X)[0] # svd <1e-15 are set to zero
    con = np.sqrt((n-p)*alpha)

    i=1
    
    while i <= ITERMAX:
        # Step 1: update residual
        r = y - X@b0[:,np.newaxis].flatten()    
        psires = rsp.psihub(r/sig0,c)*sig0

        # Step 2: Update the scale
        sig1 = np.linalg.norm(psires)/con

        # Step 3: Update the pseudo-residual
        psires = rsp.psihub(r/sig1,c)*sig1

        # Step 4: regresses X on pseudoresidual
        update = Z@psires # update should be vector not matrix
 
        # Step 6: Check convergence
        crit2 = np.linalg.norm(update)/np.linalg.norm(b0)
        
        # Step 5: update the Beta
        b0 += update
        
        if printitn >0 and i%printitn==0:
            print('hubreg: crit(%4d) = %.9f\n' %(i,crit2))
            
        if crit2 < ERRORTOL: break

        sig0 = sig1
        
    
    if i == ITERMAX: print('error!!! MAXiter = %d crit2 = %.7f\n' % (iter,crit2))
    return b0, sig1, i
    
        
    
    
    