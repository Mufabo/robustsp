'''
[B, stats] = hublasso(y, X,c,lambda,b0,sig0,...)
hublasso computes the M-Lasso estimate for a given penalty parameter 
using Huber's loss function 
INPUT: 
       yx: Numeric data vector of size N (output,respones)
       Xx: Numeric data matrix of size N x p (inputs,predictors,features). 
          Each row represents one observation, and each column represents 
          one predictor
  lambda: positive penalty parameter value
      b0: numeric initial start of the regression vector
    sig0: numeric positive scalar, initial scale estimate.
    c: Threshold constant of Huber's loss function 
  reltol: Convergence threshold. Terminate when successive 
          estimates differ in L2 norm by a rel. amount less than reltol.
          Default is 1.0e-5
printitn: print iteration number (default = 0, no printing)
 iterMAX:  default = 500, maximum number of iterations
OUTPUT:
      b0: regression coefficient vector estimate
    sig0: estimate of the scale 
  psires: pseudoresiduals
'''
import numpy as np
import robustsp as rsp
from scipy.stats import chi2
import scipy

def hublasso(yx,Xx,lambd,b0=None,sig0=None,c=None,reltol=1e-5,printitn=0,iterMAX = 500):
    # ensure that y is Nx1 and not just N and proper formats
    y = np.copy(np.asarray(yx))
    X = np.copy(np.asarray(Xx))
    y = y if not len(y.shape)==2 else y.flatten()

    n,p = X.shape
    realdata = np.isrealobj(y)

    if c is None: c = 1.3415 if realdata else 1.215
    # Default: approx 95 efficiency for Gaussian errors

    csq = c**2

    if realdata:
        qn = chi2.cdf(csq,1)
        alpha = chi2.cdf(csq,3)+csq*(1-qn) # consistency factor for scale
    else:
        qn = chi2.cdf(2*csq,2)
        alpha = chi2.cdf(2*csq,4)+csq*(1-qn) # consistency factor for scale

    con = np.sqrt(n*alpha)
    normb0 = np.linalg.norm(b0)
    b0 = b0.astype(np.longdouble)
    betaold = np.copy(b0)

    i = 0
    for _ in range(iterMAX):
        r = y-X@b0[:,np.newaxis].flatten()

        psires = rsp.psihub(r/sig0,c)*sig0
        sig1 = np.linalg.norm(psires)/con

        crit2 = np.abs(sig0-sig1)

        for jj in range(p):
            psires = rsp.psihub(r/sig1,c)*sig1 # Update the pseudo-residual

            b0[jj] = rsp.SoftThresh(b0[jj]+X[:,jj].T @ psires,lambd)
            r+=X[:,jj]*(betaold[jj]-b0[jj])


        normb = np.linalg.norm(b0)
        crit = np.sqrt(normb0**2 + normb**2 -2*np.real(betaold@b0))/normb

        if printitn > 0:
            pass
            '''
            r = (y-X@b0)/sig1
            objnew = (sig1)*np.sum(rsp.rhofun(r,c)+())
            '''
        if crit<reltol: break

        sig0 = sig1
        betaold = np.copy(b0)
        normb0 = normb

        i+=1

        if printitn > 0:
            pass
            '''
            r = (y-X@b0)/sig1
            objnew = (sig1)*np.sum(rsp.rhofun(r,c)+())
            '''
          
    if printitn > 0:
        b0[np.abs(b0)<5e-9]=0
        r = y- X@b0
        s = np.sign(b0)
        ind = np.arange(p)
        ind2 = ind[s==0]
        psires = rsp.psihub(r/sig1,c)*sig1
        s[ind2] = X[:,ind2]@psires/lambd
        fpeq = -X.T@psires + lambd*s # FP equation equal to zero
        print('lam = %.4f it = %d norm(FPeq1)= %.12f abs(FPeq2)=%.12f\n' \
        % (lambd,i, np.linalg.norm(fpeq),sig1-np.linalg.norm(psires)/con))
        
    return b0,sig0