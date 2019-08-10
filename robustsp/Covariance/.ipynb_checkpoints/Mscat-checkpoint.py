"""
[C,invC,iter,flag] = Mscat(X,loss,...)
computes M-estimator of scatter matrix for the n x p data matrix X  
using the loss function 'Huber' or 't-loss' and for a given parameter of
the loss function (i.e., q for Huber's or degrees of freedom v for 
the t-distribution). 

Data is assumed to be centered (or the symmetry center parameter = 0)
              
INPUT:
       X: the data matrix with n rows (observations) and p columns.
    loss: either 'Huber' or 't-loss' or 'Tyler'
 losspar: parameter of the loss function: q in [0,1) for Huber and 
          d.o.f. v >= 0 for t-loss. For Tyler you do not need to specify
          this value. Parameter q determines the treshold 
          c^2 as the qth quantile of chi-squared distribution with p 
          degrees of freedom distribution (Default q = 0.8). Parameter v 
          is the def.freedom of t-distribution (Default v = 3)
          if v = 0, then one computes Tyler's M-estimator
    invC: initial estimate is the inverse scatter matrix (default = 
          inverse of the sample covariance matrix) 
printitn: print iteration number (default = 0, no printing)
OUTPUT:
       C: the M-estimate of scatter using Huber's weights
    invC: the inverse of C
    iter: nr of iterations
    flag: flag (true/false) for convergence
"""
import numpy as np
from scipy.stats.distributions import chi2
import scipy as sp

def Mscat(X, loss, losspar=None,invCx=None,printitn=0,MAX_ITER = 1000,EPS = 1.0e-5):
    def tloss_consistency_factor(p,v):
        '''
        computes the concistency factor b = (1/p) E[|| x ||^2 u_v( ||x||^2)] when
        x ~N_p(0,I).
        '''
        sfun = lambda x: (x**(p/2)/(v+x) * np.exp(-x/2))
        c = 2**(p/2)*sp.special.gamma(p/2)
        q = (1/c)*\
        sp.integrate.quad(sfun,0,np.inf)[0]
        return ((v+p)/p)*q #consistency factor
    
    X = np.asarray(X);
    n,p = X.shape
    realdata = np.isrealobj(X)

    # SCM initial start 
    invC = np.linalg.pinv(X.conj().T @ X / n) if invCx==None else np.copy(invCx) 

    if loss=='Huber':
        ufun = lambda t,c: ((t<=c) + (c/t)*(t>c)) # weight function u(t)
        q = 0.9 if losspar == None else losspar
        if np.isreal(q) and np.isfinite(q) and 0<q and q<1:
            if realdata:
                upar = chi2.ppf(q, df=p) # threshold for Huber's weight u(t;.)
                b = chi2.cdf(upar,p+2)+(upar/p)*(1-q) # consistency factor
            else:
                upar = chi2.ppf(q,2*p)/2
                b = chi2.cdf(2*upar,2*(p+1))+(upar/p)*(1-q)
        else:
            raise ValueError('losspar is a real number in [0,1] and not %s for Huber-loss' % q)
        const = 1/(b*n)
    if loss == 't-loss':
        # d.o.f v=3 is used as the default parameter for t-loss
        # otherwise use d.o.f. v that was given
        upar = 3 if losspar==None else losspar
        if not np.isreal(upar) or not np.isfinite(upar) or upar < 0:
            raise ValueError('losspar should be a real number greater 0 and not %s for t-loss' % q)
        if realdata and upar !=0:
            # this is for real data
            ufun = lambda t,v: 1/(v+t) # weight function
            b = tloss_consistency_factor(p,upar)
            const = (upar+p)/(b*n)
        if not realdata and upar != 0:
            # this is for complex data
            ufun = lambda t,v: 1/(v+2*t) # weight function
            b = tloss_consistency_factor(2*p,upar)
            const = (upar+2*p)/(b*n)
        if upar==0:
            # Tylers M-estimator
            ufun = lambda t,v: 1/t
            const = p/n

    for i in range(MAX_ITER):
        t = np.real(np.sum((X@invC)*np.conj(X),axis=1)) # norms
        C = const* X.conj().T @ (X * ufun(t,upar)[:,None])
        d = np.max(np.sum(np.abs(np.eye(p)-invC@C),axis=1))

        if printitn >0 and (i+1)%printitn == 0:
            print("At iter = %d, dis=%.6f\n"%(i,d))
        invC = np.linalg.pinv(C)
        if d<=EPS: break

    if i == MAX_ITER: print("WARNING! Slow convergence: the error of the solution is %f\n'"%d)
    return C,invC,i,i==MAX_ITER-1