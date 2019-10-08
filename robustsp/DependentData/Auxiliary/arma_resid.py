import numpy as np
import scipy as sp

def arma_resid(xx, beta_hatx, p, q):
    x = np.array(xx)
    beta_hat = np.array(beta_hatx)
    if p>0:
        phi_hat = beta_hat[:p]
    else:
        phi_hat = []
        
    if q>0:
        theta_hat = beta_hat[p:]
    else:
        theta_hat = []
        
    N = len(x)
    r = max(p,q)
    a = np.zeros(N)
    
    if r==0:
        a = np.array(x)
    elif p>=1 and q >=1:
        for ii in range(r,N):
            # ARMA residuals
            xArr = x[ii-1::-1] if ii-p-1 < 0 else x[ii-1:ii-p-1:-1]
            aArr = a[ii-1::-1] if ii-q-1 < 0 else a[ii-1:ii-q-1:-1]
            a[ii] = x[ii]-phi_hat@xArr+theta_hat@aArr
    elif p==0 and q>=1:
        # MA model
        for ii in range(r,N):
            aArr = a[ii-1::-1] if ii-q-1 < 0 else a[ii-1:ii-q-1:-1]
            a[ii] = x[ii] + theta_hat@aArr
    elif p>=1 and q == 0:
        # AR models
        for ii in range(r,N):
            # AR residuals
            xArr = x[ii-1::-1] if ii-p-1 < 0 else x[ii-1:ii-p-1:-1]
            a[ii] = x[ii]-phi_hat@xArr
    return a