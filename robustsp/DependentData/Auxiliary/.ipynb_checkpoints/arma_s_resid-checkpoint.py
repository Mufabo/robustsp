import robustsp as rsp
import numpy as np

def arma_s_resid(x, beta_hat, p, q):
    phi_hat = np.array(beta_hat[:p])
    theta_hat = np.array(beta_hat[p:])
    
    N = len(x)
    r = max(p,q)
    a = np.zeros(N)
    x_sc = rsp.m_scale(x)
    
    if r == 0:
        a = np.array(x)
        return a, x_sc
    elif np.sum(np.abs(np.roots(-1*np.array([-1, *phi_hat])))>1) \
        or np.sum(np.abs(np.roots(-1*np.array([-1, *theta_hat])))>1):
        
        a_sc = 10**10
        return a, rsp.m_scale(a[p:])
    elif p>=1 and q>=1:
        # ARMA residuals
        xArr = lambda ii: x[ii-1::-1] if ii-p-1 < 0 else x[ii-1:ii-p-1:-1]
        aArr = lambda ii: a[ii-1::-1] if ii-q-1 < 0 else a[ii-1:ii-q-1:-1]
        
        for ii in range(r,N):
            a[ii] = x[ii]-phi_hat@xArr(ii)+theta_hat@aArr(ii)
            
        return a, rsp.m_scale(a[p:])
    elif p==0 and q>=0:
         # MA residuals
        aArr = lambda ii: a[ii-1::-1] if ii-q-1 < 0 else a[ii-1:ii-q-1:-1]
        
        for ii in range(r,N):
            a[ii] = x[ii]+theta_hat@aArr(ii)
    elif q==0 and p>=0:
        # AR residuals
         xArr = lambda ii: x[ii-1::-1] if ii-p-1 < 0 else x[ii-1:ii-p-1:-1]
        
         for ii in range(r,N):
            a[ii] = x[ii]-phi_hat@xArr(ii)
           
            
    return a, rsp.m_scale(a[p:])