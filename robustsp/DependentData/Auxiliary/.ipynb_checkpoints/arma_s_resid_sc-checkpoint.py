import numpy as np
import robustsp as rsp

def arma_s_resid_sc(x, beta_hat, p, q):
    
    phi_hat = beta_hat[:p] if 0<p else []
    theta_hat = beta_hat[p:] if 0<q else [] 
     
    N = len(x)
    r = max(p,q)
    a = np.zeros(N)
    x_sc = rsp.m_scale(x)
    
    poles = lambda x: np.sum(np.abs(np.roots(-1*np.array([-1,*x])))>1)

    if r==0:
        a_sc = np.array(x_sc)
        a = np.array(x)
    else:
        if poles(phi_hat) or poles(theta_hat):
            a_sc = 10**10
        elif p>=1 and q>=1:
            # ARMA Models
            for ii in range(r,N):
                # ARMA residuals
                xArr = x[ii-1::-1] if ii-p-1 < 0 else x[ii-1:ii-p-1:-1]
                aArr = a[ii-1::-1] if ii-q-1 < 0 else a[ii-1:ii-q-1:-1]
                
                a[ii] = x[ii]-np.sum(phi_hat*xArr)+\
                np.sum(theta_hat*aArr)

        elif p==0 and q>=1:
            # MA model
            for ii in range(r,N):
                aArr = a[ii-1::-1] if ii-q-1 < 0 else a[ii-1:ii-q-1:-1]
                a[ii]=x[ii]+np.sum(theta_hat*aArr)
        elif p>=1 and q==0:
            # AR model
            for ii in range(r,N):
                xArr = x[ii-1::-1] if ii-p-1 < 0 else x[ii-1:ii-p-1:-1]
                a[ii] = x[ii] - np.sum(phi_hat*xArr) # AR residuals
    
    a_sc = rsp.m_scale(a[p:])
    
    return a_sc, a[p:]