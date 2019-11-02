import numpy as np
import scipy as sp
import robustsp as rsp

def bip_resid(xx, beta_hatx, p, q):
    x = np.array(xx)
    beta_hat = np.array(beta_hatx)
    
    phi_hat = beta_hat[:p] if p>0 else []
    theta_hat = beta_hat[p:] if q>0 else []
    
    N = len(x)
    r = max(p,q)
    a_bip = np.zeros(N)
    x_sc = rsp.m_scale(x)
    kap2 = 0.8724286
    
    if np.sum(np.abs(np.roots(np.array([1, *phi_hat*-1])))>1)\
    or np.sum(np.abs(np.roots(np.array([1, *theta_hat])))>1):
        sigma_hat = x_sc
        a_bip = np.array(x)
    else:
        lamb = rsp.ma_infinity(phi_hat, -theta_hat, 100)
        sigma_hat = np.sqrt(x_sc**2 / (1+kap2*np.sum(lamb**2)))
        
        if r == 0:
            a_bip = np.array(x)
        else:
            if p>=1 and q>=1:
                # ARMA Models
                for ii in range(r,N):
                    # BIP-ARMA residuals
                    xArr = x[ii-1::-1] if ii-p-1 < 0 else x[ii-1:ii-p-1:-1]
                    abArr = a_bip[ii-1::-1] if ii-p-1 < 0 else a_bip[ii-1:ii-p-1:-1]
                    aqArr = a_bip[ii-1::-1] if ii-q-1 < 0 else a_bip[ii-1:ii-q-1:-1]
                    a_bip[ii] = x[ii]-phi_hat@(xArr-abArr+sigma_hat*rsp.eta(abArr/sigma_hat))+sigma_hat*theta_hat@rsp.eta(aqArr/sigma_hat)
                    r +=1
            elif p==0 and q>=1:
                # MA models
                for ii in range(r,N):
                    # BIP-MA residuals
                    aArr = a_bip[ii-1::-1] if ii-q-1 < 0 else a_bip[ii-1:ii-q-1:-1]
                    a_bip[ii]=x[ii]+theta_init*sigma_hat*rsp.eta(aArr/sigma_hat)
            elif p>=1 and q==0:
                # AR models
                for ii in range(r,N):
                    # BIP-AR residuals
                    xArr = x[ii-1::-1] if ii-p-1 < 0 else x[ii-1:ii-p-1:-1]
                    aArr = a_bip[ii-1::-1] if ii-p-1 < 0 else a_bip[ii-1:ii-p-1:-1]
                    a_bip[ii] = x[ii]+phi_hat@(xArr-aArr)+sigma_hat*rsp.eta(aArr/sigma_hat)

    return a_bip[p:]