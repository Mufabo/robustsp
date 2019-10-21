import numpy as np
import robustsp as rsp

def bip_s_resid(x, beta_hat,p,q):
    phi_hat = np.array(beta_hat[:p])
    theta_hat = np.array(beta_hat[p:])
    
    N = len(x)
    r = max(p,q)
    a_bip = np.zeros(N)
    x_sc = rsp.m_scale(x)
    kap2 = 0.8724286
    
    xArr = lambda ii: x[ii-1::-1] if ii-p-1 < 0 else x[ii-1:ii-p-1:-1]
    aqArr = lambda ii: a_bip[ii-1::-1] if ii-q-1 < 0 else a_bip[ii-1:ii-q-1:-1]
    apArr = lambda ii: a_bip[ii-1::-1] if ii-p-1 < 0 else a_bip[ii-1:ii-p-1:-1]
    
    if np.sum(np.abs(np.roots(-1*np.array([-1, *phi_hat])))>1) \
        or np.sum(np.abs(np.roots(-1*np.array([-1, *theta_hat])))>1):
        
        sigma_hat = x_sc
    else:
        # MA infinity approximation to compute scale used in eta function
        lamb = rsp.ma_infinity(phi_hat, -theta_hat, 100)
        
        # Scale used in eta function
        sigma_hat = np.sqrt(x_sc**2 /(1+kap2*np.sum(lamb**2)))
        
    if r == 0:
        a_sc_bip = x_sc
        a_bip = np.array(x)
        return a_bip
    else:
        if np.sum(np.abs(np.roots(-1*np.array([-1, *phi_hat])))>1) \
        or np.sum(np.abs(np.roots(-1*np.array([-1, *theta_hat])))>1):
            a_bip_sc = 10**10
        else:    
            if p>=1 and q>=1:
                # ARMA model
                for ii in range(r,N):
                    # BIP-ARMA residuals
                    a_bip[ii] = x[ii] -phi_hat@(xArr(ii)-apArr(ii)+sigma_hat*rsp.eta(apArr(ii)/sigma_hat))\
                    +theta_hat@(sigma_hat*rsp.eta(aqArr(ii)/sigma_hat))
            elif p==0 and q>=1:
                # MA model
                for ii in range(r,N):
                    # BIP-MA residuals
                    a_bip[ii] = x[ii]+theta_hat@(sigma_hat*rsp.eta(aqArr(ii)/sigma_hat))
            elif p>=1 and q==0:
                # AR model
                for ii in range(r,N):
                    # BIP-AR residuals
                    a_bip[ii] = x[ii] - phi_hat@(xArr(ii)-apArr(ii)+sigma_hat*rsp.eta(apArr(ii)/sigma_hat))

        a_bip_sc = rsp.m_scale(a_bip[p:])

        # cleaned signal
        x_filt = np.array(x)
        
        for ii in range(p,N):
            x_filt[ii] = x[ii]-a_bip[ii]+sigma_hat*rsp.eta(a_bip[ii]/sigma_hat)
            
        return a_bip, x_filt