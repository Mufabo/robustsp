import numpy as np
import robustsp as rsp

def arma_s_resid_sc(x, beta_hat, p, q):
    
    phi_hat = beta_hat[:p]
    theta_hat = beta_hat[p:] 
     
    N = len(x)
    r = max(p,q)
    a = np.zeros(N) # will contain residuals
    x_sc = rsp.m_scale(x)
    
    poles = lambda xc: np.sum(np.abs(np.roots(-1*np.array([-1,*xc])))>1)

    if r==0:
        a_sc = np.array(x_sc)
        a = np.array(x)
    else:
        if poles(phi_hat) or poles(theta_hat):
            return 10**10
        else:
            xArr = lambda ii: x[ii-1::-1] if ii-p-1 < 0 else x[ii-1:ii-p-1:-1]
            aArr = lambda ii: a[ii-1::-1] if ii-q-1 < 0 else a[ii-1:ii-q-1:-1]

            if p>=1 and q>=1:
                # ARMA Models
                for ii in range(r,N):
                    # ARMA residuals             
                    a[ii] = x[ii]-np.sum(phi_hat*xArr(ii))+theta_hat@aArr(ii)
            elif p==0 and q>=1:
                # MA model
                for ii in range(r,N):
                    a[ii]=x[ii]+theta_hat@aArr(ii)
            elif p>=1 and q==0:
                # AR model
                for ii in range(r,N):
                    a[ii] = x[ii] - phi_hat@xArr(ii) # AR residuals  
    return rsp.m_scale(a[p:])