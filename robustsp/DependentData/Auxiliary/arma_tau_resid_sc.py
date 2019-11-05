import numpy as np
import robustsp as rsp

def arma_tau_resid_sc(x, beta_hat, p, q):
    phi_hat = beta_hat[:p]
    theta_hat = beta_hat[p:]
    
    N = len(x)
    r = max(p,q)
    a = np.zeros(N)
    x_sc = rsp.tau_scale(x)
        
    if r == 0:
        return x_sc
    else:
        if np.sum(np.abs(np.roots(-1*np.array([-1, *phi_hat])))>1) \
        or np.sum(np.abs(np.roots(-1*np.array([-1, *theta_hat])))>1):
            return 10**10
        elif p>=1 and q>=1:
            # ARMA
            xArr = lambda ii: x[ii-1::-1] if ii-p-1 < 0 else x[ii-1:ii-p-1:-1]
            aArr = lambda ii: a[ii-1::-1] if ii-q-1 < 0 else a[ii-1:ii-q-1:-1]

            for ii in range(r,N):
                # Arma residuals
                a[ii] = x[ii]-phi_hat@xArr(ii)+theta_hat@aArr(ii)
        elif p==0 and q>=1:
            aArr = lambda ii: a[ii-1::-1] if ii-q-1 < 0 else a[ii-1:ii-q-1:-1]

            for ii in range(r,N):
                # MA residuals
                a[ii] = x[ii]+theta_hat@aArr(ii)

        elif q==0 and p>=1:
            xArr = lambda ii: x[ii-1::-1] if ii-p-1 < 0 else x[ii-1:ii-p-1:-1]

            for ii in range(r,N):
                # AR residuals
                a[ii] = x[ii]-phi_hat@xArr(ii)

        return rsp.tau_scale(a[p:])