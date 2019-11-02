import numpy as np
import robustsp as rsp

def m_scale(x, max_iters = 30, delta = 3.25/2, epsilon = 1e-4):
    #x = np.array(xxx)
    N = len(x)
    sigma_k = rsp.madn(x)
        
    k = 0
    w_k = np.ones(N)
    
    while k<= max_iters and sigma_k<10**5:
        w_k[x!=0] = rsp.muler_rho1(x[x!=0]/sigma_k) / (x[x!=0] / sigma_k)**2
        w_k[x==0] = 1
        sigma_k_plus1 = np.sqrt(1/(N*delta)*np.sum(w_k*x**2))
        
        if np.abs(sigma_k_plus1/sigma_k-1) > epsilon:
            sigma_k = sigma_k_plus1
            k += 1
        else:
            break        
    return sigma_k