'''
Mloc_HUB computes Huber's M-estimate of
location, i.e.,

mu_hat = arg min_mu SUM_i rho_HUB(y_i - mu)


    INPUTS: 
           y: real valued data vector of size N x 1
           c: tuning constant c>=0 . default = 1.345
            default tuning for 95 percent efficiency under 
            the Gaussian model end
           max_iters: Number of iterations. default = 1000
           tol_err: convergence error tolerance. default = 1e-5

   OUTPUT:  
           mu_hat: Hbers's M-estimate of location
'''
import numpy as np
from robustsp.AuxiliaryFunctions.madn import madn
from robustsp.AuxiliaryFunctions.whub import whub

def MLocHUB(y,c=1.345, max_iters = 1000, tol_err = 1e-5):

    y = np.asarray(y) # ensure that y is a ndarray
    
    # previously computed scale estimate
    sigma_0 = madn(y)
    # initial robust location estimate 
    mu_n = np.median(y);
    # computes huber weights
    
    for n in range(max_iters+1):
        w_n = whub(np.absolute(y - mu_n)/sigma_0,c) # compute weights
        mu_n_plus1 = np.sum(w_n*y)/(np.sum(w_n)) # compute weighted average
        if np.absolute(mu_n_plus1 - mu_n)/sigma_0 > tol_err: # breaking condition
            mu_n = mu_n_plus1 # update estimate of mean
            n = n+1 # increment iteration counter      
        else:
            break
            
    return mu_n # final estimate