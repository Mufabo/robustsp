'''
Mscale_TUK computes Tukey's M-estimate of scale.


   INPUTS: 
           y: real valued data vector of size N x 1
           c: tuning constant c>=0 . default = 4.685
               default tuning for 95 percent efficiency under the Gaussian model
           max_iters: Number of iterations. default = 1000
           tol_err: convergence error tolerance. default = 1e-5
   OUTPUT:  
           sigma_hat: Huber's M-estimate of scale
'''
import numpy as np
import random
from robustsp.AuxiliaryFunctions.madn import madn
from robustsp.AuxiliaryFunctions.wtuk import wtuk
from robustsp.AuxiliaryFunctions.rhotuk import rhotuk

def MscaleTUK(y,c=4.685, max_iters = 1000, tol_err = 1e-5):
    y = np.asarray(y) # ensure that y is a ndarray
    
    # initial scale estimate
    sigma_n = madn(y)
    
    # subtract previously computed location
    mu_hat = np.median(y)
    y = y-mu_hat
    
    # length of samples
    N = len(y)
    
    # consistency with the standard deviation at the Gaussian
    random.seed(1)
    u = np.random.randn(10000,1)
    delta = np.mean(rhotuk(u,c))
    
    for n in range(max_iters+1):
        w_n = wtuk(np.absolute(y)/sigma_n,c)
        sigma_n_plus1 = np.sqrt(1/(N*delta)*np.sum(w_n * np.square(y)))
        if np.absolute(sigma_n_plus1 / (sigma_n -1)) > tol_err:
            sigma_n = sigma_n_plus1
            n = n+1
        else:
            break
    
    return sigma_n