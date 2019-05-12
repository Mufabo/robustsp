'''
madn computes the normalized median absolute deviation estimate of scale, i.e.,

mu_hat = arg min_mu SUM_i rho_TUK(y_i - mu)


  INPUTS: 
          y: data vector of size N x 1

  OUTPUT:  
          sig: normalized median absolute deviations scale estimate
'''
import numpy as np
def madn(y):
    const = 1.20112 if np.iscomplexobj(y) else 1.4815
    sigma_0 = const*np.median(abs(y-np.median(y)))
    return sigma_0