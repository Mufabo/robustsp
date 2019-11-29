MLocHUB
=======

MLoc_HUB computes Huber's M-estimate of
location, i.e.,
mu_hat = arg min_mu SUM_i rho_HUB(y_i - mu)
    
INPUTS: 
^^^^^^   
      *  y: real valued data vector of size N x 1
      *     c: tuning constant c>=0 . default = 1.345
            default tuning for 95 percent efficiency under 
            the Gaussian model end
      *     max_iters: Number of iterations. default = 1000
      *     tol_err: convergence error tolerance. default = 1e-5
   
OUTPUT:  
^^^^^^ 
          mu_hat: Hbers's M-estimate of location

EXAMPLES:
^^^^^^^^