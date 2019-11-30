MLocTUK
=======

MLocTUK computes Tukeys's M-estimate of
location, i.e.,

.. math::
   \hat{\mu}  = arg min_{\mu} \sum_{i}  \rho_{TUK}(y_{i} - \mu)
    
INPUTS: 
^^^^^^   
      *  y:    		real valued data vector of size N x 1
      *  c: 		tuning constant c>=0 . default = 4.685
            		default tuning for 95 percent efficiency under 
            		the Gaussian model end
      *  max_iters: 	Number of iterations. default = 1000
      *  tol_err: 	convergence error tolerance. default = 1e-5
   
OUTPUT:  
^^^^^^ 
          mu_hat: Tukey's M-estimate of location

EXAMPLES:
^^^^^^^^

.. code-block:: python

   x = np.array([1, 2, 3, 4]) 
   c = 5    
   MLocTUK(x) 
   MLocTUK(x, c)