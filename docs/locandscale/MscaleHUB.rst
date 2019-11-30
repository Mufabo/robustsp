MscaleHUB
=======

MscaleHUB computes Huber's M-estimate of
scale
    
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
          sigma_hat: Huber's M-estimate of scale
EXAMPLES:
^^^^^^^^

.. code-block:: python

   import robustsp as rsp
   import numpy as np

   x = np.array([1, 2, 3])
   c = 5
   rsp.MscaleHUB(x)
   rsp.MscaleHUB(x, c)