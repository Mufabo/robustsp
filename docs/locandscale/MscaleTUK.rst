MscaleTUK
=======

MscaleTUK computes Tukey's M-estimate of
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
          scale_hat: Tukey's M-estimate of scale

EXAMPLES:
^^^^^^^^

.. code-block:: python

   import numpy as np
   import robustsp as rsp

   x = np.array([1, 2, 3])
   
   rsp.MscaleTUK(x)