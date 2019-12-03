rankflasso
=====

Computes the rank fused-Lasso regression estimates for given fused
penalty value lambda_2 and for a range of lambda_1 values

Inputs
^^^^

*  y       : 1darray, dtype either real or complex, numeric response N vector (real/complex)
*  X       : 2darray, numeric feature  N x p matrix (real/complex)
*  lambda1 : positive penalty parameter for the Lasso penalty term
*  lambda2 : positive penalty parameter for the fused Lasso penalty term
*  b0      : numeric optional initial start (regression vector) of 
             iterations. If not given, we use LSE (when p>1).
*printitn  : print iteration number (default = 0, no printing)

Outputs
^^^^

*  b      : numeric regression coefficient vector
*  iter   : positive integer, the number of iterations of IRWLS algorithm

Examples
^^^^

.. code-block:: python

  import numpy as np
  import robustsp as rsp 
