rankflassopath
=====

Computes the rank fused-Lasso regression estimates for given fused
penalty value lambda_2 and for a range of lambda_1 values

Inputs
^^^^

*   y       : numeric response 1d-array of size N (real/complex)
*   X       : numeric feature  N x p 2d-array (real/complex)
*   lambda2 : positive penalty parameter for the fused Lasso penalty term
*   L       : number of grid points for lambda1 (Lasso penalty)
*   eps     : Positive scalar, the ratio of the smallest to the 
             largest Lambda value in the grid. Default is eps = 10^-3. 
* printitn  : print iteration number (default = 0, no printing)

Outputs
^^^^

*  b        : numeric regression coefficient vector

Examples
^^^^

.. code-block:: python

  import numpy as np
  import robustsp as rsp 
