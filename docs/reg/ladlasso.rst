ladlasso
=====

ladlasso computes the LAD-Lasso regression estimates for given complex-  
or real-valued data.  If number of predictors, p, is larger than one, 
then IRWLS algorithm is used, otherwise a weighted median algorithm 
(N > 200) or elemental fits (N<200).

Inputs
^^^^

* yx      : numeric response 1d-array of size N  (real/complex)
* Xx      : numeric feature  N x p 2d-array (real/complex)
* lambd   : non-negative penalty parameter
* b0      : numeric optional initial start of the regression vector for 
            IRWLS algorithm. If not given, we use LSE (when p>1).
* intcpt  : (logical) flag to indicate if intercept is in the model
* reltol  : Convergence threshold for IRWLS. Terminate when successive 
            estimates differ in L2 norm by a rel. amount less than reltol.
* printitn: print iteration number (default = 0, no printing)

Outputs
^^^^
* b1     : (numeric) the regression coefficient vector
* iter   : (numeric) # of iterations (only when IRWLS algorithm is used)

Examples
^^^^

.. code-block:: python

  import numpy as np
  import robustsp as rsp 
