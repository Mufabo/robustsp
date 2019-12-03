ladreg
=====

ladreg computes the LAD regression estimate 

Inputs
^^^^

*        y: numeric response N vector (real/complex)
*        X: numeric feature  N x p matrix (real/complex)
*   intcpt: (logical) flag to indicate if intercept is in the model
*       b0: numeric optional initial start of the regression vector for 
            IRWLS algorithm. If not given, we use LSE (when p>1).
* printitn: print iteration number (default = 0, no printing) and
            other details


Outputs
^^^^

*       b1: (numeric) the regression coefficient vector
*     iter: (numeric) # of iterations (given when IRWLS algorithm is used)

Examples
^^^^

.. code-block:: python

  import numpy as np
  import robustsp as rsp 
