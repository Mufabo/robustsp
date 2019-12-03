Mreg
=====

Mreg computes the M-estimates of regression using an auxiliary scale
estimate. It uses the iterative reweighted least squares (IRWLS) algorithm

Inputs
^^^^

*       y : (numeric) data vector of size N x 1 (output, response vector)
*       X : (numeric) data matrix of size N x p (input, feature matrix)
            If the model has intercept, then first column of X should be a 
            vector of ones. 
* lossfun : (string) either 'huber' or 'tukey' to identify the desired 
            loss function
*      b0 : (numeric) Optional robust initial start (regression vector) of 
            iterations. If not given, we use the LAD regression estimate 
*  verbose: (logical) true of false (default). Set as true if you wish  
            to see convergence as iterations evolve.

Outputs
^^^^

* b1	:
* sig	:

Examples
^^^^

.. code-block:: python

  import numpy as np
  import robustsp as rsp 
