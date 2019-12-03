hubreg
=====

hubreg computes the joint M-estimates of regression and scale using 
Huber's criterion. Function works for both real- and complex-valued data.

Input
^^^^

*       yx: Numeric data vector of size N (output, respones)
*       Xx: Numeric data matrix of size N x p. Each row represents one 
           observation, and each column represents one predictor (feature). 
           If the model has an intercept, then first column needs to be a  
           vector of ones. 
*        c: numeric threshold constant of Huber's function
*     sig0: (numeric) initial estimator of scale [default: SQRT(1/(n-p)*RSS)]
*       b0: initial estimator of regression (default: LSE)  
* printitn: print iteration number (default = 0, no printing)
* ITERMAX:  default = 2000, maximum number of iterations
* ERRORTOL: default = 1e-5, ERROR TOLERANCE FOR HALTING CRITERION

OUTPUT
^^^^
*      b1: the regression coefficient vector estimate 
*    sig1: the estimate of scale 
*    iter: the # of iterations 



Examples
^^^^

.. code-block:: python

  import numpy as np
  import robustsp as rsp 
