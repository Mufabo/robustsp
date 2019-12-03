hublassopath
=====

hublassopath computes the M-Lasso regularization path (over grid 
of penalty parameter values) using Huber's loss function

Inputs
^^^^

* y	: numeric 1d-array of size N, can be complex or real
* X	: numeric 2d-array of size N x p
* c	: scalar, threshold for Huber's loss
		Default : 1.3415 for real data else 1.215
* intcpt: Bool, flag to indicate if intercept is in regression mode
                Default : True
* eps	: positive scalar, the ratio of the smallest to the largest Lambda 
          value in the grid
                Default : 1e-03
* L	: positive integer, number of lambda values EN/Lasso uses
		Default : 120
* reltol: Convergence threshold for IRWLS. Terminate when successive.
          estimates differ in L2 norm by a rel. amount less than reltol.
* printitn: 0 or 1, print iteration number
		Default : 0
 
Outputs
^^^^

* B    	: Fitted M-Lasso regression coefficients, a p-by-(L+1) matrix, 
          where p is the number of predictors (columns) in X, and L is 
          the  number of Lambda values.
* B0 	: estimates values of intercepts
* stats : Dictionary with following Keys:
 
          'Lambda' = lambda parameters in ascending order

          'sigma'  = estimates of the scale (a (L+1) x 1 vector)

          'gBIC'   = generalized Bayesian information criterion (gBIC) value  
                     for each lambda parameter on the grid. 

Examples
^^^^

.. code-block:: python

  import numpy as np
  import robustsp as rsp 
  
  yx = [5.5, 2, 4 ,5]
  Xx = np.eye(4)

  rsp.hublassopath(yx, Xx)