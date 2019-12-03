hublasso
=====

hublasso computes the M-Lasso estimate for a given penalty parameter 
using Huber's loss function 

Inputs
^^^^

*  yx	: Numeric 1d-array of size N (output,respones)
*  Xx	: Numeric 2d-array of size N x p (inputs,predictors,features). 
          Each row represents one observation, and each column represents 
          one predictor
* lambd	: positive penalty parameter value
* b0	: numeric 1d-array initial start  of the regression vector
* sig0	: numeric positive scalar, initial scale estimate.
*  c	: Threshold constant of Huber's loss function 
* reltol: Convergence threshold. Terminate when successive 
          estimates differ in L2 norm by a rel. amount less than reltol.
          Default is 1.0e-5
* printitn: print iteration number (default = 0, no printing)
* iterMAX:  default = 500, maximum number of iterations


Outputs
^^^^

* b0	: 1d-array regression coefficient vector estimate
* sig0	: scalar, estimate of the scale 
* psires: 1d-array, pseudoresiduals

Examples
^^^^

.. code-block:: python

  import numpy as np
  import robustsp as rsp 

  yx = np.array([ 1.375, -2.125, -0.125, 0.875])

  Xx = np.array([[ 0.8660254 , -0.28867513, -0.28867513, -0.28867513],
    [-0.28867513,  0.8660254 , -0.28867513, -0.28867513],
    [-0.28867513, -0.28867513,  0.8660254 , -0.28867513],
    [-0.28867513, -0.28867513, -0.28867513,  0.8660254 ]])

  lambd = 0.6163512806474756

  b0 = np.array([0., 0., 0., 0.])

  sig0 = 1.812638977729619

  c = 1.3415

  beta, sigma, residuals = rsp.hublasso(yx,Xx,lambd,b0,sig0,c)