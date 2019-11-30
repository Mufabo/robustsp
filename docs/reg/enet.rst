enet
=====

 enet computes the elastic net estimator using the cyclic co-ordinate 
 descent (CCD) algorithm.

Inputs
^^^^

* y: numeric 1d-arraylike of size N. Output/responses.
     If the intercept is in the model, y needs to be centered

* X: numeric 2d-array of size N x p (input/features)
     Columns are assumed to be standardized and centered
     if the intercept is in the model.

* beta : (numeric) regression vector (array) for initial start for CCD algorithm
* lambd : (numeric) a postive penalty parameter value 
* alpha  : (numeric) elastic net tuning parameter in the range [0,1]. If
           not given then use alpha = 1 (Lasso)
* printitn: 0 or 1. print iteration number (default = 0, no printing)
* iterMax: integer. Number of maximum iterations. default = 1000


Outputs
^^^^

* b1    : (numberic) the regression coefficient vector
* it    : (numeric) # of iterations

Examples
^^^^

.. code-block:: python

  import numpy as np
  import robustsp as rsp

  X = np.eye(10, 5)
  y = [1.0, 2.555555, 3.2342342, 4.73567256, 5.13131, 6, 7, 8, 9, 10]
  lambd = 2.4648666648211974
  beta = np.zeros((5,1))

  b1 = rsp.enet(y, X, beta, lambd)[0] # ouput 'it' omitted
