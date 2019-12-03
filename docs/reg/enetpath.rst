enetpath
=====

 enethpath computes the elastic net (EN) regularization path (over grid 
 of penalty parameter values). Uses pathwise CCD algorithm. 

Inputs
^^^^

* y	: Numeric 1darray of size N (output, respones)
* X 	: Nnumeric 2darray of size N x p. Each row represents one 
          observation, and each column represents one predictor (feature). 
* intcpt: Logical flag to indicate if intercept is in the model
* alpha : Numeric scalar, elastic net tuning parameter in the range [0,1].
          If not given then use alpha = 1 (Lasso)
* eps	: Positive scalar, the ratio of the smallest to the 
          largest Lambda value in the grid. Default is eps = 10^-4. 
* L 	: Positive integer, the number of lambda values EN/Lasso uses.  
	Default is L=100. 
* printitn: print iteration number (default = 0, no printing)

Outputs
^^^^

* B	: Fitted EN/Lasso regression coefficients, a p-by-(L+1) matrix, 
          where p is the number of predictors (columns) in X, and L is 
          the  number of Lambda values. If intercept is in the model, then
          B is (p+1)-by-(L+1) matrix, with first element the intercept.
* stats : Dictionary with following keys:

          'Lambda' = lambda parameters in ascending order

          'MSE' = Mean squared error (MSE)

          'BIC' = Bayesian information criterion values

Examples
^^^^

.. code-block:: python

  import numpy as np
  import robustsp as rsp
  import scipy.io
  import pkg_resources
 
  # load the data
  path = pkg_resources.resource_filename('robustsp', 'data/prostate.mat')
  X = scipy.io.loadmat(path,struct_as_record=False)['X']
  y = scipy.io.loadmat(path,struct_as_record=False)['y']

  B, stats = rsp.enetpath(y, X, intcpt = True)