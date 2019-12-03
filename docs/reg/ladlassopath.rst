ladlassopath
=====

ladlassopath computes the LAD-Lasso regularization path (over grid 
of penalty parameter values). Uses IRWLS algorithm.

Inputs
^^^^

*       yx : Numeric data vector of size N or Nx1 (output, respones)
*       Xx : Numeric data matrix of size N x p. Each row represents one 
             observation, and each column represents one predictor (feature). 
*   intcpt : Logical (true/false) flag to indicate if intercept is in the 
             regression model
*      eps : Positive scalar, the ratio of the smallest to the 
             largest Lambda value in the grid. Default is eps = 10^-3. 
*       L  : Positive integer, the number of lambda values EN/Lasso uses.  
             Default is L=120. 
*   reltol : Convergence threshold for IRWLS. Terminate when successive 
             estimates differ in L2 norm by a rel. amount less than reltol.
* printitn : print iteration number (default = 0, no printing)

Outputs
^^^^

*   B    : Fitted LAD-Lasso regression coefficients, a p-by-(L+1) matrix, 
           where p is the number of predictors (columns) in X, and L is 
           the  number of Lambda values. If intercept is in the model, then
           B is (p+1)-by-(L+1) matrix, with first element the intercept.
*  stats : structure with following keys:
 
            'Lambda' = lambda parameters in ascending order
            
	    'MeAD'   = Mean Absolute Deviation (MeAD) of the residuals
            'gBIC'   = generalized Bayesian information criterion (gBIC) value  
                       for each lambda parameter on the grid. 

Examples
^^^^

.. code-block:: python

  import numpy as np
  import robustsp as rsp 
