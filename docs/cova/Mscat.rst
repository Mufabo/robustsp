Mscat
==========

computes M-estimator of scatter matrix for the n x p data matrix X  
using the loss function 'Huber' or 't-loss' and for a given parameter of
the loss function (i.e., q for Huber's or degrees of freedom v for 
the t-distribution). 

Data is assumed to be centered (or the symmetry center parameter = 0)

Input
^^^^

*       X: the data matrix with n rows (observations) and p columns.
*    loss: either 'Huber' or 't-loss' or Tyler (see losspar and examples for Tyler)
* losspar: parameter of the loss function: q in [0,1) for Huber and 
            d.o.f. v >= 0 for t-loss. For Tyler you do not need to specify
            this value. Parameter q determines the treshold 
            c^2 as the qth quantile of chi-squared distribution with p 
            degrees of freedom distribution (Default q = 0.8). Parameter v 
            is the def.freedom of t-distribution (Default v = 3)
            if v = 0, then one computes Tyler's M-estimator
*     invC: initial estimate is the inverse scatter matrix (default = 
            inverse of the sample covariance matrix) 
* printitn: print iteration number (default = 0, no printing)

Output
^^^^
    
* C: the M-estimate of scatter using Huber's weights
* invC: the inverse of C
* iter: nr of iterations
* flag: flag (true/false) for convergence

Examples
^^^^

.. code-block:: python

   import numpy as np
   import robustsp as rsp

   X = np.array([(2, 3), (4, 6), (9, 8)])
   rsp.Mscat(X, 'Huber')
   rsp.Mscat(X, 't-loss')
   rsp.Mscat(X, 't-loss', losspar = 0) # Tyler's loss