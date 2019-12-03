signcm
==========

calculates the spatial sign covariance matrix (SCM).

Input
^^^^

* X: 2d-array of size N x p. Each row represents one observation and each column one variable

* center: bool, if True data centered using the spatial median
          Default = False
* EPS: numeric, lower bound to avoid for floats to avoid divisions by zeros
       Default = 1e-06
Output
^^^^

* spatial sign covariance matrix
* spatial median, will be empty list if center = False

Examples
^^^^

.. code-block:: python

   import robustsp as rsp
   import numpy as np

   X =  np.asarray\
      ([[2,     3],
        [4,     6],
        [9,     8]])

   rsp.signcm(X)