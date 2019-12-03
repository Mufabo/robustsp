spatmed
==========

Computes the spatial median based on (real or complex) data matrix X.

Input
^^^^

* X: 2d-array of size N x p. Each row represents one 
     observation, and each column represents one variable

* printitn: 0 or 1. print iteration number
            Default = 0, no printing

Output
^^^^

* Spatial median estimate

Examples
^^^^

.. code-block:: python

   import robustsp as rsp
   import numpy as np

   X =  np.asarray\
       ([[2,     3],
         [4,     6],
         [9,     8]])

   rsp.spatmed(X)