elemfits
=====

elemfits compute the :math:`\frac{N(N-1)}{2}` elemental fits, i.e., intercepts :math:`b_{0,ij}`
and slopes :math:`b_{1,ij}`, that define a line :math:`y = b{0}+b_1x` that passes through 
the data points :math:`(x_i,y_i)` and :math:`(x_j,y_j)`, i<j, where i, j in {1, ..., N}. 
and the respective weights :math:`| x_i - x_j |`

Inputs
^^^^
*     y : (numeric) 1darray of real-valued outputs (response vector)
*    x : (numeric) 1darray vector of inputs (feature vector) 
Outputs
^^^^
    * beta: (numeric) N*(N-1)/2 matrix of elemental fits 
    * w: (numeric) N*(N-1)/2 matrix of weights
    
Note
^^^^
 
Numpy uses C memory order whereas Matlab uses Fortran ordering, thus the 
indices of the solution elements are different

Examples
^^^^

.. code-block:: python

  import numpy as np
  import robustsp as rsp
 
  beta, w = rsp.elemfits(np.random.randn(5), np.random.randn(5))

