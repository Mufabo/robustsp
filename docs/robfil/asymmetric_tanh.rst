asymmetric_tanh
================

The function computes the smoothed assymetric tanh score function and its
derivative.

Input
^^^^

* Sigx	: numeric 1d-array, the signal
* c1	: numeric, first clipping point
* c2	: numeric, second clipping point
* x1	: numeric smoothing parameter to make the score function continuous.

Output
^^^^

* phi	    : asymmetric tanh score function
* phi_point : derviative of asymmetric tanh scor function


Examples
^^^^

.. code-block:: python

a =  [[0.5377,    0.8622,   -0.4336,    2.7694,    0.7254],
      [1.8339  ,  0.3188 ,   0.3426  , -1.3499,   -0.0631],
      [-2.2588 ,  -1.3077 ,   3.5784  ,  3.0349 ,   0.7147]]

import numpy as np
import robustsp as rsp

Sig = np.asarray(a)
c1 = 1
c2 = 3
x1 = 2

rsp.asymmetric_tanh(Sig,c1,c2,x1)

Reference
^^^^

"Robust Tracking and Geolocation for Wireless Networks in NLOS Environments." 
Hammes, U., Wolsztynski, E., and Zoubir, A.M.
IEEE Journal on Selected Topics in Signal Processing, 3(5), 889-901, 2009.