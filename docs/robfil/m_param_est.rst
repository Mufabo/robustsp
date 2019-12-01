m_param_est
================

The function computes an M-estimator of regression using the assymetric tanh score function. 

Input
^^^^

* receive : 1d-array, received signal
* Theta   : 1d-array, contains initial estimate
* C       : 2d-array, regression matrix of the model y = C * x + n
* parameter : dict, see rsp.set_parameter_book, relevant fields are 'maxiters' and 'break'


Output
^^^^

* th2 : M-estimate of regression
* Theta: M-estimates of regression for all iterations
* kk : iteration index
* residual: residuals given Theta


Examples
^^^^

.. code-block:: python

Reference
^^^^

"Robust Tracking and Geolocation for Wireless Networks in NLOS Environments." 
Hammes, U., Wolsztynski, E., and Zoubir, A.M.
IEEE Journal on Selected Topics in Signal Processing, 3(5), 889-901, 2009.
