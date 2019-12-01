ekf_toa
================

EKF for tracking with time-of-arrival (ToA) estimates.

Input
^^^^

* r_ges : 2d-array, measured distances (M x N)
* theta_init : 1d-array, initial state estimate
* BS : 2d-array, Base station positions
* parameter: dict, with fields 'P0', 'R', 'Q', 'G', 'A', 'singlescor'. 
             See rsp.set_parameters_book and create_environment_book for details

Output
^^^^

* th_hat: 1d-array, state estimates
* P_min : 2d-array, apriori covariance
* P	: 2d-array, aposteriori covariance      

Examples
^^^^

.. code-block:: pythonthon

Reference
^^^^

"Robust Tracking and Geolocation for Wireless Networks in NLOS Environments." 
Hammes, U., Wolsztynski, E., and Zoubir, A.M.
IEEE Journal on Selected Topics in Signal Processing, 3(5), 889-901, 2009.