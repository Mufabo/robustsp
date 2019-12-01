robust_starting_point
==============

The function  robust_starting_point(x,p,q) provides a robust initial estimate for robust ARMA parameter estimation based on BIP-AR(p_long) approximation. It also computes an outlier cleaned signal using BIP-AR(p_long) predictions

Input
^^^^

* x	: 1d-array, the signal
* p	: integer, AR order
* q	: integer, MA order

Output
^^^^

* beta_initial	: 1d-array, robust starting point for AR(p)and MA(q) parameters based on BIP-AR(p_long) approximation
* x_filt	: 1d-array, outlier cleaned signal using BIP-AR(p_long) predictions

Example
^^^^

.. code-block:: python

Note
^^^^
Results different from matlab