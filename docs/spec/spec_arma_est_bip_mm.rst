spec_arma_est_bip_mm
===================

The function spec_arma_est_bip_mm(x,p,q) comuptes spectral estimates using the BIP MM-estimates of the
ARMA model parameters.

Input
^^^^

* x	: 1d-array, the signal
* p	: integer, AR order
* q	: integer, MA order

Output
^^^^

* PxxdB		: Spectral estimate in dB
* Pxx		: Spectral estimate
* w		: frequency in (0, pi)
* sigma_hat	: BIP M-scale estimate of the innovations

Example
^^^^

.. code-block:: python

Reference
^^^^

"Bounded Influence Propagation $\tau$-Estimation: A New Robust Method for ARMA Model Estimation." 
Muma, M. and Zoubir, A.M.
IEEE Transactions on Signal Processing, 65(7), 1712-1727, 2017.

