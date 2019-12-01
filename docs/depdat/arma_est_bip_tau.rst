arma_est_bip_tau
==============

The function arma_est_bip_tau(x, p, q) comuptes the BIP tau-estimation of the
ARMA model parameters. It can also be used as a stand-alone
M-estimator.

Input
^^^^^^

* x		: numeric 1d-array, the data
* p		: scalar int, autoregressive order
* q		: scalar int, moving-average order


Output
^^^^
* results 		: dict, with following fields
  1. 'ar_coeffs'	: AR coefficients
  2. 'ma_coeffs'	: MA coefficients
  3. 'inno_scale'	: BIP s-estimate of the innovations scale
  4. 'cleaned_signal'	: outlier cleaned signal using BIP-ARMA(p,q) predictions
  5. 'ar_coeffs_init'	: robust starting point for BIP-AR(p) MM-estimates
  6. 'ma_coeffs_init'	: robust starting point for BIP-MA(q) MM-estimates

Examples
^^^^

Reference
^^^^

"Bounded Influence Propagation tau-Estimation: A New Robust Method for ARMA Model Estimation." 
Muma, M. and Zoubir, A.M.
IEEE Transactions on Signal Processing, 65(7), 1712-1727, 2017.