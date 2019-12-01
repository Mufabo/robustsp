arma_est_bip_m
==============

The function arma_est_bip_m(x, p, q) comuptes the BIP M-estimation of the
ARMA model parameters. It can also be used as a stand-alone
M-estimator.

Input
^^^^^^

* x		: numeric 1d-array, the data
* p		: scalar int, autoregressive order
* q		: scalar int, moving-average order
* beta_hat_s	: BIP S-estimate
* a_sc_final	: M scale estimate of residuals of BIP S-estimate

Output
^^^^
* phi_bip_mm	: vector of BIP-AR(p) MM-estimates
* theta_bip_mm	: vector of BIP-MA(q) MM-estimates
Examples
^^^^

Reference
^^^^

"Bounded Influence Propagation tau-Estimation: A New Robust Method for ARMA Model Estimation." 
Muma, M. and Zoubir, A.M.
IEEE Transactions on Signal Processing, 65(7), 1712-1727, 2017.