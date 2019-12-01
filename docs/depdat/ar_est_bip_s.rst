ar_est_bip_s
==============

The function ar_est_bip_s(x,P) comuptes BIP S-estimates of the
AR model parameters. It also computes an outlier cleaned signal using
BIP-AR(P) predictions, and the M-scale of the estimated innovations
series.

Input
^^^^^^

* x	: numeric 1d-array, the data
* P	: scalar int, autoregressive order

Output
^^^^
* phi_hat	: 1d-array of bip tau estimates for each order up to P
* x_filt	: 1d-array, cleaned version of x using robust BIP predictions
* a_scale_final : minimal tau scale of the innovations of BIP AR or AR

Examples
^^^^

Reference
^^^^

"Bounded Influence Propagation tau-Estimation: A New Robust Method for ARMA Model Estimation." 
Muma, M. and Zoubir, A.M.
IEEE Transactions on Signal Processing, 65(7), 1712-1727, 2017.
