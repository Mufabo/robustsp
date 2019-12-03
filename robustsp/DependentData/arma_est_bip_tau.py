import numpy as np
import robustsp as rsp
from scipy.optimize import least_squares as lsq
from scipy.optimize import minimize
'''
  The function arma_est_bip_tau(x,p,q) comuptes BIP tau-estimates of the
  ARMA model parameters. It also computes an outlier cleaned signal using BIP-ARMA(p,q) predictions

%INPUTS
x: 1darray, dtype=float. data (observations/measurements/signal) 
p: int. autoregressive order
q: int. moving-average order

%OUTPUTS
result.ar_coeffs: vector of BIP-AR(p) tau-estimates
result.ma_coeffs: vector of BIP-MA(q) tau-estimates
result.inno_scale: BIP s-estimate of the innovations scale
result.cleaned signal: outlier cleaned signal using BIP-ARMA(p,q) predictions
result.ar_coeffs_init: robust starting point for BIP-AR(p) tau-estimates
result.ma_coeffs_init: robust starting point for BIP-MA(q) tau-estimates

  The function "robust_starting_point" calls "sarimax" from statsmodels to compute classical ARMA parameter estimate based on cleaned
  data. Replace highlighted code by a different (nonrobust) ARMA parameter estimator if you
  do not have the toolbox.

  "Robust Statistics for Signal Processing"
  Zoubir, A.M. and Koivunen, V. and Ollila, E. and Muma, M.
  Cambridge University Press, 2018.

 "Bounded Influence Propagation $\tau$-Estimation: A New Robust Method for ARMA Model Estimation." 
  Muma, M. and Zoubir, A.M.
  IEEE Transactions on Signal Processing, 65(7), 1712-1727, 2017.


'''
def arma_est_bip_tau(x,p,q,meth='SLSQP'):
    # Robust starting point by BIP AR-tau approximation
    beta_initial = rsp.robust_starting_point(x,p,q)[0]
    
    F = lambda beta: rsp.arma_tau_resid_sc(x,beta,p,q)
    
    F_bip = lambda beta: rsp.bip_tau_resid_sc(x,beta,p,q)[0]
    
    beta_arma = minimize(F, beta_initial, method=meth)['x']
    
    beta_bip = minimize(F_bip, beta_initial, method=meth)['x']
    
    a_sc = rsp.arma_tau_resid_sc(x,beta_arma,p,q) # innovations tau-scale for ARMA model
    
    a_bip_sc, x_filt, _ = rsp.bip_tau_resid_sc(x, beta_bip, p, q) # innovations tau-scale for BIP-ARMA model
    
    # final parameter estimate uses the model that provides smallest tau_scale
    beta_hat = beta_arma if a_sc<a_bip_sc else beta_bip
    
    # final tau scale
    a_tau_sc = min(a_sc, a_bip_sc)
    
    # Output the results
    
    results = {'ar_coeffs': -beta_hat[:p],
               'ma_coeffs': -beta_hat[p:],
               'inno_scale': a_tau_sc,
               'cleaned_signal': x_filt,
               'ar_coeffs_init': -1*beta_initial[:p],
               'ma_coeffs_init': -1*beta_initial[p:]}
    
    return results