'''
  The function  arma_est_bip_m(x,p,q) comuptes the BIP M-estimation step for BIP MM estimates of the
  ARMA model parameters. It can also be used as a stand-alone
  M-estimator.

%INPUTS
x: data (observations/measurements/signal) 
p: autoregressive order
q: moving-average order
beta_hat_s: BIP S-estimate
a_sc_final: M scale estimate of residuals of BIP S-estimate

%OUTPUTS
phi_bip_mm: vector of BIP-AR(p) MM-estimates
theta_bip_mm: vector of BIP-MA(q) MM-estimates
'''

import numpy as np
from scipy.optimize import least_squares as lsq
import robustsp as rsp

def arma_est_bip_m(x, p, q, beta_hat_s, a_sc_final):
    N = len(x);
    x = np.array(x)
    F_mm = lambda beta: sp.sqrt(1/(N-p)*sp.sum(rsp.muler_rho2(rsp.arma_resid(x/a_sc_final, beta, p, q))))
    F_bip_mm = lambda beta: sp.sqrt(1/(N-p)*sp.sum(rsp.muler_rho2(rsp.bip_resid(x/a_sc_final, beta, p, q))))
    
    beta_arma_mm = lsq(F_mm, beta_hat_s,xtol=5*1e-5,ftol=5*1e-5,method='lm')[0]
    beta_bip_mm = lsq(F_bip_mm, beta_hat_s,xtol=5*1e-5,ftol=5*1e-5,method='lm')[0]
    
    a_rho2_mm = 1/(N-p)*np.sum(muler_rho2(rsp.arma_resid(x/a_sc_final,beta_arma_mm,p,q)))
    a_bip_rho2_mm = 1/(N-p)*np.sum(muler_rho2(rsp.bip_resid(x/a_sc_final,beta_bip_mm,p,q)))
    
    beta_hat = beta_arma_mm if a_rho2_mm<a_bip_rho2 else beta_bip_mm
    
    # Output the results
    phi_bip_mm = -beta_hat[:p] if p>0 else []
    theta_bip_mm = -beta_hat[p:] if q>0 else []
    
    return phi_bip_mm, theta_bip_mm