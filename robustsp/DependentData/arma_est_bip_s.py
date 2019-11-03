import numpy as np
import robustsp as rsp
from scipy.optimize import least_squares as lsq
from scipy.optimize import minimize
'''
The function  arma_est_bip_s(x,p,q) comuptes BIP S-estimates of the
ARMA model parameters. It also computes an outlier cleaned signal using BIP-ARMA(p,q) predictions

inputs:

x: 1darray-like, dtype=float. data
p: int, autoregressive order
q: int, monving-average order
meth: method used for optimization in scipy.optimize.minimize, default = 'SLSQP'
outputs

result: dictionary:
    result.ar_coeffs: 1darray of BIP-AR(p) S-estimates
    result.ma_coeffs: 1darray of BIP-MA(q) S-estimates
    result.inno_scale: float, s-estimate of the innovations scale
    result.cleaned signal: 1darray, outlier cleaned signal using BIP-ARMA(p,q) predictions
    result.ar_coeffs_init: 1darray, robust starting point for BIP-AR(p) S-estimates
    result.ma_coeffs_init: 1darray, robust starting point for BIP-MA(q) S-estimates
'''

def arma_est_bip_s(x,p,q,meth='SLSQP'):  
    if p==0 and q == 0:
        inno_scale = rsp.m_scale(x)
        print('at least p or q has to be non-zero')
    if len(x) <= p+q:
        raise ValueError('There are too many parameters to estimate for chosen data size. Reduce model order or use a larger data set.')
    
    # Robust starting point by BIP AR-S approximation
    beta_initial = rsp.robust_starting_point(x,p,q)[0] 
    
    F = lambda beta: rsp.arma_s_resid_sc(x, beta, p, q) 

    F_bip = lambda beta: rsp.bip_s_resid_sc(x, beta, p, q)[0] 
    
    beta_arma = minimize(F,beta_initial, method=meth)['x']
    
    beta_bip = minimize(F_bip,beta_initial, method=meth)['x']
    
    a_sc = rsp.arma_s_resid_sc(x, beta_arma, p, q) # innovations m-scale for ARMA model
    
    a_bip_sc, x_filt, _ = rsp.bip_s_resid_sc(x, beta_bip, p, q) # innovations m-scale for BIP-ARMA model
    
    # final parameter estimate uses the model that provides smaller
    # m-scale
    beta_hat = beta_arma if a_sc < a_bip_sc else beta_bip
    
    # final m-scale
    a_m_sc = min(a_sc, a_bip_sc)
    
    results = {'ar_coeffs': -1*beta_hat[:p],
              'ma_coeffs': -1*beta_hat[p:],
              'inno_scale': a_m_sc,
              'cleaned_signal': x_filt,
              'ar_coeffs_init': -1* beta_initial[:p],
              'ma_coeffs_init': -1* beta_initial[p:]}
    
    return results

'''
    if len(beta_initial) != 1:
        # optimize on residuals
        F = lambda beta: rsp.arma_s_resid_sc(x, beta, p, q)[1] 
        
        F_bip = lambda beta: rsp.bip_s_resid_sc(x, beta, p, q)[2]
    else:
        # optimize directly on scale
        F = lambda beta: rsp.arma_s_resid_sc(x, beta, p, q)[0] 

        F_bip = lambda beta: rsp.bip_s_resid_sc(x, beta, p, q)[0] 
    
    beta_arma = lsq(F, beta_initial,xtol=5*1e-7,ftol=5*1e-7,method='lm')['x']
    # different from matlab print('beta_arma: ', beta_arma)
    # but has lower residuals than matlab print('F(beta_arma): ',rsp.arma_s_resid_sc(x,beta_arma,p,q))
    beta_bip = lsq(F_bip, beta_initial,xtol=5*1e-7,ftol=5*1e-7,method='lm')['x']
'''