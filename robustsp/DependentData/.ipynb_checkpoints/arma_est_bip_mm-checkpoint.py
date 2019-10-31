import numpy as np
import robustsp as rsp
from scipy.optimize import least_squares as lsq

def arma_est_bip_mm(x,p,q):
    bip_s_est = rsp.arma_est_bip_s(x,p,q)
    
    beta_hat_s = np.array([*bip_s_est['ar_coeffs'], *bip_s_est['ma_coeffs']])

    N = len(x)
    
    F_mm = lambda beta: 1/(N-p)*rsp.muler_rho2(rsp.arma_s_resid(x,beta,p,q)[0])
    
    F_bip_mm = lambda beta: 1/(N-p)*rsp.muler_rho2(rsp.bip_s_resid(x,beta,p,q)[0])
    
    beta_arma_mm = lsq(F_mm, -beta_hat_s,xtol=5*1e-7,ftol=5*1e-7,method='lm')['x']
    
    beta_bip_mm = lsq(F_bip_mm, -beta_hat_s,xtol=5*1e-7,ftol=5*1e-7,method='lm')['x']

    a = rsp.arma_s_resid(x, beta_arma_mm, p, q)[0]
    
    a_sc = rsp.m_scale(a) # innovations m-scale for ARMA model
    
    a_bip = rsp.bip_s_resid(x, beta_bip_mm, p, q)[0]
    
    a_bip_sc = rsp.m_scale(a_bip) # innovations m-scale for BIP-ARMA model

    # final parameter estimate uses the model that provides smallest m_scale
    beta_hat = beta_arma_mm if a_sc<a_bip_sc else beta_bip_mm
    
    # final m-scale
    a_m_sc = min(a_sc, a_bip_sc)
    
    # Output the results
    
    results = {'ar_coeffs': -beta_hat[:p],
               'ma_coeffs': -beta_hat[p:],
               'inno_scale': a_m_sc,
               'cleaned_signal': bip_s_est['cleaned_signal'],
               'ar_coeffs_init': bip_s_est['ar_coeffs'],
               'ma_coeffs_init': bip_s_est['ma_coeffs']}
    
    return results