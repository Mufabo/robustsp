import robustsp as rsp
import numpy as np

'''
%   The function spec_arma_est_bip_tau(x,p,q) comuptes spectral estimates using the BIP tau-estimates of the
%   ARMA model parameters.
%
%% INPUTS
% x: data (observations/measurements/signal) 
% p: autoregressive order
% q: moving-average order
%
%% OUTPUTS
% PxxdB: spectral estimate in dB
% Pxx: spectral estimate
% w: frequency (0,pi)
% sigma_hat: BIP tau-scale estimate of the innovations
'''

def spec_arma_est_bip_mm(x,p,q):
    x = x - np.median(x)
    
    N = len(x)
    
    w = np.linspace(0,np.pi,int(N/2))
    # Digital frequency must be used for this calculation
    s = np.exp(1j*w)
    
    result = rsp.arma_est_bip_mm(x,p,q)
    
    beta_hat = np.zeros(p+q)
    beta_hat[:p] = result['ar_coeffs']
    beta_hat[p:] = result['ma_coeffs']
    
    Xx = np.polyval([1,*beta_hat[p:]],s)\
    / np.polyval([1,*beta_hat[:p]],s)
    
    sigma_hat = result['inno_scale']
    # BIP-ARMA tau spectral estimate 
    Pxx = (sigma_hat**2)/(2*np.pi)*np.abs(Xx)**2
    
    PxxdB = 10 *np.log10(Pxx)
    
    return PxxdB, Pxx, w, sigma_hat