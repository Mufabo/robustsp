import numpy as np
import robustsp as rsp
import statsmodels.tsa.api as tsa
'''
  The function  robust_starting_point(x,p,q) provides a robust initial estimate for robust ARMA parameter estimation based on BIP-AR(p_long) approximation. It also computes an outlier cleaned signal using BIP-AR(p_long) predictions

INPUTS
x: arraylike, data (observations/measurements/signal) 
p: int, autoregressive order
q: int, moving-average order

OUTPUTS
beta_initial: arraylike, robust starting point for AR(p)and MA(q) parameters based on BIP-AR(p_long) approximation
x_filt: arraylike, outlier cleaned signal using BIP-AR(p_long) predictions

'''
def robust_starting_point(x,p,q,enf_stat=False,enf_inv=False):
    # usually a short AR model provides best results. 
    # Change to longer model, if necessary.
    p_long = p if q==0 else min(2*(p+q),4 )
    
    x_filt = rsp.ar_est_bip_s(x,p_long)[1] 
    
    mod = tsa.SARIMAX(x_filt, order=(p, 0, q), concentrate_scale=True,
                      enforce_stationarity=enf_stat, enforce_invertibility=enf_inv)
    res = mod.fit()
    beta_initial= res.params
    beta_initial[p:] *= -1
    # Check for stationarity
    poles = lambda x: np.sum(np.abs(np.roots(-1*np.array([-1, *x]))) > 1) 
    
    if poles(beta_initial[:p]) or poles(beta_initial[p:]):
        #print('rekursion')
        #beta_initial, xfilt = robust_starting_point(x_filt,p,q,enf_stat, enf_inv)
        pass
    # necessary ?
    #if (poles(beta_initial[:p]) or poles(beta_initial[p:]))>0:
    #    beta_initial[:] = 0
    return beta_initial, x_filt