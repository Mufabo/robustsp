import numpy as np
import scipy.signal as sps

def ma_infinity(phi, theta, Q_long):
    t = [theta] if np.isscalar(theta) else theta.flatten()
    ph= phi.flatten()
    
    Q = len(t) # MA order
    P = len(ph)# AR order
    i1 = np.array([1.0, *t, *np.zeros(Q_long+P+Q)])
    i2 = np.array([1.0,  *(-ph)])
    theta_inf = sps.deconvolve(i1,i2)[0]

    theta_inf = theta_inf[1:Q_long+1]
    
    return theta_inf