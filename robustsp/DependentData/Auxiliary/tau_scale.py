import numpy as np
import robustsp as rsp

def tau_scale(x):
    b = 0.398545548533895 # E(muler_rho2) under the standard normal distribution
    sigma_m = rsp.m_scale(x)
    return np.sqrt(sigma_m**2 /len(x)  *1/b * np.sum(rsp.muler_rho2(x/sigma_m)))
