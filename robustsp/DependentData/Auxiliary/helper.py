import numpy as np
import robustsp as rsp

def poles(x):
    np.sum(np.roots(-1*np.array([-1, *x]))) > 1

def compA(xII,pred,xArr,aArr,sig):
    return xII - np.array(pred) @(np.array(xArr)-np.array(aArr)+sig*rsp.eta(np.array(aArr)/sig))
    
    
def tauEstim(phi_grid, a_bip_sc, fine_grid, a_sc):    
    poly_approx = np.polyfit(phi_grid, a_bip_sc, 5) # polynomial approximation of residual scale for BIP-AR(p) tau-estimates
    a_interp_scale = np.polyval(poly_approx, fine_grid) # interpolation of  residual scale for BIP-AR(p) tau-estimates to fine grid
    poly_approx2= np.polyfit(phi_grid, a_sc, 5) # polynomial approximation of  residual scale for AR(p) tau-estimates
    a_interp_scale2= np.polyval(poly_approx2, fine_grid) # interpolation of  residual scale for AR(p) tau-estimates to fine grid
   
    temp = np.min(a_interp_scale)
    ind_max = np.argmin(a_interp_scale)
    phi = -fine_grid[ind_max] # tau-estimate under the BIP-AR(p)

    temp2 = np.min(a_interp_scale2)
    ind_max2 = np.argmin(a_interp_scale2)
    phi2=-fine_grid[ind_max2] # tau-estimate under the AR(p)  
    return phi, phi2, temp, temp2, ind_max, ind_max2 