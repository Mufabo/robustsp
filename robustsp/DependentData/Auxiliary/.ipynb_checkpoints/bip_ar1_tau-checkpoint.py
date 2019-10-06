import numpy as np
import robustsp as rsp

def bip_ar1_tau(x,N,phi_grid,fine_grid,kap2, P):
    a_scale_final = np.zeros(P+1)
    a_scale_final[0] = rsp.tau_scale(x) # AR(0): residual scale equals observation scale
    
    # grid search for partial autocorrelations
    a_bip_sc = np.zeros(len(phi_grid))
    a_sc = np.zeros(len(phi_grid))
    for mm in range(len(phi_grid)):
        a = np.zeros(len(x)) # residuals for BIP-AR
        a2= np.zeros(len(x)) # residuals for AR
        
        lambd = rsp.ma_infinity(phi_grid[mm], 0, 100)
        sigma_hat = a_scale_final[0]/np.sqrt(1+kap2*np.sum(lambd**2)) # sigma used for BIP-model
        for ii in range(1,N):
            a[ii] = x[ii]-phi_grid[mm]*\
            (x[ii-1]-a[ii-1]+sigma_hat*rsp.eta(a[ii-1]/sigma_hat)) # residuals for BIP-AR
            a2[ii] = x[ii] - phi_grid[mm]*x[ii-1] # residuals for AR
             
        a_bip_sc[mm] = rsp.tau_scale(a[1:]) # tau-scale of residuals for BIP-AR
        a_sc[mm] = rsp.tau_scale(a2[1:]) # tau-scale of residuals for AR
    poly_approx = np.polyfit(phi_grid, a_bip_sc, 5) # polynomial approximation of tau scale objective function for BIP-AR(1) tau-estimates
    a_interp_scale = np.polyval(poly_approx,fine_grid) # interpolation of tau scale objective function for BIP-AR(1) tau-estimates to fine grid
    poly_approx2 = np.polyfit(phi_grid,a_sc,5) # polynomial approximation of  tau scale objective function for AR(1) tau-estimates
    a_interp_scale2 = np.polyval(poly_approx2, fine_grid)
    
    temp = np.min(a_interp_scale)
    ind_max = np.argmin(a_interp_scale)
    phi = fine_grid[ind_max] # tau-estimate unter the BIP-AR(1)
    temp2 = np.min(a_interp_scale2)
    ind_max2 = np.argmin(a_interp_scale2)
    phi2 = fine_grid[ind_max2] # tau-estimate under the AR(1)
    
    # final estimate maximizes robust likelihood of the two
    if temp2<temp:
        phi_s = phi2
        temp = temp2
    else:
        phi_s = phi
        
    phi_hat = phi_s # final BIP-tau-estimate for AR(1)
    
    # final AR(1) tau-scale-estimate depending on phi_hat
    lambd = rsp.ma_infinity(phi_hat, 0, 100)
    sigma_hat = a_scale_final[0]/np.sqrt(1+kap2*np.sum(lambd**2))
    a = np.zeros(len(x)) # residuals for BIP-AR
    a2= np.zeros(len(x)) # residuals for AR
    
    x_filt = np.zeros(len(x))
    
    for ii in range(1,N):
        a[ii] = x[ii]-phi_hat*(x[ii-1]-a[ii-1]+sigma_hat*rsp.eta(a[ii-1]/sigma_hat))
        a2[ii] = x[ii] - phi_hat*x[ii-1]
        x_filt[ii] = x[ii] - a[ii] + sigma_hat*rsp.eta(a[ii]/sigma_hat)
        
    if temp2<temp:
        a_scale_final[1] =  rsp.tau_scale(a[1:])
    else:
        a_scale_final[1] = rsp.tau_scale(a2[1:])
        
    return x_filt, phi_hat, a_scale_final