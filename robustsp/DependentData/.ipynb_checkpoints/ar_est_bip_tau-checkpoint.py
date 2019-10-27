import numpy as np
import robustsp as rsp
from robustsp.DependentData.Auxiliary.helper import *

def ar_est_bip_tau(xxx, P):
    x = np.array(xxx)
    N = len(x) # length of the observation vector
    phi_grid = np.arange(-.99,.991,.05) # coarse grid search
    fine_grid= np.arange(-.99,.991,.001) # finer grid via polynomial interpolation

    kap2 = 0.8724286
    phi_grid = np.arange(-.99,.991,.05) # coarse grid search
    fine_grid= np.arange(-.99,.991,.001) # finer grid via polynomial interpolation
    a_bip_sc = np.zeros([len(phi_grid)]) # residual scale for BIP-AR on finer grid 
    a_sc = np.zeros(len(phi_grid)) # residual scale for AR on finer grid

    # The following was introduced so as not to predict based
    # on highly contaminated data in the
    # first few samples.

    x_tran = x[:min(10,int(np.floor(N/2)))]
    sig_x_tran = np.median(np.abs(x-np.median(x)))

    x_tran[np.abs(x_tran)>3*sig_x_tran] = 3*sig_x_tran*np.sign(x_tran[np.abs(x_tran)>3*sig_x_tran])
    x[1:min(10,int(np.floor(N/2)))] = x_tran[1:min(10,int(np.floor(N/2)))]
    
    if P==0:
        return [], rsp.tau_scale(x)
    elif P==1:
        x_filt, phi_hat, a_scale_final = rsp.bip_ar1_tau(x,N,phi_grid,fine_grid,kap2,P)
    elif P>1:
        phi_hat = np.zeros((P,P))
        x_filt, phi_hat[0,0], a_scale_final = rsp.bip_ar1_tau(x,N,phi_grid,fine_grid,kap2,P)

        npa = lambda x: np.array(x)

        for p in range(1,P):
            for mm in range(len(phi_grid)):

                for pp in range(p):
                    phi_hat[p,pp] =\
                    phi_hat[p-1,pp]-phi_grid[mm]*phi_hat[p-1,p-pp-1]

                predictor_coeffs =\
                np.array([*phi_hat[p,:p], phi_grid[mm]])

                M = len(predictor_coeffs)

                if np.mean(np.abs(np.roots([1, *predictor_coeffs]))<1)==1:
                    lambd = rsp.ma_infinity(predictor_coeffs, 0, 100)
                    sigma_hat = a_scale_final[0]/np.sqrt(1+kap2*np.sum(lambd**2))
                else:
                    sigma_hat = 1.483*np.median(np.abs(x-np.median(x)))

                a = np.zeros(len(x))
                a2= np.zeros(len(x))

                for ii in range(p,N):

                    xArr = x[ii-1::-1] if ii-M-1 <0 else x[ii-1:ii-M-1:-1]
                    aArr = a[ii-1::-1] if ii-M-1 <0 else a[ii-1:ii-M-1:-1]

                    a[ii] = compA(x[ii],predictor_coeffs,xArr,aArr,sigma_hat)

                    a2[ii] = x[ii] - predictor_coeffs@xArr
                a_bip_sc[mm] = rsp.tau_scale(a[p+1:]) # residual scale for BIP-AR
                a_sc[mm] = rsp.tau_scale(a2[p+1:]) # residual scale for AR

            # tau-estimate under the BIP-AR(p) and AR(p)
            phi, phi2, temp, temp2, ind_max, ind_max2 = tauEstim(phi_grid, a_bip_sc, fine_grid, a_sc)

            # final estimate minimizes the residual scale of the two
            if temp2<temp:
                ind_max=ind_max2
                temp=temp2

            for pp in range(p):
                phi_hat[p,pp] = phi_hat[p-1,pp]-fine_grid[ind_max]*phi_hat[p-1,p-pp-1]

            phi_hat[p,p] = fine_grid[ind_max]

            # final AR(p) tau-scale-estimate depending on phi_hat(p,p)
            if np.mean(np.abs(np.roots([1,*phi_hat[p,:]]))<1) == 1:
                lambd = rsp.ma_infinity(phi_hat[p,:], 0, 100)
                #sigma used for bip-model
                sigma_hat = a_scale_final[0]/np.sqrt(1+kap2*np.sum(lambd**2))
            else:
                sigma_hat = 1.483*np.median(np.abs(x-np.median(x)))

            x_filt = np.zeros(len(x))

            for ii in range(p,N):
                xArr = x[ii-1::-1] if ii-M-1 <0 else x[ii-1:ii-M-1:-1]
                aArr = a[ii-1::-1] if ii-M-1 <0 else a[ii-1:ii-M-1:-1]

                a[ii] = x[ii]-phi_hat[p,:p+1]@(xArr-aArr\
                +sigma_hat*rsp.eta(aArr/sigma_hat))

                a2[ii]=x[ii]-phi_hat[p,:p+1]@xArr

            if temp2>temp:
                a_scale_final[p+1] = rsp.tau_scale(a[p+1:])
            else:
                a_scale_final[p+1] = rsp.tau_scale(a2[p+1:])

        phi_hat = phi_hat[p,:] # BIP-AR(P) tau-estimate        

        for ii in range(p,N):
            x_filt[ii] = x[ii] - a[ii] + sigma_hat*rsp.eta(a[ii]/sigma_hat)

    return phi_hat, x_filt, a_scale_final