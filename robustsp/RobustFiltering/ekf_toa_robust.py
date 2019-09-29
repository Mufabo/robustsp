import numpy as np
import scipy as sp
import robustsp as rsp

def ekf_toa_robust(r_ges, theta_init, BS, parameter={}):
    # Base station coordinates    
    x  = BS[:,0]
    y  = BS[:,1]
    M  = len(x) # M numer of BS, N number of samples
    N  = len(r_ges[0,:])
    
    if len(parameter) == 0:
        # use default parameters
        print("parameters are set to default")
        sigma_v = 1
        P0 = np.diag([100, 100, 10, 10]) # initial state covariance
        R  = 150**2 * np.diag(np.ones(M)) # measurement covariance
        Ts = 0.2 # sampling frequency
        A  = np.array([[1, 0, Ts, 0], \
                       [0, 1, 0, Ts], \
                       [0, 0, 1,  0], \
                       [0, 0, 0,  1]])
        Q  = sigma_v **2 *np.eye(2)
        G  = np.vstack([Ts**2/2*np.eye(2), Ts*np.eye(2) ])
    else:
        P0 = parameter['P0']
        R  = parameter['R']
        Q  = parameter['Q']
        G  = parameter['G']
        A  = parameter['A']
        
    if 2*parameter['dim'] != len(theta_init) or 2*parameter['dim'] != P0.shape[0]:
        raise Exception('State vector or state covariance do not match the dimensions of the BS')
        
    P = np.zeros((N,4,4))
    P[0,:,:] = P0
    th_hat = np.zeros((4,N))
    th_hat[:,0] = theta_init.flatten()
    th_hat_min = np.zeros([4,N])
    P_min = np.zeros([N,4,4])
    H = np.zeros((M,4))
    h_min = np.zeros(M)
    sigma2 = np.zeros(N)
    numberit = np.zeros(N)
    
    for kk in range(1,N):
        th_hat_min[:,kk] = A @ th_hat[:,kk-1]
        
        for ii in range(M):
            H[ii,:] = [(th_hat_min[0,kk]-x[ii])/\
           np.sqrt((th_hat_min[0,kk]-x[ii])**2 + (th_hat_min[1,kk]-y[ii])**2) ,\
           (th_hat_min[1,kk]-y[ii])/np.sqrt((th_hat_min[0,kk]-x[ii])**2 \
                   + (th_hat_min[1,kk]-y[ii])**2)\
           ,0,0]
            h_min[ii] = np.sqrt((th_hat_min[0,kk]-x[ii])**2 \
                               + (th_hat_min[1,kk]-y[ii])**2)

        P_min[kk,:,:] = A@P[kk-1,:,:]@A.T + G@Q@G.T

        # measurement residuals
        vk = r_ges[:,kk] - h_min.T

        Psi = sp.linalg.block_diag(P_min[kk,:,:],R)

        try:
            C = sp.linalg.cholesky(Psi)
        except:
            Psi = Psi + np.eye(M+4)*0.1
           
        S  = np.linalg.inv(C.T) @ np.vstack([np.eye(4), H])
        rk = np.linalg.inv(C.T) @ [*th_hat_min[:,kk],*(r_ges[:,kk]-h_min + H @ th_hat_min[:,kk])]
        
        th_hat[:,kk] = (np.linalg.pinv(S) @ rk[:,None]).flatten() 
        
        th_hat[:,kk] = rsp.m_param_est(rk,S,th_hat[:,kk],parameter)[0]

        # robust covariance estimate
        if parameter['var_est'] == 1:
            # update for robust covariance estimation
            for ii in range(M):
                h_min[ii] = np.sqrt( (th_hat[0,kk] - x[ii])**2 + \
                                   (th_hat[1,kk] - y[ii])**2)
            dd = r_ges[:,kk] - h_min.T
            sigma = 1.483*np.median(abs(dd-np.median(dd)))
            sigma2[kk] = sigma**2
            R = sigma2[kk] @ np.eye(M)
            
        K = P_min[kk,:,:] @ H.T @ np.linalg.inv(H@P_min[kk,:,:]@H.T+R)
        P[kk,:,:] = (np.eye(4) - K@H) @ P_min[kk,:,:]
        
    parameter['Rest'] = sigma2
    
    return th_hat, P_min, P, numberit, parameter

