import numpy as np
import robustsp as rsp

'''
% Masreliez type EKF for tracking with time-of-arrival (ToA) estimates.
%
%% INPUTS
% r_ges:    measured distances (M x N)
% theta_init:  initial state estimate
% BS:   base station positions
% parameter: dict, created by rsp.set_parameters_book and create_environment_book
%            
%
%% OUTPUTS
% th_hat:             state estimates
% P_min:              apriori covariance
% P:                  aposteriori covariance
'''

def ekf_toa_Masreliez(r_ges, theta_init, BS, parameter=None):
    if parameter == None:
        # use default parameters
        print("parameters are set to default")
        sigma_v = 1
        M  = len(BS) # M numer of BS, N number of samples
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
    x = BS[:,0]
    y = BS[:,1]
    
    M = len(x)
    N = len(r_ges[0,:])
    
    P = np.zeros((N,4,4))
    P[0,:,:] = P0
    th_hat = np.zeros((4,N))
    th_hat[:,0] = theta_init.flatten()
    th_hat_min = np.zeros([4,N])
    P_min = np.zeros([N,4,4])
    H = np.zeros((M,4))
    h_min = np.zeros(M)
    
    for kk in range(1,N):
        th_hat_min[:,kk] = A @ th_hat[:,kk-1]
        
        P_min[kk,:,:] = A@P[kk-1,:,:]@A.T + G@Q@G.T
        
        for ii in range(M):
            H[ii,:] = [(th_hat_min[0,kk]-x[ii])/\
           np.sqrt((th_hat_min[0,kk]-x[ii])**2 + (th_hat_min[1,kk]-y[ii])**2) ,\
           (th_hat_min[1,kk]-y[ii])/np.sqrt((th_hat_min[0,kk]-x[ii])**2 \
                   + (th_hat_min[1,kk]-y[ii])**2)\
           ,0,0]
            h_min[ii] = \
            np.sqrt((th_hat_min[0,kk]-x[ii])**2 \
                + (th_hat_min[1,kk]-y[ii])**2)

        S = H@P_min[kk,:,:]@H.T + R
            
        try:
            C = np.linalg.cholesky(S)
        except:
            print('matrix modified in Masreliez filter')
            S += 500000*np.eye(M)
            C = np.linalg.cholesky(S)
        C = C.T   

        nu = np.linalg.inv(C)@(r_ges[:,kk]-h_min)
        
        K = P_min[kk,:,:] @ H.T @ np.linalg.inv(C.T)
        
        if parameter['singlescore']:
            v, vp = rsp.asymmetric_tanh(nu, parameter['c1'], parameter['c2'], parameter['x1'])
        
        th_hat[:,kk] = th_hat_min[:, kk] + K@v
        
        P[kk,:,:] = (np.eye(4) - K@np.linalg.inv(C)@H*np.mean(vp)) @ P_min[kk,:,:]
            
    return th_hat, P_min, P, parameter