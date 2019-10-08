'''
EKF for tracking with time-of-arrival (ToA) estimates.
 
INPUTS
 r_ges:    measured distances (M x N)
 theta_init:  initial state estimate
 BS:   base station positions, M x 2 np array
OUTPUTS
 th_hat:             state estimates, 4xN matrix
 P_min:              apriori covariance
 P:                  aposteriori covariance
'''

import numpy as np
import scipy.io

def ekf_toa(r_ges, theta_init, BS, parameter=None):
    if parameter is None:
        print("parameters are set to default")
        parameter = {}
        sigma_v = 1;
        M = BS.shape[0] # M numer of BS, N number of samples
        P0= np.diag([100, 100, 10,10]) # initial state covariance
        R = 150**2 * np.diag(np.ones(M)) # measurement covariance
        Ts= 0.2 # sampling frequency
        A = np.array([[1, 0, Ts, 0],[ 0, 1, 0, Ts],[ 0, 0, 1, 0]\
                      ,[ 0, 0, 0, 1]]) # state transition matrix
        Q = sigma_v**2 * np.eye(2)
        G = np.vstack((0.5*Ts**2*np.eye(2), Ts*np.eye(2)))
    else:
        P0 = parameter['P0']
        R  = parameter['R']
        Q  = parameter['Q']
        G  = parameter['G']
        A  = parameter['A']
    ''' 
    if 2*parameter['dim'] != len(theta_init[:,0]) or 2*parameter['dim'] != len(P0[:,0]):
        disp('State vector or state covariance do not match the dimensions of the BS')
    '''    
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
    sigma2 = np.zeros(N)
    
    for kk in range(1,N):
        th_hat_min[:,kk] = A@th_hat[:,kk-1]
        
        for ii in range(M):
            H[ii,:] = [(th_hat_min[0,kk]-x[ii])/\
                       np.sqrt((th_hat_min[0,kk]-x[ii])**2 \
                               + (th_hat_min[1,kk]-y[ii])**2) ,\
                       (th_hat_min[1,kk]-y[ii])/\
                       np.sqrt((th_hat_min[0,kk]-x[ii])**2 \
                               + (th_hat_min[1,kk]-y[ii])**2)\
                       ,0,0]
            h_min[ii] = np.sqrt((th_hat_min[0,kk]-x[ii])**2 \
                               + (th_hat_min[1,kk]-y[ii])**2)
            
        '''
        if parameter['var_est'] == 1:
            sigma = 1.483*np.mean(np.abs((r_ges[:,kk] -h_min.T)-\
                                        np.median(r_ges[:,kk] -h_min.T)))
            sigma2[kk] = sigma**2
            R = sigma2[kk]*np.eye(M)
        '''
        
        P_min[kk,:,:] = A@P[kk-1,:,:]@A.T + G@Q@G.T
        K = P_min[kk,:,:]@H.T@np.linalg.inv(H@P_min[kk,:,:]@H.T+R)
        
        th_hat[:,kk] = th_hat_min[:,kk] + K@(r_ges[:,kk]- h_min.T)
        P[kk,:,:] = (np.eye(4) - K@H)@P_min[kk,:,:]
        
    parameter['Rest'] = sigma2
    parameter['K'] = K
    return th_hat, P_min, P, parameter