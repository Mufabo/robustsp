import numpy as np
import scipy as sp

parameter = {}
parameter['N'] = 3000 # length of simulated trajectory
parameter['BS'] = np.array([[2000, 7000],
                  [12000, 7000],
                  [7000, 12000],
                  [7000, 2000],
                  [7000, 7000]])

parameter['M'] = len(parameter['BS']) # number of base stations
parameter['start'] = [4300, 4300, 2,2] # mean starting position
parameter['pnlos'] = [0, 0, 0, 0, 0.25]
parameter['initial_sigma'] = [50, 50, 6, 6] # standard deviation for the initial state
parameter['mc'] = 50 # number of monte carlo runs
parameter['discardN'] = 100 # discard first N samples for calculating the error metrics
parameter['grid'] = 1
parameter['Ts'] = 0.2 # sampling frequency
parameter['A'] =np.array([[1, 0, parameter['Ts'], 0],
                 [0, 1, 0, parameter['Ts']],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])
parameter['G'] = np.vstack([parameter['Ts']**2/2*np.eye(2),parameter['Ts']*np.eye(2)])
parameter['dim'] = len(parameter['BS'][0]) # dimension of positions, default is 2 
parameter['numberBs'] = 'variable'
parameter['noisemodel'] = 'GMM1' # noise model
parameter['motionmodel'] = 'random-force' # state model
parameter['noiseStructure'] = 'Markov' # noise process model

# graphical output
parameter['figure'] = 1
parameter['plot'] = 'mse'
parameter['MarkovChain'] = np.zeros((parameter['M'],2,2))
for ii in range(parameter['M']):
    if parameter['pnlos'][ii] == 0:
        parameter['MarkovChain'][ii]=[[1, 0],[1, 0]]
    elif parameter['pnlos'][ii] == 0.1:
        parameter['MarkovChain'][ii]=[[0.99, 0.01],[0.09, 0.91]]
    elif parameter['pnlos'][ii] == 0.25:
        parameter['MarkovChain'][ii]=[[0.994, 0.006],[0.02, 0.98]]
    elif parameter['pnlos'][ii] == 0.5:
        parameter['MarkovChain'][ii]=[[0.98, 0.02],[0.02, 0.98]]
    elif parameter['pnlos'][ii] == 0.75:
        parameter['MarkovChain'][ii]=[[0.94, 0.06],[0.02, 0.98]]
    elif parameter['pnlos'][ii] == 1:
        parameter['MarkovChain'][ii]=[[0, 1],[0, 1]]
    else:
        print('Markov Matrix for Probability p_nlos is not availabe')
        
parameter['sigma_nlos'] = 400
parameter['mu_nlos'] = 1400
parameter['sigma_v'] = 1
parameter['sigma_los'] = 150
parameter['mc'] = 1

# parameters for tracker EKF
ekf = {}
ekf['R'] = parameter['sigma_los']**2 * np.diag(np.ones(parameter['M']))
ekf['Q'] = parameter['sigma_v'] * np.eye(2)
ekf['G'] = np.vstack([parameter['Ts']**2 / 2*np.eye(2), parameter['Ts']*np.eye(2)])
ekf['A'] = parameter['A']
# initialisation
ekf['X0'] = [30,0,4,10]
ekf['P0'] = np.diag(np.array(parameter['initial_sigma'])**2)
ekf['dim'] = parameter['dim']
ekf['var_est'] = 0

# robust ekf
rekf = {}
rekf['R'] = parameter['sigma_los']**2 * np.diag(np.ones(parameter['M']))
rekf['Q'] = parameter['sigma_v'] * np.eye(2)
rekf['G'] = np.vstack([parameter['Ts']**2 / 2*np.eye(2), parameter['Ts']*np.eye(2)])
rekf['A'] = parameter['A']
# intialisation
rekf['X0'] = [30, 0, 4, 10]
rekf['P0'] = np.diag(np.array(parameter['initial_sigma'])**2)
rekf['dim'] = parameter['dim']

# parameters M-estimator
rekf['break'] = 1e-4
rekf['max_iters'] = 25
rekf['singlescore'] = 'asymmetric'
rekf['c1'] = 1.5 # clipping point is c*nu^2
rekf['c2'] = 3   # clipping point is c*nu^2
rekf['var_est'] = 0 # uses robust covariance estimation
x1 = sp.optimize.fsolve(lambda x1: rekf['c1']-x1*np.tanh(0.5*x1*(rekf['c2']-rekf['c1'])) ,0)

if x1<0:
    rekf['x1'] = -x1
else:
    rekf['x1'] = x1
    
rekf['numberit'] = np.zeros((parameter['mc'], parameter['N']))
ekf_th = np.zeros((parameter['mc'],4, parameter['N']))
ekf_Hc = np.zeros((parameter['mc'],4, parameter['N']))