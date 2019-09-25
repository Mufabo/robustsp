import numpy as np
import robustsp as rsp

def create_environment_book(parameter, start, sigma_v):
    x = parameter['BS'][:,0]
    y = parameter['BS'][:,1]
    parameter['sigma_v'] = sigma_v
    d = np.zeros((parameter['M'],parameter['N']))

    # random force motion model
    xx = np.zeros([parameter['dim']*2,parameter['N']])
    xx[:,0] = start
    for ii in range(parameter['N']-1):
        xx[:,ii+1] = parameter['A']@xx[:,ii] + parameter['G']*parameter['sigma_v']@np.random.randn(2)
        
    parameter['TrueTrajectory'] = xx
        
    for jj in range(parameter['M']):
        for ii in range(parameter['N']):
            d[jj,ii] = np.sqrt((x[jj]-xx[0,ii])**2 + (y[jj]-xx[1,ii])**2)
    
    parameter['TrueDistances'] = d
    
    # create noise
    nn = np.zeros([parameter['M'],parameter['N']])
    index = np.zeros([parameter['M'],parameter['N']])
    
    for ii in range(parameter['M']):
        nn[ii,:], index[ii,:] = rsp.markov_chain_book(parameter['MarkovChain'][ii],
                                                      parameter['sigma_los'],
                                                      parameter['sigma_nlos'],
                                                      parameter['mu_nlos'],
                                                      parameter['N'])
    # alignment of variables
    parameter['start'] = start
    parameter['TrueTrajectory'] = xx
    parameter['TrueDistances'] = d
    parameter['NoiseIndices'] = index
    parameter['MeasuredDistances'] = d + nn
    
    ii = parameter['numbermc']  
    parameter['thx'] = parameter['TrueTrajectory'][0,:]
    parameter['thy'] = parameter['TrueTrajectory'][1,:]
    parameter['thvx'] = parameter['TrueTrajectory'][2,:]
    parameter['thvy'] = parameter['TrueTrajectory'][3,:]
    
    return parameter