'''
% Mreg computes the M-estimates of regression using an auxiliary scale
% estimate. It uses the iterative reweighted least squares (IRWLS) algorithm   
%
% INPUTS: 
%       y : (numeric) data vector of size N x 1 (output, response vector)
%       X : (numeric) data matrix of size N x p (input, feature matrix)
%           If the model has intercept, then first column of X should be a 
%           vector of ones. 
% lossfun : (string) either 'huber' or 'tukey' to identify the desired 
%           loss function
%      b0 : (numeric) Optional robust initial start (regression vector) of 
%           iterations. If not given, we use the LAD regression estimate 
%  verbose: (logical) true of false (default). Set as true if you wish  
%           to see convergence as iterations evolve.
% OUTPUTS:
  b1:
  sig:
'''
import numpy as np
import scipy as sci
from robuststp import *

def Mreg(yx, Xx, lossfun='huber', b0=ladreg(y,X,False), verbose=False, ITERMAX = 1000, TOL = 1.0e-5):
    y = np.copy(np.asarray(yx))
    X = np.copy(np.asarray(Xx))
    if lossfun == 'huber':
        if np.iscomplexobj(y):
            const = 1.20112
            c     = 1.214 # 95 percent ARE
        else:
            const = 1.4815
            c     = 1.345 # 95 percent ARE
        wfun = lambda x: whub(x,c) # todo
    elif lossfun == 'tukey':
        if np.iscomplexobj(y):
            const = 1.20112
            c     = 1.214 # 95 percent ARE
        else:
            const = 1.4815
            c     = 1.345 # 95 percent ARE
        wfun = lambda x: wtuk(x,c) # todo
    else:
     # TODO Error
    
    resid = np.absolute(y-X@b0)
    sig   = const*np.median(resid[resid!=0],axis=0) # auxiliary scale estimate

  
    if verbose: print('Mreg: iterations starting, using %s loss function \n' % lossfun)
    
    for iter in range(ITERMAX):
        resid[resid < .000001] = .000001
        w = wfun(resid/sig)
        Xstar = X @ w #bsxfun(@times, X, w)
        b1 = np.linalg.lstsq((Xstar.T *X),(Xstar.T *y))

        crit = sci.linalg.norm(b1-b0)/sci.linalg.norm(b0) 
        if verbose and mod(iter,1)==0:
            print('Mreg: crit(%4d) = %.9f\n' % iter,crit)
        if crit < TOL: break
        b0 = b1[:]
        resid = np.absolute(y-X@b0)

    return b1, sig
