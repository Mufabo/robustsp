"""
calculates the spatial sign covariance matrix (SCM). 

INPUT 
        X: Numeric data matrix of size N x p. Each row represents one 
          observation, and each column represents one variable.
   center: logical (true/false). If true, then center the data using
           spatial median. Default is false
   EPS: numeric, lower bound to avoid for floats to avoid divisions by zeros
        Default = 1e-06
OUTPUT: 
spatial sign covariance matrix
and spatial median (computed only if center = true)
"""
import robustsp as rsp
import numpy as np

def signcm(Xx,center=False,EPS=1e-6,):
    X = np.array(Xx)
    
    if center: 
        smed0 = rsp.spatmed(X)
        X = X - smed0[:,None]
    else: smed0 = []
        
    len = np.sqrt(np.sum(X*np.conj(X),axis=1))
    X[len != 0,:] = X
    len[len!=0] = len
    n,p = X.shape
    len[len<EPS]=EPS
    X = X / len[:,None]
    return X.T @ np.conj(X)/n , smed0