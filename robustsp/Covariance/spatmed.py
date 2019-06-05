'''
  Computes the spatial median based on (real or complex) data matrix X.
  INPUT:
         X: Numeric data matrix of size N x p. Each row represents one 
           observation, and each column represents one variable 
 printitn : print iteration number (default = 0, no printing)

 OUTPUT
      smed: Spatial median estimate
'''

import numpy as np

def spatmed(X,printitn=0,iterMAX = 500,EPS=1e-6,TOL=1e-5):
    l = np.sum(X*np.conj(X),axis=1)
    X = X[l!=0,:]
    n = len(X)
    
    smed0 = np.median(X) if np.isrealobj(X) else np.mean(X)
    norm0 = np.linalg.norm(smed0)
    
    for it in range(iterMAX):
        Xc = X - smed0
        l = np.sqrt(np.sum(Xc*np.conj(Xc),axis=1))
        l[l<EPS] = EPS
        Xpsi = Xc / l
        update = np.sum(Xpsi,axis=0)/sum(1/l)
        smed = smed0 + update
        
        dis = np.linalg.norm(update,ord=2)/norm0
        
        if printitn>0 and (i+1) % printitn == 0: print('At iter = %.3d, dis =%.7f \n' % (i,dis))
    
        if dis <= TOL: break
        smed0 = smed
        norm0 = np.linalg.norm(smed,ord=2)
    return smed