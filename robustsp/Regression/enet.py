'''
 enet computes the elastic net estimator using the cyclic co-ordinate 
 descent (CCD) algorithm.
 
 INPUT: 
       y : (numeric) 1darray of size N ( (output, respones)
         if the intercept is in the model, then y needs to be centered. 
       X : (numeric) ndarray of size N x p (input, features)
           Columns are assumed to be standardized (i.e., norm(X(:,j))=1)
           as well as centered (if intercept is in the model). 
    beta : (numeric) regression vector (array) for initial start for CCD algorithm
  lambd : (numeric) a postive penalty parameter value 
  alpha  : (numeric) elastic net tuning parameter in the range [0,1]. If
           not given then use alpha = 1 (Lasso)
 printitn: print iteration number (default = 0, no printing)
 iterMax: integer. Number of maximum iterations. default = 1000
 OUTPUT:
   b1    : (numberic) the regression coefficient vector
   it    : (numeric) # of iterations
'''
import numpy as np
from robustsp import SoftThresh
import cmath
import scipy

def enet(yx,Xx,betax,lambd,alpha=1,printitn=0, iterMAX = 1000):
    y = np.copy(np.asarray(yx,dtype=np.float64))
    X = np.copy(np.asarray(Xx))
    
    _,p = X.shape
    beta = np.copy(betax)
    # Make sure that beta and y have correct shape, namely 1d-array and not 2d-array
    if len(beta.shape) == 2:
        beta = beta.flatten()
    
    if len(y.shape) == 2:
        y = y.flatten()
        
    betaold = np.copy(beta)
    normb0 = scipy.linalg.norm(beta)
    r = y - X@beta

    # Check for correct arguments
    if lambd <= 0:
        raise TypeError('lambda has to be positive. You entered %.3f' % lambd)

    lam1 = alpha*lambd
    lam2 = (1-alpha)*lambd
    const = 1/(1+lam2)

    if printitn > 0:
        print('enet : using penalty lambda = %.5f \n' % lambd)
        
    for it in range(iterMAX):
        for jj in range(p):
            beta[jj] = const * SoftThresh(beta[jj] + X[:,jj].T @ r, lam1)            
            r = r + X[:,jj] * (betaold[jj]-beta[jj])
                   
        normb = scipy.linalg.norm(beta)
        
        crit = cmath.sqrt(normb0**2 + normb**2 - 2*np.real(betaold.T @ beta))/normb if normb != 0 else np.inf

        if printitn != 0 and iter % printitn == 0:
            print('enet: %4d crit = %.8f\n' % (iter,crit))
 
        if np.real(crit) < 1e-4:
            break
           
        betaold[:] = beta
        normb0 = normb
    return beta, it

    

