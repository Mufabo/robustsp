'''
[b1, iter] = ladlasso(y,X,lambda,...) 
 ladlasso computes the LAD-Lasso regression estimates for given complex-  
 or real-valued data.  If number of predictors, p, is larger than one, 
 then IRWLS algorithm is used, otherwise a weighted median algorithm 
 (N > 200) or elemental fits (N<200).
 INPUT: 
   yx      : numeric response 1darray of size N  (real/complex)
   Xx      : numeric feature  N x p matrix (real/complex)
   lambd : non-negative penalty parameter
   b0     : numeric optional initial start of the regression vector for 
           IRWLS algorithm. If not given, we use LSE (when p>1).
   intcpt : (logical) flag to indicate if intercept is in the model
   reltol : Convergence threshold for IRWLS. Terminate when successive 
           estimates differ in L2 norm by a rel. amount less than reltol.
 printitn : print iteration number (default = 0, no printing)
 OUTPUT:
   b1     : (numberic) the regression coefficient vector
   iter   : (numeric) # of iterations (only when IRWLS algorithm is used)
'''

import numpy as np
from robustsp import *

def ladlasso(yx,Xx,lambd,b0=None,intcpt=True,reltol=1.0e-8,printitn=0,ITERMAX = 2000):

    y = np.array(yx) #np.copy(np.asarray(yx))
    X = np.array(Xx) #np.copy(np.asarray(Xx))

    N,p = X.shape

    if intcpt:
        X = np.concatenate((np.ones((N,1)), X),1)

    # LSE is the initial start of iteration if the initial start was not given
    if b0 is None:
        b0 = np.linalg.lstsq(X[range(len(y)),:],y,rcond=None)[0]

    # we use very small error tolerance value TOL between iterations.
    iter = []

    if printitn > 0:
        print('Computing the solution for lambda = .3f\n' % lambd);

    # the case of only one predictor 
    if p==1 and not intcpt: # simple linear regression and no intercept

        b1,_,_ = wmed(np.append(y[:,np.newaxis]/X, 0), np.append(np.abs(X), lambd))

        return b1

    elif p==1 and np.isrealobj(y) and N < 200  and intcpt:         
        if lambd==0:  
            b,_ = elemfits(X[:,1],y);            
        else:
            b,_ = elemfits(np.append(X[:,1],0),np.append(y,lambd));

        res = np.repeat(y[:,np.newaxis],b.shape[1],axis=1)

        res = np.sum(np.abs(res - X @ b),axis=0);
        indx = np.nanargmin(res);

        return b[:,indx];

    else:
    #   use IRWLS always when p > 1 
        if printitn > 0:
            print('Starting the IRWLS algorithm..\n');

        if lambd >0:
            y = np.append(y,np.zeros(p));
            if intcpt:
                 X = \
                    np.vstack((X,\
                               np.hstack((np.zeros((p,1)),lambd*np.eye(p)))))
            else:
                 X =  np.vstack((X,lambd*np.eye(p)))



        for i in range(ITERMAX):
            resid = np.abs(y-X@b0) 
            resid[resid<.000001]=.000001
            Xstar = X / resid[:,None]
            b1 = np.linalg.lstsq((Xstar.T  @ X),(Xstar.T @ y[:,np.newaxis]),rcond=None)[0].flatten()
            crit = np.linalg.norm(b1-b0)/np.linalg.norm(b0);  
            if printitn !=0 and (i+1) % (printitn) == 0:
                print('ladlasso: crit(%4d) = %.9f\n' % (i,crit)) 


            if crit < reltol and i > 10:
                break 


            b0 = b1[:]

    return b1, i