'''
wmed computes the weighted median for data vector y and weights w, i.e. 
it solves the optimization problem:

$$\beta=argmin_{b} \sum_{i} |y_{i}-b|*w_{i}$$
inputs: 
   * y : (numeric) 1darray data given (real or complex) 
   * w : (numeric) 1darray positive real-valued weights. Inputs need to be of
       same length
   * verbose: (logical) true of false (default). Set as true if you wish  to see convergence as iterations evolve
   * TOL: Tolerance level breaks when update is below. default =1e-7
   * iterMAX: Maximum Number of iterations for complex y. default = 2000
   
   
outputs:
   * beta: (numeric) weighted median
   * converged: (logical) flag of convergence (in the complex-valued data case)
   * iter: (numeric) the number of iterations (complex-valued case)
   
'''

import numpy as np
from robustsp import propform

def wmed(y,w,verbose=False, TOL = 1.0e-7, iterMAX = 2000):

    y= np.asarray(y)
    w= np.asarray(w)
    
    converged = 0
    itr = 0

    N = len(y)

    if not np.isrealobj(w) or all(w<0):
        raise ValueError('input w needs to be a non-negative weight vector')

    if len(w) != N or np.sum(w >=0) != N:
        raise ValueError('wmed: nr of elements of y and w are not equal or w is not non-neg.')   

    # real value case
    if np.isrealobj(y):
        y = np.sort(y)
        w = w[np.argsort(y)]
        wcum = np.cumsum(w)
        i = np.nonzero(wcum[::-1]<0.5*np.sum(w))[0][0]
        beta = y[i] # due to equation (2.21) of the book
        return beta
    else: # complex valued
        beta0 = np.median(np.real(y)+1j*np.median(np.imag(y))) # initial guess
        abs0 = np.abs(beta0)
        
        sign = lambda x: x/np.abs(x)
      
        for itr in range(iterMAX):
            wy = np.abs(y-beta0)
            wy[wy<=10**-6] = 10**-6
            update = np.sum(w * sign(y-beta0))/np.sum(w/wy)
            beta = beta0 + update
            delta = np.abs(update)/abs0
            if verbose and (iter+1) % 10 == 0:
                print('At iter = %3d, delta=%.8f\n' % (iter,delta))
            if delta <= TOL:
                break
            beta0 = beta
            abs0 = np.abs(beta)

        converged = False if itr+1 == iterMAX else True

    return beta, itr+1, converged