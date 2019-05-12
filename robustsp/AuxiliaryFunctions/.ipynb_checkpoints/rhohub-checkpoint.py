# Huber's loss function rho: input is N x 1 data vector x which can be complex
# or real and threshold contant c

import numpy as np

def rhohub(x,c):
    rhox = np.square(np.absolute(x)) * (np.absolute(x)<=c) + (2*c*np.absolute(x)-c**2) * (np.absolute(x) > c)
            
    return rhox
    