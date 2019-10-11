import numpy as np
import scipy.io as sio
import pkg_resources

def split_into_prime(yx):
    y = np.array(yx)
    N = len(y)
    # the first 1000 prime numbers in a 1darray
    path = pkg_resources.resource_filename('robustsp', 'data/prime_numbers.mat')
    primes = sio.loadmat(path)['prime_numbers'].flatten()
    
    kk = 1
    while primes[kk-1]<=N and N<primes[-1]:
        kk += 1
        
    if kk==1:
        N_prime = 1
    elif N<primes[-1]:
        N_prime = primes[kk-2]
    elif N==primes[-1]:
        N_prime = primes[-1]
    elif N>prime_numbers[-1]:
        N_prime = primes[-1]
        print('make a longer list of prime numbers')
    
    if N==N_prime:
        return y[:,None]
    else:
        return np.hstack([y[:N_prime][:,None],y[N-N_prime:][:,None]])