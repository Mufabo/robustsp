import numpy as np
import scipy as sp
#from numba import jit

#@jit
def order_wk(y):
    # removing a robust location estimate (the sample median) from the
    # contaminated time series
    yt = y - np.median(y)

    # initializations for computing an initial robust periodogram, which is
    # used to determine the order of fitting the sine and cosine coefficients
    # in the repeated median transform
    
    N = len(y) # length of the time series
    K = int(np.floor((N-1)/2)) # number of fourier coefficients
    ARM = np.zeros(K) # repeated median estimate of cosine coefficients at w(k)
    BRM = np.zeros(K) # repeated median estimate of sine coefficients at w(k)
    
    Apuv = np.zeros((N,N))
    Bpuv = np.zeros((N,N))
    
    Ap = np.zeros(K)
    Bp = np.zeros(K)
    
    for k in range(K):
        w = 2*np.pi*(k+1)/N
        for u in range(N):
            for v in range(N):
                if u!=v:
                    Apuv[u,v] = (yt[u]*np.sin(w*v)-yt[v]*np.sin(w*u))/np.sin(w*(v-u))
                    Bpuv[u,v] = (yt[v]*np.cos(w*u)-yt[u]*np.cos(w*v))/np.sin(w*(v-u))
        
        Ap[k] = np.median(np.median(Apuv,axis=1))
        Bp[k] = np.median(np.median(Bpuv,axis=1))
        
    PSD = Ap**2 + Bp**2 # robust periodogram for determine the order

    # following 3 lines not used at all
    N_win= 3
    Smoothing_Win = np.hanning(N_win)
    PSD_smooth = np.convolve(PSD, Smoothing_Win)[N_win-1:len(PSD)-N_win+2]
    
    order = np.argsort(PSD)[::-1]
    
    wr = 2*np.pi*(order+1)/N # fourier frequencies in descending order
    return wr, PSD