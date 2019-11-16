"""
%   The biweight_filter(x) is our implementation of the
%   method described in 
%
%   "High breakdown methods of time series analysis.
%   Tatum, L.G., and Hurvich, C. M.  
%   Journal of the Royal Statistical Society. Series B (Methodological),
%   pp. 881-896, 1993.
%   
%   The code is based on an implementation by Falco Strasser, Signal Processing
%   Group, TU Darmstadt, October 2010.
%
%% INPUTS
% x: data (observations/measurements/signal), real-valued vector
%
%
%% OUTPUTS
% xFBi: Biweight filtered (outlier cleaned) signal 
% ABi:  Fourier coefficients for cosine
% BBi:  Fourier coefficients for sine

Note: Produces slightly different results than matlab, due to
different curve fitting methods
"""

import numpy as np
import robustsp as rsp
import scipy as sp

def biweight_filter(x):
    N = len(x) # length of signal
    
    xFB = np.zeros(N) # filter cleaned signal
    
    # Works for signals of prime length, therefore, signal is split into two
    # overlapping segments which are of prime length
    x_split = rsp.split_into_prime(x)
 
    # for each prime semgent do Biweight filtering
    for ii in range(x_split.shape[1]):
        x_part = x_split[:,ii] 
        wr = rsp.order_wk(x_part)[0] # Fourier frequencies in descending order
        
        # initialize with repeated median filter
        xFRM, ARM, BRM = rsp.repeated_median_filter(x_part)
        
        N_prime = len(x_part) # length of the prime time segment
        t = np.arange(N_prime)
        K = np.int((N_prime-1 )/2)
        ABi = np.zeros(K) # biweight estimate of cosine coefficients at w(K)
        BBi = np.zeros(K) # biweight estimate of sine coefficients at w(K)
        
        k = 4    # tuning constant as recommended in the paper by Tatum and Hurvich
        xb = rsp.MLocTUK(x_part, k)   # biweight location estimate
        xc = (x_part-xb)  # robustly centered time series
        
        for k in range(K):
            f = lambda x, a, b: a*np.cos(wr[k]*x)+b*np.sin(wr[k]*x)
            c, _ = sp.optimize.curve_fit(f, t, xc,p0=[ARM[k], BRM[k]])
            
            ABi[k] = c[0]
            BBi[k] = c[1]
            
            xc -= ABi[k]*np.cos(wr[k]*t)+BBi[k]*np.sin(wr[k]*t)
        # recover the core process by regression of the Biweight estimates
        # onto the independent parameters 
        if ii == 0:
            xFB1 = np.ones(t.shape)*xb
            for k in range(K):
                sumAB = ABi[k]*np.cos(wr[k]*t) + BBi[k]*np.sin(wr[k]*t)
                xFB1 += sumAB
        if ii == 1:
            xFB2 = np.ones(t.shape)*xb
            for k in range(K):
                sumAB = ABi[k]*np.cos(wr[k]*t) + BBi[k]*np.sin(wr[k]*t)
                xFB2 += sumAB
    # fuse the cleaned segments of prime length

    if ii == 0:
        xFB = np.array(xFB1)
    elif ii == 1:
        xFB[:N-N_prime] = xFB1[:N-N_prime]
        
        xFB[N-N_prime:N_prime] = (xFB1[N-N_prime:N_prime] + xFB2[:-(N-N_prime)])/2
        print(xFB2)
        xFB[N_prime:] = xFB2[-(N-N_prime):]
    
    return xFB, ABi, BBi