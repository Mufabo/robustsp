import numpy as np
import robustsp as rsp

def repeated_median_filter(x):
    N = len(x)
    
    xFRM = np.zeros(N) # filter cleaned signal
    
    # Works for signals of prime length, therefore, signal is split into two
    # overlapping segments which are of prime length
    x_split = rsp.split_into_prime(x)
    
    # for each prime segment do repeated median filtering
    for ii in range(len(x_split[0,:])):
        x_part = x_split[:,ii]
        
        # Fourier frequencies in descending order
        wr = rsp.order_wk(x_part)[0]

        # Length of the prime time segment
        N_prime = len(x_part)
        
        # time vector
        t = range(N_prime)
        
        # number of fourier coefficients
        K =(N_prime-1)/2

        # Repeated median estimate of 
        # cosine coefficients at w(k)
        ARM = np.zeros(int(K))
        
        # Repeated median estimate of sine
        # coefficients at w(k)
        BRM = np.zeros(int(K))
        
        # repeated median transform: estimate ARM and BRM starting with stongest
        # w(k), subtract from time series, repeat M times
        
        # number of iterations, as recommended in the paper by Tatum and Hurvich
        M = 2
        
        # remove a robust location estimate 
        # (the sample median)
        xm = np.median(x_part)
        xt = x_part - xm

        Auv = np.zeros((N_prime, N_prime))
        Buv = np.zeros((N_prime, N_prime))
        
        A = np.zeros(int(K))
        B = np.zeros(int(K))
        
        for m in range(M):
            for k in range(int(K)):
                for u in range(N_prime):
                    for v in range(N_prime):
                        if u != v:
                            Auv[u,v] =\
                            (xt[u]*np.sin(wr[k]*v)\
                            -xt[v]*np.sin(wr[k]*u))\
                            /np.sin(wr[k]*(v-u))
                            
                            Buv[u,v] =\
                            (xt[v]*np.cos(wr[k]*u)\
                            -xt[u]*np.cos(wr[k]*v))\
                            /np.sin(wr[k]*(v-u))
                A[k] = np.median(np.median(Auv,axis=0))

                B[k] = np.median(np.median(Buv,axis=0))
                
                xt -=  (A[k]*np.cos(wr[k]*t) + B[k]*np.sin(wr[k]*t))
                
                ARM[k] +=  A[k]
                
                BRM[k] +=  B[k]

        # recover the core process by regression of the repeated median estimates
        # onto the independent parameters

        if ii == 0:
            xFRM1 = np.array(xm)
            for k in range(int(K)):
                sumAB = ARM[k]*np.cos(wr[k]*t)+\
                BRM[k]*np.sin(wr[k]*t)

                xFRM1 = xFRM1 + sumAB
        elif ii == 1:
            xFRM2 = np.array(xm)
            for k in range(int(K)):
                sumAB = ARM[k]*np.cos(wr[k]*t)+\
                BRM[k]*np.sin(wr[k]*t)

                xFRM2 = xFRM2 + sumAB    
    if ii==0:
        xFRM=np.array(xFRM1)
    elif ii==1:
        xFRM[:N-N_prime] = xFRM1[:N-N_prime]
        xFRM[N-N_prime:N_prime] =\
        (xFRM1[N-N_prime:N_prime]+ xFRM2[N-N_prime:N_prime])/2
        xFRM[N_prime:] = xFRM2[::N-N_prime]
    return xFRM, ARM, BRM