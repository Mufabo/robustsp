import random
import numpy as np
# fix seed of random number generator for reproducibility
random.seed(2)

# Number of measurements
N = 100

# DC voltage in AWGN
x_N_minus1 = np.random.randn(N-1,1)+5

# outlier values
delta_x = np.linspace(0,10,1000)

# sensitivity curve for mean
SC_mean = np.zeros(delta_x.shape)
mu_hat = np.mean(x_N_minus1)
for ii in range(len(delta_x)):
    SC_mean[ii] = N*(np.mean(np.concatenate(x_N_minus1,delta_x[ii])) 
                     - mu_hat)

# sensitivity curve for median
SC_med = np.zeros(delta_x.shape)
mu_hat = np.median(x_N_minus1)
for ii in range(len(delta_x)):
    SC_med[ii] = N*(np.median(np.concatenate(x_N_minus1,delta_x[ii])) 
                     - mu_hat)    

# sensitivity curve for Huber's location estimator
c = 1.3415
SC_hub = np.zeros(len(delta_x))
mu_hat = MlocHUB(x_N_minus1,c)
for ii in range(len(delta_x)):
    SC_mean[ii] = N*(MlocHUB(np.concatenate(x_N_minus1,delta_x[ii])) 
                     - mu_hat)  
    
# sensitivity curve for Tukey's location estimator
c = 4.68
SC_tuk = np.zeros(len(delta_x))
mu_hat = MlocTUK(x_N_minus1,c)
for ii in range(len(delta_x)):
    SC_mean[ii] = N*(MlocTUK(np.concatenate(x_N_minus1,delta_x[ii])) 
                     - mu_hat)    
    
