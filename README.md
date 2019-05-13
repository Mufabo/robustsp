# robustsp

This package contains the functions that are currently available in the RobustSP toolbox: a Matlab toolbox for robust signal processing [[1]](https://github.com/RobustSP/toolbox) in Python. The toolbox can be freely used for non-commercial use only. Please make appropriate references to the book:

Zoubir, A. M., Koivunen, V., Ollila, E., and Muma, M. Robust Statistics for Signal Processing Cambridge University Press, 2018.

[1] https://github.com/RobustSP/toolbox

## Requirements

Python 3.
numpy
matplotlib

## Installation

There are two ways to install this package

### Installation version 1

Just run the following line in the command line:

    pip install git+https://github.com/Mufabo/robustsp.git

### Installation version 2

If you would like to have the source files of the package easily accessible.

1. Download the files.
2. Unzip the files where you want them (if you downloaded the repo as a zip file)
3. Open a command window in the freshly unzipped folder (should be called robustsp-master or something like that and contain setup.py)
4. Run in in the command window: ```pip install -e .```
    (Don't forget the dot)

## Examples

### Sensitivity Curve for Location estimation
[Link to Matlab code](https://github.com/RobustSP/toolbox/blob/master/codes/01_LocationScale/examples/sensitivity_curve_location.m)

```python
import random
import numpy as np
import matplotlib.pyplot as plt
from robustsp import *

#fix seed of random number generator for reproducibility
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
    SC_mean[ii] = N*(np.mean(np.append(x_N_minus1,delta_x[ii])) 
                     - mu_hat)

# sensitivity curve for median
SC_med = np.zeros(delta_x.shape)
mu_hat = np.median(x_N_minus1)
for ii in range(len(delta_x)):
    SC_med[ii] = N*(np.median(np.append(x_N_minus1,delta_x[ii])) 
                     - mu_hat)

# sensitivity curve for Huber's location estimator
c = 1.3415
SC_hub = np.zeros(delta_x.shape)
mu_hat = MLocHUB(x_N_minus1,c)
for ii in range(len(delta_x)):
    SC_hub[ii] = N*(MLocHUB(np.append(x_N_minus1,delta_x[ii])) 
                     - mu_hat)

# sensitivity curve for Tukey's location estimator
c = 4.68
SC_tuk = np.zeros(delta_x.shape)
mu_hat = MLocTUK(x_N_minus1,c)
for ii in range(len(delta_x)):
    SC_tuk[ii] = N*(MLocTUK(np.append(x_N_minus1,delta_x[ii])) 
                     - mu_hat)
    
plt.rcParams.update({'font.size': 18})

plt.plot(delta_x,SC_mean-np.mean(SC_mean), label ='mean', linewidth=2.0)
plt.plot(delta_x,SC_med-np.mean(SC_med), label='median', linewidth=2)
plt.plot(delta_x,SC_hub-np.mean(SC_hub), label='Huber M', linewidth=2)
plt.plot(delta_x,SC_tuk-np.mean(SC_tuk), label ='Tukey M', linewidth=2)

plt.grid(True)

plt.xlabel('Outlier value')
plt.ylabel('Sensitivity curve')
plt.legend()

plt.show()
```
