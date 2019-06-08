# robustsp

This package contains the functions that are currently available in the RobustSP toolbox: a Matlab toolbox for robust signal processing [[1]](https://github.com/RobustSP/toolbox) in Python. The toolbox can be freely used for non-commercial use only. Please make appropriate references to the book:

Zoubir, A. M., Koivunen, V., Ollila, E., and Muma, M. Robust Statistics for Signal Processing Cambridge University Press, 2018.

[1] https://github.com/RobustSP/toolbox

## Requirements

* Python 3
* numpy
* matplotlib
* scipy

The required packages can be installed via console by ```pip install name``` where name is the package name.

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
import numpy as np
import matplotlib.pyplot as plt
from robustsp import *

#fix seed of random number generator for reproducibility
np.random.seed(2)

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

### Regression with and without Outlier for Prostata Analysis
[Link to executed colab notebook](https://colab.research.google.com/drive/1aiHm7ykITroJrfeZvGkDblQB9FwnFKTt)

[Linkt ot Matlab Code (Note: Plot different than Python version, Results equal up to at least 3 decimals)](https://github.com/RobustSP/toolbox/blob/master/codes/02_Regression/examples/prost_analysis.m)
```python
import robustsp as rsp
import numpy as np
import matplotlib
#%matplotlib inline
import matplotlib.pyplot as plt
import scipy.io
import pkg_resources

path = pkg_resources.resource_filename('robustsp', 'data/prostate.mat')
names = scipy.io.loadmat(path,struct_as_record=False)['names'][0]
names = [i for i in zip(*names)][0]

X = scipy.io.loadmat(path,struct_as_record=False)['X']
y = scipy.io.loadmat(path,struct_as_record=False)['y']
n,p = X.shape

#############   LASSO   ############

B,stats = rsp.enetpath(y,X,1)

k = np.nanargmin(stats['BIC']) # ,29 , 30 in matlab
blas = np.copy(B[:,k]) # LASSO BIC solution 
bmaxlas = np.sum(np.abs(B[1:,-1])) # largest value of || \beta ||_1, 2.0733

# plot LASSO

locs = np.copy(B[1:,-1])
locs[2] = locs[2] - 0.025 # 'age' is too close, so put it down
locs[6] = locs[6] + 0.01
loc_x = np.sum(np.abs(blas[1:-1])) / bmaxlas
xx = np.sum(np.abs(B[1:,:]),axis=0)/bmaxlas # slightly different from Matlab version
Y = B[1:,:]
fig1 = plt.figure(0)
plt.subplot(221)
rsp.prostate_plot_setup(xx,Y,locs,loc_x,names)
fig1.show()

###############   LAD-LASSO   ################

Blad,statslad = rsp.ladlassopath(y,X,reltol=1e-7)
ladind = np.nanargmin(statslad['gBIC'])
blad = Blad[:,ladind] # LAD-Lasso BIC solution
bmaxlad = np.max(np.sum(np.abs(Blad[1:,:]),axis=0)) # largest solution || \beta ||_1

# Plot LAD-LASSO
plt.subplot(223)
locs = Blad[1:,-1] # stellenweise andars als matlab 
locs[1] = locs[1] + 0.02 # lweight up
locs[6] = locs[6] + 0.02 # gleason up
locs[3] = locs[3] - 0.02 # age down
loc_x = np.sum(np.abs(blad[1:])) / bmaxlad
xx = np.sum(np.abs(Blad[1:,:]),axis=0)/bmaxlad 
Y = Blad[1:,:]
rsp.prostate_plot_setup(xx,Y,locs,loc_x,names)
fig1.show()

############   Rank-LASSO   ############

Brlad, _, statsrlad = rsp.ranklassopath(y,X)
rladind = np.nanargmin(statsrlad['gBIC'])
brlad = Brlad[:,rladind]
bmaxrlad = np.max(np.sum(np.abs(Brlad),axis=0))

# Plot Rank-LASSO
plt.subplot(224)
locs = Brlad[:,-1] # stellenweise andars als matlab 
locs[6] = locs[6] + 0.02 # gleason up
locs[2] = locs[2] - 0.02 # age down
loc_x = np.sum(np.abs(brlad)) / bmaxrlad
xx = np.sum(np.abs(Brlad),axis=0)/bmaxrlad 
rsp.prostate_plot_setup(xx,Brlad,locs,loc_x,names)
fig1.show()

############   M-LASSO   ############

Bhub, _,statshub = rsp.hublassopath(y,X)
hubind = np.nanargmin(statshub['gBIC'])
bhub = Bhub[:,hubind]
bmaxhub = np.max(np.sum(np.abs(Bhub),axis=0))

# Plot Rank-LASSO
plt.subplot(222)
locs = Bhub[:,-1] # stellenweise andars als matlab 
locs[6] = locs[6] + 0.02 # gleason up
locs[2] = locs[2] - 0.03 # age down
loc_x = np.sum(np.abs(bhub)) / bmaxhub
xx = np.sum(np.abs(Bhub),axis=0)/bmaxhub 
rsp.prostate_plot_setup(xx,Bhub,locs,loc_x,names)
fig1.show()

# Outlier

yout = np.copy(y)
yout[0] = 55
fig2 = plt.figure(1)

#############   LASSO   ############

Bout,stats2 = rsp.enetpath(yout,X,1)
k = np.nanargmin(stats2['BIC']) 
blas_out = np.copy(Bout[:,k]) # LASSO BIC solution 
bmaxlas_out = np.sum(np.abs(Bout[1:,-1])) # largest value of || \beta ||_1

# plot LASSO

locs = np.copy(Bout[1:,-1])
locs[0] = locs[0] - 0.04
locs[2] = locs[2] - 0.08
locs[3] = locs[3] + 0.06
locs[5] = locs[5] + 0.02
locs[7] = locs[7] + 0.07
loc_x = np.sum(np.abs(blas_out[1:-1])) / bmaxlas_out
xx = np.sum(np.abs(Bout[1:,:]),axis=0)/bmaxlas_out # slightly different from Matlab version
Y = B[1:,:]
plt.subplot(221)
rsp.prostate_plot_setup(xx,Y,locs,loc_x,names)
fig2.show()

###############   LAD-LASSO   ################

Blad2,statslad2 = rsp.ladlassopath(yout,X,reltol=1e-7)
ladind2 = np.nanargmin(statslad['gBIC'])
blad2 = Blad2[:,ladind2] # LAD-Lasso BIC solution
bmaxlad2 = np.max(np.sum(np.abs(Blad2[1:,:]),axis=0)) # largest solution || \beta ||_1

# Plot LAD-LASSO
plt.subplot(223)
locs = Blad2[1:,-1] # stellenweise andars als matlab 
locs[5] -= 0.04
locs[7] += 0.02
locs[6] += 0.02
locs[2] -= 0.02
loc_x = np.sum(np.abs(blad2[1:])) / bmaxlad2
xx = np.sum(np.abs(Blad2[1:,:]),axis=0)/bmaxlad2 
Y = Blad2[1:,:]
rsp.prostate_plot_setup(xx,Y,locs,loc_x,names)
fig2.show()

############   Rank-LASSO   ############

Brlad2, _, statsrlad2 = rsp.ranklassopath(yout,X)
rladind2 = np.nanargmin(statsrlad2['gBIC'])
brlad2 = Brlad2[:,rladind2]
bmaxrlad2 = np.max(np.sum(np.abs(Brlad2),axis=0))
 
# Plot Rank-LASSO
plt.subplot(224)
locs = Brlad2[:,-1] 
locs[6] = locs[6] + 0.04 # gleason up
locs[2] = locs[2] - 0.025 # age down
locs[5] -= 0.015
loc_x = np.sum(np.abs(brlad2)) / bmaxrlad2
xx = np.sum(np.abs(Brlad2),axis=0)/bmaxrlad2 
rsp.prostate_plot_setup(xx,Brlad2,locs,loc_x,names)
fig2.show()

############   M-LASSO   ############

Bhub2, _,statshub2 = rsp.hublassopath(yout,X)
hubind2 = np.nanargmin(statshub2['gBIC'])
bhub2 = Bhub2[:,hubind2]
bmaxhub2 = np.max(np.sum(np.abs(Bhub2),axis=0))

# Plot Rank-LASSO
plt.subplot(222)
locs = Bhub2[:,-1] # stellenweise andars als matlab 
locs[6] = locs[6] + 0.02 # gleason up
locs[2] = locs[2] - 0.03 # age down
loc_x = np.sum(np.abs(bhub2)) / bmaxhub2
xx = np.sum(np.abs(Bhub2),axis=0)/bmaxhub2 
rsp.prostate_plot_setup(xx,Bhub2,locs,loc_x,names)
fig2.show()
```

