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