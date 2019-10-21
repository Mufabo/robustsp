import robustsp as rsp
import numpy as np
import matplotlib
# %matplotlib inline
import matplotlib.pyplot as plt
import scipy.io
import pkg_resources

# Read image of sqares (20 x 20 pixels)
path = pkg_resources.resource_filename('robustsp', 'data/images.mat') 
# contains the vectors y20 (clean data) and y20n (noisy data)
img = scipy.io.loadmat(path,struct_as_record=False)
y20 = img['y20']
y20n = img['y20n']
n = len(y20)
scaledata1 = lambda x: 3*(x-np.min(x)) / (np.max(x) - np.min(x))
scaledata2 = lambda x: 3*(x-np.min(x,axis=1)[:,None]) / (np.max(x,axis=1) - np.min(x,axis=1))[:,None]

# Plot the image
fig0 = plt.figure(0)

plt.subplot(2,2,1)
plt.imshow(np.reshape(y20,(int(np.sqrt(n)),int(np.sqrt(n))),order="F"),
              interpolation='none', cmap=plt.cm.gray)
plt.title('original image')
plt.axis('off')

# Plot the image + noise
plt.subplot(2,2,2)
plt.imshow(np.reshape(y20n,(int(np.sqrt(n)),int(np.sqrt(n))),order="F"),
              interpolation='none', cmap=plt.cm.gray)
plt.title('image + noise')
plt.axis('off')

# Plot the signal
fig1 = plt.figure(1)
plt.subplot(2,2,1)
plt.plot(range(1,n+1), y20,'*',ms=14)
plt.title('original signal')
plt.axis('off')

# Plot the noisy (measured) signal
plt.subplot(2,2,2)
plt.plot(range(1,n+1), scaledata1(y20n),'*',ms=14)
plt.title('measured signal')
plt.axis('off')
# --- Compute the LASSO solution ---

L = 20 # Grid size
Blas20n, stats = rsp.enetpath(y20n, np.eye(n), 1, L,10**-3,False)
Blas20n = Blas20n[:,1:] # Get rid of the all-zeros first column 

# Choose the best LASSO solution
ero = scaledata2(Blas20n.T).T - y20
MSElasso = np.min(np.sum(ero**2,axis=0))
indx = np.argmin(np.sum(ero**2,axis=0)) # should b 6
Blas = Blas20n[:,indx] # the best lasso solution
lam_las = stats['Lambda'][indx+1] # the best lambda value


plt.figure(0)
plt.subplot(2,2,3)
plt.imshow(np.reshape(Blas, (20, 20), order="F"),
              interpolation='none', cmap=plt.cm.gray)
plt.title('Lasso: lambda = %.3f' % lam_las)
plt.axis('off')

plt.figure(1)
plt.subplot(2,2,3)
plt.plot(range(1,n+1), scaledata1(Blas),'*',ms=14)
plt.title('Lasso')
plt.axis('off')
# --- Compute the Rank-FLasso solution ---

# start with some initial values of lambda1 and lambda2
lambda2 = 340
lambda1 = 124

B1 = rsp.rankflasso(y20n,np.eye(n),lambda1,lambda2,Blas,1)[0]
MSE_rank1 = np.sum((scaledata1(B1)-y20n)**2)

# adjust the parameters
lambda2 = 420
lambda1 = 35

B2 = rsp.rankflasso(y20n.flatten(), np.eye(400),lambda1,lambda2,B1,1)[0]
MSE_rank2 = np.sum((scaledata1(B2)-y20n)**2)

plt.figure(0)
plt.subplot(2,2,4)
plt.imshow(np.reshape(B2, (20, 20), order="F"),
              interpolation='none', cmap=plt.cm.gray)
plt.title('Rank-FLasso: lambda = %i' % lambda1)
plt.axis('off')

plt.figure(1)
plt.subplot(2,2,4)
plt.plot(range(1,n+1), scaledata1(B2),'*',ms=14)
plt.title('Rank-FLasso')
plt.axis('off')