import numpy as np

def eta(xx,c=1):
    x = np.array(xx)/c
    if np.isscalar(x):
        y = np.zeros(1)
        x = np.array([x])
    else:
        y = np.zeros(len(x))
    y[np.abs(x)<=2] = x[np.abs(x)<=2]
    lnd = lambda x,y: np.logical_and(x,y)
    y[lnd(np.abs(x)<=3,np.abs(x)>2)] = 0.016*x[lnd(np.abs(x)<=3,np.abs(x)>2)]**7\
    -.312*x[lnd(np.abs(x)<=3,np.abs(x)>2)]**5\
    +1.728*x[lnd(np.abs(x)<=3,np.abs(x)>2)]**3\
    -1.944*x[lnd(np.abs(x)<=3,np.abs(x)>2)]
    
    y= c*y
    
    return y