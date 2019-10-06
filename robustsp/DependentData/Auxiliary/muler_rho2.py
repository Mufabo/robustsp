import numpy as np

def muler_rho2(x):
        
    rho = np.zeros(len(x))
    
    rho[np.abs(x)<=2] = 0.5*x[np.abs(x)<=2]**2
    
    land = lambda x,y: np.logical_and(x,y)
    rho[land(np.abs(x)>2,np.abs(x)<=3)] = \
    0.002*x[land(np.abs(x)>2,np.abs(x)<=3)]**8 \
    -0.052*x[land(np.abs(x)>2,np.abs(x)<=3)]**6 \
    +0.432*x[land(np.abs(x)>2,np.abs(x)<=3)]**4 \
    -0.972*x[land(np.abs(x)>2,np.abs(x)<=3)]**2 + 1.792
    
    rho[np.abs(x)>3] = 3.25
    return rho