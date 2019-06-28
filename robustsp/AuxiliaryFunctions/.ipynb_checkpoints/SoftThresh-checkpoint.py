# Soft Thresholding function
import numpy as np

def SoftThresh(x,t):
    s = np.abs(x) -t
    s = (s+ np.abs(s))/2
    #y = (x/np.abs(x))*s # if not np.iscomplexobj(x) else x/np.abs(x)
    y = np.sign(x) * s if not np.iscomplexobj(x) else s * x/np.abs(x)
    return y