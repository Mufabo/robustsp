# Soft Thresholding function
import numpy as np

def SoftThresh(x,t):
    s = np.abs(x) -t
    s = (s+ np.abs(s))/2
    y = np.sign(x)*s if not np.iscomplexobj(x) else x/np.abs(x)
    return y