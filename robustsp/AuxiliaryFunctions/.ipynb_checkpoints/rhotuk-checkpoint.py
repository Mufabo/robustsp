import numpy as np

def rhotuk(x,c):
    rhox = (c**2/3) * ((1-(1-(np.abs(x)/c)**2)**3) * (np.abs(x) <= c) + (np.abs(x) > c) )
    return rhox