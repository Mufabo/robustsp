
import numpy as np
def wtuk(absx,cl):
    return np.square(1-np.square(absx/cl)) * (absx<=cl)