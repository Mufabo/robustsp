'''
Tukeys's weight function w: 
absx is N x 1 data vector  which can be complex
or real and threshold contant cl
'''
import numpy as np
def wtuk(absx,cl):
    return np.square(1-np.square(absx/cl)) * (absx<=cl)