'''
Huber's score function psi: input is N  data vector x which can be complex
or real and threshold contant c
'''
import numpy as np

def psihub(x,c):
    return (x * (np.abs(x)<=c) + c*np.sign(x) *(np.abs(x)>c))
    