'''
Huber's weight function w: 
absx is N x 1 data vector  which can be complex
or real and threshold contant cl
'''
import numpy as np
def whub(absx,cl):
    wx = 1.*(absx<=cl) + (cl*(1./absx))*(absx>cl)
    wx[absx==0] = 1
    return wx