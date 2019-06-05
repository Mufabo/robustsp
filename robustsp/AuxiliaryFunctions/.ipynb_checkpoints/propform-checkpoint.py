'''
Converts input sequence into respective 2darrays. Sequences with
just 1 dimension become column vectors

Examples:

(1,2,3) -> array([[1],
                  [2],
                  [3]])
[4,5,6] -> array([[4],
                  [5],
                  [6]])

((1,2,3),(4,5,6)) ->

array([[1,2,3],
       [4,5,6]])
'''
import numpy as np

def propform(x):
    x = np.asarray(x)
    if len(x.shape) == 1:
        x = x[:,np.newaxis]
    return x
