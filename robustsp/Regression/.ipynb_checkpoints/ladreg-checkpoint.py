'''
ladreg computes the LAD regression estimate 
INPUT: 
        y: numeric response N vector (real/complex)
        X: numeric feature  N x p matrix (real/complex)
   intcpt: (logical) flag to indicate if intercept is in the model
       b0: numeric optional initial start of the regression vector for 
           IRWLS algorithm. If not given, we use LSE (when p>1).
 printitn: print iteration number (default = 0, no printing) and
           other details 
OUTPUT:
       b1: (numberic) the regression coefficient vector
     iter: (numeric) # of iterations (given when IRWLS algorithm is used)
'''

def ladreg(yx,Xx,intcpt,b0=None,printitn=0):
    return ladlasso(yx,Xx,0,intcpt,b0,printitn)
    