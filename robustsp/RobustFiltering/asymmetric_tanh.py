'''
 The function computes the smoothed assymetric tanh score function and its
  derivative.
 
   INPUTS
  c1: first clipping point
  c2: second clipping point
  x1 smoothing parameter to make the score function continuous;   
  x1 =  fzero(@(x1)  c1-x1*tanh(0.5*x1*(c2-c1)),1);

   OUTPUTS
  phi: assymetric tanh score function
  phi_point: derivative of assymetric tanh score function
''' 
import numpy as np
    
def asymmetric_tanh(Sigx,c1,c2,x1):
    Sig = np.asarray(Sigx) # np.array maybe better
    phi = np.zeros(np.shape(Sig))
    phi_point = np.zeros(np.shape(Sig))
    
    phi[np.abs(Sig)<=c1] = Sig[np.abs(Sig)<c1]
    phi[np.abs(Sig)>c1] = x1*np.sign(Sig[np.abs(Sig)>c1])\
            *np.tanh(x1*0.5*(c2-np.abs(Sig[np.abs(Sig)>c1])))
    phi[np.abs(Sig)>c2] = 0
    
    phi_point[np.abs(Sig)<=c1] = 1
    phi_point[np.abs(Sig)>c1] = -0.5*x1**2 /np.cosh(x1*0.5*(c2-np.abs(Sig[np.abs(Sig)>c1])))**2
    phi_point[np.abs(Sig)>c2] = 0
    
    return phi.flatten(order='F'), phi_point.flatten(order='F')