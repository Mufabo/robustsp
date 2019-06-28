"""
Computes the rank fused-Lasso regression estimates for given fused
penalty value lambda_2 and for a range of lambda_1 values

INPUT: 
  y       : numeric response N x 1 vector (real/complex)
  X       : numeric feature  N x p matrix (real/complex)
  lambda2 : positive penalty parameter for the fused Lasso penalty term
  L       : number of grid points for lambda1 (Lasso penalty)
  eps     : Positive scalar, the ratio of the smallest to the 
            largest Lambda value in the grid. Default is eps = 10^-3. 
printitn : print iteration number (default = 0, no printing)
OUTPUT:
  b      : numeric regression coefficient vector
"""
def rankflassopath():

    intcpt = False
    n,p = X.shape

    B = np.repeat(np.arange(1,n+1)[:,None],n,axis=1)
    A = np.copy(B.T)
    a = A[A<B]
    b = B[A<B]

    D = -1*np.eye(p-1)
    D[p:p:(p-1)**2]=1
    onev = np.append(np.zeros(p-2),1)
    D = np.hstack((D,onev[:,np.newaxis]))

    ytilde = np.append(y[a-1]-y[b-1],
                        np.zeros(p-1)
                       )

    Xtilde = np.vstack((X[a-1,:]-X[b-1,:],lambda2*D))

    lam0 = np.max(np.abs(Xtilde.T @ np.sign(ytilde)))

    lamgrid = eps**(np.arange(L+1)/L)*lam0

    B = np.zeros((p,L+1))
    B0= np.zeros((p,L+1))
    binit = np.zeros(p)

    if printitn > 0:
        print("rankflassopath: starting iterations\n")

    for jj in range(L+1):
        B[:,jj],_ = rsp.ladlasso(ytilde,Xtilde, lamgrid[jj],binit,intcpt,printitn)
        binit = B[:,jj];
        if printitn > 0:
            print(" . ")

    B[np.abs(B)<1e-7]=0
    return B, B0, lamgrid