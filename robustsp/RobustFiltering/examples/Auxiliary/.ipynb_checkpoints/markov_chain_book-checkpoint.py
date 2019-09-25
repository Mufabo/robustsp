import numpy as np

def markov_chain_book(MC,sigma_los, sigma_nlos, mu_nlos, N):
    # transition probabilities, 2d-arrays with 1 element
    pt_losnlos = MC[0,1] # from LOS->NLOS 
    pt_nloslos = MC[1,0] # from NLOS->LOS

    m=1
  
    h = np.zeros(N+100)
    y = np.zeros(N+100)
    pp = np.random.rand() 
    
    # initialisation
    state = 0 if pp<0.5 else 1

    while m < N+100:
        if state == 0:
            # LOS
            p=np.random.rand()
            if m>100:
                y[m] = sigma_los*np.random.randn()
                h[m] = 0
            if p<pt_losnlos:
                state = 1
        else:
            q = np.random.rand()
            if m>100:
                y[m] = sigma_los*np.random.randn() \
                + mu_nlos + sigma_nlos*np.random.randn()
                h[m] = 1
            if q<pt_nloslos:
                state = 0
        m += 1
        
    # Discard first 100 samples the ensure thate the Markov Chain is
    # in the steady state
    y = y[100:]
    h = h[100:]
    
    return y,h