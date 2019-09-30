'''
evaluates the rmse, med, etc. of several tracks


input variables:

xx            estimated state vector (mc x N)

param         structure that contains the different tracks and the true
                trajectory
                
color         indicates the color of the track
  
fig           figure handle of the figure the curves are plotted


output variables

eval_filter      structure that contains the rmse, med and other metrices

'''

import numpy as np
import matplotlib.pyplot as plt

def eval_track(xx, param, color, fig1, fig2, fig3):
    # initialization
    eex = np.zeros([param['mc'],param['N']])
    eey = np.zeros([param['mc'],param['N']])
    med = np.zeros([param['mc'],param['N']])
    
    eevx = np.zeros([param['mc'],param['N']])
    eevy = np.zeros([param['mc'],param['N']])
    
    bx = np.zeros([param['mc'],param['N']])
    by = np.zeros([param['mc'],param['N']])
    
    bvx = np.zeros([param['mc'],param['N']])
    bvy = np.zeros([param['mc'],param['N']])
    
    for ii in range(param['mc']):
        for kk in range(param['N']):
            bx[ii,kk] = xx[ii][0,kk]-param['thx'][kk]
            by[ii,kk] = xx[ii][1,kk]-param['thy'][kk]
            
            bvx[ii,kk] = xx[ii][2,kk]-param['thvx'][kk]
            bvy[ii,kk] = xx[ii][3,kk]-param['thvy'][kk]
            
            eex[ii,kk] = (xx[ii][0,kk]-param['thx'][kk])**2
            eey[ii,kk] = (xx[ii][1,kk]-param['thy'][kk])**2
            
            med[ii,kk] = np.sqrt((xx[ii][0,kk]-param['thx'][kk])**2 +\
                                (xx[ii][1,kk]-param['thy'][kk])**2)
            
            eevx[ii,kk] = (xx[ii][2,kk]-param['thvx'][kk])**2
            eevy[ii,kk] = (xx[ii][3,kk]-param['thvy'][kk])**2
            
        eexy = eex+eey
        
        # mean of means 
        mom = lambda x: np.mean(np.mean(x,axis=1))
        dn = param['discardN']
        eval_filter = {'rmsex': np.sqrt(mom(eex[:,dn:])),
                      'rmsey': np.sqrt(mom(eey[:,dn:])),
                      'rmsevx': np.sqrt(mom(eevx[:,dn:])),
                      'rmsevy': np.sqrt(mom(eevy[:,dn:])),
                      'med': mom(med[:,dn:]),
                      'rmse': np.sqrt(mom(eexy[:,dn:])),
                      'biasx_v': np.mean(bx[:,dn:],axis=1),
                      'biasy_v': np.mean(by[:,dn:],axis=1),
                      'biasvx_v': np.mean(bvx[:,dn:],axis=1),
                      'biasvy_v': np.mean(bvy[:,dn:],axis=1),
                      'biasx': mom(bx[:,dn:]),
                      'biasy': mom(by[:,dn:]),
                      'biasvx': mom(bvx[:,dn:]),
                      'biasvy': mom(bvy[:,dn:])}
        
        MED = med[:,dn:]
        MED1= med[:,:]
        
        thx = np.zeros(param['N'])
        thy = np.zeros(param['N'])
        
        for kk in range(param['N']):
            thx[kk] = xx[0][0,kk]
            thy[kk] = xx[0][1,kk]
            
        if param['figure']:
            fig1
            plt.grid(True)
            plt.plot(thx,thy,color=color,lw=2,ms=12)
            plt.xlabel('x-position in m', size=16)
            plt.ylabel('y-position in m', size=16)
            
            fig2
            
            if len(np.mean(eexy,axis=1))>1:
                rmse = np.sqrt(np.mean(eexy,axis=1))
                kgrid = range(len(rmse),param['grid'])
                if param['plot'] == 'mse':
                    rmse_grid = rmse[::param['grid']]
                    plt.plot(kgrid, rmse_grid, color=color, lw =2, size=12)
                else:
                    plt.plot(kgrid,np.mean(MED1,axis=1),color=color,lw=2,size=12)
                
            if len(np.mean(eexy,axis=1))==1:
                print(np.sqrt(eexy[0]))
                plt.plot(np.sqrt(eexy[0]))#,color=color,lw=2,size=12)
            plt.xlabel('Time', size=16)
            plt.ylabel('Root Mean Square Error (RMSE)', size=12)
            
            plt.grid(True)
            
            fig3
            
            def ecdf(data):
                """ Compute ECDF """
                x = np.sort(data)
                n = x.size
                y = np.arange(1, n+1) / n
                return(x,y)
            
            cdfx, cdfy = ecdf(MED[:])
            cdfy[-1] = None # ???
            
            cdfy_grid = cdfy[::param['grid']*param['mc']]
            cdfx_grid = cdfx[::param['grid']*param['mc']]
            
            plt.xlabel('Empirical CDF', size=16)
            plt.ylabel('Error distance in m', size=16)
            
            plt.plot(cdfx_grid[0,:], cdfy_grid)#, color=color, lw=2, size=12)
            
            plt.grid(True)
            
            plt.show()
            return eval_filter#,rmse 
        
        