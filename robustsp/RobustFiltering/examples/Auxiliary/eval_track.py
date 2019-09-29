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

def eval_track(xx, param, color, fig1, fig2, fig3):
    # initialization
    eex = 