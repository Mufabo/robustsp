import matplotlib.pyplot as plt
import numpy as np

def prostate_plot_setup(xx,Y,locs,loc_x,names,axval=[0, 1.2, -0.25, 0.9]):
    # Without this my matplotlib doesnt find the Font Helvetica on
    # my pc
    import matplotlib
    matplotlib.font_manager._rebuild()
    hfont = {'fontname':'Helvetica'}
    for i in range(8):
        plt.plot(xx,Y[i,:],linewidth=2)
        plt.text(1.02,locs[i],names[i],fontsize=14)
        
    plt.grid(linewidth=3,linestyle='--')
    
    plt.yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8 ,1.0])
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8 ,1.0])
    
    plt.ylabel('Coefficients',fontsize=18,**hfont)
    plt.xlabel(r'normalized $\|\hat \beta(\lambda) \|_1$',fontsize=18,**hfont)
    plt.axvline(loc_x, color='k', linestyle='--')
    #plt.show()