# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 13:50:47 2024

@author: lloyd
"""

import numpy             as np
import matplotlib        as mpl

def custom_colormap(Name, CMAP):
# create colormap
# ---------------

# create a colormap that consists of
# - 1/5 : custom colormap, ranging from white to the first color of the colormap
# - 4/5 : existing colormap

    # set upper part: 4 * 256/4 entries
    if CMAP == 'viridis':
        upper = mpl.cm.viridis(np.arange(256))
    else:
        upper = mpl.cm.magma(np.arange(256))

    upper = upper[56:,:]
    #upper = mpl.cm.jet(np.arange(256))
    #upper = mpl.cm.magma_r(np.arange(256))
    
    # set lower part: 1 * 256/4 entries
    # - initialize all entries to 1 to make sure that the alpha channel (4th column) is 1
    lower = np.ones((int(200/3),4))
    # - modify the first three columns (RGB):
    #   range linearly between white (1,1,1) and the first color of the upper colormap
    for i in range(3):
      lower[:,i] = np.linspace(1, upper[0,i], lower.shape[0])
    
    # combine parts of colormap
    cmap = np.vstack(( lower, upper ))
    
    # convert to matplotlib colormap
    custom_cmap = mpl.colors.ListedColormap(cmap, name=Name, N=cmap.shape[0])
    
    return custom_cmap