# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 14:32:11 2024

@author: lloyd
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy.stats import binned_statistic
from matplotlib import cm

#%%

def binArray(data, axis, binstep, binsize, func=np.nanmean):
    data = np.array(data)
    dims = np.array(data.shape)
    argdims = np.arange(data.ndim)
    argdims[0], argdims[axis]= argdims[axis], argdims[0]
    data = data.transpose(argdims)
    data = [func(np.take(data,np.arange(int(i*binstep),int(i*binstep+binsize)),0),0) for i in np.arange(dims[axis]//binstep)]
    data = np.array(data).transpose(argdims)
    return data

#%%


# Generate a sample 3D dataset
x, y, z = np.indices((100, 50, 100))
data =  I.sum(axis=3)  # Replace this with your dataset

binned_data = binArray(data, 0, 1, 1, np.mean)
binned_data = binArray(binned_data, 1, 2, 2, np.mean)
binned_data = binArray(binned_data, 2, 2, 2, np.mean)

# Cut the dataset in half (example: removing one half along z-axis)
mask = z < 5
binned_data[:,0:24,:] = 0

#%%
binned_data = binned_data/np.max(binned_data)
e = 70

binned_data[:,:,e:] *= 1/np.max(binned_data[:,:,e:])

# Normalize amplitudes to [0, 1] for colormap mapping
norm = plt.Normalize(vmin=np.min(binned_data), vmax=np.max(binned_data))
colors = cm.viridis(norm(binned_data))  # Use the 'viridis' colormap
#colors = cmap_LTL

colors[..., 3] = .5  # Set alpha for transparency (4th channel in RGBA)

# Plot the voxels
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.voxels(binned_data > 0.2,
          facecolors=colors)  # Adjust alpha for transparency

sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
cbar.set_label('Intensity')

plt.show()
