#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 12:36:02 2024

@author: lawsonlloyd
"""

#%%

import plotly.graph_objects as go 
import numpy as np 

import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'

#%%

# Generate a sample 3D dataset
x, y, z = np.indices((100, 100, 100))
data_ =  (I_pos-I_neg) # I.sum(axis=3)  # Replace this with your dataset
data_ = np.abs(data_)/np.max(data_)

binned_data = binArray(data_, 0, 1, 1, np.mean)
binned_data = binArray(binned_data, 1, 1, 1, np.mean)
binned_data = binArray(binned_data, 2, 2, 2, np.mean)

# Cut the dataset in half (example: removing one half along z-axis)
mask = z < 5
binned_data[:,0:48,:] = 0

#%%
binned_data = binned_data/np.max(binned_data)
e = 70

binned_data[:,:,e:] *= 1/np.max(binned_data[:,:,e:])

x = np.linspace(0,99,100)
y = np.linspace(0,99,100)
z = np.linspace(0,99,100)

X, Y, Z = np.meshgrid(x, y, z)
#X, Y, Z = x, y, z

values = binned_data
 
fig = go.Figure(data=go.Volume( 
	x=X.flatten(), 
	y=Y.flatten(), 
	z=Z.flatten(), 
	value=values.flatten(), 
	opacity=.05,
    caps= dict(x_show=False, y_show=False, z_show=False),
    surface_count = 25,
    isomin=0.2,
    isomax=1,

	)) 

fig.show()

#%%

# Helix equation


xx = np.linspace(0,99,100)
yy = np.linspace(0,99,100)
zz = np.linspace(0,99,100)

X, Y, Z = np.meshgrid(xx, yy, zz)

norm = plt.Normalize(vmin=np.min(binned_data), vmax=np.max(binned_data))
colors = cm.viridis(norm(binned_data))  # Use the 'viridis' colormap
#colors = cmap_LTL

colors[..., 3] = .5 

fig = go.Figure()

fig.add_trace(go.Scatter3d(
	x=xx, 
	y=yy, 
	z= binned_data,
    mode='markers',
    marker=dict(
        size=12,
        colorscale='viridis',   # choose a colorscale
        opacity=.8
        )
    ))

# tight layout
#fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()

#%%

import numpy as np
import plotly.graph_objects as go

# Generate a sample 3D dataset
amplitudes = binned_data  # Replace with your data

# Cut the dataset in half (remove top half along z-axis)
#mask = z >= 5
#amplitudes[mask] = 0

# Flatten data for `plotly`
xx = np.linspace(0,99,100)
yy = np.linspace(0,99,100)
zz = np.linspace(0,99,100)

X, Y, Z = np.meshgrid(xx, yy, zz)

X, Y, Z = np.indices((100, 100, 100))

x, y, z = X.flatten(), Y.flatten(), Z.flatten()
amplitudes = amplitudes.flatten()

# Apply a mask to remove zeroed-out values
#active_mask = amplitudes > 0
#x, y, z, amplitudes = x[active_mask], y[active_mask], z[active_mask], amplitudes[active_mask]

# Create a 3D scatter plot with colormap
fig = go.Figure(
    data=go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=4,
            color=amplitudes,  # Map amplitudes to color
            colorscale='Viridis',  # Choose colormap
            opacity=0.7,  # Set transparency
            colorbar=dict(title="Intensity")  # Add colorbar
        )
    )
)

# Customize layout
fig.update_layout(
    scene=dict(
        xaxis_title='X-axis',
        yaxis_title='Y-axis',
        zaxis_title='Z-axis',
    ),
    title="3D Dataset Visualization"
)

fig.show()

#%%

import pyvista as pv
import numpy as np

# Generate 3D dataset
x, y, z = np.meshgrid(np.linspace(-1, 1, 30),
                      np.linspace(-1, 1, 30),
                      np.linspace(-1, 1, 30))
amplitude = np.sin(np.pi * x) * np.cos(np.pi * y) * np.sin(np.pi * z)

# Create a pyvista mesh
grid = pv.UniformGrid()
grid.dimensions = amplitude.shape
grid.origin = (-1, -1, -1)  # Set the origin
grid.spacing = (2 / (amplitude.shape[0] - 1),) * 3  # Spacing

# Add the data
grid["amplitude"] = amplitude.ravel(order="F")

# Cut the grid in half
cut_grid = grid.clip(normal="z", origin=(0, 0, 0))

# Plot with transparency
plotter = pv.Plotter()
plotter.add_volume(cut_grid, opacity="linear", cmap="viridis")
plotter.show()

#%%

import numpy as np
import plotly.graph_objects as go

data_ = I.sum(axis=3)  # Replace this with your dataset
data_ = np.abs(data_)/np.max(data_)

binned_data = binArray(data_, 0, 1, 1, np.mean)
binned_data = binArray(binned_data, 1, 1, 1, np.mean)
binned_data = binArray(binned_data, 2, 2, 2, np.mean)

# Cut the dataset in half (example: removing one half along z-axis)
binned_data[:,0:49,:] = 0

#binned_data[:, [0, 1, 2]] = binned_data[:, [1, 0, 2]]
binned_data =  np.transpose(binned_data, [1, 0, 2])
binned_data = binned_data[::-1,:,:]
#%%
binned_data = binned_data/np.max(binned_data)
e = 70

binned_data[:,:,e:] *= 1/np.max(binned_data[:,:,e:])

mask = binned_data < 0.075
binned_data[mask] = 0

#binned_data = np.transpose(binned_data, [1, 0, 2])
# Generate a sample 3D dataset

x, y, z = np.indices((100, 100, 100))

amplitudes = binned_data

# Flatten data for `plotly`
x, y, z = x.flatten(), y.flatten(), z.flatten()
amplitudes = amplitudes.flatten()

non_zero_mask = amplitudes != 0 

x, y, z, amplitudes = x[non_zero_mask], y[non_zero_mask], z[non_zero_mask], amplitudes[non_zero_mask]

norm = plt.Normalize(vmin=np.min(binned_data), vmax=np.max(binned_data))
colors = cm.viridis(norm(binned_data))  # Use the 'viridis' colormap
colors = colors[:,:,:,0]

custom_colorscale = [
    [0.0, "white"],  # Minimum (negative values) -> Blue
    [0.05, "grey"],  # Minimum (negative values) -> Blue
    [0.2, "blue"],  # Minimum (negative values) -> Blue
    [0.3, "purple"], # Midpoint (zero) -> White
    [0.4, "yellow"], # Midpoint (zero) -> White
    [0.6, "orange"], # Midpoint (zero) -> Wh
    [0.7, "red"], # Midpoint (zero) -> Whiteite
    [0.8, "maroon"], # Midpoint (zero) -> White
    [1.0, "pink"]    # Maximum (positive values) -> Red
]

# Create a 3D scatter plot with colormap
fig = go.Figure(
    data=go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=8,
            color=amplitudes,  # Map amplitudes to color
            colorscale=custom_colorscale,  # Choose colormap
            cmin=0.1,
            cmax=np.max(amplitudes),
            opacity=.05,  # Set transparency
            colorbar=dict(title="Intensity")  # Add colorbar
        )
    )
)

# Customize layout
fig.update_layout(
    scene=dict(
        xaxis_title='X-axis',
        yaxis_title='Y-axis',
        zaxis_title='Z-axis',
    ),
    title="3D Dataset Visualization"
)

fig.update_layout(
    scene = dict(
        xaxis = dict(visible=False),
        yaxis = dict(visible=False),
        zaxis =dict(visible=False)
        )
    )

fig.show()

