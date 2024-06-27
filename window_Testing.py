#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 13:32:27 2024

@author: lawsonlloyd
"""

import scipy
import numpy as np

mu = 0
sigma = 0.2
x = np.linspace(-2,2,100)
dx = x[1] - x[0]
wl = 22

max_r = 1/(2*dx)
r = np.linspace(-max_r,max_r,512) 
g = scipy.stats.norm.pdf(x, mu, sigma)
g = g/np.max(g)
w = np.zeros(x.shape)
w[int(len(w)/2)-int(wl/2):wl-int(wl/2)+int(len(w)/2)]= scipy.signal.boxcar(wl)
w = w/np.max(w)
gw = g*w

fig, ax = plt.subplots(1,2)
ax = ax.flatten()
#fig.set_size_inches(6, 6, forward=False)

ax[0].plot(x,g, label =  'Gaussian', color = 'black')
ax[0].plot(x,w, label = 'Window', color = 'blue')
ax[0].plot(x,gw, label = 'Windowed Gaussian', color = 'red', linestyle = 'dashed')
#ax[0].set_title('')
#ax[0].legend()
ax[0].legend(frameon = False)
g_ = abs(fftshift(fft(g, 512)))
g_ = g_/np.max(g_)
w_ = abs(fftshift(fft(w,512)))
w_ = w_/np.max(w_)
gw_ = abs(fftshift(fft(gw,512)))
gw_ = gw_/np.max(gw_)

ax[1].plot(r,g_, color = 'black')
ax[1].plot(r,w_, color = 'blue')
ax[1].plot(r,gw_, color = 'red', linestyle = 'dashed')
ax[1].set_title('FFT')
ax[0].set_xlim([-1,1])
ax[1].set_xlim([-4,4])

fig.tight_layout()

#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftshift, fft2
from scipy.signal.windows import tukey
from matplotlib.colors import Normalize
import matplotlib        as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

def gaussian_2d(shape, mean, sigma):
    """
    Generates a 2D Gaussian function.
    
    Parameters:
    - shape: tuple of the shape of the output array (nrows, ncols)
    - mean: tuple of the mean (mu_x, mu_y)
    - sigma: tuple of the standard deviation (sigma_x, sigma_y)
    
    Returns:
    - 2D Gaussian function array
    """
    x = np.linspace(0, shape[1]-1, shape[1])
    y = np.linspace(0, shape[0]-1, shape[0])
    x, y = np.meshgrid(x, y)
    gauss = np.exp(-((x - mean[0])**2 / (2 * sigma[0]**2) + (y - mean[1])**2 / (2 * sigma[1]**2)))
    return gauss / np.max(gauss)  # Normalize to 1

def apply_tukey_window(data, alpha, sigma):
    """
    Applies a 2D Tukey window to the data.
    
    Parameters:
    - data: 2D array to which the Tukey window will be applied
    - alpha: shape parameter of the Tukey window
    
    Returns:
    - Windowed data
    - The Tukey window itself
    """
    window = tukey((data.shape[0]//2), alpha).reshape(-1, 1) * tukey((data.shape[1]//2), alpha).reshape(1, -1)
    w = 4*sigma[0]
    w = window.shape[0]//2
    l = data.shape[0] #data.shape[0]
    window_zp = np.zeros(data.shape)
    window1d = np.abs(signal.windows.tukey(2*w))
    window2d = np.sqrt(np.outer(window1d,window1d))
    window2d = window2d/np.max(window2d)
    window_zp[(int(l/2)-w):(int(l/2)+w), (int(l/2)-w):(int(l/2)+w)] = window
    
    return data*window_zp, window_zp #data * window, 
    #return data * window, window


def compute_fft(data):
    """
    Computes and returns the Fourier Transform of the data.
    
    Parameters:
    - data: 2D array to transform
    
    Returns:
    - The Fourier Transform of the data
    """
    fft_data = fftshift(fft2(data,[zeropad,zeropad]))
    return np.abs(fft_data)/np.max(np.abs(fft_data))

def modified_viridis():
    """
    Returns a modified viridis colormap with white color for zero values.
    """
    from matplotlib.colors import ListedColormap
    upper = mpl.cm.viridis(np.arange(256))
    lower = np.ones((int(256/9),4))
    for i in range(3):
      lower[:,i] = np.linspace(1, upper[0,i], lower.shape[0])
    cmap = np.vstack(( lower, upper ))
    cmap_LTL = mpl.colors.ListedColormap(cmap, name='viridis_LTL', N=cmap.shape[0])
    return (cmap_LTL)

# Parameters
zeropad = 512
shape = (256, 256)
mean = (128, 128)
sigma = (10, 10)

# Tukey Window
alpha = 0.75

# Generate 2D Gaussian
gaussian = gaussian_2d(shape, mean, sigma)

# Apply Tukey window
windowed_gaussian, window = apply_tukey_window(gaussian, alpha, sigma)

# Compute Fourier Transforms
fft_gaussian = compute_fft(gaussian)
fft_windowed_gaussian = compute_fft(windowed_gaussian)
fft_window = compute_fft(window)

# Plotting
plt.figure(figsize=(18, 6))
cmap = cmap_LTL
norm = Normalize(vmin=0, vmax=np.max(gaussian))

# Plot the 2D Gaussian
plt.subplot(1, 3, 1)
plt.imshow(gaussian, cmap=cmap, norm=norm)
plt.title('2D Gaussian Function', fontsize=16)
plt.colorbar()
plt.xlabel('X', fontsize=14)
plt.ylabel('Y', fontsize=14)

# Plot the 1D x cuts of the window, unwindowed function, and windowed function
center_y = gaussian.shape[0] // 2
center_y_zp = fft_gaussian.shape[0] // 2

plt.subplot(1, 3, 2)
plt.plot(window[center_y, :], 'r-', label='Tukey Window')
plt.plot(gaussian[center_y, :], 'k-', label='Gaussian')
plt.plot(windowed_gaussian[center_y, :], 'r--', label='Windowed Gaussian')
plt.title('X Line Cuts', fontsize=16)
plt.legend(frameon=False)
plt.xlabel('X', fontsize=14)
plt.ylabel('Amplitude', fontsize=14)

# Plot the 1D x cuts of the FFTs
plt.subplot(1, 3, 3)
plt.plot(fft_gaussian[center_y_zp, :], 'k-', label='FFT of Gaussian')
plt.plot(fft_windowed_gaussian[center_y_zp, :], 'r--', label='FFT of Windowed Gaussian')
plt.plot(fft_window[center_y_zp, :], 'r-', label='FFT of Tukey Window')
plt.title('X Line Cuts of FFTs', fontsize=16)
plt.legend(frameon=False)
plt.xlabel('Frequency', fontsize=14)
plt.ylabel('Log Amplitude', fontsize=14)
plt.xlim([200,300])

plt.tight_layout()
plt.show()


