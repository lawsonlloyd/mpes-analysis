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

#%

%
