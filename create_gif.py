#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 08:24:42 2025

@author: lawsonlloyd
"""


import matplotlib.pyplot as plt
import imageio.v2 as imageio
import numpy as np
import os

# Example dataset: 10 frames of 2D data
(kx, ky), (kx_int, ky_int) = (0, 0), (4, 0.25) # Central (kx, ky) point and k-integration

Ein = .8 #Enhance excited states above this Energy, eV
energy_limits = [0.8, 3] # Energy Y Limits for Plotting Panels 3 and 4
plot_symmetry_points = False


# Temporary folder to save images
os.makedirs("frames", exist_ok=True)

# Create frames
filenames = []
for i in np.arange(len(I.delay.values)):
    
    fig, ax = plt.subplots()
    #ax.axis('off')

    # kx_frame = mpes.get_kx_E_frame(I_diff, ky, ky_int, I.delay.values[i], delay_int)
    # kx_frame = mpes.enhance_features(kx_frame, Ein, factor = 0, norm = True)
    # kx_frame = kx_frame/np.max(kx_frame.loc[{"E":slice(Ein,3)}])

    # im_kx = kx_frame.T.plot.imshow(ax = ax, cmap=cmap_LTL, add_colorbar=False, vmin=0, vmax=scale[1]) #kx, ky, t


    kx_frame = mpes.get_kx_E_frame(I_diff, ky, ky_int, I.delay.values[i], delay_int)
    kx_frame = mpes.enhance_features(kx_frame, Ein, factor = 0, norm = True)
    kx_frame = kx_frame/np.max(kx_frame.loc[{"E":slice(Ein,3)}])

    kx_edc = kx_frame.loc[{"kx":slice(kx-kx_int/2,kx+kx_int/2)}].mean(dim="kx")
    kx_edc = kx_edc/np.max(kx_edc.loc[{"E":slice(0.8,3)}])
    
    g, g1, g2, offset = two_gaussians_report(kx_edc.loc[{"E":slice(0,3)}].E.values, *p_fits_excited[i,:])
    kx_edc.plot(ax=ax, color = 'black')
    ax.plot(kx_edc.loc[{"E":slice(0,3)}].E.values, g1, color='black',linestyle = 'dashed')
    ax.plot(kx_edc.loc[{"E":slice(0,3)}].E.values, g2, color='red',linestyle = 'dashed')
    ax.plot(kx_edc.loc[{"E":slice(0,3)}].E.values, g, color='grey',linestyle = 'solid')
    #ax[0].set_title(f"$E_b = {1000*Eb}$ meV")
    ax.set_title(f"$\Delta$t = {I.delay.values[i]} fs")
    ax.set_xlim(0.5,3)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Norm. Int.')
    ax.set_xlabel('$E - E_{VBM}$, eV', color = 'black')
    #ax[0].text(1.7, .8,  f"$\Delta$t = {delay} fs", size=16)
    #ax[0].text(1.6, .95,  f'$k_x$ = {kx:.1f} $\AA^{{-1}}$', size=16)
    #ax[0].text(1.6, .825,  f'$k_y$ = {kx:.1f} $\AA^{{-1}}$', size=16)
    ax.set_aspect('auto')
    
    filename = f"frames/frame_{i:03d}.png"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    filenames.append(filename)

# Create gif
with imageio.get_writer('Scan188.gif', mode='I', duration=0.2) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Optional: cleanup
for filename in filenames:
    os.remove(filename)
os.rmdir("frames")

