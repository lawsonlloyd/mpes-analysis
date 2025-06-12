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
(kx, ky), (kx_int, ky_int) = (0, 0), (4, 0.5) # Central (kx, ky) point and k-integration

Ein = .8 #Enhance excited states above this Energy, eV
energy_limits = [0.8, 3] # Energy Y Limits for Plotting Panels 3 and 4
plot_symmetry_points = False

# Temporary folder to save images
os.makedirs("frames", exist_ok=True)

# Create frames
filenames = []
for d in I_res.loc[{"delay":slice(-100,1000)}].delay.values:
    
    fig, ax = plt.subplots()
    
    mpes.plot_kx_frame(
        I_diff, ky, ky_int, delays=d, delay_int=50,
        E_enhance = 1,
        subtract_neg=subtract_neg, neg_delays=neg_delays,
        fig = fig, ax = ax,
        cmap = cmap, scale=[0,1], energy_limits=[1,3]
    )
    
    ax.set_title(fr"$\Delta$t = {d:.0f} fs")

    filename = f"frames/frame_{d}.png"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    filenames.append(filename)

# Create gif
with imageio.get_writer('kx_dynamics.gif', mode='I', duration=0.2) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Optional: cleanup
for filename in filenames:
    os.remove(filename)
os.rmdir("frames")

