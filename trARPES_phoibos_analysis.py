#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 15:47:00 2025

@author: lawsonlloyd
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
from skimage.draw import disk
from scipy.optimize import curve_fit
import csv
from Loader import DataLoader
import xarray as xr

import phoibos

data_path = '/Users/lawsonlloyd/Desktop/Data/phoibos'

#%% PLOT Fluence Delay TRACES All Together: 915 nm

import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

save_figure = False
figure_file_name = 'Combined'
image_format = 'pdf'

# Standard 915 nm Excitation
scans = [9219, 9217, 9218, 9216, 9220, 9228]

#offsets_t0 = [-162.4, -152.7, -183.1, -118.5, -113.7, -125.2]
power = 1.05*np.asarray([153, 111, 91, 66, 47, 32, 15, 10, 8, 5])
fluence = [.2, .35, .8, 1.74, 2.4, 2.9]

# Expanded 915 nm Excitation
#scans = [9227, 9219, 9217, 9218, 9216, 9220, 9228, 9231] # Scans to analyze and fit below: 910 nm + 400 nm
#offsets_t0 = [-191.7, -162.4, -152.7, -183.1, -118.5, -113.7, -125.2, -147.6]
#fluence = [4.7, 8.3, 20.9, 41.7, 65.6, 83.2, 104.7, 151]

####
k, k_int = (0), 24
E[0], E[1], E_int = 1.35, 2.1, 0.1
subtract_neg = True
norm_trace = False

cn = 100
p_min = .1
p_max = 3.5

fig, axx = plt.subplots(1, 2)
fig.set_size_inches(10, 4, forward=False)
axx = axx.flatten()

fluence = np.array(fluence)
boundaries = np.concatenate([[fluence[0] - (fluence[1] - fluence[0]) / 2],  # Leftmost edge
                             (fluence[:-1] + fluence[1:]) / 2,  # Midpoints
                             [fluence[-1] + (fluence[-1] - fluence[-2]) / 2]])  # Rightmost edge
midpoints = (boundaries[:-1] + boundaries[1:]) / 2

cmap = cm.get_cmap('inferno_r', len(scans))    # 11 discrete colors
norm = col.BoundaryNorm(boundaries, cmap.N)  # Normalize fluence values to colors

custom_colors = ['lightsteelblue', 'royalblue', 'mediumblue', 'salmon', 'indianred', 'firebrick'] #colors for plotting the traces
cmap = mcolors.ListedColormap(custom_colors)  # Create discrete colormap
norm = mcolors.BoundaryNorm(boundaries, cmap.N)  # Normalize boundaries

i = 0
for scan_i in scans:
    
    res = phoibos.load_data(data_path, scan_i, scan_info, energy_offset, offsets_t0[i], False)
    trace_1 = phoibos.get_time_trace(res, E[0], E_int, k, k_int, subtract_neg, norm_trace)
    trace_2 = phoibos.get_time_trace(res, E[1], E_int, k, k_int, subtract_neg, norm_trace)
    trace_2 = trace_2/np.max(trace_1)
    trace_1 = trace_1/np.max(trace_1)


    t1 = trace_1.plot(ax = axx[0], color = cmap(i), linewidth = 3)
    t2 = trace_2.plot(ax = axx[1], color = cmap(i), linewidth = 3)
    
    i += 1

axx[0].set_xticks(np.arange(-1000,3500,500))
for label in axx[0].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)

axx[0].set_yticks(np.arange(-0.5,1.25,0.25))
for label in axx[0].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
    
axx[1].set_xticks(np.arange(-1000,3500,500))
for label in axx[1].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)

axx[1].set_yticks(np.arange(-0.4,1.25,0.2))
for label in axx[1].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
    
axx[0].set_xlim([-500,3000])
axx[1].set_xlim([-500,3000])
axx[0].set_ylim([-0.1,1.1])
axx[1].set_ylim([-0.1,.6])

axx[0].set_xlabel('Delay, fs')
axx[1].set_xlabel('Delay, fs')
axx[0].set_ylabel('Norm. Int.')
axx[1].set_ylabel('Norm. Int.')

axx[0].set_title('Exciton')
axx[1].set_title('CBM')

fig.text(.01, 0.975, "(a)", fontsize = 20, fontweight = 'regular')
fig.text(.5, 0.975, "(b)", fontsize = 20, fontweight = 'regular')

# Add colorbar for the discrete color mapping
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Required for colorbar to work
cbar_ax = fig.add_axes([1, 0.17, 0.02, 0.75])
cbar = fig.colorbar(sm, cax=cbar_ax, ticks=midpoints)
cbar.set_label("$n_{eh}$ ($x$10$^{13}$ cm$^{-2})$")
cbar.ax.set_yticklabels([f"{f:.2f}" for f in fluence])  # Format tick labels

params = {'lines.linewidth' : 3.5, 'axes.linewidth' : 2, 'axes.labelsize' : 20, 
              'xtick.labelsize' : 16, 'ytick.labelsize' : 16, 'axes.titlesize' : 20, 'legend.fontsize' : 16}
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update(params)

# cbar = plt.colorbar(cm, ax=axx[1])
# cbar.set_label('Fluence', rotation=90, fontsize=22)
# cbar.ax.tick_params(labelsize=20)

# cbar = plt.colorbar(cm2, ax=axx[0])
# cbar.set_label('Fluence', rotation=90, fontsize=22)
# cbar.ax.tick_params(labelsize=20)

fig.tight_layout()

if save_figure is True:
    fig.savefig(figure_file_name + '.'+ image_format, bbox_inches='tight', format=image_format) 

#%% PLOT Fluence Delay TRACES All Together: 700 nm

save_figure = True
figure_file_name = '700nm'
image_format = 'pdf'

# Standard 700 nm Excitation
scans = [9368, 9378, 9367, 9370, 9373]
power = [2.8, 4, 6.3, 8, 12.6]
fluence = [ 0.7*p/10 for p in power] #10% is 0.7mJ/cm2

####
E, E_int = [1.3, 2.0], 0.1
k, k_int = (0), 24
subtract_neg = True
norm_trace = False

fig, axx = plt.subplots(1, 2)
fig.set_size_inches(10, 4, forward=False)
axx = axx.flatten()

fluence = np.array(fluence)
boundaries = np.concatenate([[fluence[0] - (fluence[1] - fluence[0]) / 2],  # Leftmost edge
                             (fluence[:-1] + fluence[1:]) / 2,  # Midpoints
                             [fluence[-1] + (fluence[-1] - fluence[-2]) / 2]])  # Rightmost edge
midpoints = (boundaries[:-1] + boundaries[1:]) / 2


cmap = plt.colormaps['inferno_r']   # 11 discrete colors
norm = col.BoundaryNorm(boundaries, cmap.N)  # Normalize fluence values to colors

custom_colors = ['lightsteelblue', 'royalblue', 'mediumblue', 'salmon', 'firebrick'] #colors for plotting the traces
cmap = mcolors.ListedColormap(custom_colors)  # Create discrete colormap
norm = mcolors.BoundaryNorm(boundaries, cmap.N)  # Normalize boundaries

i = 0
for scan_i in scans:
    
    res = phoibos.load_data(data_path, scan_i, scan_info, energy_offset, offsets_t0[i], False)
    trace_1 = phoibos.get_time_trace(res, E[0], E_int, k, k_int, subtract_neg, norm_trace)
    trace_2 = phoibos.get_time_trace(res, E[1], E_int, k, k_int, subtract_neg, norm_trace)
    trace_2 = trace_2/np.max(trace_2)
    trace_1 = trace_1/np.max(trace_1)

    t1 = trace_1.plot(ax = axx[0], color = cmap(i), linewidth = 3, label = f'{fluence[i]:.2f} mJ / cm$^{{2}}$')
    t2 = trace_2.plot(ax = axx[1], color = cmap(i), linewidth = 3)
    
    i += 1

axx[0].set_xticks(np.arange(-1000,3500,500))
for label in axx[0].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)

axx[0].set_yticks(np.arange(-0.5,1.25,0.25))
for label in axx[0].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
    
axx[1].set_xticks(np.arange(-1000,3500,500))
for label in axx[1].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)

axx[1].set_yticks(np.arange(-0.5,1.25,0.25))
for label in axx[1].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
    
axx[0].set_xlim([-500,3000])
axx[1].set_xlim([-500,3000])
axx[0].set_ylim([-0.1,1.1])
axx[1].set_ylim([-0.1,1.1])

axx[0].set_xlabel('Delay, fs')
axx[1].set_xlabel('Delay, fs')
axx[0].set_ylabel('Norm. Int.')
axx[1].set_ylabel('Norm. Int.')

axx[0].set_title('Exciton')
axx[1].set_title('CBM')
axx[0].legend(frameon=False, fontsize = 14)

fig.text(.01, 0.975, "(a)", fontsize = 20, fontweight = 'regular')
fig.text(.5, 0.975, "(b)", fontsize = 20, fontweight = 'regular')

# Add colorbar for the discrete color mapping
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Required for colorbar to work
#cbar_ax = fig.add_axes([1, 0.17, 0.02, 0.75])
cbar = fig.colorbar(sm, cax=cbar_ax, ticks=midpoints)
#cbar.set_label("$n_{eh}$ ($x$10$^{13}$ cm$^{-2})$")
cbar.set_label("$mJ/cm^{2}$")

cbar.ax.set_yticklabels([f"{f:.2f}" for f in fluence])  # Format tick labels

params = {'lines.linewidth' : 3.5, 'axes.linewidth' : 2, 'axes.labelsize' : 20, 
              'xtick.labelsize' : 16, 'ytick.labelsize' : 16, 'axes.titlesize' : 20, 'legend.fontsize' : 16}
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update(params)

# cbar = plt.colorbar(cm, ax=axx[1])
# cbar.set_label('Fluence', rotation=90, fontsize=22)
# cbar.ax.tick_params(labelsize=20)

# cbar = plt.colorbar(cm2, ax=axx[0])
# cbar.set_label('Fluence', rotation=90, fontsize=22)
# cbar.ax.tick_params(labelsize=20)

fig.tight_layout()

if save_figure is True:
    fig.savefig(figure_file_name + '.'+ image_format, bbox_inches='tight', format=image_format)
    
#%% PLOT Fluence Delay TRACES All Together: 680 nm

save_figure = False
figure_file_name = 'Combined'
image_format = 'pdf'

# Standard 915 nm Excitation
scans = [9399, 9400]
fluence = [8, 10]

####
E, E_int = [1.3, 2.0], 0.1
k, k_int = (0), 24
subtract_neg = True
norm_trace = False

cn = 100
p_min = .1
p_max = 3.5

fig, axx = plt.subplots(1, 2)
fig.set_size_inches(10, 4, forward=False)
axx = axx.flatten()

fluence = np.array(fluence)
boundaries = np.concatenate([[fluence[0] - (fluence[1] - fluence[0]) / 2],  # Leftmost edge
                             (fluence[:-1] + fluence[1:]) / 2,  # Midpoints
                             [fluence[-1] + (fluence[-1] - fluence[-2]) / 2]])  # Rightmost edge
midpoints = (boundaries[:-1] + boundaries[1:]) / 2


cmap = plt.colormaps['inferno_r']   # 11 discrete colors
norm = col.BoundaryNorm(boundaries, cmap.N)  # Normalize fluence values to colors

custom_colors = ['lightsteelblue', 'royalblue', 'mediumblue', 'salmon', 'indianred', 'firebrick'] #colors for plotting the traces
cmap = mcolors.ListedColormap(custom_colors)  # Create discrete colormap
norm = mcolors.BoundaryNorm(boundaries, cmap.N)  # Normalize boundaries

i = 0
for scan_i in scans:
    
    res = phoibos.load_data(data_path, scan_i, scan_info, energy_offset, offsets_t0[i], False)
    trace_1 = phoibos.get_time_trace(res, E[0], E_int, k, k_int, subtract_neg, norm_trace)
    trace_2 = phoibos.get_time_trace(res, E[1], E_int, k, k_int, subtract_neg, norm_trace)
    trace_2 = trace_2/np.max(trace_2)
    trace_1 = trace_1/np.max(trace_1)


    t1 = trace_1.plot(ax = axx[0], color = cmap(i), linewidth = 3)
    t2 = trace_2.plot(ax = axx[1], color = cmap(i), linewidth = 3)
    
    i += 1

axx[0].set_xticks(np.arange(-1000,3500,500))
for label in axx[0].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)

axx[0].set_yticks(np.arange(-0.5,1.25,0.25))
for label in axx[0].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
    
axx[1].set_xticks(np.arange(-1000,3500,500))
for label in axx[1].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)

axx[1].set_yticks(np.arange(-0.5,1.25,0.25))
for label in axx[1].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
    
axx[0].set_xlim([-500,3000])
axx[1].set_xlim([-500,3000])
axx[0].set_ylim([-0.1,1.1])
axx[1].set_ylim([-0.1,1.1])

axx[0].set_xlabel('Delay, fs')
axx[1].set_xlabel('Delay, fs')
axx[0].set_ylabel('Norm. Int.')
axx[1].set_ylabel('Norm. Int.')

axx[0].set_title('Exciton')
axx[1].set_title('CBM')

fig.text(.01, 0.975, "(a)", fontsize = 20, fontweight = 'regular')
fig.text(.5, 0.975, "(b)", fontsize = 20, fontweight = 'regular')

# Add colorbar for the discrete color mapping
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Required for colorbar to work
cbar_ax = fig.add_axes([1, 0.17, 0.02, 0.75])
cbar = fig.colorbar(sm, cax=cbar_ax, ticks=midpoints)
cbar.set_label("$n_{eh}$ ($x$10$^{13}$ cm$^{-2})$")
cbar.ax.set_yticklabels([f"{f:.2f}" for f in fluence])  # Format tick labels

params = {'lines.linewidth' : 3.5, 'axes.linewidth' : 2, 'axes.labelsize' : 20, 
              'xtick.labelsize' : 16, 'ytick.labelsize' : 16, 'axes.titlesize' : 20, 'legend.fontsize' : 16}
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update(params)

# cbar = plt.colorbar(cm, ax=axx[1])
# cbar.set_label('Fluence', rotation=90, fontsize=22)
# cbar.ax.tick_params(labelsize=20)

# cbar = plt.colorbar(cm2, ax=axx[0])
# cbar.set_label('Fluence', rotation=90, fontsize=22)
# cbar.ax.tick_params(labelsize=20)

fig.tight_layout()

if save_figure is True:
    fig.savefig(figure_file_name + '.'+ image_format, bbox_inches='tight', format=image_format)
#%% PLOT Fluence Delay TRACES All Together: 640 nm

save_figure = True
figure_file_name = '640nm'
image_format = 'pdf'

# Standard 640 nm Excitation
scans = [9411, 9409, 9412, 9410]
power = [2, 4, 8, 12.5]
fluence = [ 0.59*p/10 for p in power] #10% is 0.59mJ/cm2

####
E, E_int = [1.3, 2.0], 0.1
k, k_int = (0), 24
subtract_neg = True
norm_trace = False

cn = 100
p_min = .1
p_max = 3.5

fig, axx = plt.subplots(1, 2)
fig.set_size_inches(10, 4, forward=False)
axx = axx.flatten()

fluence = np.array(fluence)
boundaries = np.concatenate([[fluence[0] - (fluence[1] - fluence[0]) / 2],  # Leftmost edge
                             (fluence[:-1] + fluence[1:]) / 2,  # Midpoints
                             [fluence[-1] + (fluence[-1] - fluence[-2]) / 2]])  # Rightmost edge
midpoints = (boundaries[:-1] + boundaries[1:]) / 2


cmap = plt.colormaps['inferno_r']   # 11 discrete colors
norm = col.BoundaryNorm(boundaries, cmap.N)  # Normalize fluence values to colors

custom_colors = ['lightsteelblue', 'royalblue', 'salmon', 'firebrick'] #colors for plotting the traces
cmap = mcolors.ListedColormap(custom_colors)  # Create discrete colormap
norm = mcolors.BoundaryNorm(boundaries, cmap.N)  # Normalize boundaries

i = 0
for scan_i in scans:
    
    res = phoibos.load_data(data_path, scan_i, scan_info, energy_offset, offsets_t0[i], False)
    trace_1 = phoibos.get_time_trace(res, E[0], E_int, k, k_int, subtract_neg, norm_trace)
    trace_2 = phoibos.get_time_trace(res, E[1], E_int, k, k_int, subtract_neg, norm_trace)
    trace_2 = trace_2/np.max(trace_2)
    trace_1 = trace_1/np.max(trace_1)


    t1 = trace_1.plot(ax = axx[0], color = cmap(i), linewidth = 3)
    t2 = trace_2.plot(ax = axx[1], color = cmap(i), linewidth = 3, label = f'{fluence[i]:.2f} mJ / cm$^{{2}}$')
    
    i += 1

axx[0].set_xticks(np.arange(-1000,3500,500))
for label in axx[0].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)

axx[0].set_yticks(np.arange(-0.5,1.25,0.25))
for label in axx[0].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
    
axx[1].set_xticks(np.arange(-1000,3500,500))
for label in axx[1].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)

axx[1].set_yticks(np.arange(-0.5,1.25,0.25))
for label in axx[1].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
    
axx[0].set_xlim([-500,3000])
axx[1].set_xlim([-500,3000])
axx[0].set_ylim([-0.1,1.1])
axx[1].set_ylim([-0.1,1.1])

axx[0].set_xlabel('Delay, fs')
axx[1].set_xlabel('Delay, fs')
axx[0].set_ylabel('Norm. Int.')
axx[1].set_ylabel('Norm. Int.')

axx[0].set_title('Exciton')
axx[1].set_title('CBM')
axx[1].legend(frameon=False)

fig.text(.01, 0.975, "(a)", fontsize = 20, fontweight = 'regular')
fig.text(.5, 0.975, "(b)", fontsize = 20, fontweight = 'regular')

# Add colorbar for the discrete color mapping
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Required for colorbar to work
#cbar_ax = fig.add_axes([1, 0.17, 0.02, 0.75])
cbar = fig.colorbar(sm, cax=cbar_ax, ticks=midpoints)
cbar.set_label("$n_{eh}$ ($x$10$^{13}$ cm$^{-2})$")
cbar.set_label("$mJ/cm^{2}$")

cbar.ax.set_yticklabels([f"{f:.2f}" for f in fluence])  # Format tick labels

params = {'lines.linewidth' : 3.5, 'axes.linewidth' : 2, 'axes.labelsize' : 20, 
              'xtick.labelsize' : 16, 'ytick.labelsize' : 16, 'axes.titlesize' : 20, 'legend.fontsize' : 16}
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update(params)

# cbar = plt.colorbar(cm, ax=axx[1])
# cbar.set_label('Fluence', rotation=90, fontsize=22)
# cbar.ax.tick_params(labelsize=20)

# cbar = plt.colorbar(cm2, ax=axx[0])
# cbar.set_label('Fluence', rotation=90, fontsize=22)
# cbar.ax.tick_params(labelsize=20)

fig.tight_layout()

if save_figure is True:
    fig.savefig(figure_file_name + '.'+ image_format, bbox_inches='tight', format=image_format)
#%% PLOT Fluence Delay TRACES All Together: 640 nm

save_figure = False
figure_file_name = 'Combined'
image_format = 'pdf'

# Standard 915 nm Excitation
scans = [9411, 9409, 9412, 9410
fluence = [.35, .8, 1.74, 2.4]

####
E, E_int = [1.3, 2.0], 0.1
k, k_int = (0), 24
subtract_neg = True
norm_trace = False

cn = 100
p_min = .1
p_max = 3.5

fig, axx = plt.subplots(1, 2)
fig.set_size_inches(10, 4, forward=False)
axx = axx.flatten()

fluence = np.array(fluence)
boundaries = np.concatenate([[fluence[0] - (fluence[1] - fluence[0]) / 2],  # Leftmost edge
                             (fluence[:-1] + fluence[1:]) / 2,  # Midpoints
                             [fluence[-1] + (fluence[-1] - fluence[-2]) / 2]])  # Rightmost edge
midpoints = (boundaries[:-1] + boundaries[1:]) / 2


cmap = plt.colormaps['inferno_r']   # 11 discrete colors
norm = col.BoundaryNorm(boundaries, cmap.N)  # Normalize fluence values to colors

custom_colors = ['lightsteelblue', 'royalblue', 'mediumblue', 'salmon', 'indianred', 'firebrick'] #colors for plotting the traces
cmap = mcolors.ListedColormap(custom_colors)  # Create discrete colormap
norm = mcolors.BoundaryNorm(boundaries, cmap.N)  # Normalize boundaries

i = 0
for scan_i in scans:
    
    res = phoibos.load_data(data_path, scan_i, scan_info, energy_offset, offsets_t0[i], False)
    trace_1 = phoibos.get_time_trace(res, E[0], E_int, k, k_int, subtract_neg, norm_trace)
    trace_2 = phoibos.get_time_trace(res, E[1], E_int, k, k_int, subtract_neg, norm_trace)
    trace_2 = trace_2/np.max(trace_2)
    trace_1 = trace_1/np.max(trace_1)


    t1 = trace_1.plot(ax = axx[0], color = cmap(i), linewidth = 3)
    t2 = trace_2.plot(ax = axx[1], color = cmap(i), linewidth = 3)
    
    i += 1

axx[0].set_xticks(np.arange(-1000,3500,500))
for label in axx[0].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)

axx[0].set_yticks(np.arange(-0.5,1.25,0.25))
for label in axx[0].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
    
axx[1].set_xticks(np.arange(-1000,3500,500))
for label in axx[1].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)

axx[1].set_yticks(np.arange(-0.4,1.25,0.2))
for label in axx[1].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
    
axx[0].set_xlim([-500,3000])
axx[1].set_xlim([-500,3000])
axx[0].set_ylim([-0.1,1.1])
axx[1].set_ylim([-0.1,1.1])

axx[0].set_xlabel('Delay, fs')
axx[1].set_xlabel('Delay, fs')
axx[0].set_ylabel('Norm. Int.')
axx[1].set_ylabel('Norm. Int.')

axx[0].set_title('Exciton')
axx[1].set_title('CBM')

fig.text(.01, 0.975, "(a)", fontsize = 20, fontweight = 'regular')
fig.text(.5, 0.975, "(b)", fontsize = 20, fontweight = 'regular')

# Add colorbar for the discrete color mapping
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Required for colorbar to work
cbar_ax = fig.add_axes([1, 0.17, 0.02, 0.75])
cbar = fig.colorbar(sm, cax=cbar_ax, ticks=midpoints)
cbar.set_label("$n_{eh}$ ($x$10$^{13}$ cm$^{-2})$")
cbar.ax.set_yticklabels([f"{f:.2f}" for f in fluence])  # Format tick labels

params = {'lines.linewidth' : 3.5, 'axes.linewidth' : 2, 'axes.labelsize' : 20, 
              'xtick.labelsize' : 16, 'ytick.labelsize' : 16, 'axes.titlesize' : 20, 'legend.fontsize' : 16}
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update(params)

# cbar = plt.colorbar(cm, ax=axx[1])
# cbar.set_label('Fluence', rotation=90, fontsize=22)
# cbar.ax.tick_params(labelsize=20)

# cbar = plt.colorbar(cm2, ax=axx[0])
# cbar.set_label('Fluence', rotation=90, fontsize=22)
# cbar.ax.tick_params(labelsize=20)

fig.tight_layout()

if save_figure is True:
    fig.savefig(figure_file_name + '.'+ image_format, bbox_inches='tight', format=image_format) 
    
#%% PLOT Fluence Delay TRACES All Together: 400 nm

save_figure = False
figure_file_name = '400nm'
image_format = 'pdf'

# Standard 400 nm Excitation
scans = [9525, 9517, 9526]
#scans = [9526]
power = [20, 36.3, 45]
fluence = [0.3, 0.54, 0.68]

####
E, E_int = [1.3, 2.05], 0.1
k, k_int = (0), 24
subtract_neg = True
norm_trace = False

fig, axx = plt.subplots(1, 2)
fig.set_size_inches(10, 4, forward=False)
axx = axx.flatten()

fluence = np.array(fluence)
boundaries = np.concatenate([[fluence[0] - (fluence[1] - fluence[0]) / 2],  # Leftmost edge
                             (fluence[:-1] + fluence[1:]) / 2,  # Midpoints
                             [fluence[-1] + (fluence[-1] - fluence[-2]) / 2]])  # Rightmost edge
midpoints = (boundaries[:-1] + boundaries[1:]) / 2

custom_colors = ['lightsteelblue', 'royalblue', 'mediumblue'] #colors for plotting the traces
cmap = mcolors.ListedColormap(custom_colors)  # Create discrete colormap
norm = mcolors.BoundaryNorm(boundaries, cmap.N)  # Normalize boundaries

i = 0
for scan_i in scans:
    
    res = phoibos.load_data(data_path, scan_i, scan_info, energy_offset, offsets_t0[i], False)
    trace_1 = phoibos.get_time_trace(res, E[0], E_int, k, k_int, subtract_neg, norm_trace)
    trace_2 = phoibos.get_time_trace(res, E[1], E_int, k, k_int, subtract_neg, norm_trace)
    trace_2 = trace_2/np.max(trace_2)
    trace_1 = trace_1/np.max(trace_1)

    t1 = trace_1.plot(ax = axx[0], color = cmap(i), linewidth = 3)
    t2 = trace_2.plot(ax = axx[1], color = cmap(i), linewidth = 3, label = f'{fluence[i]} mJ / cm$^{{2}}$')
    
    test = exp_rise_monoexp_decay(trace_1.Delay.values, 1.02, 258, 6700)
    axx[0].plot(trace_1.Delay.values, test, color = 'maroon')

    test2 = exp_rise_monoexp_decay(trace_1.Delay.values, 1., 134, 8000)
    axx[0].plot(trace_1.Delay.values, 0.9*test2/np.max(test2), color = 'green')
    
    test3 = exp_rise_biexp_decay(trace_1.Delay.values, 3, 200, .9, 280, 4000)
    axx[1].plot(trace_1.Delay.values, test3/np.max(test3), color = 'pink')

    #t1 = trace_1.plot(ax = axx[1], color = 'royalblue', linewidth = 3)
    #t2 = trace_2.plot(ax = axx[1], color = 'blue', linewidth = 3, label = f'{fluence[i]} mJ / cm$^{{2}}$')

    i += 1

axx[0].set_xticks(np.arange(-1000,3500,500))
for label in axx[0].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)

axx[0].set_yticks(np.arange(-0.5,1.25,0.25))
for label in axx[0].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
    
axx[1].set_xticks(np.arange(-1000,3500,500))
for label in axx[1].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)

axx[1].set_yticks(np.arange(-0.5,1.25,0.25))
for label in axx[1].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
    
axx[0].set_xlim([-500,3000])
axx[1].set_xlim([-500,3000])
axx[0].set_ylim([-0.1,1.1])
axx[1].set_ylim([-0.1,1.1])

axx[0].set_xlabel('Delay, fs')
axx[1].set_xlabel('Delay, fs')
axx[0].set_ylabel('Norm. Int.')
axx[1].set_ylabel('Norm. Int.')

axx[0].set_title('Exciton')
axx[1].set_title('CBM')
axx[1].legend(frameon=False)
fig.text(.01, 0.975, "(a)", fontsize = 20, fontweight = 'regular')
fig.text(.5, 0.975, "(b)", fontsize = 20, fontweight = 'regular')

# Add colorbar for the discrete color mapping
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Required for colorbar to work
#cbar_ax = fig.add_axes([1, 0.17, 0.02, 0.75])
cbar = fig.colorbar(sm, cax=cbar_ax, ticks=midpoints)
#cbar.set_label("$n_{eh}$ ($x$10$^{13}$ cm$^{-2})$")
cbar.set_label("$mJ/cm^{2}$")

cbar.ax.set_yticklabels([f"{f:.2f}" for f in fluence])  # Format tick labels

params = {'lines.linewidth' : 3.5, 'axes.linewidth' : 2, 'axes.labelsize' : 20, 
              'xtick.labelsize' : 16, 'ytick.labelsize' : 16, 'axes.titlesize' : 20, 'legend.fontsize' : 16}
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update(params)

fig.tight_layout()

if save_figure is True:
    fig.savefig(figure_file_name + '.'+ image_format, bbox_inches='tight', format=image_format)

#%% PLOT Fluences and Wavelengths Together

save_figure = False
figure_file_name = 'Combined'
image_format = 'pdf'

scans = [9219, 9217, 9218, 9216, 9220, 9228]

power = 1.05*np.asarray([153, 111, 91, 66, 47, 32, 15, 10, 8, 5])
fluence = [1, 2, 3, 4]

scans = [9373, 9400, 9412, 9526]
custom_colors = ['lightsteelblue', 'royalblue', 'salmon', 'firebrick'] #colors for plotting the traces
custom_colors = ['brown', 'red', 'purple', 'blue']

# Expanded 915 nm Excitation
#scans = [9227, 9219, 9217, 9218, 9216, 9220, 9228, 9231] # Scans to analyze and fit below: 910 nm + 400 nm
#offsets_t0 = [-191.7, -162.4, -152.7, -183.1, -118.5, -113.7, -125.2, -147.6]
#fluence = [4.7, 8.3, 20.9, 41.7, 65.6, 83.2, 104.7, 151]

####
k, k_int = (0), 24
E[0], E[1], E_int = 1.3, 2.05, 0.1
subtract_neg = True
norm_trace = False

fig, axx = plt.subplots(1, 2)
fig.set_size_inches(10, 4, forward=False)
axx = axx.flatten()

fluence = np.array(fluence)
boundaries = np.concatenate([[fluence[0] - (fluence[1] - fluence[0]) / 2],  # Leftmost edge
                             (fluence[:-1] + fluence[1:]) / 2,  # Midpoints
                             [fluence[-1] + (fluence[-1] - fluence[-2]) / 2]])  # Rightmost edge
midpoints = (boundaries[:-1] + boundaries[1:]) / 2

cmap = cm.get_cmap('inferno_r', len(scans))    # 11 discrete colors
norm = col.BoundaryNorm(boundaries, cmap.N)  # Normalize fluence values to colors

cmap = mcolors.ListedColormap(custom_colors)  # Create discrete colormap
norm = mcolors.BoundaryNorm(boundaries, cmap.N)  # Normalize boundaries

i = 0
for scan_i in scans:
    
    res = phoibos.load_data(data_path, scan_i, scan_info, energy_offset, offsets_t0[i], False)
    trace_1 = phoibos.get_time_trace(res, E[0], E_int, k, k_int, subtract_neg, norm_trace)
    trace_2 = phoibos.get_time_trace(res, E[1], E_int, k, k_int, subtract_neg, norm_trace)
    trace_1 = trace_1/np.max(trace_2)
    trace_2 = trace_2/np.max(trace_2)

    t1 = trace_1.plot(ax = axx[0], color = cmap(i), linewidth = 3)
    t2 = trace_2.plot(ax = axx[1], color = cmap(i), linewidth = 3)
    
    i += 1

axx[0].set_xticks(np.arange(-1000,3500,500))
for label in axx[0].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)

axx[0].set_yticks(np.arange(-0.5,1.25,0.25))
for label in axx[0].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
    
axx[1].set_xticks(np.arange(-1000,3500,500))
for label in axx[1].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)

axx[1].set_yticks(np.arange(-0.5,1.25,0.25))
for label in axx[1].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
    
axx[0].set_xlim([-500,3000])
axx[1].set_xlim([-500,3000])
axx[0].set_ylim([-0.1,1.1])
axx[1].set_ylim([-0.1,1.1])

axx[0].set_xlabel('Delay, fs')
axx[1].set_xlabel('Delay, fs')
axx[0].set_ylabel('Norm. Int.')
axx[1].set_ylabel('Norm. Int.')

axx[0].set_title('Exciton')
axx[1].set_title('CBM')

fig.text(.01, 0.975, "(a)", fontsize = 20, fontweight = 'regular')
fig.text(.5, 0.975, "(b)", fontsize = 20, fontweight = 'regular')

# Add colorbar for the discrete color mapping
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Required for colorbar to work
cbar_ax = fig.add_axes([1, 0.17, 0.02, 0.75])
cbar = fig.colorbar(sm, cax=cbar_ax, ticks=midpoints)
cbar.set_label("$n_{eh}$ ($x$10$^{13}$ cm$^{-2})$")
cbar.ax.set_yticklabels([f"{f:.2f}" for f in fluence])  # Format tick labels

params = {'lines.linewidth' : 3.5, 'axes.linewidth' : 2, 'axes.labelsize' : 20, 
              'xtick.labelsize' : 16, 'ytick.labelsize' : 16, 'axes.titlesize' : 20, 'legend.fontsize' : 16}
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update(params)

# cbar = plt.colorbar(cm, ax=axx[1])
# cbar.set_label('Fluence', rotation=90, fontsize=22)
# cbar.ax.tick_params(labelsize=20)

# cbar = plt.colorbar(cm2, ax=axx[0])
# cbar.set_label('Fluence', rotation=90, fontsize=22)
# cbar.ax.tick_params(labelsize=20)

fig.tight_layout()

if save_figure is True:
    fig.savefig(figure_file_name + '.'+ image_format, bbox_inches='tight', format=image_format) 

#%% Plot Cold Data Scans

%matplotlib inline

save_figure = False
figure_file_name = 'Cold_Data_phoibos'
image_format = 'pdf'

# Scans to plot
# Standard 915 nm Excitation
scans = [9219, 9217, 9218, 9216, 9220, 9228]

# Combined
scans = [9241, 9240, 9137] #915 nm (top 3) ; 700 nm, 640 nm, 400 nm

#power = 1.05*np.asarray([153, 111, 91, 66, 47, 32, 15, 10, 8, 5])
fluence = [.2, .35, .8, 1.74, 2.4, 2.9]
fluence = [.2, .8, 2.9, 4.5, 3.6, 0]
ev = [1.355, 1.355, 3.10]
# Specify energy and Angle ranges
E, E_int = [1.325, 2.075], 0.1
E, E_int = [1.325, 2.025], 0.1

k, k_int = 0, 20
subtract_neg = True
norm_trace = False

# Plot
fig, ax = plt.subplots(1, 3)
fig.set_size_inches(11, 4, forward=False)
plt.gcf().set_dpi(300)
ax = ax.flatten()

for i in np.arange(len(scans)):
    scan_i = scans[i]
    res = phoibos.load_data(data_path, scan_i, scan_info, 19.72 , _ , False)
    WL = scan_info[str(scan_i)].get("Wavelength")
    temp = scan_info[str(scan_i)].get("Temperature")

    if i == 3:
        E, E_int = [1.3, 1.92], 0.1

#    res = phoibos.load_data(scan_i, energy_offset, offsets_t0[i])
    trace_1 = phoibos.get_time_trace(res, E[0], E_int, k, k_int, subtract_neg, norm_trace)
    trace_2 = phoibos.get_time_trace(res, E[1], E_int, k, k_int, subtract_neg, norm_trace)
    norm = np.max([trace_1,trace_2])
    #trace_2 = trace_2/np.max(trace_1)
    #trace_1 = trace_1/np.max(trace_1)
    trace_2 = trace_2/norm
    trace_1 = trace_1/norm
    
    t1 = trace_1.plot(ax = ax[i], color = 'black', linewidth = 3)
    t2 = trace_2.plot(ax = ax[i], color = 'crimson', linewidth = 3)
    #ax[i].text(1000, .9, f"$n_{{eh}} = {fluence[i]:.2f} x 10^{{13}} cm^{{-2}}$", fontsize = 14, fontweight = 'regular')

    # Set major ticks at every 500
    xticks = np.arange(-1000, 3500, 500)
    ax[i].set_xticks(xticks)
    
    # Hide every other label by replacing with an empty string
    xtick_labels = [str(tick/1000) if i % 2 == 0 else "" for i, tick in enumerate(xticks)]
    ax[i].set_xticklabels(xtick_labels)
    ax[i].set_yticks(np.arange(-0.5,1.25,0.25))
    for label in ax[i].yaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
    ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax[i].set_xlim([-500,3000])
    ax[i].set_ylim([-0.1,1.1])
    ax[i].set_xlabel('Delay, ps')
    ax[i].set_ylabel('Nom. Int.')
    ax[i].set_title(f'$hv_{{ex}} $ = {ev[i]} eV', fontsize = 22)
    ax[i].text(1000, .95, f'T = {temp} K', fontsize = 18)
    #ax[0].set_title('Exciton')
    #ax[1].set_title('CBM')

fig.text(.01, 1, "(a)", fontsize = 20, fontweight = 'regular')
fig.text(.33, 1, "(b)", fontsize = 20, fontweight = 'regular')
fig.text(.66, 1, "(c)", fontsize = 20, fontweight = 'regular')

# Add colorbar for the discrete color mapping
# sm = cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])  # Required for colorbar to work
# cbar_ax = fig.add_axes([1, 0.17, 0.02, 0.75])
# cbar = fig.colorbar(sm, cax=cbar_ax, ticks=midpoints)
# cbar.set_label("$n_{eh}$ ($x$10$^{13}$ cm$^{-2})$")
# cbar.ax.set_yticklabels([f"{f:.2f}" for f in fluence])  # Format tick labels

params = {'lines.linewidth' : 3.5, 'axes.linewidth' : 2, 'axes.labelsize' : 20, 
              'xtick.labelsize' : 16, 'ytick.labelsize' : 16, 'axes.titlesize' : 20, 'legend.fontsize' : 16}
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update(params)

fig.tight_layout()

if save_figure is True:
    fig.savefig(figure_file_name + '.'+ image_format, bbox_inches='tight', format=image_format)

#%% Plot Neg, Pos, and Difference Angle-Energy Panels

%matplotlib inline

save_figure = False
figure_file_name = 'ARPRES_Panels_diff'
image_format = 'pdf'

E, E_int = [1.375, 2.125], 0.1

colormap = cmap_LTL # 'bone_r'# 'Purples'

fig, axx = plt.subplots(1, 3)
fig.set_size_inches(12, 4, forward=True)
plt.gcf().set_dpi(300)
axx = axx.flatten()

E_inset = .8

res_neg = res.loc[{'Delay':slice(-1000,-300)}]
res_pos = res.loc[{'Delay':slice(0,5000)}]
res_neg_mean = res_neg.mean(axis=2)
res_pos_mean = res_pos.mean(axis=2)

neg_enh = phoibos.enhance_features(res_neg_mean, E_inset, 1, True)
pos_enh = phoibos.enhance_features(res_pos_mean, E_inset, 1, True)
diff_enh = phoibos.enhance_features(res_diff_E_Ang, E_inset, 1, True) 
diff_enh = diff_enh / np.max(np.abs(diff_enh))

im1 = neg_enh.T.plot.imshow(ax = axx[0], cmap = colormap)
im2 = pos_enh.T.plot.imshow(ax = axx[1], cmap = colormap)
im3 = diff_enh.T.plot.imshow(ax = axx[2], cmap = 'RdBu_r')
axx[0].set_ylim(-1,2.65)
axx[1].set_ylim(-1,2.65)
axx[2].set_ylim(-1,2.65)

axx[2].axhline(E_inset, color = 'black', linewidth = 1)
axx[2].axhline(E_inset, color = 'black', linewidth = 1)
axx[2].axhline(E_inset, color = 'black', linewidth = 1)

axx[0].set_title('$\Delta$t < -300 fs')
axx[1].set_title('$\Delta$t > 0 fs ')
axx[2].set_title(f"Scan{scan}")
axx[2].set_title('Difference')

axx[0].set_ylabel('$E - E_{VBM}, eV$')
axx[1].set_ylabel('$E - E_{VBM}, eV$')
axx[2].set_ylabel('$E - E_{VBM}, eV$')

fig.text(.01, .95, "(a)", fontsize = 18, fontweight = 'regular')
fig.text(.33, .95, "(b)", fontsize = 18, fontweight = 'regular')
fig.text(.66, .95, "(c)", fontsize = 18, fontweight = 'regular')

params = {'lines.linewidth' : 3.5, 'axes.linewidth' : 2, 'axes.labelsize' : 20, 
              'xtick.labelsize' : 16, 'ytick.labelsize' : 16, 'axes.titlesize' : 20, 'legend.fontsize' : 16}
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update(params)

fig.tight_layout()
plt.show()

if save_figure is True:
    fig.savefig(figure_file_name + '.'+ image_format, bbox_inches='tight', format=image_format) 

#%% Waterfall difference Panel

save_figure = True
figure_file_name = 'WaterFallDifference_phoibos400nm'
image_format = 'pdf'

#E, E_int = [1.3, 2.0], 0.1
scan = 9526
res = phoibos.load_data(data_path, scan, scan_info, _, _, False)

A, A_int = 0, 20
E_inset = 0.9

colormap, scale = cmap_LTL2, [-1,1] #'bone_r'
subtract_neg = True
norm_trace = False

###
WL = scan_info[str(scan)].get("Wavelength")
per = (scan_info[str(scan)].get("Percent"))
Temp = float(scan_info[str(scan)].get("Temperature"))

res_neg = res.loc[{'Delay':slice(-1000,-300)}]
res_pos = res.loc[{'Delay':slice(0,5000)}]

res_neg_mean = res_neg.mean(axis=2)
res_pos_mean = res_pos.mean(axis=2)

#res_diff_E_Ang = res_pos_mean - res_neg_mean

res_diff_E_Ang = res_pos_mean - res_neg_mean
#res_diff_E_Ang = res.loc[{'Delay':slice(-100,0)}].mean(axis=2) - res_neg_mean
#res_diff_E_Ang = res.loc[{'Delay':slice(250,350)}].mean(axis=2) - res_neg_mean

res_diff_E_Ang = res_diff_E_Ang/np.max(np.abs(res_diff_E_Ang))

res_sum_Angle = res.loc[{'Angle':slice(-A-A_int/2,A+A_int/2)}].sum(axis=0)
res_sum_Angle = res_sum_Angle/np.max(res_sum_Angle)

res_diff = res - res_neg.mean(axis=2)
res_diff_sum_Angle = res_diff.loc[{'Angle':slice(-A-A_int/2,A+A_int/2)}].sum(axis=0)
res_diff_sum_Angle = res_diff_sum_Angle/np.max(res_diff_sum_Angle)

res_sum_Angle = phoibos.enhance_features(res_sum_Angle, E_inset, _ , True)

res_diff_E_Ang = phoibos.enhance_features(res_diff_E_Ang, E_inset, _ , True)
res_diff_sum_Angle = phoibos.enhance_features(res_diff_sum_Angle, E_inset, _ , True)

trace_1 = phoibos.get_time_trace(res, E[0], E_int, A, A_int, subtract_neg, norm_trace)
trace_2 = phoibos.get_time_trace(res, E[1], E_int, A, A_int, subtract_neg, norm_trace)

trace_2 = trace_2/trace_1.max()
trace_1 = trace_1/trace_1.max()

############
### PLOT ###
############

fig, axx = plt.subplots(1, 2)
fig.set_size_inches(12, 4, forward=False)
plt.gcf().set_dpi(300)
axx = axx.flatten()

im1 = res_sum_Angle.plot.imshow(ax = axx[0], cmap = cmap_LTL, vmin = 0, vmax = scale[1])
#plt.colorbar(im1, ax=axx[0], extend='neither')

im2 = res_diff_sum_Angle.plot.imshow(ax = axx[1], cmap = 'RdBu_r', vmin = -1, vmax = 1)
#plt.colorbar(im2, ax=axx[1], extend='neither')

#fig.colorbar(im2, ax=axx[1])
#im_dyn = axx[2].plot(trace_1.Delay.loc[{"Delay":slice(0,50000)}].values, \
 #                  0.6*np.exp(-trace_1.Delay.loc[{"Delay":slice(0,50000)}].values/18000) +

  #                 0.3*np.exp(-trace_1.Delay.loc[{"Delay":slice(0,50000)}].values/2000))
#axx[0].axhline(E[0],  color = 'black')
#axx[0].axhline(E[1],  color = 'red')
axx[0].axhline(E_inset,  color = 'grey', linestyle = 'dashed')
axx[0].set_title('Integrated')
axx[1].set_title('Difference')
axx[0].set_xlim(-750,3000)
axx[0].set_ylim(E_inset-0.25,res.Energy.values.max())
axx[1].set_xlim(-750,3000)
axx[1].set_ylim(E_inset-0.25,res.Energy.values.max())
axx[1].axhline(E_inset,  color = 'grey', linestyle = 'dashed')

axx[0].set_xlabel('Delay, fs')
axx[1].set_xlabel('Delay, fs')
axx[0].set_ylabel('$E - E_{VBM}, eV$')
axx[1].set_ylabel('$E - E_{VBM}, eV$')

fig.text(.01, .95, "(d)", fontsize = 18, fontweight = 'regular')
fig.text(.5, .95, "(e)", fontsize = 18, fontweight = 'regular')

fig.tight_layout()

if save_figure is True:
    fig.savefig(figure_file_name + '.'+ image_format, bbox_inches='tight', format=image_format) 
    
#%% Total Excited State Population 

save_figure = False
figure_file_name = 'Combined'
image_format = 'pdf'

# Standard 915 nm Excitation
scans = [9219, 9217, 9218, 9216, 9220, 9228]
#offsets_t0 = [-162.4, -152.7, -183.1, -118.5, -113.7, -125.2]
power = 1.05*np.asarray([153, 111, 91, 66, 47, 32, 15, 10, 8, 5])
fluence = [.2, .35, .8, 1.74, 2.4, 2.9]

# Expanded 915 nm Excitation
#scans = [9227, 9219, 9217, 9218, 9216, 9220, 9228, 9231] # Scans to analyze and fit below: 910 nm + 400 nm
#offsets_t0 = [-191.7, -162.4, -152.7, -183.1, -118.5, -113.7, -125.2, -147.6]
#fluence = [4.7, 8.3, 20.9, 41.7, 65.6, 83.2, 104.7, 151]

####
k, k_int = (0), 24
E[0], E[1], E_int = 1.35, 2.1, 0.1
subtract_neg = True
norm_trace = False

cn = 100
p_min = .1
p_max = 3.5

fig, axx = plt.subplots()
fig.set_size_inches(5, 4, forward=False)
plt.gcf().set_dpi(300)

fluence = np.array(fluence)
boundaries = np.concatenate([[fluence[0] - (fluence[1] - fluence[0]) / 2],  # Leftmost edge
                             (fluence[:-1] + fluence[1:]) / 2,  # Midpoints
                             [fluence[-1] + (fluence[-1] - fluence[-2]) / 2]])  # Rightmost edge
midpoints = (boundaries[:-1] + boundaries[1:]) / 2


cmap = cm.get_cmap('inferno_r', len(scans))    # 11 discrete colors
norm = col.BoundaryNorm(boundaries, cmap.N)  # Normalize fluence values to colors

custom_colors = ['lightsteelblue', 'royalblue', 'mediumblue', 'salmon', 'indianred', 'firebrick'] #colors for plotting the traces
cmap = mcolors.ListedColormap(custom_colors)  # Create discrete colormap
norm = mcolors.BoundaryNorm(boundaries, cmap.N)  # Normalize boundaries

intensity = np.zeros(len(scans))
i = 0
for scan_i in scans:
    
    res = phoibos.load_data(data_path, scan_i, scan_info, energy_offset, offsets_t0[i], False)
    trace_1 = phoibos.get_time_trace(res, E[0], E_int, k, k_int, subtract_neg, norm_trace)
    trace_2 = phoibos.get_time_trace(res, E[1], E_int, k, k_int, subtract_neg, norm_trace)
    trace_2 = trace_2/np.max(trace_1)
    trace_1 = trace_1/np.max(trace_1)

    combined = trace_1 + trace_2
    combined = phoibos.get_time_trace(res, 1.7, 1, k, k_int, subtract_neg, norm_trace)
    
    intensity[i] = np.max(combined)
    combined = combined/np.max(combined)

    t3 = combined.plot(ax = axx, color = cmap(i), linewidth = 3)
    
    i += 1

axx.set_xticks(np.arange(-1000,3500,500))
for label in axx.xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)

axx.set_yticks(np.arange(-0.5,1.25,0.25))
for label in axx.yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
    

axx.set_xlim([-500,3000])
axx.set_ylim([-0.1,1.1])

axx.set_xlabel('Delay, fs')
axx.set_ylabel('Norm. Int.')

axx.set_title('Excited State Population')

fig.text(.01, 0.975, "(a)", fontsize = 20, fontweight = 'regular')

# Add colorbar for the discrete color mapping
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Required for colorbar to work
cbar_ax = fig.add_axes([1, 0.17, 0.02, 0.75])
cbar = fig.colorbar(sm, cax=cbar_ax, ticks=midpoints)
cbar.set_label("$n_{eh}$ ($x$10$^{13}$ cm$^{-2})$")
cbar.ax.set_yticklabels([f"{f:.2f}" for f in fluence])  # Format tick labels

params = {'lines.linewidth' : 3.5, 'axes.linewidth' : 2, 'axes.labelsize' : 20, 
              'xtick.labelsize' : 16, 'ytick.labelsize' : 16, 'axes.titlesize' : 20, 'legend.fontsize' : 16}
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update(params)

fig.tight_layout()

if save_figure is True:
    fig.savefig(figure_file_name + '.'+ image_format, bbox_inches='tight', format=image_format) 

#%%

plt.plot(fluence, intensity, '-o', color = 'black')
plt.plot(fluence, .01+(1.025e8)*fluence, color = 'pink', linestyle = 'dashed')
plt.ylim(0,2.75e8)
plt.xlim(0,3)

#%% # PLOT Fluence Delay TRACES All Together

save_figure = False
figure_file_name = 'phoibosfluencetraces'

fig, axx = plt.subplots(1, 3)
fig.set_size_inches(12, 4, forward=False)
plt.gcf().set_dpi(300)
axx = axx.flatten()

# Standard 915 nm Excitation
scans = [9219, 9217, 9218, 9216, 9220, 9228]
offsets_t0 = [-162.4, -152.7, -183.1, -118.5, -113.7, -125.2]
power = 1.05*np.asarray([153, 111, 91, 66, 47, 32, 15, 10, 8, 5])
fluence = [.2, .35, .8, 1.74, 2.4, 2.9]

# Expanded 915 nm Excitation
scans = [9227, 9219, 9217, 9218, 9216, 9220, 9228, 9231] # Scans to analyze and fit below: 910 nm + 400 nm
offsets_t0 = [-191.7, -162.4, -152.7, -183.1, -118.5, -113.7, -125.2, -147.6]
fluence = [4.7, 8.3, 20.9, 41.7, 65.6, 83.2, 104.7, 151]
colors = ['grey', 'black', 'purple', 'green', 'blue', 'magenta', 'orange', 'red']

# 400 nm Excitation
# scans = [9525, 9526]
# offsets_t0 = [-77, -200.6]
# colors = ['grey', 'maroon']
# fluence = [20, 45]

# ALL
trans_percent = [float(scan_info[str(s)].get("Percent")) for s in scans] # Retrieve the percentage
power = [4.7, 8.3, 20.9, 41.7, 65.6, 83.2, 104.7, 151, 20, 36.3, 45]

#####
#####

k, k_int = (0), 24
E[0], E[1], E3, E_int = 1.35, 2.1, 0.1,  0.1
subtract_neg = True
norm_trace = False

cn = 100
p_min = 0
p_max = 155

colors1 = plt.cm.jet(np.linspace(0, 1, cn)) 
cm2 = plt.cm.ScalarMappable(norm = plt.Normalize(vmin=p_min,vmax=p_max), cmap=plt.cm.plasma)

colors2 = plt.cm.plasma(np.linspace(0, 1, cn))
cm = plt.cm.ScalarMappable(norm = plt.Normalize(vmin=p_min,vmax=p_max), cmap=plt.cm.plasma)

colors3 = plt.cm.plasma(np.linspace(0, 1, cn)) 

fluence_cbar = np.linspace(p_min, p_max, cn)

pop = []
i = 0
for scan_i in scans:
    
    res = phoibos.load_data(data_path, scan_i, scan_info, energy_offset, offsets_t0[i], False)
    trace_1 = phoibos.get_time_trace(res, E[0], E_int, k , k_int, subtract_neg, norm_trace)
    trace_2 = phoibos.get_time_trace(res, E[1], E_int, k , k_int, subtract_neg, norm_trace)
#    trace_2 = trace_2/np.max(trace_1)
#    trace_1 = trace_1/np.max(trace_1)
    trace_1 = trace_1
    trace_2 = trace_2
    
    trace_3 = trace_1 + trace_2
    
    pop.append(trace_3.max())
    
    #trace_2 = trace_2/np.max(trace_1)

    j_fluence = (np.abs(fluence_cbar-fluence[i])).argmin()

    t1 = trace_1.plot(ax = axx[0], color = cmap(i) )
    t2 = trace_2.plot(ax = axx[1], color = cmap(i) , label = f"{fluence[i]}")
    #t1 = trace_1.plot(ax = axx[0], color = colors[i])
    #t2 = trace_2.plot(ax = axx[1], color = colors[i], label = f"{fluence[i]}")
    t3 = trace_3.plot(ax = axx[2], color = cmap(i), label = f"{fluence[i]}")

    i += 1

axx[0].set_title('Exciton')
axx[1].set_title('CBM')
axx[0].set_xlim([-500,3000])
axx[1].set_xlim([-500,3000])
axx[2].set_xlim([-500,3000])
axx[0].set_ylabel('Intensity, a.u.')
axx[1].set_ylabel('Intensity, a.u.')
axx[1].legend(frameon=False, loc = 'upper right', fontsize = 11)

# cbar = plt.colorbar(cm, ax=axx[1])
# cbar.set_label('Fluence', rotation=90, fontsize=22)
# # cbar.ax.tick_params(labelsize=20)

# cbar = plt.colorbar(cm2, ax=axx[0])
# cbar.set_label('Fluence', rotation=90, fontsize=22)
# cbar.ax.tick_params(labelsize=20)

fig.tight_layout()

if save_figure is True:
    fig.savefig((figure_file_name +'.svg'), format='svg')

#%%

plt.plot(fluence, pop, '-o')
    
#%%

def func(x, a, b, tau1, tau2):
    return a*np.exp(-x/tau1)+b*np.exp(-x/tau2)

delays_trunc = trace_1.loc[{"Delay":slice(0,20000)}].Delay.values
trace_trunc =  trace_1.loc[{"Delay":slice(0,20000)}].values

delays = trace_1.Delay.values
trace =  trace_1.values

popt, pcov = curve_fit(func, delays_trunc, trace_trunc, p0=(1,1,2000,15000))

fit = func(delays_trunc, *popt)

fig = plt.figure()
plt.plot(delays, trace, 'o', color = 'grey')
plt.plot(delays_trunc, fit, color = 'blue')
plt.title(f"Biexp: tau_1 = {round(popt[2])}, tau_2 = {round(popt[3],0)}")
plt.xlim(-600,20000)
plt.ylabel('Int.')
plt.xlabel('Delay, fs')
#print(popt)

save_figure = False
figure_file_name = f"Long_Delays_{scan}"
#plt.rcParams['svg.fonttype'] = 'none'
new_rc_params = {'text.usetex': False, "svg.fonttype": 'none'}
plt.rcParams.update(new_rc_params)

if save_figure is True:
    fig.savefig((figure_file_name +'.svg'), format='svg')

#%% Do the EDC Fits: Functions

###################
##### Fit EDCs ####
###################

##### VBM #########

def fit_vbm_dynamics(res, k, k_int):
    e1 = -.2
    e2 = 0.6
    p0 = [1, 0, .2, 0] # Fitting params initial guess [amp, center, width, offset]
    bnds = ((0.5, -1, 0.0, 0), (1.5, 0.5, 1, .5))
    
    centers_VBM = np.zeros(len(res.Delay))
    p_fits_VBM = np.zeros((len(res.Delay),4))
    p_err_VBM = np.zeros((len(res.Delay),2))
    
    (kx), k_int = k, k_int
    edc_gamma = res.loc[{"Angle":slice(kx-k_int/2,kx+k_int/2)}].sum(dim=("Angle"))
    edc_gamma = edc_gamma/np.max(edc_gamma)
    
    n = len(res.Delay)
    for t in np.arange(n):
        edc_i = edc_gamma.loc[{"Energy":slice(e1,e2)}][:,t].values
        edc_i = edc_i/np.max(edc_i)
        
        try:
            popt, pcov = curve_fit(gaussian, edc_gamma.loc[{"Energy":slice(e1,e2)}].Energy.values, edc_i, p0, method=None, bounds = bnds)
        except ValueError:
            print('oops')
            popt = [0,0,0,0]
            
        centers_VBM[t] = popt[1]
        p_fits_VBM[t,:] = popt
        perr = np.sqrt(np.diag(pcov))
        p_err_VBM[t,:] = perr[1:2+1]
        
    return centers_VBM, p_fits_VBM, p_err_VBM

##### CBM AND EXCITON #####

def fit_ex_cbm_dynamics(res, delay_int):
    delay_int = 50
    e1 = 1.1
    e2 = 3
    p0 = [1, 0.3,  1.3, 2.1,  0.2, 0.2, 0] # Fitting params initial guess [amp, center, width, offset]
    bnds = ((0.5, 0.1, 1.0, 1.5, 0.1, 0.1, 0), (1.5, 0.7, 1.5, 2.3, 0.9, 0.9, .3))
    
    centers_CBM = np.zeros(len(res.Delay))
    centers_EX = np.zeros(len(res.Delay))
    Ebs = np.zeros(len(res.Delay))
    
    p_fits_excited = np.zeros((len(res.Delay),7))
    p_err_excited = np.zeros((len(res.Delay),7))
    p_err_eb = np.zeros((len(res.Delay)))
    
    n = len(res.Delay.values)
    for t in range(n):
    
        kx_frame = res.loc[{"Delay":slice(res.Delay.values[t]-delay_int/2, res.Delay.values[t]+delay_int/2)}].mean(dim="Delay")
        kx_frame = kx_frame - res.loc[{"Delay":slice(-1000,-150)}].mean(dim="Delay")
    
        kx_edc_i = kx_frame.loc[{"Angle":slice(-12,12)}].sum(dim="Angle")
        kx_edc_i = kx_edc_i/np.max(kx_edc_i.loc[{"Energy":slice(0.8,3)}])
        
        try:
            popt, pcov = curve_fit(two_gaussians, kx_edc_i.loc[{"Energy":slice(e1,e2)}].Energy.values, kx_edc_i.loc[{"Energy":slice(e1,e2)}].values, p0, method=None, bounds = bnds)
        except ValueError:
            print('Oops!')
            popt = [0,0,0,0]
       
        centers_EX[t] = popt[2]
        centers_CBM[t] = popt[3]
        Eb = round(popt[3] - popt[2],3)
        Ebs[t] = Eb
        perr = np.sqrt(np.diag(pcov))
        p_fits_excited[t,:] = popt
        
        p_err_excited[t,:] = perr 
        p_err_eb[t] = np.sqrt(perr[3]**2+perr[2]**2)
        
    return centers_EX, centers_CBM, Ebs, p_fits_excited, p_err_excited, p_err_eb

#%% # Do the Fitting for VBM, EXCITON, AND CBM

res = phoibos.load_data(data_path, scan, scan_info, energy_offset, delay_offset, False)

centers_VBM, p_fits_VBM, p_err_VBM = fit_vbm_dynamics(res, -3, 4)

centers_EX, centers_CBM, Ebs, p_fits_excited, p_err_excited, p_err_eb = fit_ex_cbm_dynamics(res, delay_int)

#%% Plot and Fit EDCs of the VBM

%matplotlib inline

save_figure = False
figure_file_name = 'EDC_phoibos'

#I_res = I.groupby_bins('delay', 50)
#I_res = I_res.rename({"delay_bins":"delay"})
#I_res = I_res/np.max(I_res)
res_n = res/np.max(res)

fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1.5, 1], 'height_ratios':[1]})
fig.set_size_inches(10, 4, forward=False)
ax = ax.flatten()

### Plot EDCs at GAMMA vs time
(kx), k_int = (-3), 6
edc_gamma = res.loc[{"Angle":slice(kx-k_int/2,kx+k_int/2)}].sum(dim=("Angle"))
edc_gamma = edc_gamma/np.max(edc_gamma)

im = edc_gamma.plot.imshow(ax = ax[0], cmap = cmap_LTL, add_colorbar = False)
# cbar_ax = fig.add_axes([.51, 0.275, 0.025, 0.5])
cbar = fig.colorbar(im, ax = ax[0], ticks = [0,1])
cbar.ax.set_yticklabels(['min', 'max'])  # vertically oriented colorbar

ax[0].set_ylim([-1,1])
ax[0].set_xlim([edc_gamma.Delay[0], edc_gamma.Delay[-1]])
#plt.axhline(0, linestyle = 'dashed', color = 'black')
#plt.axvline(0, linestyle = 'dashed', color = 'black')
ax[0].set_xlabel('Delay, fs')
ax[0].set_ylabel('E - E$_{VBM}$, eV')

pts = [-120, 0, 50, 100, 500]
delay_int = 50
colors = ['black', 'red', 'orange', 'purple', 'blue', 'green', 'grey']
n = len(pts)
colors = mpl.cm.inferno(np.linspace(0,.8,n))
                    
for i in range(n):
    edc = res.loc[{"Angle":slice(kx-k_int/2,kx+k_int/2), "Delay":slice(pts[i]-delay_int/2,pts[i]+delay_int/2)}].sum(dim=("Angle","Delay"))
    edc = edc/np.max(edc)
    
    e = edc.plot(ax = ax[1], color = colors[i], label = f"{pts[i]} fs")

#plt.legend(frameon = False)
ax[1].set_xlim([-1.5, 1]) 
#ax[1].set_ylim([0, 1.1])
ax[1].set_xlabel('E - E$_{VBM}$, eV')
ax[1].set_ylabel('Norm. Int.')
#ax[1].axvline(0, color = 'black', linestyle = 'dashed', linewidth = 0.5)
ax[1].legend(frameon=False, loc = 'upper left', fontsize = 11)
#ax[1].set_yscale('log')
#plt.ax[1].gca().set_aspect(2)

# #ax[1].plot(edc_gamma.Energy.values, edc_gamma[:,t].values/edc_gamma.loc[{"Energy":slice(e1,e2)}][:,t].values.max(), color = 'pink')
# ax[1].plot(edc_gamma.Energy.values, gauss_test, linestyle = 'dashed', color = 'grey')
# #plt.axvline(trunc_e, linestyle = 'dashed', color = 'black')
# ax[1].set_xlim([-2,1.5])
# ax[1].set_xlabel('Energy, eV')
# ax[1].set_ylabel('Norm. Int, arb. u.')
# #plt.gca().set_aspect(3)

# # PLOT VBM SHIFT DYNAMICS
# #fig = plt.figure()
# ax[2].plot(edc_gamma.Delay.values, 1000*(p_fits_VBM[:,1] - p_fits_VBM[0:10,1].mean()), color = 'grey')
# ax[2].set_xlim([edc_gamma.Delay.values[1], edc_gamma.Delay.values[-1]])
# ax[2].set_ylim([-30,20])
# ax[2].set_xlabel('Delay, fs')
# ax[2].set_ylabel('Energy Shift, meV')
# #plt.axhline(0, linestyle = 'dashed', color = 'black')
# #plt.axvline(0, linestyle = 'dashed', color = 'black')

# # PLOT VBM PEAK WIDTH DYNAMICS
# ax2 = ax[2].twinx()
# ax2.plot(edc_gamma.Delay.values, 1000*(p_fits_VBM[:,2] - 0*p_fits_VBM[0:10,2].mean()), color = 'pink')
# #ax2.set_ylim([-75,50])
# ax2.set_ylabel('Energy Width Shift, meV')

fig.tight_layout()

if save_figure is True:
    fig.savefig((figure_file_name +'.svg'), format='svg', dpi = 300)

#%% Plot VBM Fit Results

figure_file_name = 'EDC_phoibos_fits1'
save_figure = True

fig, ax = plt.subplots(1, 2)
fig.set_size_inches(10, 4, forward=False)
ax = ax.flatten()

# VBM FIT TESTS FOR ONE POINT
t = 15
gauss_test = gaussian(edc_gamma.Energy.values, *p_fits_VBM[t,:])
ax[0].plot(edc_gamma.Energy.values, edc_gamma[:,t].values/edc_gamma.loc[{"Energy":slice(-0.2,0.5)}][:,t].values.max(), color = 'black')
ax[0].plot(edc_gamma.Energy.values, gauss_test, linestyle = 'dashed', color = 'grey')
#plt.axvline(trunc_e, linestyle = 'dashed', color = 'black')
ax[0].set_xlim([-1,1])
ax[0].set_xlabel('E - E$_{VBM}$, eV')
ax[0].set_ylabel('Norm. Int.')
ax[0].axvline(e1, linestyle = 'dashed', color = 'pink')
ax[0].axvline(e2, linestyle = 'dashed', color = 'pink')

# PLOT VBM SHIFT DYNAMICS

t = 39 # Show only after 50 (?) fs
y_vb, y_vb_err = 1000*(p_fits_VBM[:,1] - p_fits_VBM[0:10,1].mean()), 1000*p_err_VBM[:,0]
y_vb_w, y_vb_w_err = 1000*(p_fits_VBM[:,2]),  1000*p_err_VBM[:,1]

ax[1].plot(res.Delay.values, y_vb, color = 'navy', label = '$\Delta E_{VBM}$')
ax[1].fill_between(res.Delay.values, y_vb - y_vb_err, y_vb + y_vb_err, color = 'navy', alpha = 0.5)

ax[1].set_xlim([edc_gamma.Delay.values[1], edc_gamma.Delay.values[-1]])
ax[1].set_ylim([-40,75])
ax[1].set_xlabel('Delay, fs')
ax[1].set_ylabel('$\Delta E_{VBM}$, meV', color = 'navy')
ax[1].legend(frameon=False)

#PLOT VBM PEAK WIDTH DYNAMICS
ax2 = ax[1].twinx()
ax2.plot(res.Delay.values, y_vb_w, color = 'maroon', label = '$\sigma_{VBM}$')
ax2.fill_between(res.Delay.values, y_vb_w - y_vb_w_err, y_vb_w + y_vb_w_err, color = 'maroon', alpha = 0.5)
ax2.set_ylim([160,280])
ax2.legend(frameon=False, loc = 'upper left')
ax2.set_ylabel('${VBM}$ Peak Width, meV', color = 'maroon')

fig.tight_layout()

if save_figure is True:
    fig.savefig((figure_file_name +'.svg'), format='svg')

#%% TEST: CBM EDC Fitting: Extract Binding Energy

E_trace, E_int = [1.35, 2.1], .12 # Energies for Plotting Time Traces ; 1st Energy for MM
delay, delay_int = 50, 50

kx_frame = res.loc[{"Delay":slice(delay-delay_int/2, delay+delay_int/2)}].mean(dim="Delay")
kx_frame = kx_frame - res.loc[{"Delay":slice(-1000,-150)}].mean(dim="Delay")

kx_edc = kx_frame.loc[{"Angle":slice(-12,12)}].sum(dim="Angle")
kx_edc = kx_edc/np.max(kx_edc.loc[{"Energy":slice(0.8,3)}])

##### X and CBM ####
e1 = 1.15
e2 = 3
p0 = [1, 0.3,  1.35, 2.1,  0.2, 0.2, 0] # Fitting params initial guess [amp, center, width, offset]
bnds = ((0.5, 0.1, 1.0, 1.5, 0.1, 0.1, 0), (1.5, 0.7, 1.5, 2.3, 0.9, 0.9, .3))

popt_2, _ = curve_fit(two_gaussians, kx_edc.loc[{"Energy":slice(e1,e2)}].Energy.values, kx_edc.loc[{"Energy":slice(e1,e2)}].values, p0, method=None, bounds = bnds)
g, g1, g2, offset = two_gaussians_report(kx_edc.loc[{"Energy":slice(0,3)}].Energy.values, *popt_2)
Eb = round(popt_2[3] - popt_2[2],2)

fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1], 'height_ratios':[1]})
fig.set_size_inches(10, 4, forward=False)
ax = ax.flatten()

kx_edc.plot(ax=ax[0], color = 'black')
ax[0].plot(kx_edc.loc[{"Energy":slice(0,3)}].Energy.values, g1, color='black',linestyle = 'dashed')
ax[0].plot(kx_edc.loc[{"Energy":slice(0,3)}].Energy.values, g2, color='red',linestyle = 'dashed')
ax[0].plot(kx_edc.loc[{"Energy":slice(0,3)}].Energy.values, g, color='grey',linestyle = 'solid')
ax[0].set_title(f"$\Delta t = {delay}$ fs : $E_B = {1000*Eb}$ meV")
ax[0].set_xlim(0.5,3)
ax[0].set_ylim(0, 1.1)

delay, delay_int = 250, 50

kx_frame = res.loc[{"Delay":slice(delay-delay_int/2, delay+delay_int/2)}].mean(dim="Delay")
kx_frame = kx_frame - res.loc[{"Delay":slice(-1000,-150)}].mean(dim="Delay")

kx_edc = kx_frame.loc[{"Angle":slice(-12,12)}].sum(dim="Angle")
kx_edc = kx_edc/np.max(kx_edc.loc[{"Energy":slice(0.8,3)}])

##### X and CBM ####
e1 = 1.15
e2 = 3
p0 = [1, 0.3,  1.35, 2.1,  0.2, 0.2, 0] # Fitting params initial guess [amp, center, width, offset]
bnds = ((0.5, 0.1, 1.0, 1.5, 0.1, 0.1, 0), (1.5, 0.7, 1.5, 2.3, 0.9, 0.9, .3))

popt_2, _ = curve_fit(two_gaussians, kx_edc.loc[{"Energy":slice(e1,e2)}].Energy.values, kx_edc.loc[{"Energy":slice(e1,e2)}].values, p0, method=None, bounds = bnds)
g, g1, g2, offset = two_gaussians_report(kx_edc.loc[{"Energy":slice(0,3)}].Energy.values, *popt_2)
Eb = round(popt_2[3] - popt_2[2],2)

kx_edc.plot(ax=ax[1], color = 'black')
ax[1].plot(kx_edc.loc[{"Energy":slice(0,3)}].Energy.values, g1, color='black',linestyle = 'dashed')
ax[1].plot(kx_edc.loc[{"Energy":slice(0,3)}].Energy.values, g2, color='red',linestyle = 'dashed')
ax[1].plot(kx_edc.loc[{"Energy":slice(0,3)}].Energy.values, g, color='grey',linestyle = 'solid')
ax[1].set_title(f"$\Delta t = {delay}$ fs : $E_B = {1000*Eb}$ meV")
ax[1].set_xlim(0.5,3)
ax[1].set_ylim(0, 1.1)

#%% Plot Excited State EDC Fits and Binding Energy

figure_file_name = 'EDC_fits_phoibos_excted2'
save_figure = False

fig, ax = plt.subplots(1, 3,  gridspec_kw={'width_ratios': [1, 1.25, 1.25], 'height_ratios':[1]})
fig.set_size_inches(13, 4, forward=False)
ax = ax.flatten()

kx_edc.plot(ax=ax[0], color = 'black')
ax[0].plot(kx_edc.loc[{"Energy":slice(0,3)}].Energy.values, g1, color='black',linestyle = 'dashed')
ax[0].plot(kx_edc.loc[{"Energy":slice(0,3)}].Energy.values, g2, color='red',linestyle = 'dashed')
ax[0].plot(kx_edc.loc[{"Energy":slice(0,3)}].Energy.values, g, color='grey',linestyle = 'solid')
ax[0].set_title(f"$\Delta t = {delay}$ fs : $E_B = {1000*Eb}$ meV")
ax[0].set_xlim(0.5,3)
ax[0].set_ylim(0, 1.1)
ax[0].set_ylabel('Norm. Int.')
ax[0].set_xlabel('$E - E_{VBM}$, eV', color = 'black')

# PLOT CBM and EX SHIFT DYNAMICS
#fig = plt.figure()
t = 11 #Show only after 50 (?) fs
tt = 11
y_ex, y_ex_err = 1*(centers_EX[t:] - 0*centers_EX[-12].mean()), 1*p_err_excited[t:,2]
y_cb, y_cb_err = 1*(centers_CBM[tt:]- 0*centers_CBM[-12].mean()),  1*p_err_excited[tt:,3]

ax[1].plot(res.Delay.values[t:], y_ex, color = 'black', label = 'EX')
ax[1].fill_between(res.Delay.values[t:], y_ex - y_ex_err, y_ex + y_ex_err, color = 'grey', alpha = 0.5)
ax[1].set_xlim([0, edc_gamma.Delay.values[-1]])
#ax[1].set_ylim([1.1,2.3])
ax[1].set_ylim([1.15, 1.4])
ax[1].set_xlabel('Delay, fs')

ax2 = ax[1].twinx()
ax2.plot(res.Delay.values[tt:], y_cb, color = 'red', label = 'CBM')
ax2.fill_between(res.Delay.values[tt:], y_cb - y_cb_err, y_cb + y_cb_err, color = 'pink', alpha = 0.5)
ax2.set_ylim([1.95,2.2])
#ax[1].Energyrrorbar(I.Delay.values[t:], 1*(centers_EX[t:]), yerr = p_err_excited[t:,2], marker = 'o', color = 'black', label = 'EX')
#ax[1].Energyrrorbar(I.Delay.values[t:], 1*(centers_CBM[t:]), yerr = p_err_excited[t:,3], marker = 'o', color = 'red', label = 'CBM')
#ax[1].set_ylim([1.1,2.3])
ax[1].set_ylabel('$E_{EX}$, eV', color = 'black')
ax2.set_ylabel('$E_{CBM}$, eV', color = 'red')
ax[1].set_title(f"From {round(res.Delay.values[t])} fs")
ax[1].legend(frameon=False, loc = 'upper right')
ax2.legend(frameon=False, loc = 'upper left' )

ax[2].plot(res.Delay.values[t:], 1000*Ebs[t:], color = 'purple', label = '$E_{B}$')
ax[2].fill_between(res.Delay.values[t:], 1000*Ebs[t:] - 1000*p_err_eb[t:], 1000*Ebs[t:] + 1000*p_err_eb[t:], color = 'violet', alpha = 0.5)
ax[2].set_xlim([0, edc_gamma.Delay.values[-1]])
ax[2].set_ylim([625,825])
ax[2].set_xlabel('Delay, fs')
ax[2].set_ylabel('$E_{B}$, meV', color = 'black')
ax[2].legend(frameon=False)

# # PLOT VBM PEAK WIDTH DYNAMICS
# ax2 = ax[1].twinx()
# ax2.plot(edc_gamma.Delay.values, 1000*(p_fits_VBM[:,2] - 0*p_fits_VBM[0:10,2].mean()), color = 'maroon')
# #ax2.set_ylim([-75,50])
# ax2.set_ylabel('${VBM}$ Peak Width, meV', color = 'maroon')

fig.tight_layout()

if save_figure is True:
    fig.savefig((figure_file_name +'.svg'), format='svg')


#%%

def plot_band_dynamics(ax):

    M, F = 0, 0
    t = 1
    tt = 1
    
    y_vb, y_vb_err = 1000*(p_fits_VBM[:,1] - 1*p_fits_VBM[0:7,1].mean())+F*offset[i], 1000*(p_err_VBM[:,0])
    y_vb_w, y_vb_w_err = 1000*(p_fits_VBM[:,2]),  1000*p_err_VBM[:,1]
    
    y_ex, y_ex_err = 1*(centers_EX[t:] - M*1*centers_EX[-4:].mean())+F*0.01*offset[i], 1*p_err_excited[t:,2]
    y_cb, y_cb_err = 1*(centers_CBM[tt:] - M*1*centers_CBM[-4:].mean())+F*0.01*offset[i],  1*p_err_excited[tt:,3]
    
    y_eb =  1000*Ebs[:] - M*1000*Ebs[-4:].mean() + F*7.5*offset[i]
    y_eb_err = 1000*p_err_eb[:]
    
    #########################
    # PLOT VBM SHIFT DYNAMICS
    colors = ['black', 'blue', 'purple', 'green', 'orange', 'red']
    #colors = mpl.cm.jet(np.linspace(0,.9,len(scans)))
    ax[0].plot(res.Delay.values, y_vb, color = colors[i], label = '$\Delta E_{VBM}$')
    ax[0].fill_between(res.Delay.values, y_vb - y_vb_err, y_vb + y_vb_err, color = colors[i], alpha = 0.5)
    #ax[0].axhline(offset[i], color = 'grey', linestyle = 'dashed')
    ax[0].set_xlim([-20, edc_gamma.Delay.values[-1]])
    ax[0].set_ylim([-30,20])
    ax[0].set_xlabel('Delay, fs')
    ax[0].set_ylabel('$\Delta E_{VBM}$, meV', color = 'navy')
    #ax[0].legend(frameon=False)
    
    # PLOT CBM and EX SHIFT DYNAMICS
    #colors = mpl.cm.jet(np.linspace(0,.9,len(scans)))
    ax[1].plot(res.Delay.values[t:], y_ex, color = colors[i], label = 'EX')
    ax[1].fill_between(res.Delay.values[t:], y_ex - y_ex_err, y_ex + y_ex_err, color = colors[i], alpha = 0.5)
    #ax[1].axhline(0.01*offset[i], color = 'grey', linestyle = 'dashed')
    ax[1].set_xlim([-20, edc_gamma.Delay.values[-1]])
    #ax[1].set_ylim([1.1,2.3])
    ax[1].set_ylim([1.225,1.425])
    ax[1].set_xlabel('Delay, fs')
    ax[1].set_ylabel('$E_{EX}$, eV', color = 'black')

    #colors = mpl.cm.jet(np.linspace(0,.9,len(scans)))
#    ax2 = ax[1].twinx()
    ax[2].plot(res.Delay.values[tt:], y_cb, color = colors[i], label = 'CBM')
    ax[2].fill_between(res.Delay.values[tt:], y_cb - y_cb_err, y_cb + y_cb_err, color = colors[i], alpha = 0.5)
    #ax[2].axhline(0.01*offset[i], color = 'grey', linestyle = 'dashed')
    ax[2].set_ylim([1.95,2.2])
    ax[2].set_xlim([-20, edc_gamma.Delay.values[-1]])
    ax[2].set_ylabel('$E_{CBM}$, eV', color = 'red')
    #ax[1].set_title(f"From {round(res.Delay.values[t])} fs")
    #ax[1].legend(frameon=False, loc = 'upper right')
    #ax2.legend(frameon=False, loc = 'upper left' )
    
   # colors = mpl.cm.jet(np.linspace(0,.9,len(scans)))
    ax[3].plot(res.Delay.values[:], y_eb, color = colors[i], label = '$E_{B}$')
    ax[3].fill_between(res.Delay.values[:], y_eb - y_eb_err, y_eb + y_eb_err, color = colors[i], alpha = 0.5)
    #ax[3].axhline(7.5*offset[i], color = 'grey', linestyle = 'dashed')
    ax[3].set_xlim([-20, edc_gamma.Delay.values[-1]])
    ax[3].set_ylim([600,900])
    ax[3].set_xlabel('Delay, fs')
    ax[3].set_ylabel('$E_{B}$, meV', color = 'black')
    #ax[2].legend(frameon=False)
#    plt.show()

#%% Plot Band Dynamics

scans = [9219, 9217, 9218, 9216, 9220, 9228]
offset = np.linspace(0,100,6)
energy_offset = + 19.72
delay_offset = -80
i = 0
(kx), k_int = (-3), 4

peaks_dynamics = np.zeros((6,4,76))

save_figure = True
figure_file_name = 'phoibos_power_fits2'

fig, ax = plt.subplots(2, 2)
fig.set_size_inches(12, 8, forward=False)
ax = ax.flatten()

for scan_i in scans:

    res = phoibos.load_data(data_path, scan_i, scan_info, energy_offset, delay_offset, False)
    
    centers_VBM, p_fits_VBM, p_err_VBM = fit_vbm_dynamics(res, kx, 4)
    centers_EX, centers_CBM, Ebs, p_fits_excited, p_err_excited, p_err_eb = fit_ex_cbm_dynamics(res, delay_int)
    
    # peaks_dynamics[i,0,:] = centers_VBM
    # peaks_dynamics[i,1,:] = centers_EX
    # peaks_dynamics[i,2,:] = centers_CBM
    # peaks_dynamics[i,3,:] = Ebs
    
    plot_band_dynamics(ax)

    i += 1

fig.tight_layout()

if save_figure is True:
    fig.savefig((figure_file_name +'.svg'), format='svg')

#%%

def fit_vbm_int(res, k, k_int):
    e1 = -.2
    e2 = 0.6
    p0 = [1, 0, .2, 0] # Fitting params initial guess [amp, center, width, offset]
    bnds = ((0.5, -1, 0.0, 0), (1.5, 0.5, 1, .5))
    
    (kx), k_int = k, k_int
    edc_gamma = res.loc[{"Angle":slice(kx-k_int/2,kx+k_int/2), "Delay":slice(0,3000)}].sum(dim=("Angle", "Delay"))
    edc_gamma = edc_gamma/np.max(edc_gamma)
    
    edc_i = edc_gamma.loc[{"Energy":slice(e1,e2)}].values
    edc_i = edc_i/np.max(edc_i)
    
    try:
        popt, pcov = curve_fit(gaussian, edc_gamma.loc[{"Energy":slice(e1,e2)}].Energy.values, edc_i, p0, method=None, bounds = bnds)
    except ValueError:
        print('oops')
        popt = [0,0,0,0]
        
    centers_VBM_i = popt[1]
    p_fits_VBM_i = popt
    perr = np.sqrt(np.diag(pcov))
    p_err_VBM_i = perr[1:2+1]
        
    return centers_VBM_i, p_fits_VBM_i, p_err_VBM_i

##### CBM AND EXCITON #####

def fit_ex_cbm_int(res):
    delay_int = 50
    e1 = 1.1
    e2 = 3
    p0 = [1, 0.3,  1.3, 2.1,  0.2, 0.2, 0] # Fitting params initial guess [amp, center, width, offset]
    bnds = ((0.5, 0.1, 1.0, 1.5, 0.1, 0.1, 0), (1.5, 0.7, 1.5, 2.3, 0.9, 0.9, .3))
    
    centers_CBM = np.zeros(len(res.Delay))
    centers_EX = np.zeros(len(res.Delay))
    Ebs = np.zeros(len(res.Delay))
    
    p_fits_excited = np.zeros((len(res.Delay),7))
    p_err_excited = np.zeros((len(res.Delay),7))
    p_err_eb = np.zeros((len(res.Delay)))

    #kx_frame = res.loc[{"Delay":slice(res.Delay.values[t]-delay_int/2, res.Delay.values[t]+delay_int/2)}].mean(dim="Delay")
    kx_frame = res - res.loc[{"Delay":slice(-1000,-200)}].mean(dim="Delay")

    kx_frame = kx_frame.loc[{"Delay":slice(200,300)}].mean(dim="Delay")
    kx_edc_i = kx_frame.loc[{"Angle":slice(-12,12)}].sum(dim="Angle")
    kx_edc_i = kx_edc_i/np.max(kx_edc_i.loc[{"Energy":slice(0.8,3)}])
    
    try:
        popt, pcov = curve_fit(two_gaussians, kx_edc_i.loc[{"Energy":slice(e1,e2)}].Energy.values, kx_edc_i.loc[{"Energy":slice(e1,e2)}].values, p0, method=None, bounds = bnds)
    except ValueError:
        print('Oops!')
        popt = [0,0,0,0]
   
    centers_EX_i = popt[2]
    centers_CBM_i = popt[3]
    Eb = round(popt[3] - popt[2],3)
    Ebs_i = Eb
    perr = np.sqrt(np.diag(pcov))
    p_fits_excited_i = popt
    
    p_err_excited_i = perr[2:3+1] 
    p_err_eb_i = np.sqrt(perr[3]**2+perr[2]**2)
        
    return centers_EX_i, centers_CBM_i, Ebs_i, p_fits_excited_i, p_err_excited_i, p_err_eb_i


#%%
save_figure = False
figure_file_name = 'phoibos_power_fits2'

scans = [9219, 9217, 9218, 9216, 9220, 9228]
offset = np.linspace(0,100,6)
i = 0
(kx), k_int = (-3), 4

centers_VBM, p_fits_VBM, p_err_VBM = [], [], []
centers_EX, centers_CBM, Ebs, p_fits_excited, p_err_excited, p_err_eb = [], [], [], [], [], []

for scan_i in scans:
    res = phoibos.load_data(data_path, scan_i, scan_info, energy_offset, delay_offset, False)
    
    centers_VBM_i, p_fits_VBM_i, p_err_VBM_i = fit_vbm_int(res, kx, 4)
    centers_EX_i, centers_CBM_i, Ebs_i, p_fits_excited_i, p_err_excited_i, p_err_eb_i = fit_ex_cbm_int(res)
    
    i += 1
    
    centers_VBM.append(centers_VBM_i)
    p_fits_VBM.append(p_fits_VBM_i)
    p_err_VBM.append(p_err_VBM_i)
    
    centers_EX.append(centers_EX_i)
    centers_CBM.append(centers_CBM_i)
    Ebs.append(Ebs_i)
    p_fits_excited.append(p_fits_excited_i)
    p_err_excited.append(p_err_excited_i)
    p_err_eb.append(p_err_eb_i)
    
p_err_eb = np.asarray(p_err_eb)
p_err_excited = np.asarray(p_err_excited)
p_err_VBM = np.asarray(p_err_VBM)

#%% Plot Peak Energies and Eb Change

fig, ax = plt.subplots(1, 2)
fig.set_size_inches(12, 4, forward=False)
ax = ax.flatten()

y1, y_vb_err = 1000*(centers_VBM - centers_VBM[0]), 1000*p_err_VBM[:,0]
y2, y_ex_err = 1000*(centers_EX - centers_EX[0]), 1000*p_err_excited[:,0]
y3, y_cb_err = 1000*(centers_CBM - centers_CBM[0]), 1000*p_err_excited[:,1]
y4, y_eb_err = 1000*(Ebs - Ebs[0]), 1000*p_err_eb

colors = ['grey', 'black', 'red']
i = 0
#ax[0].plot(y1, color = 'grey')
ax[0].errorbar(x = range(0,6), y = y1, yerr = y_vb_err, marker = 'o', color = 'grey', label = 'VBM')
ax[0].errorbar(x = range(0,6), y = y2, yerr = y_ex_err, marker = 'o', color = 'black', label = 'ex')
ax[0].errorbar(x = range(0,6), y = y3, yerr = y_cb_err, marker = 'o', color = 'red', label = 'CBM')
ax[0].axhline(0, color = 'grey', linestyle = 'dashed')
#ax[0].fill_between(y1 - y_vb_err, y1 + y_vb_err, color = colors[i], alpha = 0.5)

ax[1].errorbar(x = range(0,6), y = y4, yerr = y_eb_err, marker = 'o', color = 'purple', label = '$E_{b}$')
ax[1].axhline(0, color = 'grey', linestyle = 'dashed')

fig.tight_layout()

if save_figure is True:
    fig.savefig((figure_file_name +'.svg'), format='svg')
