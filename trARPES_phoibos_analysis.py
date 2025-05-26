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
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.draw import disk
from scipy.optimize import curve_fit
import csv
from Loader import DataLoader
import xarray as xr

import phoibos

#%%

data_path = '/Users/lawsonlloyd/Desktop/Data/phoibos'
data_path = 'R:\Lawson\Data\phoibos'

data_path_info = '/Users/lawsonlloyd//GitHub/mpes-analysis'
data_path_info = 'R:\Lawson\mpes-analysis'

energy_offset, delay_offset, force_offset = 19.62,  0, False

filename = '2024 Bulk CrSBr Phoibos.csv'
scan_info = phoibos.get_scan_info(data_path_info, filename, {})
 
#%% PLOT Fluence Delay TRACES All Together: 915 nm

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
E, E_int = [1.35, 2.1], 0.1
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
    
    res = phoibos.load_data(data_path, scan_i, scan_info, energy_offset, delay_offset, force_offset)
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
    
    res = phoibos.load_data(data_path, scan_i, scan_info, energy_offset, delay_offset, force_offset)
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
    
    res = phoibos.load_data(data_path, scan_i, scan_info, energy_offset, delay_offset, force_offset)
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
    
    res = phoibos.load_data(data_path, scan_i, scan_info, energy_offset, delay_offset, force_offset)
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

custom_colors = ['lightsteelblue', 'royalblue', 'mediumblue'] #colors for plotting the traces

i = 0
for scan_i in scans:
    
    res = phoibos.load_data(data_path, scan_i, scan_info, energy_offset, delay_offset, force_offset)
    trace_1 = phoibos.get_time_trace(res, E[0], E_int, k, k_int, subtract_neg, norm_trace)
    trace_2 = phoibos.get_time_trace(res, E[1], E_int, k, k_int, subtract_neg, norm_trace)
    trace_2 = trace_2/np.max(trace_2)
    trace_1 = trace_1/np.max(trace_1)

    t1 = trace_1.plot(ax = axx[0], color = custom_colors[i], linewidth = 3)
    t2 = trace_2.plot(ax = axx[1], color = custom_colors[i], linewidth = 3, label = f'{fluence[i]} mJ / cm$^{{2}}$')
    
    #test = exp_rise_monoexp_decay(trace_1.Delay.values, 1.02, 258, 6700)
    #axx[0].plot(trace_1.Delay.values, test, color = 'maroon')

    #test2 = exp_rise_monoexp_decay(trace_1.Delay.values, .8, 400, 6000)
    #test2 = exciton_model(trace_1.Delay.values, 1., 450, 8000)
    #axx[0].plot(trace_1.Delay.values, 0.85*test2/np.max(test2), color = 'green', linestyle = 'dashed')
    
    #test3 = exp_rise_biexp_decay(trace_1.Delay.values, 1, 350, .92, 240, 2500)
    #test4= exp_rise_biexp_decay(trace_1.Delay.values, 1, 250, .9, 300, 4000)
    #test5 = (np.exp(-trace_1.Delay.values/400))*(1-np.exp(-trace_1.Delay.values/300))
    
    #axx[1].plot(trace_1.Delay.values, test3/np.max(test3), color = 'pink')
    #axx[1].plot(trace_1.Delay.values, test4/np.max(test4), color = 'red', linestyle= 'dashed')
    #axx[1].plot(trace_1.Delay.values, test5/np.max(test5), color = 'green', linestyle= 'dashed')

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
    
    res = phoibos.load_data(data_path, scan_i, scan_info, energy_offset, delay_offset, force_offset)
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


#%%

plt.plot(fluence, intensity, '-o', color = 'black')
plt.plot(fluence, .01+(1.025e8)*fluence, color = 'pink', linestyle = 'dashed')
plt.ylim(0,2.75e8)
plt.xlim(0,3)

####

#%% PLOT Fluence Delay TRACES All Together

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

#%% Do the EDC Fits: Functions

def fit_vbm_peak(res, k, k_int, delay, delay_int):
    e1 = -.2
    e2 = 0.6
    p0 = [1, 0, .2, 0] # Fitting params initial guess [amp, center, width, offset]
    bnds = ((0.5, -1, 0.0, 0), (1.5, 0.5, 1, .5))
    
    (kx), k_int = k, k_int
    edc_gamma = res.loc[{"Angle":slice(kx-k_int/2,kx+k_int/2), "Delay":slice(delay-delay_int/2,delay+delay_int/2)}].mean(dim=("Angle", "Delay"))
    edc_gamma = edc_gamma/np.max(edc_gamma)
            
    try:
        popt, pcov = curve_fit(gaussian,  edc_gamma.loc[{"Energy":slice(e1,e2)}].Energy.values, edc_gamma.loc[{"Energy":slice(e1,e2)}].values, p0, method=None, bounds = bnds)
    except ValueError:
        print('oops')
        popt = [-1,0,0,0]
        pcov = [-1,0,0,0]
        
    perr = np.sqrt(np.diag(pcov))
    
    return popt, perr

def fit_ex_cbm_dynamics(res, delay, delay_int):
    e1 = 1.1
    e2 = 3
    p0 = [1, 0.3,  1.3, 2.1,  0.2, 0.2, 0] # Fitting params initial guess [amp, center, width, offset]
    bnds = ((0.5, 0.1, 1.0, 1.5, 0.1, 0.1, 0), (1.5, 0.7, 1.5, 2.3, 0.9, 0.9, .3))
    
    res_diff = res - res.loc[{"Delay":slice(-1000,-200)}].mean(dim="Delay")
    res_diff = res_diff.loc[{"Delay":slice(delay-delay_int/2, delay+delay_int/2)}].mean(dim="Delay")

    edc_i = res_diff.loc[{"Angle":slice(-12,12)}].mean(dim="Angle")
    edc_i = edc_i/np.max(edc_i.loc[{"Energy":slice(1,3)}])
    
    try:
        popt, pcov = curve_fit(two_gaussians, edc_i.loc[{"Energy":slice(e1,e2)}].Energy.values, edc_i.loc[{"Energy":slice(e1,e2)}].values, p0, method=None, bounds = bnds)
    except ValueError:
        print('Oops!')
        popt = [0,0,0,0]
    
    perr = np.sqrt(np.diag(pcov))
    
    return popt, perr

#%% # Do the Fitting for VBM, EXCITON, AND CBM

scan = 9220 

# Load the Data
res = phoibos.load_data(data_path, scan, scan_info, energy_offset, delay_offset, force_offset)

# Get the Dynamic Peak Fits Fits
centers_VBM = np.zeros(len(res.Delay))
p_fits_VBM = np.zeros((len(res.Delay),4))
p_err_VBM = np.zeros((len(res.Delay),2))

centers_VBM[t] = popt[1]
p_fits_VBM[t,:] = popt
perr = np.sqrt(np.diag(pcov))
p_err_VBM[t,:] = perr[1:2+1]
#centers_VBM, p_fits_VBM, p_err_VBM = 

popt, perr = fit_vbm_peak(res, -3, 4, 500, 200)

centers_CBM = np.zeros(len(res.Delay))
centers_EX = np.zeros(len(res.Delay))
Ebs = np.zeros(len(res.Delay))

p_fits_excited = np.zeros((len(res.Delay),7))
p_err_excited = np.zeros((len(res.Delay),7))
p_err_eb = np.zeros((len(res.Delay)))
    n = len(res.Delay.values)
    for t in range(n):
    
        centers_EX[t] = popt[2]
        centers_CBM[t] = popt[3]
        Eb = round(popt[3] - popt[2],3)
        Ebs[t] = Eb
        perr = np.sqrt(np.diag(pcov))
        p_fits_excited[t,:] = popt
        
        p_err_excited[t,:] = perr 
        p_err_eb[t] = np.sqrt(perr[3]**2+perr[2]**2)
    
centers_EX, centers_CBM, Ebs, p_fits_excited, p_err_excited, p_err_eb 

popt, perr = fit_ex_cbm_peaks(res, delay, delay_int)

#%% Plot and Fit EDCs of the VBM

%matplotlib inline

save_figure = False
figure_file_name = 'EDC_phoibos'

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

delay, delay_int = 500, 1000

res = phoibos.load_data(data_path, scan, scan_info, energy_offset, delay_offset, False)

kx_frame = res.loc[{"Delay":slice(delay-delay_int/2, delay+delay_int/2)}].mean(dim="Delay")
kx_frame = kx_frame - res.loc[{"Delay":slice(-1000,-200)}].mean(dim="Delay")

kx_edc = kx_frame.loc[{"Angle":slice(-12,12)}].sum(dim="Angle")
kx_edc = kx_edc/np.max(kx_edc.loc[{"Energy":slice(1,3)}])

##### X and CBM ####
e1 = 1.15
e2 = 2.8
p0 = [1, 0.3,  1.35, 2.1,  0.2, 0.2, 0] # Fitting params initial guess [amp, center, width, offset]
bnds = ((0.5, 0.1, 1.0, 1.5, 0.1, 0.1, 0), (1.5, 0.7, 1.5, 2.3, 0.9, 0.9, .3))

popt_2, _ = curve_fit(two_gaussians, kx_edc.loc[{"Energy":slice(e1,e2)}].Energy.values, kx_edc.loc[{"Energy":slice(e1,e2)}].values, p0, method=None, bounds = bnds)
g, g1, g2, offset = two_gaussians_report(kx_edc.loc[{"Energy":slice(0,3)}].Energy.values, *popt_2)
Eb = round(popt_2[3] - popt_2[2],2)

fig, ax = plt.subplots()
ax = np.ravel(ax)
fig.set_size_inches(6, 4, forward=False)
#ax = ax.flatten()

kx_edc.plot(ax=ax[0], color = 'black')
ax[0].plot(kx_edc.loc[{"Energy":slice(0,3)}].Energy.values, g1, color='black',linestyle = 'dashed')
ax[0].plot(kx_edc.loc[{"Energy":slice(0,3)}].Energy.values, g2, color='red',linestyle = 'dashed')
ax[0].plot(kx_edc.loc[{"Energy":slice(0,3)}].Energy.values, g, color='grey',linestyle = 'solid')
ax[0].set_title(f"$E_B = {1000*Eb}$ meV")
ax[0].set_xlim(0.5,3)
ax[0].set_ylim(0, 1.1)
ax[0].text(1.6, 0.95, fr"$\Delta$t = {delay} $\pm$ {delay_int} fs", fontsize = 18, fontweight = 'regular')

#%% Plot Excited State EDC Fits and Binding Energy

figure_file_name = 'EDC_fits_phoibos_excted2'
save_figure = False

delay, delay_int = 50, 50

res = phoibos.load_data(data_path, scan, scan_info, energy_offset, delay_offset, False)

kx_frame = res.loc[{"Delay":slice(delay-delay_int/2, delay+delay_int/2)}].mean(dim="Delay")
kx_frame = kx_frame - res.loc[{"Delay":slice(-1000,-200)}].mean(dim="Delay")

kx_edc = kx_frame.loc[{"Angle":slice(-12,12)}].sum(dim="Angle")
kx_edc = kx_edc/np.max(kx_edc.loc[{"Energy":slice(1,3)}])

##### X and CBM ####
e1 = 1.15
e2 = 2.8
p0 = [1, 0.3,  1.35, 2.1,  0.2, 0.2, 0] # Fitting params initial guess [amp, center, width, offset]
bnds = ((0.5, 0.1, 1.0, 1.5, 0.1, 0.1, 0), (1.5, 0.7, 1.5, 2.3, 0.9, 0.9, .3))

popt_2, _ = curve_fit(two_gaussians, kx_edc.loc[{"Energy":slice(e1,e2)}].Energy.values, kx_edc.loc[{"Energy":slice(e1,e2)}].values, p0, method=None, bounds = bnds)
g, g1, g2, offset = two_gaussians_report(kx_edc.loc[{"Energy":slice(0,3)}].Energy.values, *popt_2)
Eb = round(popt_2[3] - popt_2[2],2)

fig, ax = plt.subplots(1, 3,  gridspec_kw={'width_ratios': [1, 1.25, 1.25], 'height_ratios':[1]})
fig.set_size_inches(13, 4, forward=False)
ax = ax.flatten()

kx_edc.plot(ax=ax[0], color = 'black')
ax[0].plot(kx_edc.loc[{"Energy":slice(0,3)}].Energy.values, g1, color='black',linestyle = 'dashed')
ax[0].plot(kx_edc.loc[{"Energy":slice(0,3)}].Energy.values, g2, color='red',linestyle = 'dashed')
ax[0].plot(kx_edc.loc[{"Energy":slice(0,3)}].Energy.values, g, color='grey',linestyle = 'solid')
ax[0].set_title(f"$E_B = {1000*Eb}$ meV")
ax[0].set_xlim(0.5,3)
ax[0].set_ylim(0, 1.1)
ax[0].set_ylabel('Norm. Int.')
ax[0].set_xlabel('$E - E_{VBM}$, eV', color = 'black')
ax[0].text(1.6, 0.95, fr"$\Delta$t = {delay} fs", fontsize = 16, fontweight = 'regular')

# PLOT CBM and EX SHIFT DYNAMICS
t = np.abs(res.Delay.values-0).argmin() # Show only after 0 fs
tt = np.abs(res.Delay.values-100).argmin() #Show Only after 100

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
#ax[1].set_title(f"From {round(res.Delay.values[t])} fs")
ax[1].legend(frameon=False, loc = 'upper right')
ax2.legend(frameon=False, loc = 'upper left' )

Eb_mean = np.mean(1000*Ebs[tt:])
Eb_std = np.std(1000*Ebs[tt:]) #/np.sqrrlen(Ebs[tt:])

ax[2].plot(res.Delay.values[tt:], 1000*Ebs[tt:], color = 'purple', label = '$E_{B}$')
ax[2].fill_between(res.Delay.values[tt:], 1000*Ebs[tt:] - 1000*p_err_eb[tt:], 1000*Ebs[tt:] + 1000*p_err_eb[tt:], color = 'violet', alpha = 0.5)
ax[2].set_xlim([0, edc_gamma.Delay.values[-1]])
ax[2].set_ylim([625,825])
ax[2].set_xlabel('Delay, fs')
ax[2].set_ylabel('$E_{B}$, meV', color = 'black')
ax[2].legend(frameon=False)
ax[2].set_title(f"$E_B = {Eb_mean:.0f} \pm {Eb_std:.0f}$ meV")

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

#%% Plot 4-Panel VB and Excited State Peak Energy Dynamics

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

#%% Fit Peak Shifts as Function of Fluence

save_figure = False
figure_file_name = 'phoibos_power_fits2'

scans = [9219, 9217, 9218, 9216, 9220, 9228]
offset = np.linspace(0,100,6)

(kx), k_int = (-3), 4
delay, delay_int = 125, 250

centers_VBM, p_fits_VBM, p_err_VBM = [], [], []
centers_EX, centers_CBM, Ebs, p_fits_excited, p_err_excited, p_err_eb = [], [], [], [], [], []

i = 0
for scan_i in scans:
    res = phoibos.load_data(data_path, scan_i, scan_info, energy_offset, delay_offset, False)
    
    centers_VBM_i, p_fits_VBM_i, p_err_VBM_i = fit_vbm_int(res, kx, 4)
    centers_EX_i, centers_CBM_i, Ebs_i, p_fits_excited_i, p_err_excited_i, p_err_eb_i = fit_ex_cbm_int(res, delay, delay_int)
    
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

x = range(0,6)
colors = ['grey', 'black', 'red']
i = 0
ax[0].axhline(0, color = 'black', linestyle = 'dashed', linewidth = 2)
ax[1].axhline(0, color = 'black', linestyle = 'dashed', linewidth = 2)

#ax[0].plot(y1, color = 'grey')
#ax[0].errorbar(x = x, y = y1, yerr = y_vb_err, marker = 'o', color = 'grey', label = 'VBM')
ax[0].plot(x, y1, color = 'grey', marker = 'o', markersize = 12)
ax[0].plot(x, y2, color = 'black', marker = 'o', markersize = 12)
ax[0].plot(x, y3, color = 'red', marker = 'o', markersize = 12)
ax[1].plot(x, y4, color = 'purple', marker = 'o', markersize = 12)

ax[0].fill_between(x, y1 - y_vb_err, y1 + y_vb_err, color = 'grey', alpha = 0.5, label = 'VBM')
ax[0].fill_between(x, y2 - y_ex_err, y2 + y_ex_err, color = 'black', alpha = 0.5, label = 'Exciton')
ax[0].fill_between(x, y3 - y_cb_err, y3 + y_cb_err, color = 'crimson', alpha = 0.5, label = 'CBM')
ax[1].fill_between(x, y4 - y_eb_err, y4 + y_eb_err, color = 'purple', alpha = 0.5, label = '$E_{b}$')

#ax[0].errorbar(x = x, y = y2, yerr = y_ex_err, marker = 'o', color = 'black', label = 'ex')
#ax[0].errorbar(x = x, y = y3, yerr = y_cb_err, marker = 'o', color = 'red', label = 'CBM')
#ax[0].fill_between(y1 - y_vb_err, y1 + y_vb_err, color = colors[i], alpha = 0.5)

#ax[1].errorbar(x = x, y = y4, yerr = y_eb_err, marker = 'o', color = 'purple', label = '$E_{b}$')

ax[0].set_xlabel('Fluence', fontsize = 20)
ax[1].set_xlabel('Fluence', fontsize = 20)

ax[0].set_ylabel('$\Delta$E, meV', fontsize = 20)
ax[1].set_ylabel('$\Delta$E, meV', fontsize = 20)

ax[0].legend(frameon=False)
ax[1].legend(frameon=False)

fig.tight_layout()

if save_figure is True:
    fig.savefig((figure_file_name +'.svg'), format='svg')
