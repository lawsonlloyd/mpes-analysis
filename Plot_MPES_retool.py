#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 08:47:53 2024

@author: lawsonlloyd
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as col
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
from skimage.draw import disk
from scipy.optimize import curve_fit
from scipy import signal
from scipy.fft import fft, fftshift
#from lmfit import Parameters, minimize, report_fit
from obspy.imaging.cm import viridis_white
import cmocean
import xarray as xr
from math import nan

from Loader import DataLoader
from main import main
from Manager import DataHandler, FigureHandler, PlotHandler, ValueHandler, SliderManager, EventHandler, CheckButtonManager, ClickButtonManager
import mpes
from mpes import cmap_LTL, cmap_LTL2

#%% Specifiy filename of h5 file in your path.
# Include manual energy and time delay offsets for the axes, if required.

data_path = 'path_to_your_data'
filename = 'your_file_name.h5'

data_path = 'R:\Lawson\Data\metis'
data_path = '/Users/lawsonlloyd/Desktop/Data/metis'

filename, offsets = 'Scan062_binned_200x200x300_CrSBr_RT_Static_rebinned.h5', [0,0]
filename, offsets = 'Scan129_binned_100x100x200x67_CrSBr_XUVPolScan.h5', [-.3, 0]
filename, offsets = 'Scan138_binned_200x200x300_CrSBr_Integrated_XUV_Pol.h5', [0,0]
#filename, offsets = 'Scan177_120K_120x120x115_binned.h5', [0.363, 0]

filename, offsets = 'Scan162_RT_120x120x115x50_binned.h5', [0.8467, -120]
#filename, offsets = 'Scan163_120K_120x120x115x75_binned.h5',  [0.6369, -132]
#filename, offsets = 'Scan188_120K_120x120x115x77_binned.h5', [0.5660, -110]

#%% Load the data and axes information

data_loader = DataLoader(data_path + '//' + filename, offsets)

I = data_loader.load()
I_res = I/np.max(I)

I_diff = I_res - I_res.loc[{"delay":slice(-500,-200)}].mean(dim="delay")
I_diff = I_diff/np.max(I_diff)

#%% This sets the plots to plot in the IDE window

%matplotlib inline
cmap_plot = cmap_LTL

def lorentzian(x, amp_1, mean_1, stddev_1, offset):
    
    b = (x - mean_1)/(stddev_1/2)
    l1 = amp_1/(1+b**2) + offset
    
    return l1
    
def gaussian(x, amp_1, mean_1, stddev_1, offset):
    
    g1 = amp_1 * np.exp(-0.5*((x - mean_1) / stddev_1)**2)+offset
    
    return g1

def two_gaussians(x, amp_1, amp_2, mean_1, mean_2, stddev_1, stddev_2, offset):
    
    g1 = amp_1 * np.exp(-0.5*((x - mean_1) / stddev_1)**2)
    g2 = amp_2 * np.exp(-0.5*((x - mean_2) / stddev_2)**2)
    
    g = g1 + g2 + offset
    return g

def two_gaussians_report(x, amp_1, amp_2, mean_1, mean_2, stddev_1, stddev_2, offset):
    
    g1 = amp_1 * np.exp(-0.5*((x - mean_1) / stddev_1)**2)
    g2 = amp_2 * np.exp(-0.5*((x - mean_2) / stddev_2)**2)
    
    g = g1 + g2 + offset
    return g, g1, g2, offset

def objective(params, x, data):
    
    g1, g2, offset = two_gaussians(x, **params)
    fit = g1+g2+offset
    resid = np.abs(data-fit)**2
    
    return resid

#%% Define E = 0 wrt VBM

fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1], 'height_ratios':[1]})
fig.set_size_inches(10, 4, forward=False)
ax = ax.flatten()

figure_file_name = 'vbm'
save_figure = False

### Plot EDCs at GAMMA vs time

a, b = 3.508, 4.763 # CrSBr values
X, Y = np.pi/a, np.pi/b
x, y = -2*X, 0

(kx, ky), k_int = (x, y), 0.1
delay, delay_int = 0, 1000

edc_gamma = mpes.get_edc(I_res, kx, ky, (k_int, k_int), delay, delay_int)
edc_gamma = edc_gamma/np.max(edc_gamma)

mpes.plot_momentum_maps(
    I_res, E=0, E_int=E_int, delays=0, delay_int=1000,
    fig = fig, ax = ax[0],
    cmap=cmap_LTL, scale=[0, 1],
    fontsize=16, figsize=(8, 6), colorbar=False, panel_labels = False
)
    
mpes.find_E0(edc_gamma, fig, ax)

# cbar_ax = fig.add_axes([.51, 0.275, 0.025, 0.5])
rect = (Rectangle((kx-k_int/2, ky-k_int/2), k_int, k_int, linewidth=1.5,\
                         edgecolor='k', facecolor='None'))
fig.axes[0].add_patch(rect)

if save_figure is True:
    mpes.save_figure(fig, name = f'test', image_format = 'pdf')

#%% Plots EDC Comparisons


edc_pos = (I_res.loc[{"delay":slice(200,1100)}].mean(dim=("kx","ky","delay")))
edc_neg = (I_res.loc[{"delay":slice(-200,-90)}].mean(dim=("kx","ky","delay")))

edc_pos = edc_pos/np.max(edc_neg.loc[{"E":slice(-0.5,0.5)}])
edc_neg = edc_neg/np.max(edc_neg.loc[{"E":slice(-0.5,0.5)}])
edc_diff = edc_pos - edc_neg
edc_diff = edc_diff/np.max(edc_diff.loc[{"E":slice(1,3)}])

edc_pos = np.log(edc_pos)
edc_neg = np.log(edc_neg)

plt.plot(I_res.E, edc_pos , color = 'green', label = 't > 200 fs')
plt.plot(I_res.E, edc_neg, color = 'grey', label = 'Neg Delays') 
plt.plot(I_res.E, edc_diff, color = 'purple', label = 'Neg Delays') 

plt.axvline(1.325, linestyle = 'dashed', color = 'black')
plt.axvline(2.1, linestyle = 'dashed', color = 'crimson')

plt.xlim(-.5,2.75)
plt.ylim(-.2, 1.1)
plt.xlabel('Energy, eV')
plt.ylabel('Int, logscale')
plt.legend(frameon=False)

#a.plot(I_res.E.loc[{"E":slice(0.75,3)}], edc_diff.loc[{"E":slice(0.75,3)}], color = 'purple', label = 'Difference')
#plt.ylim(-.1,1.2)

#%% Define t0 from Exciton Rise

fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1], 'height_ratios':[1]})
fig.set_size_inches(10, 4, forward=False)
ax = ax.flatten()

figure_file_name = 'define_t0'
save_figure = False

### Plot EDCs at GAMMA vs time

(kx, ky), (kx_int, ky_int) = (0, 0), (3.8, 1.8)
#(kx, ky), (kx_int, ky_int) = (0, .7), (1.5, 0.5)
E, E_int = 1.55, 0.1

mpes.plot_momentum_maps(
    I_res, E=E, E_int=E_int, delays=200, delay_int=700,
    fig = fig, ax = ax[0],
    cmap=cmap_LTL, scale=[0, 1],
    fontsize=16, figsize=(8, 6), colorbar=False, panel_labels = False
)
    
rect = (Rectangle((kx-kx_int/2, ky-ky_int/2), kx_int, ky_int, linewidth=1.5,\
                         edgecolor='red', facecolor='None'))
fig.axes[0].add_patch(rect)

trace_ex = mpes.get_time_trace(I_res, E, E_int, (kx, ky), (kx_int, ky_int), True, True, neg_delays)
mpes.find_t0(trace_ex, fig, ax)

if save_figure is True:
    mpes.save_figure(fig, name = f'test', image_format = 'pdf')

#%% Plot Momentum Maps at Constant Energy

E, E_int = [1.35, 1.35, 1.35, 2.15, 2.15, 2.15], 0.2
delays, delay_int = [500, 500], 1000

fig, ax, im = mpes.plot_momentum_maps(
    I_res, E=E, E_int=0.2, delays=delays, delay_int=delay_int,
    cmap=cmap_LTL, scale=[0, 1],
    fontsize=16, figsize=(8, 6), colorbar=True, panel_labels = False
)
     
mpes.save_figure(fig, name = f'test', image_format = 'pdf')

#%% Plot Overview: MM, Delay Traces, k-cut, and Waterfall Panel

fig, axs = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 1.5], 'height_ratios':[1, 1]})
fig.set_size_inches(8, 6, forward=False)
axs = axs.flatten()

# Plot momentum map
mpes.plot_momentum_maps(
    I_res, E=1.3, E_int=0.2, delays=500, delay_int=1000,
    fig = fig, ax = axs[0],
    cmap='BuPu',
    panel_labels=False, fontsize=16,
    nrows=2, figsize=(8, 6)
)

# Plot time traces
E, E_int = [1.35, 2.05], .12 # Energies for Plotting Time Traces ; 1st Energy for MM
(kx, ky), (kx_int, ky_int) = (0, 0), (4, 4) # Central (kx, ky) point and k-integration
norm_trace = True
subtract_neg = True
neg_delays = [-300, -50]

mpes.plot_time_traces(
    I, E, E_int, (kx, ky), (kx_int, ky_int),
    norm_trace, subtract_neg, neg_delays,
    fig = fig, ax = axs[1],
    colors = ['black', 'maroon'],
    fontsize=16
)

# Plot kx-E frame
(kx, ky), (kx_int, ky_int) = (0, 0), (4, 0.5) # Central (kx, ky) point and k-integration
mpes.plot_kx_frame(
    I_res, ky, ky_int, delays=[500], delay_int=1000,
    subtract_neg=subtract_neg, neg_delays=neg_delays,
    fig = fig, ax = axs[2],
    cmap = 'BuPu', scale=[0,1], energy_limits=[1,3]
)

# # Plot ky-E frame
# (kx, ky), (kx_int, ky_int) = (0, 0), (0.5, 0.5) # Central (kx, ky) point and k-integration
# mpes.plot_ky_frame(
#     I_res, ky, ky_int, delays=[500], delay_int=1000,
#     subtract_neg=subtract_neg, neg_delays=neg_delays,
#     fig = fig, ax = axs[2],
#     cmap = 'BuPu', scale=[0,1], energy_limits=[1,3]
# )

# Plot waterfall
(kx, ky), (kx_int, ky_int) = (0, 0), (4, 4) # Central (kx, ky) point and k-integration
mpes.plot_waterfall(
    I_res, kx, kx_int, ky, ky_int,
    fig = fig, ax = axs[3],
    cmap=cmap_LTL, scale=[0,1], energy_limits=[1,3]
)

rect = (Rectangle((kx-kx_int/2, ky-ky_int/2), kx_int, ky_int, linewidth=.5,\
                         edgecolor='fuchsia', facecolor='fuchsia', alpha = 0.3))
if kx_int < 4:
    axs[0].add_patch(rect) #Add rectangle to plot
    
colors = ['black', 'maroon'],

for i in np.arange(len(E)):
    rect2 = (Rectangle((kx-kx_int/2, E[i]-E_int/2), kx_int, E_int , linewidth=.5,\
                         edgecolor=colors[i], facecolor=colors[i], alpha = 0.5))
    rect3 = (Rectangle((-500, E[i]-E_int/2), 2000, E_int , linewidth=.5,\
                         edgecolor=colors[i], facecolor=colors[i], alpha = 0.35))
    axs[2].add_patch(rect2) #Add rectangle to plot
    axs[3].add_patch(rect3) #Add rectangle to plot

mpes.save_figure(fig, name = f'test', image_format = 'pdf')

#%% Plot MM, kx-E, and ky-E Frames

fig, ax = plt.subplots(1, 3, height_ratios=[1])
fig.set_size_inches(12, 6, forward=False)
ax = ax.flatten()

# Momenutm Map
mpes.plot_momentum_maps(
    I_res, 1.3, E_int=0.2, delays=500, delay_int=1000,
    fig = fig, ax = ax[0],
    cmap='BuPu',
    panel_labels=False, fontsize=16,
    nrows=2, figsize=(8, 6)
)

# Plot kx frame
mpes.plot_kx_frame(
    I_res, 0, 0.5, delays=[500], delay_int=1000,
    subtract_neg=subtract_neg, neg_delays=neg_delays,
    fig = fig, ax = ax[1],
    cmap = 'BuPu'
)

# Plot ky frame
mpes.plot_ky_frame(
    I_res, 0, 0.5, delays=[500], delay_int=1000,
    subtract_neg=subtract_neg, neg_delays=neg_delays,
    fig = fig, ax = ax[2],
    cmap = 'BuPu'
)

mpes.save_figure(fig, name = f'test', image_format = 'pdf')


save_figure = False
figure_file_name = 'k-integrated 4Panel_'
image_format = 'pdf'

E_trace, E_int = [2.05, 2.3, 2.5, 2.7, 2.9, 3.1], .12 # Energies for Plotting Time Traces ; 1st Energy for MM
(kx, ky), (kx_int, ky_int) = (0, 0), (4, 4) # Central (kx, ky) point and k-integration
colors = ['black', 'crimson'] #colors for plotting the traces

colors = colors[::-1]
E_trace = E_trace[::-1]

E_trace, E_int = [1.33, 2.15], .12 # Energies for Plotting Time Traces ; 1st Energy for MM
(kx, ky), (kx_int, ky_int) = (0, 0), (4, 5) # Central (kx, ky) point and k-integration
delay, delay_int = 500, 1000 #kx frame

#E_trace, E_int = [1.39, 2.24], .12 # Energies for Plotting Time Traces ; 1st Energy for MM

colors = ['black', 'crimson'] #colors for plotting the traces

subtract_neg, neg_delays = True, [-200,-100] #If you want to subtract negative time delay baseline
norm_trace = False
subtract_neg_maps = True

E_MM ,E_MM_int = 1.4, 0.2
Ein = .9 #Enhance excited states above this Energy, eV
energy_limits = [.25, 3] # Energy Y Limits for Plotting Panels 3 and 4

# Define Data Slices
if subtract_neg_maps is True:
    mm_frame = mpes.get_momentum_map(I_diff, E_MM, E_MM_int, 500, 2000)
    kx_frame = mpes.get_kx_E_frame(I_diff, ky, ky_int, delay, delay_int)
    waterfall = mpes.get_waterfall(I_diff, kx, kx_int, ky, ky_int)
    scale = [-1, 1]
    cmap_plot = cmocean.cm.balance
else:
    scale = [0, 1]
    cmap_plot = cmap_LTL
    mm_frame = mpes.get_momentum_map(I_res, E_MM, E_MM_int, 500, 2000)
    kx_frame = mpes.get_kx_E_frame(I_res, ky, ky_int, delay, delay_int)
    waterfall = mpes.get_waterfall(I_res, kx, kx_int, ky, ky_int)

### Do the Plotting ###
fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 1.5], 'height_ratios':[1, 1]})
fig.set_size_inches(8, 6, forward=False)
ax = ax.flatten()

plot_symmetry_points = False

### FIRST PLOT: MM of the First Energy
mm_frame = mm_frame/np.max(mm_frame)
im = mm_frame.plot.imshow(ax = ax[0], vmin = 0, vmax = 1, cmap = cmap_LTL, add_colorbar=False)

ax[0].set_aspect(1)
ax[0].set_xlim(-2,2)
ax[0].set_ylim(-2,2)
ax[0].set_xticks(np.arange(-2,2.2,1))
for label in ax[0].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
    
ax[0].set_yticks(np.arange(-2,2.1,1))
for label in ax[0].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
ax[0].set_xlabel('$k_x$, $\AA^{-1}$', fontsize = 18)
ax[0].set_ylabel('$k_y$, $\AA^{-1}$', fontsize = 18)
ax[0].set_title('$E$ = ' + str(E_MM) + ' eV', fontsize = 18)
ax[0].tick_params(axis='both', labelsize=16)
rect = (Rectangle((kx-kx_int/2, ky-ky_int/2), kx_int, ky_int, linewidth=.5,\
                         edgecolor='fuchsia', facecolor='fuchsia', alpha = 0.3))
if kx_int < 4:
    ax[0].add_patch(rect) #Add rectangle to plot

### SECOND PLOT: kx Frame
kx_frame = mpes.enhance_features(kx_frame, Ein, factor = 0, norm = True)

im2 = kx_frame.T.plot.imshow(ax=ax[2], cmap=cmap_plot, add_colorbar=False, vmin=scale[0], vmax=scale[1]) #kx, ky, t
ax[2].set_aspect(1)
ax[2].set_xticks(np.arange(-2,2.2,1))
for label in ax[2].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
ax[2].set_yticks(np.arange(-2,4.1,.25))
for label in ax[2].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax[2].set_xlabel('$k_x$, $\AA^{-1}$', fontsize = 18)
ax[2].set_ylabel('$E - E_{VBM}, eV$', fontsize = 18)
ax[2].set_title(f'$k_y$ = {ky} $\pm$ {ky_int/2} $\AA^{{-1}}$', fontsize = 18)
ax[2].tick_params(axis='both', labelsize=16)
ax[2].set_xlim(-2,2)
ax[2].set_ylim(energy_limits[0],energy_limits[1])
ax[2].text(-1.9, energy_limits[1]-0.3,  f"$\Delta$t = {delay} $\pm$ {delay_int/2:.0f} fs", size=14)
ax[2].axhline(Ein, linestyle = 'dashed', color = 'black', linewidth = 1)

if plot_symmetry_points is True:
    ax[2].axvline(0, linestyle = 'dashed', color = 'pink', linewidth = 1.5)
    ax[2].axvline(X, linestyle = 'dashed', color = 'pink', linewidth = 1.5)
    ax[2].axvline(-X, linestyle = 'dashed', color = 'pink', linewidth = 1.5)
    ax[2].axvline(2*X, linestyle = 'dashed', color = 'pink', linewidth = 1.5)
    ax[2].axvline(-2*X, linestyle = 'dashed', color = 'pink', linewidth = 1.5)
#ax[0].fill_between(I.E.values, I.kx.values - kx_int/2, I.kx.values + kx_int/2, color = 'pink', alpha = 0.5)
ax[2].set_aspect("auto")

### THIRD PLOT: WATERFALL

waterfall = mpes.enhance_features(waterfall, Ein, factor = 0, norm = True)

im3 = waterfall.plot(ax = ax[3], vmin = scale[0], vmax = scale[1], cmap = cmap_plot, add_colorbar=False)
ax[3].set_xlabel('Delay, fs', fontsize = 18)
ax[3].set_ylabel('E - E$_{VBM}$, eV', fontsize = 18)
ax[3].set_yticks(np.arange(-1,3.5,0.25))
ax[3].set_xlim(I.delay[1],I.delay[-1])
ax[3].set_ylim(energy_limits)
ax[3].set_title('$k$-Integrated')
ax[3].axhline(Ein, linestyle = 'dashed', color = 'black', linewidth = 1)

for label in ax[3].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
hor = I.delay[-1] - I.delay[1]
ver =  energy_limits[1] - energy_limits[0]
aspra = hor/ver 
#ax[1].set_aspect(aspra)
ax[3].set_aspect("auto")
#ax[3].set_title(f'$k_x$ = {kx} $\pm$ {kx_int/2} $\AA^{{-1}}$', fontsize = 18)
#ax[3].text(500, 2.4, f'$k_y$ = {ky} $\pm$ {ky_int/2} $\AA^{{-1}}$', fontsize = 14)

#fig.colorbar(waterfall, ax=ax[1], shrink = 0.8, ticks = [0, color_max/2, color_max])

### THIRD PLOT: DELAY TRACES (Angle-Integrated)
trace_norms = []
trace_norm_i = len(E_trace)
for i in np.arange(len(E_trace)):
    
    trace = mpes.get_time_trace(I, E_trace[i], E_int, (kx, ky), (kx_int, ky_int), norm_trace, subtract_neg, neg_delays)
    trace_norms.append(np.max(trace))

for i in np.arange(len(E_trace)):
    trace = mpes.get_time_trace(I, E_trace[i], E_int, (kx, ky), (kx_int, ky_int), norm_trace, subtract_neg, neg_delays)

    trace = trace/np.max(trace_norms)
        
    trace.plot(ax = ax[1], color = colors[i], label = str(E_trace[i]) + ' eV')
#    ax[3].axhline(E_trace[i], linestyle = 'dashed', color = colors[i], linewidth = 1.5)

#   ax[1].axhline(E_trace[i]-E_int/2, linestyle = 'dashed', color = colors[i], linewidth = 1.5)
#   ax[1].axhline(E_trace[i]+E_int/2, linestyle = 'dashed', color = colors[i], linewidth = 1.5)
    
    rect2 = (Rectangle((kx-kx_int/2, E_trace[i]-E_int/2), kx_int, E_int , linewidth=.5,\
                         edgecolor=colors[i], facecolor=colors[i], alpha = 0.5))
#    if kx_int < 4:
    ax[2].add_patch(rect2) #Add rectangle to plot
    
    rect3 = (Rectangle((-500, E_trace[i]-E_int/2), 2000, E_int , linewidth=.5,\
                         edgecolor=colors[i], facecolor=colors[i], alpha = 0.35))
    ax[3].add_patch(rect3) #Add rectangle to plot

#    ax[2].axhline(E_trace[i], linestyle = 'dashed', color = colors[i], linewidth = 1.5)


ax[1].set_xlim(I.delay[1], I.delay[-1])

if norm_trace is True:
    ax[1].set_ylim(-0.1, 1.1)
else:
    ax[1].set_ylim(-0.1, 1.1*np.max(1))
    
ax[1].set_ylabel('Norm. Int.', fontsize = 18)
ax[1].set_xlabel('Delay, fs', fontsize = 18)
ax[1].legend(frameon = False, loc = 'upper right', fontsize = 14)
ax[1].set_title('Delay Traces')

fig.text(.03, 0.975, "(a)", fontsize = 18, fontweight = 'regular')
fig.text(.03, 0.5, "(c)", fontsize = 18, fontweight = 'regular')
fig.text(.44, 0.975, "(b)", fontsize = 18, fontweight = 'regular')
fig.text(.44, 0.5, "(d)", fontsize = 18, fontweight = 'regular')

params = {'lines.linewidth' : 2.5, 'axes.linewidth' : 2, 'axes.labelsize' : 20, 
              'xtick.labelsize' : 16, 'ytick.labelsize' : 16, 'axes.titlesize' : 20, 'legend.fontsize' : 16}
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update(params)

fig.tight_layout()
#ax[1].set_tick_params(axis='both', labelsize=16)
#plt.gca().set_aspect(200)

if save_figure is True:
    fig.savefig(figure_file_name + '.'+ image_format, bbox_inches='tight', format=image_format) 
#%% Excited State kx Panels

save_figure = False
figure_file_name = 'kx_PANELS_alldelays_0.25dky'
image_format = 'pdf'

(kx, ky), (kx_int, ky_int) = (0, 0), (4, 0.25) # Central (kx, ky) point and k-integration
delays, delay_int = [-120, 0, 50, 100, 200, 500], 50 #kx frame

Ein = .8 #Enhance excited states above this Energy, eV
energy_limits = [0.8, 3] # Energy Y Limits for Plotting Panels 3 and 4
plot_symmetry_points = False

# Plot kx frame
mpes.plot_kx_frame(
    I_res, 0, 0.5, delays, delay_int=100,
    subtract_neg=subtract_neg, neg_delays=neg_delays,
    nrows = 2, ncols = 3,
    cmap = 'BuPu'
)

fig.text(.03, 0.975, "(a)", fontsize = 18, fontweight = 'regular')
fig.text(.03, 0.5, "(d)", fontsize = 18, fontweight = 'regular')
fig.text(.355, 0.975, "(b)", fontsize = 18, fontweight = 'regular')
fig.text(.355, 0.5, "(e)", fontsize = 18, fontweight = 'regular')
fig.text(.68, 0.975, "(c)", fontsize = 18, fontweight = 'regular')
fig.text(.68, 0.5, "(f)", fontsize = 18, fontweight = 'regular')

#ax[1].set_tick_params(axis='both', labelsize=16)
#plt.gca().set_aspect(200)

if save_figure is True:
    mpes.save_figure(fig, name = f'test', image_format = 'pdf')

#%% Excited State ky Panels

save_figure = False
figure_file_name = 'ky_PANELS_alldelays_0.25dky'
image_format = 'pdf'

(kx, ky), (kx_int, ky_int) = (0, 0), (4, 0.25) # Central (kx, ky) point and k-integration
delays, delay_int = [-120, 0, 50, 100, 200, 500], 50 #kx frame

Ein = .8 #Enhance excited states above this Energy, eV
energy_limits = [0.8, 3] # Energy Y Limits for Plotting Panels 3 and 4
plot_symmetry_points = False

# Plot kx frame
mpes.plot_ky_frame(
    I_res, 0, 0.5, delays, delay_int=100,
    subtract_neg=subtract_neg, neg_delays=neg_delays,
    nrows = 2, ncols = 3,
    cmap = 'BuPu'
)

fig.text(.03, 0.975, "(a)", fontsize = 18, fontweight = 'regular')
fig.text(.03, 0.5, "(d)", fontsize = 18, fontweight = 'regular')
fig.text(.355, 0.975, "(b)", fontsize = 18, fontweight = 'regular')
fig.text(.355, 0.5, "(e)", fontsize = 18, fontweight = 'regular')
fig.text(.68, 0.975, "(c)", fontsize = 18, fontweight = 'regular')
fig.text(.68, 0.5, "(f)", fontsize = 18, fontweight = 'regular')

if save_figure is True:
    mpes.save_figure(fig, name = f'test', image_format = 'pdf')