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

E_MM = 0
frame_mm = mpes.get_momentum_map(I_res, E_MM, 0.2, delays, delay_int)
frame_mm = frame_mm / np.max(frame_mm)
im = frame_mm.plot.imshow(ax = ax[0], vmin = scale[0], vmax = scale[1], cmap = cmap_plot, add_colorbar=False)

# cbar_ax = fig.add_axes([.51, 0.275, 0.025, 0.5])
cbar = fig.colorbar(im, ax = ax[0], ticks = [0,1])
cbar.ax.set_yticklabels(['min', 'max'])  # vertically oriented colorbar
rect = (Rectangle((kx-k_int/2, ky-k_int/2), k_int, k_int, linewidth=1.5,\
                         edgecolor='k', facecolor='None'))
fig.axes[0].add_patch(rect)
ax[0].set_ylim([-2,2])
ax[0].set_xlim([-2,2])
ax[0].set_title(f"E = {E_MM} eV")
ax[0].set_xlabel('$k_x$, $\AA^{-1}$', fontsize = 18)
ax[0].set_ylabel('$k_y$, $\AA^{-1}$', fontsize = 18)
ax[0].set_aspect(1)
#plt.axhline(0, linestyle = 'dashed', color = 'black')
#plt.axvline(0, linestyle = 'dashed', color = 'black')
#ax[0].set_xlabel('Delay, fs')
#ax[0].set_ylabel('E - E$_{VBM}$, eV')

pts = [-120]
colors = ['black']
n = len(pts)
colors = mpl.cm.inferno(np.linspace(0,.8,n))
                    
edc = mpes.get_edc(I_res, kx, ky, (k_int, k_int), delay, delay_int)
edc = edc/np.max(edc)
    
e = edc.plot(ax = ax[1], color = 'k', label = f"{pts[0]} fs")

#plt.legend(frameon = False)
ax[1].set_xlim([-1, 1]) 
#ax[1].set_ylim([0, 1.1])
ax[1].set_xlabel('E - E$_{VBM}$, eV')
ax[1].set_ylabel('Norm. Int.')
#ax[1].axvline(0, color = 'black', linestyle = 'dashed', linewidth = 0.5)
#ax[1].set_yscale('log')
#plt.ax[1].gca().set_aspect(2)

###################
##### Fit EDCs ####
###################

##### VBM #########
e1 = -.15
e2 = 0.6
p0 = [1, .02, 0.17, 0] # Fitting params initial guess [amp, center, width, offset]
bnds = ((0.5, -.5, 0.0, 0), (1.5, 0.5, 1, .5))
    
try:
    popt, pcov = curve_fit(gaussian, edc.loc[{"E":slice(e1,e2)}].E.values, edc.loc[{"E":slice(e1,e2)}].values, p0, method=None, bounds = bnds)
except ValueError:
    popt = p0
    print('oops!')
    
perr = np.sqrt(np.diag(pcov))
        
vb_fit = gaussian(edc.E, *popt)
ax[1].plot(edc.E, vb_fit, linestyle = 'dashed', color = 'pink', label = 'Fit')
ax[1].legend(frameon=False, loc = 'upper left', fontsize = 11)

print(round(popt[1],4))
print(round(perr[1],4))

if save_figure is True:
    fig.savefig((figure_file_name +'.svg'), format='svg')

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

from scipy.special import erf

fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1], 'height_ratios':[1]})
fig.set_size_inches(10, 4, forward=False)
ax = ax.flatten()

figure_file_name = 'define_t0'
save_figure = False

### Plot EDCs at GAMMA vs time

(kx, ky), (kx_int, ky_int) = (0, 0), (3.8, 1.8)
#(kx, ky), (kx_int, ky_int) = (0, .7), (1.5, 0.5)
E, E_int = 1.55, 0.1

trace_ex = I_res.loc[{"kx":slice(kx-kx_int/2,kx+kx_int/2), "ky":slice(ky-ky_int/2,ky+ky_int/2), "E":slice(E-E_int,E+E_int)}].mean(dim=("kx","ky","E"))
trace_ex = trace_ex - trace_ex[2:5].mean()
trace_ex = trace_ex/np.max(trace_ex)

im = I_res.loc[{"E":slice(E-E_int,E+E_int), "delay":slice(-200,500)}].mean(dim=("E","delay")).T.plot.imshow(ax = ax[0], cmap = cmap_LTL, add_colorbar = False)
# cbar_ax = fig.add_axes([.51, 0.275, 0.025, 0.5])
cbar = fig.colorbar(im, ax = ax[0], ticks = [0,1])
cbar.ax.set_yticklabels(['min', 'max'])  # vertically oriented colorbar
rect = (Rectangle((kx-kx_int/2, ky-ky_int/2), kx_int, ky_int, linewidth=1.5,\
                         edgecolor='k', facecolor='None'))
fig.axes[0].add_patch(rect)
ax[0].set_ylim([-2,2])
ax[0].set_xlim([-2,2])
ax[0].set_aspect(1)
ax[0].set_title(f"E = {E} eV")

#plt.axhline(0, linestyle = 'dashed', color = 'black')
#plt.axvline(0, linestyle = 'dashed', color = 'black')

t0 = 0
tau = 25
def rise_erf(t, t0, tau):
    r = 0.5 * (1 + erf((t - t0) / (tau)))
    return r

rise = rise_erf(I_res.delay, -30, 45)

p0 = [-30, 45]
popt, pcov = curve_fit(rise_erf, trace_ex.loc[{"delay":slice(-200,40)}].delay.values ,
                                trace_ex.loc[{"delay":slice(-200,40)}].values,
                                p0, method="lm")

perr = np.sqrt(np.diag(pcov))

rise_fit = rise_erf(np.linspace(-200,200,50), *popt)

ax[1].plot(I_res.delay, trace_ex, 'ko')
ax[1].plot(np.linspace(-200,200,50), rise_fit, 'pink')
#ax[1].plot(I_res.delay, rise, 'red')
ax[1].set_xlabel('Delay, fs')
ax[1].set_ylabel('Norm. Int.')
ax[1].axvline(0, color = 'grey', linestyle = 'dashed')

ax[1].set_xlim([-150, 150]) 
ax[1].set_ylim(-.1,1.05)
#ax[1].axvline(30)

print(round(popt[0],3), round(perr[0],1))
print(round(popt[1],3), round(perr[1],1))

if save_figure is True:
    fig.savefig((figure_file_name +'.svg'), format='svg')

#%% Plot Momentum Maps at Constant Energy

E, E_int = [1.35, 1.35, 1.35, 2.15, 2.15, 2.15], 0.2
delays, delay_int = [500, 500], 1000

fig, ax, im = mpes.plot_momentum_maps(
    I_res, E=E, E_int=0.2, delays=delays, delay_int=delay_int,
    cmap=cmap_LTL, scale=[0, 1],
    fontsize=16, figsize=(8, 6), colorbar=True, panel_labels = False
)
     
mpes.save_figure(fig, name = f'test', image_format = 'pdf')
#%% Plot Overview: Extract MM, Delay Traces, k-cut, and Waterfall Panel

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

#%% Plot Dynamics Overview: Extract Traces At Different Energies with Waterfall Panel

save_figure = False
figure_file_name = 'k-integrated 4Panel_'
image_format = 'pdf'

E_trace, E_int = [2.05, 2.3, 2.5, 2.7, 2.9, 3.1], .12 # Energies for Plotting Time Traces ; 1st Energy for MM
(kx, ky), (kx_int, ky_int) = (0, 0), (4, 4) # Central (kx, ky) point and k-integration
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

#%% Plot Waterfall With Different Traces

E_trace, E_int = [2.05, 2.3, 2.5, 2.7, 2.9, 3.1], .1 # Energies for Plotting Time Traces ; 1st Energy for MM
(kx, ky), (kx_int, ky_int) = (0, 0), (4, 4) # Central (kx, ky) point and k-integration

colors = ['crimson', 'violet', 'purple', 'midnightblue', 'orange', 'grey'] #colors for plotting the traces
colors = colors[::-1]
E_trace = E_trace[::-1]

fig, axs = plt.subplots(2, 1)
fig.set_size_inches(8, 8, forward=False)
axs = axs.flatten()

# Plot waterfall
plot_waterfall(
    I_res, kx, kx_int, ky, ky_int,
    fig = fig, ax = axs[0],
    cmap= cmocean.cm.balance, scale=[0,1]
)

# Plot time traces
mpes.plot_time_traces(
    I, E_trace, E_int, (kx, ky), (kx_int, ky_int),
    norm_trace=False, subtract_neg=True, neg_delays=[-110,-70],
    fig = fig, ax = axs[1],
    colors = colors,
    fontsize=16
)

#%% Plot Dynamics: Extract Traces At Different Energies and Momenta: Distinct k Points

save_figure = False
figure_file_name = 'distinct_K_Points'
image_format = 'pdf'

subtract_neg, neg_delays = True, [-110,-70] #If you want to subtract negative time delay baseline
norm_trace = False

E_ex, E_cbm, E_int = 1.25, 2.05, 0.2

(kx, ky), (kx_int, ky_int) = ((-2*X, -1.5*X, -X, -X/2, 0.0, X/2, X, 1.5*X, 2*X), 0), (.25, .25) # Central (kx, ky) point and k-integration
colors = ['crimson', 'purple', 'blue', 'darkblue', 'dodgerblue', 'yellow', 'darkorange', 'lightcoral', 'black'] #colors for plotting the traces

#######################
### Do the Plotting ###
#######################

fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 1.5], 'height_ratios':[1, 1]})
fig.set_size_inches(8, 6, forward=False)
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
plot_kx_frame(
    I_res, 0, 0.5, delays=[500], delay_int=1000,
    subtract_neg=subtract_neg, neg_delays=neg_delays,
    fig = fig, ax = ax[2],
    cmap = 'BuPu'
)

### for the Exciton

# Plot time traces
mpes.plot_time_traces(
    I, E_ex, E_int, (kx, ky), (kx_int, ky_int),
    norm_trace=True, subtract_neg=True, neg_delays=[-110,-70],
    fig = fig, ax = ax[1],
    colors = colors, legend=False,
    fontsize=16
)

### for the CBM

# Plot time traces
mpes.plot_time_traces(
    I, E_cbm, E_int, (kx, ky), (kx_int, ky_int),
    norm_trace=True, subtract_neg=True, neg_delays=[-110,-70],
    fig = fig, ax = ax[3],
    colors = colors, legend=False,
    fontsize=16
)

ax[1].set_title(f"Exciton")
ax[3].set_title(f"CBM")

for i in np.arange(len(kx)):
    
    rect1 = (Rectangle((kx[i]-kx_int/2, E_ex-E_int/2), kx_int, E_int , linewidth=.5,\
                         edgecolor=colors[i], facecolor=colors[i], alpha = 0.75))
    ax[2].add_patch(rect1) #Add rectangle to plot

    rect2 = (Rectangle((kx[i]-kx_int/2, E_cbm-E_int/2), kx_int, E_int , linewidth=.5,\
                         edgecolor=colors[i], facecolor=colors[i], alpha = 0.75))
    ax[2].add_patch(rect2) #Add rectangle to plot
    rect = (Rectangle((kx[i]-kx_int/2, ky-ky_int/2), kx_int, ky_int, linewidth=.5,\
                             edgecolor='grey', facecolor='grey', alpha = 0.7))
    if kx_int < 4:
        ax[0].add_patch(rect) #Add rectangle to plot


# ax[2].text(0+0.04, 2.75, f"$\Gamma$", size=12)
# ax[2].text(-X+0.04, 2.75, f"$X$", size=12)
# ax[2].text(X+0.04, 2.75, f"$X$", size=12)
# ax[2].text(-2*X+0.04, 2.75, f"$\Gamma$", size=14)
# ax[2].text(2*X-0.35, 2.75, f"$\Gamma$", size=14)
# ax[2].text(-1.82, 2.475,  f"$\Delta$t = {delay} fs", size=13)

k_point_label = ['$\Gamma_{-1,0}$', '$-X$', '$\Gamma_{0}$', '$+X$', '$\Gamma_{1,0}$']
k_point_label = ['$\Gamma_{-1,0}$', '$-3X/2$', '$-X$','$-X/2$', '$\Gamma_{0}$', '$+X/2$', '$+X$', '$+3X/2$', '$\Gamma_{1,0}$']

fig.text(.03, 0.975, "(a)", fontsize = 18, fontweight = 'regular')
fig.text(.03, 0.5, "(c)", fontsize = 18, fontweight = 'regular')
fig.text(.425, 0.975, "(b)", fontsize = 18, fontweight = 'regular')
fig.text(.425, 0.5, "(d)", fontsize = 18, fontweight = 'regular')

params = {'lines.linewidth' : 2.5, 'axes.linewidth' : 2, 'axes.labelsize' : 20, 
              'xtick.labelsize' : 16, 'ytick.labelsize' : 16, 'axes.titlesize' : 20, 'legend.fontsize' : 16}
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update(params)

fig.tight_layout()

#ax[1].set_tick_params(axis='both', labelsize=16)
#plt.gca().set_aspect(200)

if save_figure is True:
    fig.savefig(figure_file_name + '.'+ image_format, bbox_inches='tight', format=image_format) 

#%% Plot Excited State kx Panels

save_figure = False
figure_file_name = 'kx_PANELS_alldelays_0.25dky'
image_format = 'pdf'
cmap_plot = 'seismic'
cmap_plot = cmocean.cm.balance
scale = 1

(kx, ky), (kx_int, ky_int) = (0, 0), (4, 0.25) # Central (kx, ky) point and k-integration
delays, delay_int = [-120, 0, 50, 100, 200, 500], 50 #kx frame

Ein = .8 #Enhance excited states above this Energy, eV
energy_limits = [0.8, 3] # Energy Y Limits for Plotting Panels 3 and 4
plot_symmetry_points = False

#######################
### Do the Plotting ###
#######################

# Plot kx frame
plot_kx_frame(
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


params = {'lines.linewidth' : 2.5, 'axes.linewidth' : 2, 'axes.labelsize' : 20, 
              'xtick.labelsize' : 16, 'ytick.labelsize' : 16, 'axes.titlesize' : 20, 'legend.fontsize' : 16}
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update(params)

fig.tight_layout()
#ax[1].set_tick_params(axis='both', labelsize=16)
#plt.gca().set_aspect(200)

if save_figure is True:
    fig.savefig(figure_file_name + '.'+ image_format, bbox_inches='tight', format=image_format) 

#%% Excited State ky Panels

save_figure = False
figure_file_name = 'ky_PANELS_kx=negX'
image_format = 'pdf'
cmap_plot = cmocean.cm.balance
scale = [-1, 1]

(kx, ky), (kx_int, ky_int) = (-X, 0), (0.4, 0) # Central (kx, ky) point and k-integration

delays, delay_int = [-120, 0, 50, 100, 200, 500], 50 #kx frame

Ein = .75 #Enhance excited states above this Energy, eV
energy_limits = [0.75, 3] # Energy Y Limits for Plotting Panels 3 and 4
plot_symmetry_points = False

#######################
### Do the Plotting ###
#######################

fig, ax = plt.subplots(2, 3)
fig.set_size_inches(9, 5, forward=False)
ax = ax.flatten()
plt.gcf().set_dpi(1000)

for i in np.arange(len(delays)):
    
    ky_frame = mpes.get_ky_E_frame(I_diff, kx, kx_int, delays[i], delay_int)
    ky_frame = mpes.enhance_features(ky_frame, Ein, factor = 0, norm = True)
    ky_frame = ky_frame/np.max(ky_frame.loc[{"E":slice(Ein,3)}])
        
    im_ky = ky_frame.T.plot.imshow(ax=ax[i], cmap=cmap_plot, add_colorbar=False, vmin=scale[0], vmax=scale[1]) #kx, ky, t
    ax[i].set_aspect(1)
    ax[i].set_xticks(np.arange(-2,2.2,1))
    for label in ax[i].xaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
    ax[i].set_yticks(np.arange(-2,4.1,.25))
    for label in ax[i].yaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
    ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax[i].set_xlabel('$k_y$, $\AA^{-1}$', fontsize = 18)
    ax[i].set_ylabel('$E - E_{VBM}, eV$', fontsize = 18)
#    ax[i].set_title(f'$k_y$ = {ky} $\pm$ {ky_int/2} $\AA^{{-1}}$', fontsize = 18)
    ax[i].set_title( f"$\Delta$t = {delays[i]} fs", fontsize = 18)
    ax[i].tick_params(axis='both', labelsize=16)
    ax[i].set_xlim(-2,2)
    ax[i].set_ylim(energy_limits[0],energy_limits[1])
#    ax[i].text(-1.9, energy_limits[1]-0.3,  f"$\Delta$t = {delay} $\pm$ {delay_int/2:.0f} fs", size=14)
    ax[i].axhline(Ein, linestyle = 'dashed', color = 'black', linewidth = 1)
    
    if plot_symmetry_points is True:
        ax[2].axvline(0, linestyle = 'dashed', color = 'pink', linewidth = 1.5)
        ax[2].axvline(X, linestyle = 'dashed', color = 'pink', linewidth = 1.5)
        ax[2].axvline(-X, linestyle = 'dashed', color = 'pink', linewidth = 1.5)
        ax[2].axvline(2*X, linestyle = 'dashed', color = 'pink', linewidth = 1.5)
        ax[2].axvline(-2*X, linestyle = 'dashed', color = 'pink', linewidth = 1.5)
    #ax[0].fill_between(I.E.values, I.kx.values - kx_int/2, I.kx.values + kx_int/2, color = 'pink', alpha = 0.5)
    ax[i].set_aspect("auto")

fig.text(.03, 0.975, "(a)", fontsize = 18, fontweight = 'regular')
fig.text(.03, 0.5, "(d)", fontsize = 18, fontweight = 'regular')
fig.text(.355, 0.975, "(b)", fontsize = 18, fontweight = 'regular')
fig.text(.355, 0.5, "(e)", fontsize = 18, fontweight = 'regular')
fig.text(.68, 0.975, "(c)", fontsize = 18, fontweight = 'regular')
fig.text(.68, 0.5, "(f)", fontsize = 18, fontweight = 'regular')

cbar_ax = fig.add_axes([1, 0.325, 0.025, 0.5])
cbar = fig.colorbar(im_kx, cax=cbar_ax, ticks = [-1, 0, 1])
cbar.ax.set_yticklabels(['-1', 0, '1'])  # vertically oriented colorbar

params = {'lines.linewidth' : 2.5, 'axes.linewidth' : 2, 'axes.labelsize' : 20, 
              'xtick.labelsize' : 16, 'ytick.labelsize' : 16, 'axes.titlesize' : 20, 'legend.fontsize' : 16}
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update(params)

fig.tight_layout()
#ax[1].set_tick_params(axis='both', labelsize=16)
#plt.gca().set_aspect(200)

if save_figure is True:
    fig.savefig(figure_file_name + '.'+ image_format, bbox_inches='tight', format=image_format) 
    
#%% Plot MM + k vs E

save_figure = False
figure_file_name = 'MM_ARPES_delay_frames_200'

E_MM, E_int = 0.4, .12 # Energies for Plotting Time Traces ; 1st Energy for MM
k, k_int = (0, 0), .2 # Central (kx, ky) point and k-integration
delay, delay_int = 50, 50

colors = ['black', 'red'] #colors for plotting the traces
cmap_plot = cmap_LTL

i = 0

(kx, ky) = k
I_res = I/I.max()
I_res_enh = mpes.enhance_features(I_res, Ein, factor = 0, norm = True)

frame = mpes.get_momentum_map(I, E_MM, E_int, delay, delay_int)

#Norm to t0 Frame
frame_t0 = mpes.get_kx_E_frame(I_res_enh, ky, k_int, 25, delay_int)
frame_t0_2 = mpes.get_ky_E_frame(I_res_enh, ky, k_int, 25, delay_int)

n = [frame_t0.loc[{"E":slice(-3,Ein)}].max().values, frame_t0.loc[{"E":slice(Ein,5)}].max().values]
n2 = [frame_t0_2.loc[{"E":slice(-3,Ein)}].max().values, frame_t0_2.loc[{"E":slice(Ein,5)}].max().values]

frame2 = mpes.get_kx_E_frame(I_res_enh, ky, k_int, delay, delay_int)
frame3 = mpes.get_ky_E_frame(I_res_enh, kx, k_int, delay, delay_int)

Ein = 0.75

f23 = mpes.enhance_features(frame2, Ein, factor = n, norm = True)
f33 = mpes.enhance_features(frame3, Ein, factor = n2, norm = True)
#f23 = frame2
#f33 = frame3

#######################
### Do the Plotting ###
#######################
    
fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [.75, 1, 1], 'height_ratios':[1]})
fig.set_size_inches(10, 4, forward=False)
ax = ax.flatten()

### FIRST PLOT: MM of the First Energy
im = frame.plot.imshow(ax = ax[0], cmap = cmap_plot, add_colorbar=False)
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
ax[0].set_title(f'$E$ = {E_MM} $eV$', fontsize = 18)
ax[0].tick_params(axis='both', labelsize=16)
ax[0].text(-1.9, 1.55,  f"$\Delta$t = {delay} fs", size=16)
ax[0].axvline(0, linestyle = 'dashed', color = 'pink', linewidth = 1.5)
ax[0].axvline(x, linestyle = 'dashed', color = 'pink', linewidth = 1.5)
ax[0].axvline(-x, linestyle = 'dashed', color = 'pink', linewidth = 1.5)
ax[0].axvline(2*x, linestyle = 'dashed', color = 'pink', linewidth = 1.5)
ax[0].axvline(-2*x, linestyle = 'dashed', color = 'pink', linewidth = 1.5)

### SECOND PLOT: kx vs E
im2 = f23.T.plot.imshow(ax=ax[1], cmap=cmap_plot, add_colorbar=False, vmin=0, vmax=1) #kx, ky, t
ax[1].set_aspect(1)

ax[1].set_xticks(np.arange(-2,2.2,1))
for label in ax[1].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
    
ax[1].set_yticks(np.arange(-2,2.1,1))
for label in ax[1].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
ax[1].set_xlabel('$k_x$, $\AA^{-1}$', fontsize = 18)
ax[1].set_ylabel('$E - E_{VBM}, eV$', fontsize = 18)
ax[1].set_title(f'$k_y$ = {kx} $\AA^{{-1}}$', fontsize = 18)
ax[1].tick_params(axis='both', labelsize=16)
ax[1].set_xlim(-2,2)
ax[1].set_ylim(-3,3)
ax[1].text(-1.9, 2.5,  f"$\Delta$t = {delay} fs", size=16)
ax[1].axhline(Ein, linestyle = 'dashed', color = 'black', linewidth = 1)
ax[1].axvline(0, linestyle = 'dashed', color = 'pink', linewidth = 1.5)
ax[1].axvline(x, linestyle = 'dashed', color = 'pink', linewidth = 1.5)
ax[1].axvline(-x, linestyle = 'dashed', color = 'pink', linewidth = 1.5)
ax[1].axvline(2*x, linestyle = 'dashed', color = 'pink', linewidth = 1.5)
ax[1].axvline(-2*x, linestyle = 'dashed', color = 'pink', linewidth = 1.5)

### THIRD PLOT: ky vs E
im3 = f33.T.plot.imshow(ax=ax[2], cmap=cmap_plot, add_colorbar=False, vmin=0, vmax=1) #kx, ky, t
ax[2].set_aspect(1)

ax[2].set_xticks(np.arange(-2,2.2,1))
for label in ax[2].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
    
ax[2].set_yticks(np.arange(-2,2.1,1))
for label in ax[2].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
ax[2].set_xlabel('$k_y$, $\AA^{-1}$', fontsize = 18)
ax[2].set_ylabel('$E - E_{VBM}, eV$', fontsize = 18)
ax[2].set_title(f'$k_x$ = {kx} $\AA^{{-1}}$', fontsize = 18)
ax[2].tick_params(axis='both', labelsize=16)
ax[2].set_xlim(-2,2)
ax[2].set_ylim(-2,3)
ax[2].text(-1.9, 2.5,  f"$\Delta$t = {delay} fs", size=16)
ax[2].axhline(Ein, linestyle = 'dashed', color = 'black', linewidth = 1)
ax[2].axvline(0, linestyle = 'dashed', color = 'pink', linewidth = 1.5)
ax[2].axvline(y, linestyle = 'dashed', color = 'pink', linewidth = 1.5)
ax[2].axvline(-y, linestyle = 'dashed', color = 'pink', linewidth = 1.5)
ax[2].axvline(2*y, linestyle = 'dashed', color = 'pink', linewidth = 1.5)
ax[2].axvline(-2*y, linestyle = 'dashed', color = 'pink', linewidth = 1.5)

cbar_ax = fig.add_axes([1, 0.275, 0.025, 0.5])
cbar = fig.colorbar(im, cax=cbar_ax, ticks = [0,frame.max()])
cbar.ax.set_yticklabels(['min', 'max'])  # vertically oriented colorbar

params = {'lines.linewidth' : 2.5, 'axes.linewidth' : 2, 'axes.labelsize' : 20, 
              'xtick.labelsize' : 16, 'ytick.labelsize' : 16, 'axes.titlesize' : 20, 'legend.fontsize' : 16}
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update(params)

fig.tight_layout()
#ax[1].set_tick_params(axis='both', labelsize=16)
#plt.gca().set_aspect(200)

if save_figure is True:
    fig.savefig(figure_file_name + '.'+ image_format, bbox_inches='tight', format=image_format) 

#%% Extract k-Dispersion and Eb momentum-depenence

save_figure = False
figure_file_name = 'eb-dispersion_120k'
image_format = 'pdf'
cmap_plot = cmap_LTL2

E_trace, E_int = [1.35, 2.05], .1 # Energies for Plotting Time Traces ; 1st Energy for MM
(kx, ky), (kx_int, ky_int) = (0.0, 0.0), (0.2, 0.2) # Central (kx, ky) point and k-integration
delay, delay_int = 500, 1000
subtract_neg  = True

if subtract_neg is True:
    I_ = I_diff
    cmap_plot = cmap_LTL2
    scale = [-1, 1]

else:
    I_ = I_res
    scale = [0, 1]
    
kx_frame = mpes.get_kx_E_frame(I_, ky, ky_int, delay, delay_int)
kx_frame = kx_frame/np.max(kx_frame.loc[{"E":slice(0.8,3)}])

e1, e2 = 1.1, 3
k1, k2 = -1.83, 1.8
ax_kx = kx_frame.loc[{"kx":slice(k1,k2)}].kx.values
kx_fits = np.zeros((len(kx_frame.loc[{"kx":slice(k1,k2)}].kx.values),2))
kx_fits_error = np.zeros(kx_fits.shape)
eb_kx = np.zeros(kx_fits.shape[0])
eb_kx_error = np.zeros(kx_fits.shape[0])

i = 0
for k in ax_kx:
    kx_int = 0.2
    
    kx_edc = kx_frame.loc[{"kx":slice(k-kx_int/2,k+kx_int/2)}].mean(dim="kx")
    kx_edc = kx_edc/np.max(kx_edc.loc[{"E":slice(0.8,3)}])
    
    ##### X and CBM ####
    p0 = [1, 0.3,  1.35, 2.1,  0.2, 0.2, 0] # Fitting params initial guess [amp, center, width, offset]
    bnds = ((0.5, 0.1, 1.1, 1.9, 0.1, 0.1, 0), (1.5, 0.7, 1.5, 2.3, 0.9, 0.9, .3))
    
    popt, pcov = curve_fit(two_gaussians, kx_edc.loc[{"E":slice(e1,e2)}].E.values, kx_edc.loc[{"E":slice(e1,e2)}].values, p0, method=None, bounds = bnds)
    perr = np.sqrt(np.diag(pcov))
    g, g1, g2, offset = two_gaussians_report(kx_edc.loc[{"E":slice(0,3)}].E.values, *popt)
    kx_fits[i,0] = popt[2]
    kx_fits[i,1] = popt[3]
    kx_fits_error[i,0] = perr[2]
    kx_fits_error[i,1] = perr[3]
    eb_kx[i] = kx_fits[i,1] - kx_fits[i,0]
    eb_kx_error[i] = np.sqrt(perr[3]**2+perr[2]**2)
    
    if k < 0.02 and k > -0.7:
        kx_fits[i,:] = nan
        eb_kx[i] = nan
        eb_kx_error[i] = nan
        kx_fits_error[i,:] = nan
    i += 1

## Plot ##

fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1], 'height_ratios':[1]})
fig.set_size_inches(9, 3.5, forward=False)
ax = ax.flatten()

##### X and CBM ####
#Fit to Time an kx integrated
kx, kx_int = 0, 3.8
kx_edc = kx_frame.loc[{"kx":slice(kx-kx_int/2,kx+kx_int/2)}].mean(dim="kx")
kx_edc = kx_edc/np.max(kx_edc.loc[{"E":slice(0.8,3)}])
p0 = [1, 0.3,  1.35, 2.1,  0.2, 0.2, 0] # Fitting params initial guess [amp, center, width, offset]
bnds = ((0.5, 0.1, 1.1, 1.9, 0.1, 0.1, 0), (1.5, 0.7, 1.5, 2.3, 0.9, 0.9, .3))

popt, pcov = curve_fit(two_gaussians, kx_edc.loc[{"E":slice(e1,e2)}].E.values, kx_edc.loc[{"E":slice(e1,e2)}].values, p0, method=None, bounds = bnds)
perr = np.sqrt(np.diag(pcov))
g, g1, g2, offset = two_gaussians_report(kx_edc.loc[{"E":slice(0,3)}].E.values, *popt)
Eb = round(popt[3] - popt[2],3)
Eb_err = np.sqrt(perr[3]**2+perr[2]**2) 
print(f'The kx mean is {np.nanmean(1000*eb_kx)} +- {1000*np.nanstd(eb_kx)}')

im2 = kx_frame.T.plot.imshow(ax=ax[0], cmap=cmap_plot, add_colorbar=False, vmin=scale[0], vmax=scale[1]) #kx, ky, t
#ax[0].set_aspect(1)

ax[0].set_xticks(np.arange(-2,2.2,1))
for label in ax[0].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
ax[0].set_yticks(np.arange(-2,4.1,.25))
for label in ax[0].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
ax[0].set_xlabel('$k_x$, $\AA^{-1}$', fontsize = 18)
ax[0].set_ylabel('$E - E_{VBM}, eV$', fontsize = 18)
ax[0].set_title(f'$k_y$ = {ky} $\AA^{{-1}}$', fontsize = 18)
ax[0].tick_params(axis='both', labelsize=16)
ax[0].set_xlim(-2,2)
ax[0].set_ylim(0.8,3)
#ax[0].text(-1.9, 2.7,  f"$\Delta$t = {delay} $\pm$ {delay_int/2:.0f} fs", size=16)

ax[0].text(0+0.05, 2.75, f"$\Gamma$", size=18)
ax[0].text(-X+0.05, 2.75, f"$X$", size=18)
ax[0].text(X+0.05, 2.75, f"$X$", size=18)
ax[0].text(-2*X+0.05, 2.75, f"$\Gamma$", size=18)
ax[0].text(2*X+0.05, 2.75, f"$\Gamma$", size=18)

ax[0].axhline(Ein, linestyle = 'dashed', color = 'black', linewidth = 1.5)
ax[0].axvline(0, linestyle = 'dashed', color = 'black', linewidth = 1.5)
ax[0].axvline(X, linestyle = 'dashed', color = 'black', linewidth = 1.5)
ax[0].axvline(-X, linestyle = 'dashed', color = 'black', linewidth = 1.5)
ax[0].axvline(2*X, linestyle = 'dashed', color = 'black', linewidth = 1.5)
ax[0].axvline(-2*X, linestyle = 'dashed', color = 'black', linewidth = 1.5)
#ax[0].axhline(popt[2], linestyle = 'dashed', color = 'k', linewidth = 1.5)
#ax[0].axhline(popt[3], linestyle = 'dashed', color = 'r', linewidth = 1.5)
ax[0].plot(ax_kx, kx_fits[:,0], 'o', color = 'black')
ax[0].plot(ax_kx, kx_fits[:,1], 'o', color = 'crimson')
#ax[0].fill_between(kx_frame.loc[{"kx":slice(k1,k2)}].kx.values, kx_fits[:,0] - kx_fits_error[:,0], kx_fits[:,0] + kx_fits_error[:,0], color = 'grey', alpha = 0.5)
#ax[0].fill_between(kx_frame.loc[{"kx":slice(k1,k2)}].kx.values, kx_fits[:,1] - kx_fits_error[:,1], kx_fits[:,1] + kx_fits_error[:,1], color = 'crimson', alpha = 0.5)

#ax[0].fill_between(I.E.values, I.kx.values - kx_int/2, I.kx.values + kx_int/2, color = 'pink', alpha = 0.5)
rect = (Rectangle((kx-kx_int/2, .5), kx_int, 3, linewidth=2.5,\
                         edgecolor='purple', facecolor='purple', alpha = 0.3))
#if kx_int < 4:
    #ax[0].add_patch(rect) #Add rectangle to plot

ax[1].fill_between(ax_kx, 1000*eb_kx - 1000*eb_kx_error, 1000*eb_kx + 1000*eb_kx_error, color = 'violet', alpha = 0.5)
ax[1].plot(ax_kx, 1000*eb_kx, color = 'darkviolet')
ax[1].set_xlim(-2,2)
ax[1].set_ylim(700,900)
ax[1].set_ylabel('$E_{b}, meV$', fontsize = 18)
ax[1].set_xlabel('$k_x$, $\AA^{-1}$', fontsize = 18)
ax[1].set_title(f"$E_b = {1000*np.nanmean(eb_kx):.0f} \pm {np.nanmean(1000*eb_kx_error):.0f}$ meV")
ax[1].set_title(f"Extracted $E_b$")

ax[1].set_aspect('auto')

fig.text(.02, 0.975, "(a)", fontsize = 20, fontweight = 'regular')
fig.text(.51, 0.975, "(b)", fontsize = 20, fontweight = 'regular')

params = {'lines.linewidth' : 2.5, 'axes.linewidth' : 2, 'axes.labelsize' : 20, 
              'xtick.labelsize' : 16, 'ytick.labelsize' : 16, 'axes.titlesize' : 20, 'legend.fontsize' : 16}
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update(params)

fig.tight_layout()

if save_figure is True:
    fig.savefig(figure_file_name + '.'+ image_format, bbox_inches='tight', format=image_format) 
    
print(f"{popt[2]:.3f} +- {perr[2]:.3f}")
print(f"{popt[3]:.3f} +- {perr[3]:.3f}")
print(f"{1000*Eb:.3f} +- {1000*Eb_err:.3f}")

#%% TEST: CBM EDC Fitting to Extract EXCITON Binding Energy and Peak Positions

save_figure = False
figure_name = 'kx_excited'

E_trace, E_int = [1.35, 2.05], .1 # Energies for Plotting Time Traces ; 1st Energy for MM
(kx, ky), (kx_int, ky_int) = (0, 0), (3.8, 0.2) # Central (kx, ky) point and k-integration
delay, delay_int = 500, 1000

kx_frame = mpes.get_kx_E_frame(I_, ky, ky_int, delay, delay_int)
kx_frame = kx_frame/np.max(kx_frame.loc[{"E":slice(0.8,3)}])

kx_edc = kx_frame.loc[{"kx":slice(kx-kx_int/2,kx+kx_int/2)}].mean(dim="kx")
kx_edc = kx_edc/np.max(kx_edc.loc[{"E":slice(0.8,3)}])

##### X and CBM ####
p0 = [1, 0.3,  1.35, 2.1,  0.2, 0.2, 0] # Fitting params initial guess [amp, center, width, offset]
bnds = ((0.5, 0.1, 1.0, 1.5, 0.1, 0.1, 0), (1.5, 0.7, 1.5, 2.3, 0.9, 0.9, .3))

popt, pcov = curve_fit(two_gaussians, kx_edc.loc[{"E":slice(e1,e2)}].E.values, kx_edc.loc[{"E":slice(e1,e2)}].values, p0, method=None, bounds = bnds)
perr = np.sqrt(np.diag(pcov))
g, g1, g2, offset = two_gaussians_report(kx_edc.loc[{"E":slice(0,3)}].E.values, *popt)
Eb = round(popt[3] - popt[2],3)

fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1.5, 1], 'height_ratios':[1]})
fig.set_size_inches(10, 4, forward=False)
ax = ax.flatten()

im2 = kx_frame.T.plot.imshow(ax=ax[0], cmap=cmap_plot, add_colorbar=False, vmin=0, vmax=1) #kx, ky, t
ax[0].set_aspect(1)

ax[0].set_xticks(np.arange(-2,2.2,1))
for label in ax[0].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
ax[0].set_yticks(np.arange(-2,4.1,.25))
for label in ax[0].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
ax[0].set_xlabel('$k_x$, $\AA^{-1}$', fontsize = 18)
ax[0].set_ylabel('$E - E_{VBM}, eV$', fontsize = 18)
ax[0].set_title(f'$k_y$ = {ky} $\AA^{{-1}}$', fontsize = 18)
ax[0].tick_params(axis='both', labelsize=16)
ax[0].set_xlim(-2,2)
ax[0].set_ylim(0.8,3)
ax[0].text(-1.9, 2.7,  f"$\Delta$t = {delay} fs", size=16)
ax[0].axhline(Ein, linestyle = 'dashed', color = 'grey', linewidth = 1.5)
ax[0].axvline(0, linestyle = 'dashed', color = 'grey', linewidth = 1.5)
ax[0].axvline(X, linestyle = 'dashed', color = 'grey', linewidth = 1.5)
ax[0].axvline(-X, linestyle = 'dashed', color = 'grey', linewidth = 1.5)
ax[0].axvline(2*X, linestyle = 'dashed', color = 'grey', linewidth = 1.5)
ax[0].axvline(-2*X, linestyle = 'dashed', color = 'grey', linewidth = 1.5)
ax[0].axhline(popt[2], linestyle = 'dashed', color = 'k', linewidth = 1.5)
ax[0].axhline(popt[3], linestyle = 'dashed', color = 'r', linewidth = 1.5)
#ax[0].fill_between(I.E.values, I.kx.values - kx_int/2, I.kx.values + kx_int/2, color = 'pink', alpha = 0.5)
rect = (Rectangle((kx-kx_int/2, .5), kx_int, 3, linewidth=2.5,\
                         edgecolor='purple', facecolor='purple', alpha = 0.3))
if kx_int < 4:
    ax[0].add_patch(rect) #Add rectangle to plot

#kx_edc.plot(ax=ax[1], color = 'purple', alpha = 0.8)
ax[1].plot(kx_edc.loc[{"E":slice(0,3)}].E.values, g, color='grey',linestyle = 'solid',linewidth = 3)
ax[1].plot(kx_edc.loc[{"E":slice(0,3)}].E.values, g1, color='black',linestyle = 'dashed')
ax[1].plot(kx_edc.loc[{"E":slice(0,3)}].E.values, g2, color='red',linestyle = 'dashed')
kx_edc.plot(ax=ax[1], color = 'purple', alpha = 0.6)

ax[1].set_title(f"$E_b = {1000*Eb}$ meV")
ax[1].text(1.9, .8,  f"$\Delta$t = {delay} fs", size=16)
ax[1].text(1.8, .95,  f'$k_x$ = {kx:.1f} $\AA^{{-1}}$', size=16)

ax[1].set_xlim(0.5,3)
ax[1].set_ylim(0, 1.1)
ax[1].set_xlabel('$E - E_{VBM}, eV$', fontsize = 18)
ax[1].set_ylabel('Norm. Int.', fontsize = 18)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update(params)

fig.text(.02, 0.975, "(a)", fontsize = 20, fontweight = 'regular')
fig.text(.59, 0.975, "(b)", fontsize = 20, fontweight = 'regular')

fig.tight_layout()

if save_figure is True:
    fig.savefig(figure_file_name + '.'+ image_format, bbox_inches='tight', format=image_format) 
    
print(f"{popt[2]:.3f} +- {perr[2]:.3f}")
print(f"{popt[3]:.3f} +- {perr[3]:.3f}")
print(f"{1000*Eb:.3f} +- {1000*np.sqrt(perr[3]**2+perr[2]**2):.3f}")

#%% # Do the Excited State Fits for all Delay Times

# Momenta and Time Integration
(kx, ky), (kx_int, ky_int) = (0, 0), (4, 0.25) # Central (kx, ky) point and k-integration
delay_int = 40

# Fitting Paramaters
e1, e2 = 1.1, 3
p0 = [1, 0.3,  1.3, 2.1,  0.2, 0.2, 0] # Fitting params initial guess [amp, center, width, offset]
bnds = ((0.5, 0.1, 1.0, 1.5, 0.1, 0.1, 0), (1.5, 0.7, 1.5, 2.3, 0.9, 0.9, .3))

centers_CBM = np.zeros(len(I.delay))
centers_EX = np.zeros(len(I.delay))
Ebs = np.zeros(len(I.delay))

p_fits_excited = np.zeros((len(I.delay),7))
p_err_excited = np.zeros((len(I.delay),7))
p_err_eb = np.zeros((len(I.delay)))

n = len(I.delay.values)
for t in range(n):

    kx_frame = mpes.get_kx_E_frame(I_, ky, ky_int, I.delay.values[t], delay_int)

    kx_edc_i = kx_frame.loc[{"kx":slice(kx-kx_int/2,kx+kx_int/2)}].sum(dim="kx")
    kx_edc_i = kx_edc_i/np.max(kx_edc_i.loc[{"E":slice(0.8,3)}])
    
    try:
        popt, pcov = curve_fit(two_gaussians, kx_edc_i.loc[{"E":slice(e1,e2)}].E.values, kx_edc_i.loc[{"E":slice(e1,e2)}].values, p0, method=None, bounds = bnds)
    except ValueError:
        popt = [0,0,0,0]
   
    centers_EX[t] = popt[2]
    centers_CBM[t] = popt[3]
    Eb = round(popt[3] - popt[2],3)
    Ebs[t] = Eb
    perr = np.sqrt(np.diag(pcov))
    p_fits_excited[t,:] = popt
    
    p_err_excited[t,:] = perr 
    p_err_eb[t] = np.sqrt(perr[3]**2+perr[2]**2)

#%% Plot Excited State EDC Fits and Binding Energy

figure_file_name = 'Eb_delays_allkx'
save_figure = False
image_format = 'pdf'

fig, ax = plt.subplots(1, 3,  gridspec_kw={'width_ratios': [1.25,1.5,1.5], 'height_ratios':[1]})
fig.set_size_inches(14, 4, forward=False)
ax = ax.flatten()

kx_frame = mpes.get_kx_E_frame(I_diff, ky, ky_int, 500, delay_int)
kx_frame = kx_frame/np.max(kx_frame.loc[{"E":slice(0.8,3)}])

kx_edc = kx_frame.loc[{"kx":slice(kx-kx_int/2,kx+kx_int/2)}].mean(dim="kx")
kx_edc = kx_edc/np.max(kx_edc.loc[{"E":slice(0.8,3)}])

kx_edc.plot(ax=ax[0], color = 'darkgreen', label = 'Data')
ax[0].plot(kx_edc.loc[{"E":slice(0,3)}].E.values, g, color='grey',linestyle = 'solid', label = 'Fit', linewidth = 5)
ax[0].plot(kx_edc.loc[{"E":slice(0,3)}].E.values, g1, color='black',linestyle = 'dashed')
ax[0].plot(kx_edc.loc[{"E":slice(0,3)}].E.values, g2, color='red',linestyle = 'dashed')
#ax[0].set_title(f"$E_b = {1000*Eb}$ meV")
ax[0].set_title(f"$\Delta$t = {delay} fs")
ax[0].set_xticks(np.arange(0,3.2,.5))
for label in ax[0].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
ax[0].set_yticks(np.arange(0,1.5,0.25))
for label in ax[0].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
ax[0].set_xlim(0.5,3)
ax[0].set_ylim(0, 1.1)
ax[0].set_ylabel('Norm. Int.')
ax[0].set_xlabel('$E - E_{VBM}$, eV', color = 'black')
#ax[0].text(1.7, .8,  f"$\Delta$t = {delay} fs", size=16)
#ax[0].text(1.6, .95,  f'$k_x$ = {kx:.1f} $\AA^{{-1}}$', size=16)
#ax[0].text(1.6, .825,  f'$k_y$ = {kx:.1f} $\AA^{{-1}}$', size=16)
ax[0].set_aspect('auto')
ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax[0].legend(frameon=False)
# PLOT CBM and EX SHIFT DYNAMICS
#fig = plt.figure()
t = np.abs(I.delay.values-0).argmin() # Show only after 50 (?) fs
tt = np.abs(I.delay.values-100).argmin()
y_ex, y_ex_err = 1*(centers_EX[t:] - 0*centers_EX[-tt].mean()), 1*p_err_excited[t:,2]
y_cb, y_cb_err = 1*(centers_CBM[tt:]- 0*centers_CBM[-tt].mean()),  1*p_err_excited[tt:,3]

ax[1].plot(I.delay.values[t:], y_ex, color = 'black', label = 'Exciton')
ax[1].fill_between(I.delay.values[t:], y_ex - y_ex_err, y_ex + y_ex_err, color = 'grey', alpha = 0.5)
ax[1].set_xlim([0, I.delay.values[-1]])
#ax[1].set_ylim([1.1,2.3])
ax[1].set_xlabel('Delay, fs')
ax[1].set_xticks(np.arange(-400,1200,100))
for label in ax[1].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
ax[1].set_yticks(np.arange(0.5,2,0.05))
for label in ax[1].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
ax[1].set_xlim([0, I.delay.values[-1]])
ax[1].set_ylim([1.15,1.375])

ax2 = ax[1].twinx()
ax2.plot(I.delay.values[tt:], y_cb, color = 'red', label = 'CBM')
ax2.fill_between(I.delay.values[tt:], y_cb - y_cb_err, y_cb + y_cb_err, color = 'pink', alpha = 0.5)
ax2.set_ylim([2.075,2.25])
#ax[1].errorbar(I.delay.values[t:], 1*(centers_EX[t:]), yerr = p_err_excited[t:,2], marker = 'o', color = 'black', label = 'EX')
#ax[1].errorbar(I.delay.values[t:], 1*(centers_CBM[t:]), yerr = p_err_excited[t:,3], marker = 'o', color = 'red', label = 'CBM')
#ax[1].set_ylim([1.1,2.3])
ax[1].set_ylabel('$E_{Exciton}$, eV', color = 'black')
ax2.set_ylabel('$E_{CBM}$, eV', color = 'red')
#ax[1].set_title(f"From {round(I.delay.values[t])} fs")
ax[1].legend(frameon=False, loc = 'lower left')
ax2.legend(frameon=False, loc = 'lower right')
ax[1].arrow(250, 1.355, -80, 0, head_width = .025, width = 0.005, head_length = 40, fc='black', ec='black')
ax[1].arrow(900, 1.285, 80, 0, head_width = .025, width = 0.005, head_length = 40, fc='red', ec='red')
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax[2].plot(I.delay.values[tt:], 1000*Ebs[tt:], color = 'purple', label = '$E_{b}$')
ax[2].fill_between(I.delay.values[tt:], 1000*Ebs[tt:] - 1000*p_err_eb[tt:], 1000*Ebs[tt:] + 1000*p_err_eb[tt:], color = 'violet', alpha = 0.5)
ax[2].set_xlim([0, I.delay.values[-1]])
ax[2].set_ylim([700,900])
ax[2].set_xlabel('Delay, fs')
ax[2].set_ylabel('$E_{b}$, meV', color = 'black')
ax[2].legend(frameon=False, loc = 'lower right')
ax[2].set_xticks(np.arange(-400,1200,100))
for label in ax[2].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
ax[2].set_yticks(np.arange(600,1000,25))
for label in ax[2].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
ax[2].set_ylim([700, 900])    
ax[2].set_xlim([0, I.delay.values[-1]])

fig.text(.02, 0.975, "(a)", fontsize = 20, fontweight = 'regular')
fig.text(.32, 0.975, "(b)", fontsize = 20, fontweight = 'regular')
fig.text(0.69, 0.975, "(c)", fontsize = 20, fontweight = 'regular')

# # PLOT VBM PEAK WIDTH DYNAMICS
# ax2 = ax[1].twinx()
# ax2.plot(edc_gamma.delay.values, 1000*(p_fits_VBM[:,2] - 0*p_fits_VBM[0:10,2].mean()), color = 'maroon')
# #ax2.set_ylim([-75,50])
# ax2.set_ylabel('${VBM}$ Peak Width, meV', color = 'maroon')

fig.tight_layout()

if save_figure is True:
    fig.savefig(figure_file_name + '.'+ image_format, bbox_inches='tight', format=image_format) 

#%% Fourier Transform Analaysis

figure_file_name = 'FFT_PEAKS'
save_figure = False
image_format = 'pdf'

def monoexp(t, A, tau):
   return A * np.exp(-t/tau) * (t >= 0)
    
i_start = np.abs(I.delay.values-200).argmin()
waitingfreq = (1/2.99793E10)*np.fft.fftshift(np.fft.fftfreq(len(I.delay.values[i_start:]), d=20E-15));
delay_trunc = I.delay.values[i_start:]

pk, color = Ebs, 'purple'
#pk, color = centers_EX, 'black'
#pk, color = centers_CBM, 'crimson'
pk = pk[i_start:] - np.mean(pk[i_start:])

omega = 2*np.pi/136
#pk = np.sin(delay_trunc*omega)
trace = np.abs(np.fft.fftshift(np.fft.fft(pk)))

fig, ax = plt.subplots(1, 3)
fig.set_size_inches(12, 4, forward=False)
ax = ax.flatten()

# .plot(delay_trunc, centers_EX[i_start:]-np.mean(centers_EX[i_start:]), 'black')
# plt.plot(delay_trunc, centers_CBM[i_start:]-np.mean(centers_CBM[i_start:]), 'crimson')
# plt.plot(delay_trunc, Ebs[i_start:]-np.mean(Ebs[i_start:]), 'purple')
# plt.plot(delay_trunc, np.sin(delay_trunc*omega)*0.015, 'blue')

peaks = [centers_EX, centers_CBM, Ebs]
peak_labels = ['Exciton', 'CBM', '$E_{b}$']
colors = ['black', 'crimson', 'purple']
phonons = [110, 240, 350]

for i in [0,1,2]:
    
    pk = peaks[i]
    
    p0 = [1, 200]
    #popt, pcov = curve_fit(monoexp, I.delay.values[i_start], pk[i_start], p0, method=None)
    popt, pcov = curve_fit(monoexp, I.delay.values[i_start:], pk[i_start:], p0, method=None)

    pk_fit = monoexp(I.delay.values, *popt)
    trace = pk - pk_fit#- np.mean(pk[i_start:])
    trace = trace[i_start:]
    fft_trace = np.abs(np.fft.fftshift(np.fft.fft(trace)))

    ax[i].plot(waitingfreq, fft_trace, color = colors[i]) 
    ax[i].axvline(phonons[0], color = 'grey', linestyle = 'dashed') 
    ax[i].axvline(phonons[1], color = 'grey', linestyle = 'dashed') 
    ax[i].axvline(phonons[2], color = 'grey', linestyle = 'dashed')
    #plt.axvline(45, color = 'grey', linestyle = 'dashed')
    
    ax[i].set_xticks(np.arange(-1000,1000,100))
    for label in ax[i].xaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
    ax[i].set_yticks(np.arange(0,0.3,0.02))
    for label in ax[i].yaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
        
    ax[i].set_xlim(0,800)
    ax[i].set_ylim(0,0.2)
    ax[i].set_xlabel('Wavenumber, $cm^{-1}$')
    ax[i].set_ylabel('Amplitude')
    ax[i].set_title(f'{peak_labels[i]}')

fig.text(.02, .98, "(a)", fontsize = 20, fontweight = 'regular')
fig.text(.34, .98, "(b)", fontsize = 20, fontweight = 'regular')
fig.text(.67, .98, "(c)", fontsize = 20, fontweight = 'regular')

fig.tight_layout()

if save_figure is True:
    fig.savefig(figure_file_name + '.'+ image_format, bbox_inches='tight', format=image_format) 

#%% Plot and Fit EDCs of the VBM

%matplotlib inline

save_figure = False
figure_file_name = 'EDC_metis'
image_format = 'pdf'

#I_res = I.groupby_bins('delay', 50)
#I_res = I_res.rename({"delay_bins":"delay"})
#I_res = I_res/np.max(I_res)
I_res = I/np.max(I)

fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 1], 'height_ratios':[1, 1]})
fig.set_size_inches(10, 8, forward=False)
ax = ax.flatten()
### Plot EDCs at GAMMA vs time

(kx, ky), k_int = (-2*X, 0), 0.12
edc_gamma = I_res.loc[{"kx":slice(kx-k_int/2,kx+k_int/2), "ky":slice(ky-k_int/2,ky+k_int/2)}].sum(dim=("kx","ky"))
edc_gamma = edc_gamma/np.max(edc_gamma)

im = edc_gamma.plot.imshow(ax = ax[0], cmap = cmap_LTL, add_colorbar = False)
# cbar_ax = fig.add_axes([.51, 0.275, 0.025, 0.5])
cbar = fig.colorbar(im, ax = ax[0], ticks = [0,1])
cbar.ax.set_yticklabels(['min', 'max'])  # vertically oriented colorbar

ax[0].set_ylim([-1,1])
ax[0].set_xlim([edc_gamma.delay[0],edc_gamma.delay[-1]])
#plt.axhline(0, linestyle = 'dashed', color = 'black')
#plt.axvline(0, linestyle = 'dashed', color = 'black')
ax[0].set_xlabel('Delay, fs')
ax[0].set_ylabel('E - E$_{VBM}$, eV')
ax[0].set_title('EDCs at $\Gamma$')

delays, delay_int  = [-120, 0, 50, 100, 500], 30
colors = ['black', 'red', 'orange', 'purple', 'blue', 'green', 'grey']
n = len(pts)
colors = mpl.cm.inferno(np.linspace(0,.8,n))
                    
for i in range(n):
    edc = I_res.loc[{"kx":slice(kx-k_int/2,kx+k_int/2), "ky":slice(ky-k_int/2,ky+k_int/2), "delay":slice(delays[i]-delay_int/2,delays[i]+delay_int/2)}].sum(dim=("kx","ky","delay"))
    edc = edc/np.max(edc)
    
    e = edc.plot(ax = ax[1], color = colors[i], label = f"{pts[i]} fs")

#plt.legend(frameon = False)
ax[1].set_xlim([-1.5, 1]) 
#ax[1].set_ylim([0, 1.1])
ax[1].set_xlabel('E - E$_{VBM}$, eV')
ax[1].set_ylabel('Norm. Int.')
#ax[1].set_title('EDCs at $\Gamma$')
#ax[1].axvline(0, color = 'black', linestyle = 'dashed', linewidth = 0.5)
ax[1].legend(frameon=False, loc = 'upper left', fontsize = 11)
#ax[1].set_yscale('log')
#plt.ax[1].gca().set_aspect(2)

###################
##### Fit EDCs ####
###################

##### VBM #########
e1 = -.2
e2 = 0.6
p0 = [1, 0, .2, 0] # Fitting params initial guess [amp, center, width, offset]
bnds = ((0.5, -1, 0.0, 0), (1.5, 0.5, 1, .5))

centers_VBM = np.zeros(len(I.delay))
p_fits_VBM = np.zeros((len(I.delay),4))
p_err_VBM = np.zeros((len(I.delay),2))

n = len(I.delay)
for t in np.arange(n):
    edc_i = edc_gamma.loc[{"E":slice(e1,e2)}][:,t].values
    edc_i = edc_i/np.max(edc_i)
    
    try:
        popt, pcov = curve_fit(gaussian, edc_gamma.loc[{"E":slice(e1,e2)}].E.values, edc_i, p0, method=None, bounds = bnds)
    except ValueError:
        popt = [0,0,0,0]
        
    centers_VBM[t] = popt[1]
    p_fits_VBM[t,:] = popt
    perr = np.sqrt(np.diag(pcov))
    p_err_VBM[t,:] = perr[1:2+1]

# VBM FIT TESTS FOR ONE POINT
t = 9
gauss_test = gaussian(edc_gamma.E.values, *p_fits_VBM[t,:])
ax[2].plot(edc_gamma.E.values, edc_gamma.loc[{"delay":slice(-120-10,-120+10)}]/edc_gamma.loc[{"delay":slice(-120-10,-120+10),"E":slice(e1,e2)}].max(), color = 'black', label = 'Data')
ax[2].plot(edc_gamma.E.values, gauss_test, linestyle = 'dashed', color = 'grey', label = 'Fit')
#plt.axvline(trunc_e, linestyle = 'dashed', color = 'black')
ax[2].set_xlim([-2,1.5])
ax[2].set_xlabel('E - E$_{VBM}$, eV')
ax[2].set_ylabel('Norm. Int.')
ax[2].set_title(f'$\Delta$t = {-120} fs')
ax[2].legend(frameon=False, loc = 'upper left')
#ax[0].axvline(0, linestyle = 'dashed', color = 'grey')
#ax[0].axvline(e2, linestyle = 'dashed', color = 'black')

# PLOT VBM SHIFT DYNAMICS

t = 39 # Show only after 50 (?) fs
y_vb, y_vb_err = 1000*(p_fits_VBM[:,1] - p_fits_VBM[0:10,1].mean()), 1000*p_err_VBM[:,0]
y_vb_w, y_vb_w_err = 1000*(p_fits_VBM[:,2]),  1000*p_err_VBM[:,1]

ax[3].plot(I.delay.values, y_vb, color = 'navy', label = '$\Delta E_{VBM}$')
ax[3].fill_between(I.delay.values, y_vb - y_vb_err, y_vb + y_vb_err, color = 'navy', alpha = 0.5)

ax[3].set_xlim([edc_gamma.delay.values[1], edc_gamma.delay.values[-1]])
ax[3].set_ylim([-30,15])
ax[3].set_xlabel('Delay, fs')
ax[3].set_ylabel('$\Delta E_{VBM}$, meV', color = 'navy')
ax[3].set_title('Peak Dynamics')
ax[3].legend(frameon=False, loc = 'upper left')
#ax[3].arrow(250, 1.3, -75, 0, head_width = .025, width = 0.005, head_length = 40, fc='black', ec='navy')
#ax[3].arrow(650, 1.12, 75, 0, head_width = .025, width = 0.005, head_length = 40, fc='red', ec='maroon')

# PLOT VBM PEAK WIDTH DYNAMICS
ax2 = ax[3].twinx()
ax2.plot(I.delay.values, y_vb_w, color = 'maroon', label = '$\sigma_{VBM}$')
ax2.fill_between(I.delay.values, y_vb_w - y_vb_w_err, y_vb_w + y_vb_w_err, color = 'maroon', alpha = 0.5)
ax2.set_ylim([125,275])
ax2.legend(frameon=False, loc = 'upper right')
ax2.set_ylabel('${VBM}$ Peak Width, meV', color = 'maroon')

# #ax[1].plot(edc_gamma.E.values, edc_gamma[:,t].values/edc_gamma.loc[{"E":slice(e1,e2)}][:,t].values.max(), color = 'pink')
# ax[1].plot(edc_gamma.E.values, gauss_test, linestyle = 'dashed', color = 'grey')
# #plt.axvline(trunc_e, linestyle = 'dashed', color = 'black')
# ax[1].set_xlim([-2,1.5])
# ax[1].set_xlabel('Energy, eV')
# ax[1].set_ylabel('Norm. Int, arb. u.')
# #plt.gca().set_aspect(3)

# # PLOT VBM SHIFT DYNAMICS
# #fig = plt.figure()
# ax[2].plot(edc_gamma.delay.values, 1000*(p_fits_VBM[:,1] - p_fits_VBM[0:10,1].mean()), color = 'grey')
# ax[2].set_xlim([edc_gamma.delay.values[1], edc_gamma.delay.values[-1]])
# ax[2].set_ylim([-30,20])
# ax[2].set_xlabel('Delay, fs')
# ax[2].set_ylabel('Energy Shift, meV')
# #plt.axhline(0, linestyle = 'dashed', color = 'black')
# #plt.axvline(0, linestyle = 'dashed', color = 'black')

# # PLOT VBM PEAK WIDTH DYNAMICS
# ax2 = ax[2].twinx()
# ax2.plot(edc_gamma.delay.values, 1000*(p_fits_VBM[:,2] - 0*p_fits_VBM[0:10,2].mean()), color = 'pink')
# #ax2.set_ylim([-75,50])
# ax2.set_ylabel('Energy Width Shift, meV')

fig.text(.02, 1, "(a)", fontsize = 20, fontweight = 'regular')
fig.text(.5, 1, "(b)", fontsize = 20, fontweight = 'regular')
fig.text(.02, 0.5, "(c)", fontsize = 20, fontweight = 'regular')
fig.text(.5, 0.5, "(d)", fontsize = 20, fontweight = 'regular')
fig.tight_layout()

if save_figure is True:
    fig.savefig(figure_file_name + '.'+ image_format, bbox_inches='tight', format=image_format) 


#%% Plot Difference of MMs

%matplotlib inline

save_figure = False
figure_file_name = 'MM_DIFFERENCE'

E, E_int  = [.5, .1], .2
delays = [-160, 1200] #Integration range for delays

MM_1  = get_momentum_map(I, E[0], E_int, [100, 2000])
MM_2  = get_momentum_map(I, E[1], E_int, [100, 2000])
diff_MM = MM_2 - MM_1

########################
%matplotlib inline
fig, ax = plt.subplots(1, 3, squeeze = False)
ax = ax.flatten()
fig.set_size_inches(8, 5, forward=False)
plt.gcf().set_dpi(300)

extent =  extent=[ax_kx[0],ax_kx[-1],ax_ky[0],ax_ky[-1]]
im = ax[0].imshow(np.transpose(MM_1), origin='lower', cmap=cmap_plot, clim=[0,1], interpolation='none', extent=extent) #kx, ky, t
im = ax[1].imshow(np.transpose(MM_2), origin='lower', cmap=cmap_plot, clim=[0,1], interpolation='none', extent=extent) #kx, ky, t
im = ax[2].imshow(np.transpose(diff_MM), origin='lower', cmap='seismic', clim=[0,1], interpolation='none', extent=extent) #kx, ky, t

for i in np.arange(3):
    ax[i].set_aspect(1)    
    ax[i].set_xlim(-2,2)
    ax[i].set_ylim(-2,2)
    ax[i].set_xticks(np.arange(-2,2.2,1))
    for label in ax[i].xaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
    ax[i].set_yticks(np.arange(-2,2.1,1))
    for label in ax[i].yaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
    #ax[0].set_box_aspect(1)
    ax[i].set_xlabel('$k_x$, $A^{-1}$', fontsize = 14)
    ax[i].set_ylabel('$k_y$, $A^{-1}$', fontsize = 14)
    ax[i].tick_params(axis='both', labelsize=12)

ax[0].set_title('$E$ = ' + str((E[0])) + ' eV', fontsize = 14)
ax[1].set_title('$E$ = ' + str((E[1])) + ' eV', fontsize = 14)
ax[2].set_title('$\\Delta$MM ', fontsize = 14)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([1, 0.325, 0.025, 0.35])
fig.colorbar(im, cax=cbar_ax, ticks = [-1,0,1])

fig.tight_layout()
plt.show()

if save_figure is True:
    fig.savefig((figure_file_name +'.svg'), format='svg')
