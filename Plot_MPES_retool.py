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
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
from skimage.draw import disk
from scipy.optimize import curve_fit
from scipy.optimize import curve_fit
from scipy import signal
from scipy.fft import fft, fftshift
from obspy.imaging.cm import viridis_white
import xarray as xr

from Loader import DataLoader
from main import main
from Manager import DataHandler, FigureHandler, PlotHandler, ValueHandler, SliderManager, EventHandler, CheckButtonManager, ClickButtonManager

import mpes

#%% Specifiy filename of h5 file in your path.
# Include manual energy and time delay offsets for the axes, if required.

data_path = 'path_to_your_data'
filename = 'your_file_name.h5'

data_path = 'R:\Lawson\Data\metis'
data_path = '/Users/lawsonlloyd/Desktop/Data/'

#filename, offsets = 'Scan682_binned.h5', [0,0]
filename, offsets = 'Scan162_binned_100x100x200x150_CrSBr_RT_750fs_New_2.h5', [0.2, -90] # Axis Offsets: [Energy (eV), delay (fs)]
filename, offsets = 'Scan162_RT_120x120x115x50_binned.h5', [0.8467, -120]
filename, offsets = 'Scan803_binned.h5', [0.2, -102]

#filename, offsets = 'Scan163_binned_100x100x200x150_CrSBr_120K_1000fs_rebinned_distCorrected_New_2.h5', [0, 100]
#filename, offsets = 'Scan188_binned_100x100x200x155_CrSBr_120K_1000fs_rebinned_ChargeingCorrected_DistCorrected.h5', [0.05, 65]
#filename, offsets = 'Scan188_binned_100x100x200x155_CrSBr_120K_1000fs_rebinned_ChargeingCorrected_DistCorrected.h5', [0.05, 65]

#filename, offsets = 'Scan62_binned_200x200x300_CrSBr_RT_Static_rebinned.h5', [0,0]


#%% Load the data and axes information

data_loader = DataLoader(data_path + '//' + filename)
value_manager =  ValueHandler()

#I, ax_kx, ax_ky, ax_E, ax_delay = data_loader.load()
#data_handler = DataHandler(value_manager, I, ax_kx, ax_ky, ax_E, ax_delay, *offsets)

I = data_loader.load()
#data_handler = DataHandler(value_manager, I, ax_kx, ax_ky, ax_E, ax_delay, *offsets)

I = I.assign_coords(E=(I.E-offsets[0]))
I = I.assign_coords(delay=(I.delay-offsets[1]))

#%% #This sets the plots to plot in the IDE window

%matplotlib inline

#%% Define E = 0 wrt VBM

fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1], 'height_ratios':[1]})
fig.set_size_inches(10, 4, forward=False)
ax = ax.flatten()

figure_file_name = 'vbm'
save_figure = True

### Plot EDCs at GAMMA vs time

I_res = I/np.max(I)

a, b = 3.508, 4.763 # CrSBr values
X, Y = np.pi/a, np.pi/b
x, y = -2*X, 0*Y

(kx, ky), k_int = (x, y), 0.1
edc_gamma = I_res.loc[{"kx":slice(kx-k_int/2,kx+k_int/2), "ky":slice(ky-k_int/2,ky+k_int/2)}].sum(dim=("kx","ky"))
edc_gamma = edc_gamma/np.max(edc_gamma)

E_MM = 0
im = I_res.loc[{"E":slice(E_MM-0.05,E_MM+0.05), "delay":slice(-300,-100)}].mean(dim=("E","delay")).T.plot.imshow(ax = ax[0], cmap = cmap_LTL, add_colorbar = False)
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
                    
edc = I_res.loc[{"kx":slice(kx-k_int/2,kx+k_int/2), "ky":slice(ky-k_int/2,ky+k_int/2), "delay":slice(-120-20,-120+20)}].mean(dim=("kx","ky","delay"))
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

#%%
edc_pos = (I_res.loc[{"delay":slice(200,1100)}].mean(dim=("kx","ky","delay")))
edc_neg = (I_res.loc[{"delay":slice(-200,-150)}].mean(dim=("kx","ky","delay")))

edc_pos = edc_pos/np.max(edc_neg.loc[{"E":slice(-0.5,0.5)}])
edc_neg = edc_neg/np.max(edc_neg.loc[{"E":slice(-0.5,0.5)}])
edc_diff = edc_pos - edc_neg
edc_diff = edc_diff/np.max(edc_diff.loc[{"E":slice(1,3)}])

edc_pos = np.log(edc_pos)
edc_neg = np.log(edc_neg)
#edc_diff = np.log(edc_pos - edc_neg)

plt.plot(I_res.E, edc_pos , color = 'green', label = 't > 200 fs')
plt.plot(I_res.E, edc_neg, color = 'grey', label = 'Neg Delays') 

plt.xlim(-.5,2.75)
plt.xlabel('Energy, eV')
plt.ylabel('Int, logscale')
plt.legend(frameon=False)
a = plt.twinx()

a.plot(I_res.E.loc[{"E":slice(0.75,3)}], edc_diff.loc[{"E":slice(0.75,3)}], color = 'purple', label = 'Difference')
plt.ylim(-.1,1.2)

#%% Define t0 from Exciton Rise

from scipy.special import erf

fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1], 'height_ratios':[1]})
fig.set_size_inches(10, 4, forward=False)
ax = ax.flatten()

figure_file_name = 'define_t0'
save_figure = False

### Plot EDCs at GAMMA vs time

(kx, ky), (kx_int, ky_int) = (0, 0), (3.8, 2)
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

print(round(popt[0],3), round(popt[1],1))
print(round(perr[0],3), round(perr[1],1))

if save_figure is True:
    fig.savefig((figure_file_name +'.svg'), format='svg')

#%% Plot Momentum Maps at Constant Energy

E, E_int = [1.25, 1.25, 1.25, 2.05, 2.05, 2.05], 0.2 # Energies and Total Energy Integration Window to Plot MMs
delays, delay_int = [0, 100, 500, 0, 100, 500], 100 #Integration range for delays

#######################

%matplotlib inline

figure_file_name = f'MMs_Delays' 
save_figure = True

#cmap_plot = viridis_white
cmap_plot = cmap_LTL
#cmap_plot = 'turbo'

fig, ax = plt.subplots(2, 3, squeeze = False)
fig.set_size_inches(8, 5, forward=False)
plt.gcf().set_dpi(600)
ax = ax.flatten()

frame_norms = []
for i in np.arange(len(E)):

    norm_frame_ex = I_res.loc[{"E":slice(1.25-E_int/2,1.25+E_int/2)}].mean(dim="E").max()
    norm_frame_cbm = I_res.loc[{"E":slice(2.05-E_int/2,2.05+E_int/2)}].mean(dim="E").max()

    frame = mpes.get_momentum_map(I_res, E[i], E_int, delays[i], delay_int)
    frame = frame/np.max(frame)
    
    im = frame.plot.imshow(ax = ax[i], vmin = 0, vmax = 1, cmap = cmap_plot, add_colorbar=False)
    ax[i].set_aspect(1)
    ax[i].set_xlim(-2,2)
    ax[i].set_ylim(-2,2)
    
    ax[i].set_xticks(np.arange(-2,2.2,1))
    for label in ax[i].xaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
       
    ax[i].set_yticks(np.arange(-2,2.1,1))
    for label in ax[i].yaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
    ax[i].set_xlabel('$k_x$, $\AA^{-1}$', fontsize = 18)
    ax[i].set_ylabel('$k_y$, $\AA^{-1}$', fontsize = 18)
    ax[i].set_title(f"$\Delta$t = {delays[i]} fs", fontsize = 18)
    ax[i].tick_params(axis='both', labelsize=16)
    ax[i].text(-1.9, 1.5,  f"E = {E[i]:.2f} eV", size=14)

cbar_ax = fig.add_axes([1, 0.325, 0.025, 0.35])
cbar = fig.colorbar(im, cax=cbar_ax, ticks = [0,frame.max()])
cbar.ax.set_yticklabels(['min', 'max'])  # vertically oriented colorbar

plt.rcParams['svg.fonttype'] = 'none'
fig.tight_layout()
 
#fig = mpes_helper.plot_momentum_maps(I, E, E_int, delays, delay_int, cmap_plot)

#add_bz(fig, x, y)

# y, x = np.pi/4.76, np.pi/3.52

# for r in np.arange(1,2,2):
    
#     rect = (Rectangle((0-r*x, 0-y), 2*x, 2*y, linewidth=1.5,\
#                          edgecolor='k', facecolor='None'))
#     #rect2 = (Rectangle((0-r*x, 0+y), 2*x, 2*y, linewidth=1.5,\
#                           #   edgecolor='k', facecolor='None'))
#     fig.axes[0].add_patch(rect)

# fig.axes[0].plot(0,0, 'ok', markersize = 4)
# fig.axes[0].plot(0,2*y, 'ok', markersize = 4)
# fig.axes[0].plot(0,-2*y, 'ok', markersize = 4)
# fig.axes[0].plot(2*x, 0,  'ok', markersize = 4)
# fig.axes[0].plot(-2*x, 0, 'ok', markersize = 4)

# fig.axes[0].plot(x, 0, 'k', marker = '_', markersize = 6)
# fig.axes[0].plot(0, y, 'k', marker = '|', markersize = 8)

#fig.axes[0].set_xlim(-3,3)

if save_figure is True:
    fig.savefig((figure_file_name +'.svg'), format='svg')


#%% Plot Dynamics: Extract Traces At Different Energies and Momenta

save_figure = False
figure_file_name = 'k-integrated 4Panel'

E_trace, E_int = [0.85, 1.4, 1.9], .1 # Energies for Plotting Time Traces ; 1st Energy for MM
(kx, ky), (kx_int, ky_int) = (0, 0), (4, 3) # Central (kx, ky) point and k-integration

colors = ['black', 'red', 'purple'] #colors for plotting the traces

subtract_neg, neg_delays = True, [-110,-70] #If you want to subtract negative time delay baseline
norm_trace = True

E_MM ,E_MM_int = 1.4, 0.2

delay, delay_int = 500, 600 #kx frame

Ein = .65 #Enhance excited states above this Energy, eV
energy_limits = [0, 2.7] # Energy Y Limits for Plotting Panels 3 and 4

#######################
### Do the Plotting ###
#######################

i = 0
frame = get_momentum_map(I, E_MM, E_MM_int, 500, 2000)

fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 1.5], 'height_ratios':[1, 1]})
fig.set_size_inches(8, 6, forward=False)
ax = ax.flatten()

plot_symmetry_points = False

### FIRST PLOT: MM of the First Energy
im = frame.plot.imshow(ax = ax[i], clim = None, cmap = cmap_plot, add_colorbar=False)
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
rect = (Rectangle((kx-kx_int/2, ky-ky_int/2), kx_int, ky_int, linewidth=3.5,\
                         edgecolor='purple', facecolor='purple', alpha = 0.25))
if kx_int < 4:
    ax[0].add_patch(rect) #Add rectangle to plot

### kx Frame
kx_frame = get_kx_E_frame(I, ky, ky_int, delay, delay_int)
kx_frame = kx_frame/np.max(kx_frame)
kx_frame = kx_frame/np.max(kx_frame.loc[{"E":slice(Ein,3)}])

im2 = kx_frame.T.plot.imshow(ax=ax[2], cmap=cmap_plot, add_colorbar=False, vmin=0, vmax=1) #kx, ky, t
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

### SECOND PLOT: WATERFALL
I_norm = I/np.max(I)
I_neg_mean = I_norm.loc[{"delay":slice(-130,-95)}].mean(axis=(3))

#I_norm = I_norm.loc[{"kx":slice(kx-kx_int/2, kx+kx_int/2), "ky":slice(ky-ky_int/2, ky+ky_int/2)}].sum(dim=("kx","ky"))
#I_neg_mean = I_neg_mean.loc[{"kx":slice(kx-kx_int/2, kx+kx_int/2), "ky":slice(ky-ky_int/2, ky+ky_int/2)}].sum(dim=("kx","ky"))

I_diff = I_norm - I_neg_mean
I_diff = I_diff.loc[{"kx":slice(kx-kx_int/2, kx+kx_int/2), "ky":slice(ky-ky_int/2, ky+ky_int/2)}].mean(dim=("kx","ky"))
I_diff = I_diff/np.max(I_diff)

I_k_int = I_norm.loc[{"kx":slice(kx-kx_int/2, kx+kx_int/2), "ky":slice(ky-ky_int/2, ky+ky_int/2)}].mean(dim=("kx","ky"))
I_k_int = I_k_int/np.max(I_k_int)

I3 = enhance_features(I_diff, Ein, factor = 0, norm = True)

color_max = 1
#waterfall = I3.plot(ax = ax[1], vmin = -1, vmax = 1, cmap = cmap_LTL)
waterfall = I3.plot(ax = ax[3], vmin = 0, vmax = 1, cmap = cmap_LTL, add_colorbar=False)
#waterfall = ax[1].imshow(diff_ang, clim = clim, origin = 'lower', cmap = cmap_LTL, extent=[ax_delay_offset[0], ax_delay_offset[-1], ax_E_offset[0], ax_E_offset[-1]])
ax[3].set_xlabel('Delay, fs', fontsize = 18)
ax[3].set_ylabel('E - E$_{VBM}$, eV', fontsize = 18)
ax[3].set_yticks(np.arange(-1,3.5,0.25))
ax[3].set_xlim(I.delay[1],I.delay[-1])
ax[3].set_ylim(energy_limits)
ax[3].set_title('k-Integrated')
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
for i in np.arange(len(E_trace)):
    
    trace = mpes.get_time_trace(I, E_trace[i], E_int, (kx, ky), (kx_int, ky_int), norm_trace, subtract_neg, neg_delays)
    trace_norms.append(np.max(trace))

    if i == 0:
        trace = trace/np.max(trace)
    elif i == 1:
        trace = trace/trace_norms[0]
        
    trace.plot(ax = ax[1], color = colors[i], label = str(E_trace[i]) + ' eV')
    ax[3].axhline(E_trace[i], linestyle = 'dashed', color = colors[i], linewidth = 1.5)

#   ax[1].axhline(E_trace[i]-E_int/2, linestyle = 'dashed', color = colors[i], linewidth = 1.5)
#   ax[1].axhline(E_trace[i]+E_int/2, linestyle = 'dashed', color = colors[i], linewidth = 1.5)
    
    rect2 = (Rectangle((kx-kx_int/2, E_trace[i]-E_int/2), kx_int, E_int , linewidth=.5,\
                         edgecolor=colors[i], facecolor=colors[i], alpha = 0.5))
#    if kx_int < 4:
    ax[2].add_patch(rect2) #Add rectangle to plot
#    ax[2].axhline(E_trace[i], linestyle = 'dashed', color = colors[i], linewidth = 1.5)


ax[1].set_xlim(I.delay[1], I.delay[-1])

if norm_trace is True:
    ax[1].set_ylim(-0.1, 1.1)
else:
    ax[1].set_ylim(-0.1, 1.1*np.max(1))
    
ax[1].set_ylabel('Norm. Int.', fontsize = 18)
ax[1].set_xlabel('Delay, fs', fontsize = 18)
ax[1].legend(frameon = False)
ax[1].set_title('Delay Traces')

params = {'lines.linewidth' : 2.5, 'axes.linewidth' : 2, 'axes.labelsize' : 20, 
              'xtick.labelsize' : 16, 'ytick.labelsize' : 16, 'axes.titlesize' : 20, 'legend.fontsize' : 16}
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update(params)

fig.tight_layout()
#ax[1].set_tick_params(axis='both', labelsize=16)
#plt.gca().set_aspect(200)

if save_figure is True:
    fig.savefig((figure_file_name +'.svg'), format='svg', dpi =300)


#%% Plot Dynamics: Extract Traces At Different Energies and Momenta

save_figure = False
figure_file_name = 'distinct_K_Points'

subtract_neg, neg_delays = True, [-110,-70] #If you want to subtract negative time delay baseline
norm_trace = False

#######################
### Do the Plotting ###
#######################

i = 0
E_MM = 1.3
frame = get_momentum_map(I, E_MM, 0.1, 500, 2000)

fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 1.5], 'height_ratios':[1, 1]})
fig.set_size_inches(8, 6, forward=False)
ax = ax.flatten()

### FIRST PLOT: MM of the First Energy
im = frame.plot.imshow(ax = ax[i], clim = None, cmap = cmap_plot, add_colorbar=False)
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

### kx Frame
energy_limits = [0.8, 3]
delay, delay_int = 250, 200
kx_frame = get_kx_E_frame(I, ky, ky_int, delay, delay_int)
kx_frame = kx_frame/np.max(kx_frame.loc[{"E":slice(0.8,3)}])

im2 = kx_frame.T.plot.imshow(ax=ax[2], cmap=cmap_plot, add_colorbar=False, vmin=0, vmax=1) #kx, ky, t
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
ax[2].text(-1.9, 2.7,  f"$\Delta$t = {delay} $\pm$ {delay_int/2:.0f} fs", size=14)
ax[2].axhline(Ein, linestyle = 'dashed', color = 'black', linewidth = 1)
ax[2].axvline(0, linestyle = 'dashed', color = 'pink', linewidth = 1.5)
ax[2].axvline(X, linestyle = 'dashed', color = 'pink', linewidth = 1.5)
ax[2].axvline(-X, linestyle = 'dashed', color = 'pink', linewidth = 1.5)
ax[2].axvline(2*X, linestyle = 'dashed', color = 'pink', linewidth = 1.5)
ax[2].axvline(-2*X, linestyle = 'dashed', color = 'pink', linewidth = 1.5)
#ax[0].fill_between(I.E.values, I.kx.values - kx_int/2, I.kx.values + kx_int/2, color = 'pink', alpha = 0.5)

ax[2].set_aspect("auto")

### SECOND PLOT: WATERFALL

#ax[3].text(500, 2.4, f'$k_y$ = {ky} $\pm$ {ky_int/2} $\AA^{{-1}}$', fontsize = 14)

#fig.colorbar(waterfall, ax=ax[1], shrink = 0.8, ticks = [0, color_max/2, color_max])

### THIRD PLOT: DELAY TRACES (Angle-Integrated)
E, E_int = 1.25, 0.2
(kx, ky), (kx_int, ky_int) = ((-2*X, -X, 0.0, X, 2*X), 0), (.25, .25) # Central (kx, ky) point and k-integration
colors = ['crimson', 'purple', 'darkblue', 'dodgerblue', 'lightcoral'] #colors for plotting the traces
k_point_label = ['$\Gamma_{-1,0}$', '$-X$', '$\Gamma_{0}$', '$+X$', '$\Gamma_{1,0}$']

trace_norms = []
for i in np.arange(len(kx)):
    
    print(kx[i])
    trace = mpes.get_time_trace(I, E, E_int, (kx[i], ky), (kx_int, ky_int), norm_trace, subtract_neg, neg_delays)
    trace_norms.append(np.max(trace))
    trace = trace/np.max(trace)
        
    trace.plot(ax = ax[1], color = colors[i], label = k_point_label[i])

#   ax[1].axhline(E_trace[i]-E_int/2, linestyle = 'dashed', color = colors[i], linewidth = 1.5)
#   ax[1].axhline(E_trace[i]+E_int/2, linestyle = 'dashed', color = colors[i], linewidth = 1.5)
    
    rect2 = (Rectangle((kx[i]-kx_int/2, E-E_int/2), kx_int, E_int , linewidth=.5,\
                         edgecolor=colors[i], facecolor=colors[i], alpha = 0.75))
#    if kx_int < 4:
    ax[2].add_patch(rect2) #Add rectangle to plot
#    ax[2].axhline(E_trace[i], linestyle = 'dashed', color = colors[i], linewidth = 1.5)
    rect = (Rectangle((kx[i]-kx_int/2, ky-ky_int/2), kx_int, ky_int, linewidth=.5,\
                             edgecolor='purple', facecolor='purple', alpha = 0.75))
    if kx_int < 4:
        ax[0].add_patch(rect) #Add rectangle to plot

ax[1].set_xlim(I.delay[1], I.delay[-1])

if norm_trace is True:
    ax[1].set_ylim(-0.5, 1.1)
else:
    ax[1].set_ylim(-0.1, 1.1*np.max(1))
    
ax[1].set_ylabel('Norm. Int.', fontsize = 18)
ax[1].set_xlabel('Delay, fs', fontsize = 18)
ax[1].legend(frameon = False, bbox_to_anchor=(1, 1), fontsize = 12)
ax[1].set_title(f"Exciton, $E$ = {E} eV")

### for the CBM

E, E_int = 2.05, 0.2

trace_norms = []
for i in np.arange(len(kx)):
    
    trace = mpes.get_time_trace(I, E, E_int, (kx[i], ky), (kx_int, ky_int), norm_trace, subtract_neg, neg_delays)
    trace_norms.append(np.max(trace))
    trace = trace/np.max(trace)
        
    trace.plot(ax = ax[3], color = colors[i], label = k_point_label[i])

#   ax[1].axhline(E_trace[i]-E_int/2, linestyle = 'dashed', color = colors[i], linewidth = 1.5)
#   ax[1].axhline(E_trace[i]+E_int/2, linestyle = 'dashed', color = colors[i], linewidth = 1.5)
    
    rect2 = (Rectangle((kx[i]-kx_int/2, E-E_int/2), kx_int, E_int , linewidth=.5,\
                         edgecolor=colors[i], facecolor=colors[i], alpha = 0.75))
#    if kx_int < 4:
    ax[2].add_patch(rect2) #Add rectangle to plot
#    ax[2].axhline(E_trace[i], linestyle = 'dashed', color = colors[i], linewidth = 1.5)


ax[3].set_xlim(I.delay[1], I.delay[-1])

if norm_trace is True:
    ax[3].set_ylim(-0.5, 1.1)
else:
    ax[3].set_ylim(-0.1, 1.1*np.max(1))
    
ax[3].set_ylabel('Norm. Int.', fontsize = 18)
ax[3].set_xlabel('Delay, fs', fontsize = 18)
#ax[3].legend(frameon = False, fontsize = 12)
ax[3].set_title(f"CBM, $E$ = {E} eV")

params = {'lines.linewidth' : 2.5, 'axes.linewidth' : 2, 'axes.labelsize' : 20, 
              'xtick.labelsize' : 16, 'ytick.labelsize' : 16, 'axes.titlesize' : 20, 'legend.fontsize' : 16}
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update(params)

fig.tight_layout()
#ax[1].set_tick_params(axis='both', labelsize=16)
#plt.gca().set_aspect(200)

if save_figure is True:
    fig.savefig((figure_file_name +'.svg'), format='svg', dpi =300)
    
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
I_res_enh = enhance_features(I_res, Ein, factor = 0, norm = True)

frame = get_momentum_map(I, E_MM, E_int, delay, delay_int)

#Norm to t0 Frame
frame_t0 = get_kx_E_frame(I_res_enh, ky, k_int, 25, delay_int)
frame_t0_2 = get_ky_E_frame(I_res_enh, ky, k_int, 25, delay_int)

n = [frame_t0.loc[{"E":slice(-3,Ein)}].max().values, frame_t0.loc[{"E":slice(Ein,5)}].max().values]
n2 = [frame_t0_2.loc[{"E":slice(-3,Ein)}].max().values, frame_t0_2.loc[{"E":slice(Ein,5)}].max().values]

frame2 = get_kx_E_frame(I_res_enh, ky, k_int, delay, delay_int)
frame3 = get_ky_E_frame(I_res_enh, kx, k_int, delay, delay_int)

Ein = 0.75

f23 = enhance_features(frame2, Ein, factor = n, norm = True)
f33 = enhance_features(frame3, Ein, factor = n2, norm = True)
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
    fig.savefig((figure_file_name +'.svg'), format='svg')

#%%

from lmfit import Parameters, minimize, report_fit

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


#%% TEST: CBM EDC Fitting to Extract EXCITON Binding Energy and Peak Positions

save_figure = False
figure_name = 'kx_excited'

E_trace, E_int = [1.35, 2.05], .1 # Energies for Plotting Time Traces ; 1st Energy for MM
(kx, ky), (kx_int, ky_int) = (0, 0), (.5, 0.2) # Central (kx, ky) point and k-integration
delay, delay_int = 500, 500

kx_frame = get_kx_E_frame(I, ky, ky_int, delay, delay_int)
kx_frame = kx_frame/np.max(kx_frame.loc[{"E":slice(0.8,3)}])

kx_edc = kx_frame.loc[{"kx":slice(kx-kx_int/2,kx+kx_int/2)}].mean(dim="kx")
kx_edc = kx_edc/np.max(kx_edc.loc[{"E":slice(0.8,3)}])

##### X and CBM ####
e1 = 1.15
e2 = 3
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
ax[0].set_title(f'$k_y$ = {ky} $\pm$ {ky_int/2} $\AA^{{-1}}$', fontsize = 18)
ax[0].tick_params(axis='both', labelsize=16)
ax[0].set_xlim(-2,2)
ax[0].set_ylim(0.8,3)
ax[0].text(-1.9, 2.7,  f"$\Delta$t = {delay} fs", size=16)
ax[0].axhline(Ein, linestyle = 'dashed', color = 'black', linewidth = 1)
ax[0].axvline(0, linestyle = 'dashed', color = 'pink', linewidth = 1.5)
ax[0].axvline(X, linestyle = 'dashed', color = 'pink', linewidth = 1.5)
ax[0].axvline(-X, linestyle = 'dashed', color = 'pink', linewidth = 1.5)
ax[0].axvline(2*X, linestyle = 'dashed', color = 'pink', linewidth = 1.5)
ax[0].axvline(-2*X, linestyle = 'dashed', color = 'pink', linewidth = 1.5)
ax[0].axhline(popt[2], linestyle = 'dashed', color = 'k', linewidth = 1.5)
ax[0].axhline(popt[3], linestyle = 'dashed', color = 'r', linewidth = 1.5)
#ax[0].fill_between(I.E.values, I.kx.values - kx_int/2, I.kx.values + kx_int/2, color = 'pink', alpha = 0.5)
rect = (Rectangle((kx-kx_int/2, .5), kx_int, 3, linewidth=2.5,\
                         edgecolor='purple', facecolor='purple', alpha = 0.3))
if kx_int < 4:
    ax[0].add_patch(rect) #Add rectangle to plot

#kx_edc.plot(ax=ax[1], color = 'purple', alpha = 0.8)
ax[1].plot(kx_edc.loc[{"E":slice(0,3)}].E.values, g1, color='black',linestyle = 'dashed')
ax[1].plot(kx_edc.loc[{"E":slice(0,3)}].E.values, g2, color='red',linestyle = 'dashed')
ax[1].plot(kx_edc.loc[{"E":slice(0,3)}].E.values, g, color='grey',linestyle = 'solid')
kx_edc.plot(ax=ax[1], color = 'purple', alpha = 0.8)

ax[1].set_title(f"$E_b = {1000*Eb}$ meV")
ax[1].text(1.9, .8,  f"$\Delta$t = {delay} fs", size=16)
ax[1].text(1.4, .95,  f'$k_x$ = {kx:.1f} $\pm$ {kx_int/2} $\AA^{{-1}}$', size=14)

ax[1].set_xlim(0.5,3)
ax[1].set_ylim(0, 1.1)
ax[1].set_xlabel('$E - E_{VBM}, eV$', fontsize = 18)
ax[1].set_ylabel('Norm. Int.', fontsize = 18)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update(params)

fig.tight_layout()

if save_figure is True:
    fig.savefig((figure_file_name +'.svg'), format='svg')
    
print(f"{popt[2]:.3f} +- {perr[2]:.3f}")
print(f"{popt[3]:.3f} +- {perr[3]:.3f}")
print(f"{1000*Eb:.3f} +- {1000*np.sqrt(perr[3]**2+perr[2]**2):.3f}")
#%% # Do the Excited State Fits for all Delay Times

##### CBM AND EXCITON #####
delay_int = 40
e1 = 1.1
e2 = 3
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

    kx_frame = get_kx_E_frame(I, ky, k_int, I.delay.values[t], delay_int)

    kx_edc_i = kx_frame.loc[{"kx":slice(-2,2)}].sum(dim="kx")
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

figure_file_name = 'EDC_fits_metis_excted2'
save_figure = False

fig, ax = plt.subplots(1, 3,  gridspec_kw={'width_ratios': [1, 1.25, 1.25], 'height_ratios':[1]})
fig.set_size_inches(13, 4, forward=False)
ax = ax.flatten()

kx_edc.plot(ax=ax[0], color = 'black')
ax[0].plot(kx_edc.loc[{"E":slice(0,3)}].E.values, g1, color='black',linestyle = 'dashed')
ax[0].plot(kx_edc.loc[{"E":slice(0,3)}].E.values, g2, color='red',linestyle = 'dashed')
ax[0].plot(kx_edc.loc[{"E":slice(0,3)}].E.values, g, color='grey',linestyle = 'solid')
ax[0].set_title(f"$\Delta t = {delay}$ fs : $E_B = {1000*Eb}$ meV")
ax[0].set_xlim(0.5,3)
ax[0].set_ylim(0, 1.1)
ax[0].set_ylabel('Norm. Int.')
ax[0].set_xlabel('$E - E_{VBM}$, eV', color = 'black')

# PLOT CBM and EX SHIFT DYNAMICS
#fig = plt.figure()
t = 7 # Show only after 50 (?) fs
tt = 12
y_ex, y_ex_err = 1*(centers_EX[t:] - 0*centers_EX[-12].mean()), 1*p_err_excited[t:,2]
y_cb, y_cb_err = 1*(centers_CBM[tt:]- 0*centers_CBM[-12].mean()),  1*p_err_excited[tt:,3]

ax[1].plot(I.delay.values[t:], y_ex, color = 'black', label = 'EX')
ax[1].fill_between(I.delay.values[t:], y_ex - y_ex_err, y_ex + y_ex_err, color = 'grey', alpha = 0.5)
ax[1].set_xlim([0, I.delay.values[-1]])
#ax[1].set_ylim([1.1,2.3])
ax[1].set_xlim([0, I.delay.values[-1]])
ax[1].set_ylim([1,1.4])
ax[1].set_xlabel('Delay, fs')

ax2 = ax[1].twinx()
ax2.plot(I.delay.values[tt:], y_cb, color = 'red', label = 'CBM')
ax2.fill_between(I.delay.values[tt:], y_cb - y_cb_err, y_cb + y_cb_err, color = 'pink', alpha = 0.5)
ax2.set_ylim([2,2.4])
#ax[1].errorbar(I.delay.values[t:], 1*(centers_EX[t:]), yerr = p_err_excited[t:,2], marker = 'o', color = 'black', label = 'EX')
#ax[1].errorbar(I.delay.values[t:], 1*(centers_CBM[t:]), yerr = p_err_excited[t:,3], marker = 'o', color = 'red', label = 'CBM')
#ax[1].set_ylim([1.1,2.3])
ax[1].set_ylabel('$E_{EX}$, eV', color = 'black')
ax2.set_ylabel('$E_{CBM}$, eV', color = 'red')
ax[1].set_title(f"From {round(I.delay.values[t])} fs")
ax[1].legend(frameon=False, loc = 'upper right')
ax2.legend(frameon=False, loc = 'upper left' )

ax[2].plot(I.delay.values[tt:], 1000*Ebs[tt:], color = 'purple', label = '$E_{B}$')
ax[2].fill_between(I.delay.values[tt:], 1000*Ebs[tt:] - 1000*p_err_eb[tt:], 1000*Ebs[tt:] + 1000*p_err_eb[tt:], color = 'violet', alpha = 0.5)
ax[2].set_xlim([0, I.delay.values[-1]])
ax[2].set_ylim([700,950])
ax[2].set_xlabel('Delay, fs')
ax[2].set_ylabel('$E_{B}$, meV', color = 'black')
ax[2].legend(frameon=False)

# # PLOT VBM PEAK WIDTH DYNAMICS
# ax2 = ax[1].twinx()
# ax2.plot(edc_gamma.delay.values, 1000*(p_fits_VBM[:,2] - 0*p_fits_VBM[0:10,2].mean()), color = 'maroon')
# #ax2.set_ylim([-75,50])
# ax2.set_ylabel('${VBM}$ Peak Width, meV', color = 'maroon')

fig.tight_layout()

if save_figure is True:
    fig.savefig((figure_file_name +'.svg'), format='svg')

#%% Plot and Fit EDCs of the VBM

%matplotlib inline

save_figure = False
figure_file_name = 'EDC_metis'

#I_res = I.groupby_bins('delay', 50)
#I_res = I_res.rename({"delay_bins":"delay"})
#I_res = I_res/np.max(I_res)
I_res = I/np.max(I)

fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1.5, 1], 'height_ratios':[1]})
fig.set_size_inches(10, 4, forward=False)
ax = ax.flatten()
### Plot EDCs at GAMMA vs time

(kx, ky), k_int = (-1.9, 0), 0.25
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

pts = [-120, 0, 50, 100, 500]
colors = ['black', 'red', 'orange', 'purple', 'blue', 'green', 'grey']
n = len(pts)
colors = mpl.cm.inferno(np.linspace(0,.8,n))
                    
for i in range(n):
    edc = I_res.loc[{"kx":slice(kx-k_int/2,kx+k_int/2), "ky":slice(ky-k_int/2,ky+k_int/2), "delay":slice(pts[i]-5,pts[i]+5)}].sum(dim=("kx","ky","delay"))
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

fig.tight_layout()

if save_figure is True:
    fig.savefig((figure_file_name +'.svg'), format='svg', dpi = 300)

#%% Fit VBM

figure_file_name = 'EDC_metis_fits'
save_figure = False

fig, ax = plt.subplots(1, 2)
fig.set_size_inches(10, 4, forward=False)
ax = ax.flatten()

# VBM FIT TESTS FOR ONE POINT
t = 9
gauss_test = gaussian(edc_gamma.E.values, *p_fits_VBM[t,:])
ax[0].plot(edc_gamma.E.values, edc_gamma[:,t].values/edc_gamma.loc[{"E":slice(e1,e2)}][:,t].values.max(), color = 'black')
ax[0].plot(edc_gamma.E.values, gauss_test, linestyle = 'dashed', color = 'grey')
#plt.axvline(trunc_e, linestyle = 'dashed', color = 'black')
ax[0].set_xlim([-2,1.5])
ax[0].set_xlabel('E - E$_{VBM}$, eV')
ax[0].set_ylabel('Norm. Int.')
#ax[0].axvline(0, linestyle = 'dashed', color = 'grey')
#ax[0].axvline(e2, linestyle = 'dashed', color = 'black')

# PLOT VBM SHIFT DYNAMICS

t = 39 # Show only after 50 (?) fs
y_vb, y_vb_err = 1000*(p_fits_VBM[:,1] - p_fits_VBM[0:10,1].mean()), 1000*p_err_VBM[:,0]
y_vb_w, y_vb_w_err = 1000*(p_fits_VBM[:,2]),  1000*p_err_VBM[:,1]

ax[1].plot(I.delay.values, y_vb, color = 'navy', label = '$\Delta E_{VBM}$')
ax[1].fill_between(I.delay.values, y_vb - y_vb_err, y_vb + y_vb_err, color = 'navy', alpha = 0.5)

ax[1].set_xlim([edc_gamma.delay.values[1], edc_gamma.delay.values[-1]])
ax[1].set_ylim([-30,20])
ax[1].set_xlabel('Delay, fs')
ax[1].set_ylabel('$\Delta E_{VBM}$, meV', color = 'navy')
ax[1].legend(frameon=False)

# PLOT VBM PEAK WIDTH DYNAMICS
ax2 = ax[1].twinx()
ax2.plot(I.delay.values, y_vb_w, color = 'maroon', label = '$\sigma_{VBM}$')
ax2.fill_between(I.delay.values, y_vb_w - y_vb_w_err, y_vb_w + y_vb_w_err, color = 'maroon', alpha = 0.5)
ax2.set_ylim([150,250])
ax2.legend(frameon=False, loc = 'upper left')
ax2.set_ylabel('${VBM}$ Peak Width, meV', color = 'maroon')

fig.tight_layout()

if save_figure is True:
    fig.savefig((figure_file_name +'.svg'), format='svg')


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


#%% Do Fourier Transform Analysis

def window_MM(kspace_frame, kx, ky, kx_int, ky_int, win_type, alpha):    
    
    ### Deconvolve k-space momentum broadening, Gaussian with FWHM 0.063A-1
    fwhm = 0.063
    fwhm_pixel = fwhm/dkx
    sigma = fwhm_pixel/2.355
    gaussian_kx = signal.gaussian(len(ax_kx), std = sigma)
    gaussian_kx = gaussian_kx/np.max(gaussian_kx)
    gaussian_ky = signal.gaussian(len(ax_ky), std = sigma)
    gaussian_ky = gaussian_ky/np.max(gaussian_ky)
    
    gaussian_kxky = np.outer(gaussian_kx, gaussian_ky)
    gaussian_kxky = gaussian_kxky/np.max(gaussian_kxky)
    gaussian_kxky = np.outer(gaussian_kx, gaussian_ky)
    #kx_cut_deconv = signal.deconvolve(kx_cut, gaussian_kx)
    
    ### Symmetrize Data
    #kspace_frame_sym = xr.DataArray(np.zeros(kspace_frame.shape), coords = {"ky": ax_kx, "kx": ax_ky})
    kspace_frame_ = kspace_frame.values
    kspace_frame_rev = kspace_frame_[:,::-1]
    
    kspace_frame_sym = kspace_frame_ + kspace_frame_rev
    kspace_frame_sym =  kspace_frame_sym/2
    kspace_frame_sym = xr.DataArray(kspace_frame_sym, coords = {"ky": ax_ky, "kx": ax_kx})

    ### Generate the Windows to Apodize the signal
    k_x_i = np.abs(ax_kx.values-(kx-kx_int/2)).argmin()
    k_x_f = np.abs(ax_kx.values-(kx+kx_int/2)).argmin()
    k_y_i = np.abs(ax_kx.values-(ky-ky_int/2)).argmin()
    k_y_f = np.abs(ax_kx.values-(ky+ky_int/2)).argmin()
    #I_res.indexes["kx"].get_indexer([kx-kx_int/2], method = 'nearest')[0]
    
    # kx Axis
    win_1_tuk = np.zeros(kspace_frame.shape[0])
    win_1_box = np.zeros(kspace_frame.shape[0])

    tuk_1 = signal.windows.tukey(k_x_f-k_x_i, alpha = alpha)
    box_1 = signal.windows.boxcar(k_x_f-k_x_i)
    win_1_tuk[k_x_i:k_x_f] = tuk_1
    win_1_box[k_x_i:k_x_f] = box_1

    # ky Axis
    win_2_tuk = np.zeros(kspace_frame.shape[1])
    win_2_box = np.zeros(kspace_frame.shape[1])

    tuk_2 = signal.windows.tukey(k_y_f-k_y_i, alpha = alpha)
    box_2 = signal.windows.boxcar(k_y_f-k_y_i)
    win_2_tuk[k_y_i:k_y_f] = tuk_2
    win_2_box[k_y_i:k_y_f] = box_2

    # Combine to (kx, ky) Window
    window_2D_tukey = np.outer(win_2_tuk, win_1_tuk) # 2D tukey
    window_2D_box = np.outer(win_2_box, win_1_box) # 2D Square Window
    window_tukey_box = np.outer(win_2_tuk, win_1_box) # Tukey + Box

    if win_type == 'gaussian':
        win_1_gauss = np.zeros(kspace_frame.shape[0])
        gaus_1 = signal.windows.gaussian(k_x_f-k_x_i, alpha)
        win_1_gauss[k_x_i:k_x_f] = gaus_1
        window_2D_gaussian = np.outer(win_1_gauss, win_2_box)
        kspace_window = window_2D_gaussian
        
    if win_type == 1:
        kspace_window = xr.DataArray(window_2D_tukey, coords = {"ky": ax_ky, "kx": ax_kx})

    if win_type == 'square':
        kspace_window = window_2D_box

    if win_type == 'tukey, square':
        kspace_window = xr.DataArray(window_tukey_box, coords = {"ky": ax_ky, "kx": ax_kx})

    kspace_frame_sym_win = kspace_frame_sym*kspace_window
    kspace_frame_win = kspace_frame*(kspace_window)
        
    return kspace_frame_sym, kspace_frame_win, kspace_frame_sym_win, kspace_window

def FFT_MM(MM_frame, zeropad):
    
    I_MM = MM_frame

    ##############################
    
    # Define real-space axis
    k_step = dkx
    #k_step_y = k-step
    #k_length_y = len(ax_ky)
    zplength = zeropad #5*k_length+1
    max_r = 1/(2*k_step)

    #r_axis = np.linspace(-max_r, max_r, num = k_length)
    r_axis = np.linspace(-max_r, max_r, num = zplength)
    freq = np.fft.fftshift(np.fft.fftfreq(len(ax_kx), d=dkx))  # Frequency array

    #r_axis = r_axis/(10)

    # Shuo Method ?
    N = 1 #(zplength)Fs
    Fs = 1/((2*np.max(ax_kx.values))/len(ax_kx.values))
    r_axis = np.arange(0,zplength)*Fs/1
    r_axis = r_axis - (np.max(r_axis)/2)
    r_axis = r_axis/(1*zplength)

    ### Do the FFT operations to get --> |Psi(x,y)|^2 ###
    I_MM = np.abs(I_MM)/np.max(I_MM)
    root_I_MM = np.sqrt(I_MM)
    fft_frame = np.fft.fft2(root_I_MM, [zplength, zplength])
    fft_frame = np.fft.fftshift(fft_frame, axes = (0,1))

    fft_frame = np.abs(fft_frame) 
    I_xy = np.square(np.abs(fft_frame)) #frame squared
    I_xy = I_xy/np.max(I_xy)
    
    ### Take x and y cuts and extract bohr radius
    ky_cut = I_MM[:,int(len(I_MM[0])/2)-1-4:int(len(I_MM[0])/2)-1+4].sum(axis=1)
    ky_cut = ky_cut/np.max(ky_cut)
    kx_cut = I_MM[int(len(I_MM[0])/2)-1-4:int(len(I_MM[0])/2)-1+4,:].sum(axis=0)
    kx_cut = kx_cut/np.max(kx_cut)

    y_cut = I_xy[:,int(zplength/2)-1] # real space Psi*^2 cut
    x_cut = I_xy[int(zplength/2)-1,:]
    x_cut = x_cut/np.max(x_cut)
    y_cut = y_cut/np.max(y_cut)

    r2_cut_y = fft_frame[:,int(zplength/2)-1] #real space Psi cut
    r2_cut_y = np.square(np.abs(r2_cut_y*r_axis)) #|r*Psi(r)|^2
    r2_cut_y = r2_cut_y/np.max(r2_cut_y)

    x_brad = (np.abs(x_cut[int(zplength/2)-10:int(zplength/2)+200] - 0.5)).argmin()
    y_brad = (np.abs(y_cut[int(zplength/2)-10:] - 0.5)).argmin()
    x_brad = int(zplength/2)-10 + x_brad
    y_brad = int(zplength/2)-10 + y_brad
    x_brad = r_axis[x_brad]
    y_brad = r_axis[y_brad]
    
    ###
    r2_cut_x = fft_frame[int(zplength/2)-1,:]
    r2_cut_x = np.square(np.abs(r2_cut_x[0:1090]*r_axis[0:1090]))
    r2_cut_x = r2_cut_x/np.max(r2_cut_x)

    rdist_brad_x = np.argmax(r2_cut_x[int(zplength/2)-10:int(zplength/2)+90])
    rdist_brad_y = np.argmax(r2_cut_y[int(zplength/2)-10:int(zplength/2)+150])

    rdist_brad_x = r_axis[int(zplength/2)-10 + rdist_brad_x]
    rdist_brad_y = r_axis[int(zplength/2)-10 + rdist_brad_y]

    return r_axis, I_xy, x_cut, y_cut, rdist_brad_x, rdist_brad_y, x_brad, y_brad
    
#%% Do the 2D FFT of MM to Extract Real-Space Information

E, E_int  = 1.35, 0.200 #Energy and total width in eV
kx, kx_int = (1.1--0.1)/2, 1.5
ky, ky_int = 0, 1.2
delays, delay_int = 550, 650 

win_type = 1 #0, 1 = 2D Tukey, 2, 3
alpha = 0.25
zeropad = 2048

frame_pos = get_momentum_map(I_res, E, E_int, delays, delay_int)  # Get Positive Delay MM frame (takes mean over ranges)
frame_neg = get_momentum_map(I_res, E, E_int, -130, 50) # Get Negative Delay MM frame (takes mean over ranges)
frame_diff = frame_pos - frame_neg

testing = 0
if testing == 1:
    ax_kx, ax_ky = np.linspace(-2,2,1000), np.linspace(-2,2,1000)
    dkx = (ax_kx[1] - ax_kx[0])
    g_test = gaussian(ax_kx, *[1, 0, 0.15, 0])
    kspace_frame_test = np.zeros((g_test.shape[0], g_test.shape[0]))
    i, f = round(0.45*len(g_test)), round(0.8*len(g_test))
    kspace_frame_test[:,i:f] = np.tile(g_test, (f-i,1)).T
    kspace_frame = xr.DataArray(kspace_frame_test, coords = {"ky": ax_kx, "kx": ax_ky})

elif testing == 0:
    kspace_frame = frame_pos/np.max(frame_pos) #Define MM of itnerested for FFT
    ax_kx, ax_ky = I.kx, I.ky
    dkx = (ax_kx.values[1] - ax_kx.values[0])

background = frame_pos.loc[{"kx":slice(-1.8,1.8), "ky":slice(0.5,.8)}].mean(dim="ky")
 
#kspace_frame = kspace_frame - background
kspace_frame_sym, kspace_frame_win, kspace_frame_sym_win, kspace_window = window_MM(kspace_frame, kx, ky, kx_int, ky_int, win_type, alpha) # Window the MM

MM_frame = kspace_frame_win # Choose which kspace frame to FFT
MM_frame = kspace_frame_sym_win
#MM_frame = window_tukey_box
r_axis, rspace_frame, x_cut, y_cut, rdist_brad_x, rdist_brad_y, x_brad, y_brad = FFT_MM(MM_frame, zeropad) # Do the 2D FFT and extract real-space map and cuts

r_axis = r_axis/10

#%% # Plot MM, Windowed Map, I_xy, and r-space cuts

### PLOT ###

save_figure = False
figure_file_name = 'MM_FFT' 

fig, ax = plt.subplots(2, 2)
fig.set_size_inches(6,8)
plt.gcf().set_dpi(300)
ax = ax.flatten()

im0 = ax[0].imshow(kspace_frame/np.max(kspace_frame), clim = None, origin = 'lower', vmax = 1, cmap=cmap_LTL, interpolation = 'none', extent = [ax_kx[0], ax_kx[-1], ax_ky[0], ax_ky[-1]])
im1 = ax[1].imshow(MM_frame/np.max(MM_frame), clim = None, origin = 'lower', vmax = 1, cmap=cmap_LTL, interpolation = 'none', extent = [ax_kx[0], ax_kx[-1], ax_ky[0], ax_ky[-1]])
im2 = ax[2].imshow(rspace_frame/np.max(rspace_frame), clim = None, origin='lower', cmap=cmap_LTL, interpolation='none', extent = [r_axis[0], r_axis[-1], r_axis[0], r_axis[-1]]) #kx, ky, t
#single_k_circle = plt.Circle((single_ky, single_kx), single_rad, color='red', linestyle = 'dashed', linewidth = 1.5, clip_on=False, fill=False)
#ax[1].add_patch(single_k_circle)
ax[0].set_aspect(1)
ax[1].set_aspect(1)
ax[2].set_aspect(1)

#ax[0].axhline(y,color='black')
#ax[0].axvline(x,color='bl ack')

ax[0].set_xticks(np.arange(-2,2.2,1))
for label in ax[0].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
   # label.set_xticklabels(tick_labels.astype(int))
    
ax[0].set_yticks(np.arange(-2,2.2,1))
for label in ax[0].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)

ax[1].set_xticks(np.arange(-2,2.2,1))
for label in ax[1].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
   # label.set_xticklabels(tick_labels.astype(int))
    
ax[1].set_yticks(np.arange(-2,2.1,1))
for label in ax[1].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)

ax[2].set_xticks(np.arange(-8,8,1))
for label in ax[2].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
    
ax[2].set_yticks(np.arange(-8,8.1,1))
for label in ax[2].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
    
ax[3].set_xticks(np.arange(0,5.2,.5))
#for label in ax[3].xaxis.get_ticklabels()[1::2]:
    #label.set_visible(False)    

ax[0].axvline(0, color='black', linewidth = 1, linestyle = 'dashed')
ax[0].axhline(0, color='black', linewidth = 1, linestyle = 'dashed')
ax[0].axvline(-1.1, color='blue', linewidth = 1, linestyle = 'dashed')
ax[0].axvline(1.1, color='blue', linewidth = 1, linestyle = 'dashed')
    
ax[0].set_xlim(-2,2)
ax[0].set_ylim(-2,2)
#ax[0].set_box_aspect(1)
ax[0].set_xlabel('$k_x$, $\AA^{-1}$', fontsize = 16)
ax[0].set_ylabel('$k_y$,  $\AA^{-1}$', fontsize = 16)
ax[0].tick_params(axis='both', labelsize=10)
ax[0].set_title('$E$ = ' + str(E) + ' eV, ' + '$\Delta$E = ' + str(E_int) + ' eV', fontsize = 14)
ax[0].set_title('$E$ = ' + str(E) + ' eV ', fontsize = 14)
#fig.suptitle('E = ' + str(E) + ' eV, $\Delta$E = ' + str(1000*E_int) + ' meV,  $\Delta$$k_{rad}$ = ' + str(window_k_width) + ' $\AA^{-1}$', fontsize = 18)
ax[0].text(-1.9, 1.5,  f"$\Delta$t = {round(delays-delay_int/2)} to {round(delay_int+delay_int/2)} fs", size=12)

ax[1].set_xlim(-2,2)
ax[1].set_ylim(-2,2)
#ax[0].set_box_aspect(1)
ax[1].set_xlabel('$k_x$, $\AA^{-1}$', fontsize = 16)
ax[1].set_ylabel('$k_y$,  $\AA^{-1}$', fontsize = 16)
ax[1].tick_params(axis='both', labelsize=10)
ax[1].set_title(f'$\Delta$k = ({kx_int}, {ky_int})', fontsize = 15)
ax[1].axvline(0, color='black', linewidth = 1, linestyle = 'dashed')
ax[1].axhline(0, color='black', linewidth = 1, linestyle = 'dashed')
ax[1].axvline(-1.1, color='blue', linewidth = 1, linestyle = 'dashed')
ax[1].axvline(1.1, color='blue', linewidth = 1, linestyle = 'dashed')
 
ax[2].set_xlim(-2,2)
ax[2].set_ylim(-2,2)
#ax[0].set_box_aspect(1)
ax[2].set_xlabel('$r_x$, nm', fontsize = 16)
ax[2].set_ylabel('$r_y$, nm', fontsize = 16)
ax[2].tick_params(axis='both', labelsize=10)
ax[2].set_title('2D FFT', fontsize = 15)

#ax[2].plot(r_axis, x_cut/np.max(1), color = 'black', label = '$r_b$')
ax[3].plot(r_axis, x_cut/np.max(x_cut), color = 'black', label = '$r_x$')
#ax[3].plot(r_axis, r2_cut_x, color = 'black', linestyle = 'dashed')
ax[3].plot(r_axis, y_cut/np.max(y_cut), color = 'red', label = '$r_y$')
#ax[3].plot(r_axis, r2_cut_y, color = 'red', linestyle = 'dashed')

ax[3].axvline(x_brad, linestyle = 'dashed', color = 'black', linewidth = 1.5)
ax[3].axvline(y_brad, linestyle = 'dashed', color = 'red', linewidth = 1.5)
ax[3].axvline(rdist_brad_x, linestyle = 'dashed', color = 'black', linewidth = .5)
ax[3].axvline(rdist_brad_y, linestyle = 'dashed', color = 'red', linewidth = .5)

ax[3].set_xlim([0, 2])
ax[3].set_ylim([-0.025, 1.025])
ax[3].set_xlabel('$r$, nm', fontsize = 16)
ax[3].set_ylabel('Norm. Int.', fontsize = 16)
ax[3].set_title(f"$r^*_{{x,y}}$ = ({round(x_brad,2)}, {round(y_brad,2)}) nm", fontsize = 14)
ax[3].tick_params(axis='both', labelsize=10)
ax[3].set_yticks(np.arange(-0,1.5,0.5))
ax[3].set_aspect(2)
ax[3].set_xlabel('$r$, nm')
ax[3].legend(frameon=False, fontsize = 12)
ax[3].text(1.05, 0.55,  f"({np.round(rdist_brad_x,2)}, {np.round(rdist_brad_y,2)})", size=10)

fig.subplots_adjust(right=0.58, top = 1.1)
fig.tight_layout()
new_rc_params = {'text.usetex': False, "svg.fonttype": 'none'}
plt.rcParams.update(new_rc_params)

if save_figure is True:
    fig.savefig((figure_file_name +'.svg'), format='svg')
    
#print("x: " + str(round(rdist_brad_x,3)))
#print("y: " + str(round(rdist_brad_y,3)))

#%% #Do line fits analysis to cross check values

test_frame = kspace_frame_sym
test_frame_win = kspace_frame_sym_win

kx_cut = test_frame.loc[{"ky":slice(-.4,.4)}].mean(dim="ky")
ky_cut = test_frame.loc[{"kx":slice(0.2,.6)}].mean(dim="kx")
kx_win_cut = test_frame_win.loc[{"ky":slice(-.4,.4)}].mean(dim="ky")
ky_win_cut = test_frame_win.loc[{"kx":slice(.2,.6)}].mean(dim="kx")
window_kx_cut = kspace_window.loc[{"ky":slice(-.4,.4)}].mean(dim="ky")
window_ky_cut = kspace_window.loc[{"kx":slice(0.2,.6)}].mean(dim="kx")

#kx_win_cut = MM_frame.loc[{"ky":slice(-.25,.25)}].sum(dim="ky")
#ky_win_cut = MM_frame.loc[{"kx":slice(-.05,1.1)}].sum(dim="kx")

#ky_cut = MM_frame[:,int(len(MM_frame[0])/2)-1-4:int(len(MM_frame[0])/2)-1+4].sum(axis=1)
#kx_cut = MM_frame[int(len(MM_frame[0])/2)-1-4:int(len(MM_frame[0])/2)-1+4,:].sum(axis=0)

ky_cut = ky_cut/np.max(ky_cut)
kx_cut = kx_cut/np.max(kx_cut)

kx_win_cut = kx_win_cut/np.max(kx_win_cut)
ky_win_cut = ky_win_cut/np.max(ky_win_cut)

# Fit kx Cut
xlim = [-0.5, 1]
p0 = [0.5, 0.5, 0.325, 0.4]
bnds = ((0.1, -0.5, .2, 0), (1.5, 1.2, .5, 0.8))
popt_kx, pcov = curve_fit(gaussian, ax_kx.loc[{"kx":slice(xlim[0],xlim[1])}], kx_cut.loc[{"kx":slice(xlim[0],xlim[1])}], p0, method=None, bounds = bnds)
g_fit_kx = gaussian(ax_kx, *popt_kx)
k_sig_fit_x = popt_kx[2]
#plt.plot(ax_kx, kx_cut) ; plt.plot(ax_kx, g_fit_kx)

# Fit ky Cut
ylim = 0.3
p0 = [.8, 0, .08, 0.2]
bnds = ((0.1, -0.2, 0, 0), (1, 0.2, .8, .5))
popt_ky, pcov = curve_fit(gaussian, ax_ky.loc[{"ky":slice(-ylim,ylim)}], ky_cut.loc[{"ky":slice(-ylim,ylim)}], p0, method=None, bounds = bnds)
g_fit_ky = gaussian(ax_ky, *popt_ky)

#popt_ky, pcov = curve_fit(lorentzian, ax_ky.loc[{"ky":slice(-ylim,ylim)}], ky_cut.loc[{"ky":slice(-ylim,ylim)}], p0, method=None, bounds = bnds)
#g_fit_ky = lorentzian(ax_ky, *popt_ky)

k_sig_fit_y = popt_ky[2]
#plt.plot(ax_ky, ky_cut) ; plt.plot(ax_ky, g_fit_ky)

#Fit r-y Cut
p0 = [1, 0, 0.2, 0]
bnds = ((0.5, -1, .1, 0), (1.2, 2, 5, 0.4))
popt_ry, pcov_r = curve_fit(gaussian, r_axis, y_cut/np.max(y_cut), p0, method=None, bounds = bnds)
g_fit_ry = gaussian(r_axis, *popt_ry)
r_sig_fit_y = popt_ry[2]

#Fit r-x Cut
p0 = [1, 0, 0.2, 0]
bnds = ((0.5, -1, .1, 0), (1.2, 2, 5, 0.4))
popt_rx, pcov_r = curve_fit(gaussian, r_axis, x_cut/np.max(x_cut), p0, method=None, bounds = bnds)
g_fit_rx = gaussian(r_axis, *popt_rx)
r_sig_fit_x = popt_rx[2]

# Do the FFT of the fit in the k-space to get verify r-space
g_fit_fft = np.abs(np.fft.fftshift(np.fft.fft((g_fit_kx-popt_kx[3])**0.5, zeropad)))  # Compute FFT
g_fit_fft = g_fit_fft / np.max(g_fit_fft)
g_fit_fft_x = g_fit_fft**2

g_fit_fft = np.abs(np.fft.fftshift(np.fft.fft((g_fit_ky-popt_ky[3])**0.5, zeropad)))  # Compute FFT
g_fit_fft = g_fit_fft / np.max(np.abs(g_fit_fft))
g_fit_fft_y = g_fit_fft**2

### Fourier Transform Relation: k-space to r-space
r_sig_x = 0.1*1/(2*k_sig_fit_x) #Ang to nm
r_sig_rad_x = np.sqrt(2)*r_sig_x # Rad from Gaussian Relation Considering fit to k-data before fft

r_sig_y = 0.1*1/(2*k_sig_fit_y) #Ang to nm
r_sig_rad_y = np.sqrt(2)*r_sig_y # Rad from Gaussian Relation Considering fit to k-data before fft

r_sig_rad_fit_x = np.sqrt(2)*r_sig_fit_x #Rad from fit to r-data from fft
r_sig_rad_fit_y = np.sqrt(2)*r_sig_fit_y #Rad from fit to r-data from fft

#print("predicted x: " + str(round(x_pr,3)))

#####################################################
save_figure = False
figure_file_name = '2DFFT_Windowing' 

fig, ax = plt.subplots(2, 3, sharey=False, gridspec_kw={'width_ratios': [.75, 1, 1], 'height_ratios':[1, 1]})
fig.set_size_inches(8, 4, forward=False)
plt.gcf().set_dpi(300)
ax = ax.flatten()

test_frame.plot.imshow(ax = ax[0], cmap = cmap_LTL, origin = 'lower')

ax[1].plot(ax_ky, g_fit_kx, linewidth = 4, color = 'pink', label = 'Fit to k-Data')
ax[1].plot(ax_kx, kx_cut, color =  'purple', linewidth = 2, label = 'Data')
ax[1].plot(ax_kx, kx_win_cut, color =  'black', linestyle = 'solid', linewidth = 1.5, label = 'Win. Data.')
ax[1].plot(ax_kx, window_kx_cut, color = 'grey', linestyle = 'solid', linewidth = 1.5, label =  'WINDOW')

ax[2].plot(ax_ky, g_fit_ky, linewidth = 3, color = 'pink', label = 'Fit to k-Data')
#ax[2].plot(ax_kx, ky_win, color =  'grey', linestyle = 'solid', linewidth = 1.5)
ax[2].plot(ax_ky, ky_cut, color =  'purple', linewidth = 2, label = 'Data')
ax[2].plot(ax_ky, ky_win_cut, color =  'black', linestyle = 'solid', linewidth = 2, label = 'Win. Data.')
ax[2].plot(ax_ky, window_ky_cut, color = 'grey', linestyle = 'solid', linewidth = 1.5, label =  'WINDOW')

#ax[0].axhline(xi, color='black', linewidth = 1, linestyle = 'dashed')
#ax[0].axvline(yi, color='black', linewidth = 1, linestyle = 'dashed')
ax[1].set_xlim(-2,2)
ax[1].set_ylim(0,1.1)
ax[2].set_xlim(-1.5,1.5)
ax[2].set_ylim(0,1.1)
#ax[1].set_aspect(60)
#ax[2].set_aspect(30)

ax[3].imshow(rspace_frame/np.max(rspace_frame), cmap = cmap_LTL, origin = 'lower', aspect = 1, extent = [r_axis[0], r_axis[-1], r_axis[0], r_axis[-1]])
ax[3].set_xlim(-2,2)
ax[3].set_ylim(-2,2)

ax[4].plot(r_axis, g_fit_fft_x, linewidth = 3, color = 'pink', label = 'FFT of k-fit (w/o offset)')
#ax[4].plot(r_axis, g_fit_rx/np.max(g_fit_rx), linewidth = 3, color = 'green', label = 'fit to FFT of Win. k-data')
#ax[5].plot(r_axis, y_cut, color =  'grey', linestyle = 'solid', linewidth = 1.5)
ax[4].plot(r_axis, g_fit_rx/np.max(g_fit_rx), linewidth = 3, color = 'green', label = 'fit to FFT of Win. k-data')
ax[4].plot(r_axis, x_cut, color =  'black', linewidth = 2, label = 'FFT of Win k-data')
#ax[5].plot(r_axis, kx_win_cut, color =  'black', linestyle = 'dashed', linewidth = 1.5, label = 'FFT Data')
ax[4].set_xlim(-2,2)

ax[5].plot(r_axis, g_fit_fft_y, linewidth = 4, color = 'pink', label = 'FFT of k-fit (w/o offset)')
ax[5].plot(r_axis, g_fit_ry/np.max(g_fit_ry), linewidth = 3, color = 'green', label = 'fit to FFT of Win. k-data')
#ax[5].plot(r_axis, y_cut, color =  'grey', linestyle = 'solid', linewidth = 1.5)
ax[5].plot(r_axis, y_cut, color =  'black', linewidth = 2, label = 'FFT of Win k-data')
#ax[5].plot(r_axis, kx_win_cut, color =  'black', linestyle = 'dashed', linewidth = 1.5, label = 'FFT Data')
ax[5].set_xlim(-2,2)

print(f"Pred. Rx (radius) from Fit of k Peak ({round(k_sig_fit_x,4)} A^-1): {round(r_sig_rad_x,3)} nm")
#print(f"Rx (radius) from Fit of Real-space After FFT: {round(r_sig_rad_fit_x,3)} nm")
print(f"Pred. Ry (radius) from Fit of k Peak ({round(k_sig_fit_y,4)} A^-1): {round(r_sig_rad_y,3)} nm")

print(f"Rx (radius) from Fit of Real-space After FFT: {round(r_sig_rad_fit_x,3)} nm")
print(f"Ry (radius) from Fit of Real-space After FFT: {round(r_sig_rad_fit_y,3)} nm")

ax[2].legend(frameon=False, fontsize = 12)
ax[5].legend(frameon=False, fontsize = 12)
#fig.subplots_adjust(right=0.8)

fig.tight_layout()
plt.show()

if save_figure is True:
    fig.savefig((figure_file_name +'.svg'), format='svg')
    
#%% #Plot Momentum MAPS

save_figure = False
figure_file_name = '2DFFT_Windowing' 

fig, ax = plt.subplots(1, 3, sharey=True)
plt.gcf().set_dpi(300)
ax = ax.flatten()

im = ax[0].imshow(kspace_frame, origin='lower', cmap=cmap_LTL, extent = [ax_kx[0],ax_kx[-1],ax_ky[0],ax_ky[-1]])
im = ax[1].imshow(kspace_frame_sym, origin='lower', cmap=cmap_LTL, extent = [ax_kx[0],ax_kx[-1],ax_ky[0],ax_ky[-1]])
im = ax[2].imshow(kspace_frame_sym_win, origin='lower', cmap=cmap_LTL, extent = [ax_kx[0],ax_kx[-1],ax_ky[0],ax_ky[-1]])

for i in np.arange(3):
    ax[i].axhline(0, color='black', linewidth = 1, linestyle = 'dashed')
    ax[i].axhline(0, color='black', linewidth = 1, linestyle = 'dashed')
    ax[i].axvline(0, color='black', linewidth = 1, linestyle = 'dashed')
    ax[i].axvline(0, color='black', linewidth = 1, linestyle = 'dashed')
    
    ax[i].axvline(-1.1, color='blue', linewidth = 1, linestyle = 'dashed')
    ax[i].axvline(1.1, color='blue', linewidth = 1, linestyle = 'dashed')
    
    ax[i].set_aspect(1)
    #ax[0].axhline(y,color='black')
    #ax[0].axvline(x,color='black')
    
    ax[i].set_xticks(np.arange(-2,2.2,1))
    for label in ax[i].xaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
        
    ax[i].set_yticks(np.arange(-2,2.1,1))
    for label in ax[i].yaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
    
    ax[i].set_xlim(-2,2)
    ax[i].set_ylim(-2,2)
    #ax[0].set_box_aspect(1)
    ax[i].set_xlabel('$k_x$', fontsize = 14)
    ax[i].set_ylabel('$k_y$', fontsize = 14)
    ax[i].tick_params(axis='both', labelsize=12)
    ax[i].set_title('$E$ = ' + str(E) + ' eV', fontsize = 16)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([1, 0.35, 0.025, 0.3])
fig.colorbar(im, cax=cbar_ax, ticks = [10,100])

#fig.colorbar(im, fraction=0.046, pad=0.04)
fig.tight_layout()
plt.show()

if save_figure is True:
    fig.savefig((figure_file_name +'.svg'), format='svg')    