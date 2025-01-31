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
from obspy.imaging.cm import viridis_white
import xarray as xr

from Loader import DataLoader
from Main import main
from Manager import DataHandler, FigureHandler, PlotHandler, ValueHandler, SliderManager, EventHandler, CheckButtonManager, ClickButtonManager

#%% Specifiy filename of h5 file in your path.
# Include manual energy and time delay offsets for the axes, if required.

data_path = 'path_to_your_data'
filename = 'your_file_name.h5'

#data_path = 'R:\Lawson\Data\metis'
data_path = '/Users/lawsonlloyd/Desktop/Data/'
#filename, offsets = 'Scan682_binned.h5', [0,0]

filename, offsets = 'Scan162_binned_100x100x200x150_CrSBr_RT_750fs_New_2.h5', [0.2, -90] # Axis Offsets: [Energy (eV), delay (fs)]
#filename, offsets = 'Scan163_binned_100x100x200x150_CrSBr_120K_1000fs_rebinned_distCorrected_New_2.h5', [0, 100]
#filename, offsets = 'Scan188_binned_100x100x200x155_CrSBr_120K_1000fs_rebinned_ChargeingCorrected_DistCorrected.h5', [0.05, 65]

#filename, offsets = 'Scan62_binned_200x200x300_CrSBr_RT_Static_rebinned.h5', [0,0]

#filename, offsets = 'Scan383_binned_LTL.h5', [7.2,0]

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

#%% Useful Functions and Definitions for Manipulating Data

# Partition Data into + and - Delays
def get_data_chunks(neg_times, t0, ax_delay_offset):
    if I.ndim > 3:
        tnf1 = (np.abs(ax_delay_offset - neg_times[0])).argmin()
        tnf2 = (np.abs(ax_delay_offset - neg_times[1])).argmin()

        I_neg = I[:,:,:,tnf1:tnf2+1] #Sum over delay/polarization/theta...
        neg_length = I_neg.shape[3]
        I_neg = I_neg
        I_neg_sum = I_neg.sum(axis=(3))/neg_length
    
        I_pos = I[:,:,:,t0+1:]
        pos_length = I_pos.shape[3]
        I_pos = I_pos #Sum over delay/polarization/theta...
        I_pos_sum = I_pos.sum(axis=(3))/pos_length
    
        I_sum = I[:,:,:,:].sum(axis=(3))
        
        return I_sum, I_pos, I_pos_sum, I_neg, I_neg_sum

    else:
        I_neg = I[:,:,:] #Sum over delay/polarization/theta...
        I_pos = I[:,:,:]
        I_sum = I

# Function for Creating MM Constant Energy kx, ky slice 
def get_momentum_map(I_res, E, E_int, delays, delay_int):
    # Momentum Maps at specified Energies and Delays
        
    frame = I_res.loc[{"E":slice(E-E_int/2, E+E_int/2), "delay":slice(delays-delay_int/2, delays+delay_int/2)}].mean(dim=("E","delay")).T
                             
    return frame

def get_kx_E_frame(I_res, ky, ky_int, delay, delay_int):
    
    frame = I_res.loc[{"ky":slice(ky-ky_int/2, ky+ky_int/2), "delay":slice(delay-delay_int/2, delay+delay_int/2)}].mean(dim="ky").mean(dim="delay")
    
    return frame

def get_ky_E_frame(I_res, kx, kx_int, delay, delay_int):

    frame = I_res.loc[{"kx":slice(kx-kx_int/2, kx+kx_int/2), "delay":slice(delay-delay_int/2, delay+delay_int/2)}].mean(dim="kx").mean(dim="delay")
    
    return frame

# Fucntion for Extracting time Traces
def get_time_trace(I_res, E, E_int, k , k_int, subtract_neg, norm_trace):
    
    (kx, ky) = k
    (kx_int, ky_int) = k_int
    
    trace = I_res.loc[{"E":slice(E-E_int/2, E+E_int/2), "kx":slice(kx-kx_int/2, kx+kx_int/2), "ky":slice(ky-ky_int/2, ky+ky_int/2)}].sum(axis=(0,1,2))

    if subtract_neg is True : 
        trace = trace - np.mean(trace.loc[{"delay":slice(-100,-50)}])
    
    if norm_trace is True : 
        trace = trace/np.max(trace)
    
    return trace

def enhance_features(I_res, Ein, factor, norm):
    
    I1 = I_res.loc[{"E":slice(-3.5,Ein)}]
    I2 = I_res.loc[{"E":slice(Ein,3.5)}]

    if norm is True:
        I1 = I1/np.max(I1)
        I2 = I2/np.max(I2)
    else:
        I1 = I1/factor[0]
        I2 = I2/factor[1]
        
    I3 = xr.concat([I1, I2], dim = "E")
    
    return I3

#%% Useful Functions and Definitions for Plotting Data

def plot_momentum_maps(I, E, E_int, delays, delay_int, cmap_plot):
            
    fig, ax = plt.subplots(1, len(E), squeeze = False)
    ax = ax.flatten()
    fig.set_size_inches(8, 5, forward=False)
    plt.gcf().set_dpi(300)
    
    for i in np.arange(len(E)):
            
        frame = get_momentum_map(I, E[i], E_int, delays[i], delay_int)
        
        im = frame.plot.imshow(ax = ax[i], clim = None, cmap = cmap_plot, add_colorbar=False)
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
        ax[i].set_xlabel('$k_x$, $\AA^{-1}$', fontsize = 18)
        ax[i].set_ylabel('$k_y$, $\AA^{-1}$', fontsize = 18)
        ax[i].set_title('$E$ = ' + str((E[i])) + ' eV', fontsize = 18)
        ax[i].tick_params(axis='both', labelsize=16)
        #ax[i].annotate(('E = '+ str(round(tMaps[i],2)) + ' eV'), xy = (-1.85, 1.6), fontsize = 14, weight = 'bold')
        ax[i].text(-1.9, 1.5,  f"$\Delta$t = {delays[i]} fs", size=14)

    cbar_ax = fig.add_axes([1, 0.325, 0.025, 0.35])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks = [0,frame.max()])
    cbar.ax.set_yticklabels(['min', 'max'])  # vertically oriented colorbar

    plt.rcParams['svg.fonttype'] = 'none'
    fig.tight_layout()
    
    return fig
                
#%%


#I_sum, I_pos, I_pos_sum, I_neg, I_neg_sum = get_data_chunks([-180,-100], t0, ax_delay_offset) #Get the Neg and Pos delay time arrays
def custom_colormap(CMAP, lower_portion_percentage):
    # create a colormap that consists of
    # - 1/5 : custom colormap, ranging from white to the first color of the colormap
    # - 4/5 : existing colormap
    
    # set upper part: 4 * 256/4 entries
    
    upper =  CMAP(np.arange(256))
    upper = upper[56:,:]
    
    # - initialize all entries to 1 to make sure that the alpha channel (4th column) is 1
    lower_portion = int(1/lower_portion_percentage) - 1
    
    lower = np.ones((int(200/lower_portion),4))
    # - modify the first three columns (RGB):
    #   range linearly between white (1,1,1) and the first color of the upper colormap
    for i in range(3):
      lower[:,i] = np.linspace(1, upper[0,i], lower.shape[0])
    
    # combine parts of colormap
    cmap = np.vstack(( lower, upper ))
    
    # convert to matplotlib colormap
    custom_cmap = mpl.colors.ListedColormap(cmap, name='custom', N=cmap.shape[0])
    
    return custom_cmap

cmap_LTL = custom_colormap(mpl.cm.viridis, 0.2)

#%% Plot Momentum Maps at Constant Energy

E, E_int = [1.35, 2.1], 0.2 # Energies and Total Energy Integration Window to Plot MMs
delays, delay_int = [200, 200], 100 #Integration range for delays

#######################

%matplotlib inline

figure_file_name = f'MM_delays_1' 
save_figure = True

#cmap_plot = viridis_white
cmap_plot = cmap_LTL

fig = plot_momentum_maps(I, E, E_int, delays, delay_int, cmap_plot)

if save_figure is True:
    fig.savefig((figure_file_name +'.svg'), format='svg')


#%% Plot Dynamics: Extract Traces At Different Energies and Momenta

save_figure = False
figure_file_name = ''

E_trace, E_int = [1.35, 2.1], .12 # Energies for Plotting Time Traces ; 1st Energy for MM
k, k_int = (0, 0), (4, .4) # Central (kx, ky) point and k-integration

colors = ['black', 'red'] #colors for plotting the traces

subtract_neg = True #If you want to subtract negative time delay baseline
norm_trace = True

#######################
### Do the Plotting ###
#######################
(kx, ky) = k
(kx_int, ky_int) = k_int

i = 0
frame = get_momentum_map(I, E_trace[i], E_int, 500, 2000)

fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [.75, 1, 1], 'height_ratios':[1]})
fig.set_size_inches(12 , 4, forward=False)
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
ax[0].set_title('$E$ = ' + str((E_trace[i])) + ' eV', fontsize = 18)
ax[0].tick_params(axis='both', labelsize=16)

### SECOND PLOT: WATERFALL
Ein = 0.7
I_norm = I/np.max(I)
I_neg_mean = I_norm.loc[{"delay":slice(-130,-95)}].mean(axis=(3))

#I_norm = I_norm.loc[{"kx":slice(kx-kx_int/2, kx+kx_int/2), "ky":slice(ky-ky_int/2, ky+ky_int/2)}].sum(dim=("kx","ky"))
#I_neg_mean = I_neg_mean.loc[{"kx":slice(kx-kx_int/2, kx+kx_int/2), "ky":slice(ky-ky_int/2, ky+ky_int/2)}].sum(dim=("kx","ky"))

I_diff = I_norm - I_neg_mean
I_diff = I_norm.loc[{"kx":slice(kx-kx_int/2, kx+kx_int/2), "ky":slice(ky-ky_int/2, ky+ky_int/2)}].sum(dim=("kx","ky"))
I_diff = I_diff/np.max(I_diff)

I3 = enhance_features(I_diff, Ein, factor = 0, norm = True)

energy_limits = [-.5, 3]
color_max = 1
#waterfall = I3.plot(ax = ax[1], vmin = -1, vmax = 1, cmap = cmap_LTL)
waterfall = I3.plot(ax = ax[1], vmin = 0, vmax = 1, cmap = cmap_LTL, add_colorbar=False)

#waterfall = ax[1].imshow(diff_ang, clim = clim, origin = 'lower', cmap = cmap_LTL, extent=[ax_delay_offset[0], ax_delay_offset[-1], ax_E_offset[0], ax_E_offset[-1]])
ax[1].set_xlabel('Delay, fs', fontsize = 18)
ax[1].set_ylabel('E - E$_{VBM}$, eV', fontsize = 18)
ax[1].set_yticks(np.arange(-1,3.5,0.5))
ax[1].set_xlim(I.delay[1],I.delay[-1])
ax[1].set_ylim(energy_limits)
ax[1].set_title('k-Integrated')

for label in ax[1].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)

hor = I.delay[-1] - I.delay[1]
ver =  energy_limits[1] - energy_limits[0]
aspra = hor/ver 
#ax[1].set_aspect(aspra)
ax[1].set_aspect("auto")
#fig.colorbar(waterfall, ax=ax[1], shrink = 0.8, ticks = [0, color_max/2, color_max])

### THIRD PLOT: DELAY TRACES (Angle-Integrated)
trace_norms = []
for i in np.arange(len(E_trace)):
    
    trace = get_time_trace(I, E_trace[i], E_int, k , k_int, subtract_neg, norm_trace)
    trace_norms.append(np.max(trace))
    
    trace.plot(ax = ax[2], color = colors[i], label = str(E_trace[i]) + ' eV')
    ax[1].axhline(E_trace[i], linestyle = 'dashed', color = colors[i], linewidth = 1.5)

#   ax[1].axhline(E_trace[i]-E_int/2, linestyle = 'dashed', color = colors[i], linewidth = 1.5)
 #   ax[1].axhline(E_trace[i]+E_int/2, linestyle = 'dashed', color = colors[i], linewidth = 1.5)
    rect = (Rectangle((kx-kx_int, ky-ky_int), 2*kx_int, 2*ky_int, linewidth=1.5,\
                     edgecolor=colors[i], facecolor='None'))
    ax[0].add_patch(rect)
    
ax[2].set_xlim(I.delay[1], I.delay[-1])

if norm_trace is True:
    ax[2].set_ylim(-0.1, 1.1)
else:
    ax[2].set_ylim(-0.1*np.max(trace_norms), 1.1*np.max(trace_norms))
    
ax[2].set_ylabel('Norm. Int.', fontsize = 18)
ax[2].set_xlabel('Delay, fs', fontsize = 18)
ax[2].legend(frameon = False)
ax[2].set_title('Delay Traces')

params = {'lines.linewidth' : 2.5, 'axes.linewidth' : 2, 'axes.labelsize' : 20, 
              'xtick.labelsize' : 16, 'ytick.labelsize' : 16, 'axes.titlesize' : 20, 'legend.fontsize' : 16}
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update(params)

fig.tight_layout()
#ax[1].set_tick_params(axis='both', labelsize=16)
#plt.gca().set_aspect(200)

if save_figure is True:
    fig.savefig((figure_file_name +'.svg'), format='svg', dpi =300)
    
#%% Plot kx vs E

save_figure = True
figure_file_name = 'MM_ARPES_delay_frames_200'

E_trace, E_int = [1.35, 2.1], .12 # Energies for Plotting Time Traces ; 1st Energy for MM
k, k_int = (0, 0), .2 # Central (kx, ky) point and k-integration
delay, delay_int = 200, 50

colors = ['black', 'red'] #colors for plotting the traces
cmap_plot = cmap_LTL

i = 0

(kx, ky) = k
I_res = I/I.max()
I_res_enh = enhance_features(I_res, Ein, factor = 0, norm = True)

frame = get_momentum_map(I, E_trace[i], E_int, delay, delay_int)

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
f#33 = frame3

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
ax[0].set_title(f'$E$ = {E_trace[i]} $eV$', fontsize = 18)
ax[0].tick_params(axis='both', labelsize=16)
ax[0].text(-1.9, 1.55,  f"$\Delta$t = {delay} fs", size=16)

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
ax[1].set_ylim(-2,3)
ax[1].text(-1.9, 2.5,  f"$\Delta$t = {delay} fs", size=16)
ax[1].axhline(Ein, linestyle = 'dashed', color = 'black', linewidth = 1)
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

def gaussian(x, amp_1, mean_1, stddev_1, offset):
    
    g1 = amp_1 * np.exp(-0.5*((x - mean_1) / stddev_1)**2)+offset
    
    return g1

def two_gaussians(x, amp_1, amp_2, mean_1, mean_2, stddev_1, stddev_2, offset):
    
    g1 = amp_1 * np.exp(-0.5*((x - mean_1) / stddev_1)**2)
    g2 = amp_2 * np.exp(-0.5*((x - mean_2) / stddev_2)**2)
    
    return g1, g2, offset

def objective(params, x, data):
    
    g1, g2, offset = two_gaussians(x, **params)
    fit = g1+g2+offset
    resid = np.abs(data-fit)**2
    
    return resid


#%% Plot and Fit EDCs

%matplotlib inline

save_figure = True
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

n = len(I.delay)
for t in np.arange(n):
    edc_i = edc_gamma.loc[{"E":slice(e1,e2)}][:,t].values
    edc_i = edc_i/np.max(edc_i)
    
    try:
        popt, _ = curve_fit(gaussian, edc_gamma.loc[{"E":slice(e1,e2)}].E.values, edc_i, p0, method=None, bounds = bnds)
    except ValueError:
        popt = [0,0,0,0]
        
    centers_VBM[t] = popt[1]
    p_fits_VBM[t,:] = popt 

# VBM FIT TESTS
t = 6
gauss_test = gaussian(edc_gamma.E.values, *p_fits_VBM[t,:])

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

#%%

figure_file_name = 'EDC_metis_fits'
save_figure = True

fig, ax = plt.subplots(1, 2)
fig.set_size_inches(10, 4, forward=False)
ax = ax.flatten()

t = 9
ax[0].plot(edc_gamma.E.values, edc_gamma[:,t].values/edc_gamma.loc[{"E":slice(e1,e2)}][:,t].values.max(), color = 'black')
ax[0].plot(edc_gamma.E.values, gauss_test, linestyle = 'dashed', color = 'grey')
#plt.axvline(trunc_e, linestyle = 'dashed', color = 'black')
ax[0].set_xlim([-2,1.5])
ax[0].set_xlabel('E - E$_{VBM}$, eV')
ax[0].set_ylabel('Norm. Int.')
#ax[0].axvline(e1, linestyle = 'dashed', color = 'black')
#ax[0].axvline(e2, linestyle = 'dashed', color = 'black')

# PLOT VBM SHIFT DYNAMICS
#fig = plt.figure()
ax[1].plot(edc_gamma.delay.values, 1000*(p_fits_VBM[:,1] - p_fits_VBM[0:10,1].mean()), color = 'navy')
ax[1].set_xlim([edc_gamma.delay.values[1], edc_gamma.delay.values[-1]])
ax[1].set_ylim([-30,20])
ax[1].set_xlabel('Delay, fs')
ax[1].set_ylabel('$\Delta E_{VBM}$, meV', color = 'navy')


# PLOT VBM PEAK WIDTH DYNAMICS
ax2 = ax[1].twinx()
ax2.plot(edc_gamma.delay.values, 1000*(p_fits_VBM[:,2] - 0*p_fits_VBM[0:10,2].mean()), color = 'maroon')
#ax2.set_ylim([-75,50])
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
