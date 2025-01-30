#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 08:47:53 2024

@author: lawsonlloyd
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
from skimage.draw import disk
from scipy.optimize import curve_fit
from obspy.imaging.cm import viridis_white

from Loader import DataLoader
from Main import main
from Manager import DataHandler, FigureHandler, PlotHandler, ValueHandler, SliderManager, EventHandler, CheckButtonManager, ClickButtonManager

#%% Specifiy filename of h5 file in your path.
# Include manual energy and time delay offsets for the axes, if required.

data_path = 'path_to_your_data'
filename = 'your_file_name.h5'

data_path = 'R:\Lawson\Data\metis'
#data_path = '/Users/lawsonlloyd/Desktop/Data/'
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
        
    frame = I_res.loc[{"E":slice(E-E_int/2, E+Eint/2), "delay":slice(delays-delay_int/2, delays+delay_int/2)}].mean(dim=("E","delay")).T
                             
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
    
    trace = I_res.loc[{"E":slice(E-E_int/2, E+Eint/2), "kx":slice(kx-kx_int/2, kx+kx_int/2), "ky":slice(ky-ky_int/2, ky+ky_int/2)}].sum(axis=(0,1,2))

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
        
        im = frame.plot(ax = ax[i], clim = None, cmap = cmap_plot, add_colorbar=False)
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

cmap_LTL = plot_manager.custom_colormap(plt.cm.viridis, 0.2) #choose colormap based and percentage of total map for new white transition map

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
    fig.savefig((figure_file_name +'.pdf'), format='pdf')


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
im = frame.plot(ax = ax[i], clim = None, cmap = cmap_plot, add_colorbar=False)
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
    
    trace = get_time_traces(I, E_trace[i], E_int, k , k_int, subtract_neg, norm_trace)
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
    fig.savefig((figure_file_name +'.svg'), format='svg')
    
#%% Plot kx vs E

save_figure = True
figure_file_name = 'MM_ARPES_delay_frames_'

E_trace, E_int = [1.35, 2.1], .12 # Energies for Plotting Time Traces ; 1st Energy for MM
k, k_int = (0, 0), .2 # Central (kx, ky) point and k-integration
delay, delay_int = 25, 50

colors = ['black', 'red'] #colors for plotting the traces
cmap_plot = cmap_LTL

i = 0

(kx, ky) = k
I_res = I/I.max()
I_res_enh = enhance_features(I_res, Ein, factor = 0, norm = True)

frame = get_momentum_map(I, E_trace[i], E_int, delay, delay_int)

#Norm to t0 Frame
frame_t0 = get_kx_E_frame(I_res_enh, ky, k_int, 0, delay_int)
frame_t0_2 = get_ky_E_frame(I_res_enh, ky, k_int, 0, delay_int)

n = [frame_t0.loc[{"E":slice(-3,Ein)}].max().values, frame_t0.loc[{"E":slice(Ein,5)}].max().values]
n2 = [frame_t0_2.loc[{"E":slice(-3,Ein)}].max().values, frame_t0_2.loc[{"E":slice(Ein,5)}].max().values]

frame2 = get_kx_E_frame(I_res_enh, ky, k_int, delay, delay_int)
frame3 = get_ky_E_frame(I_res_enh, kx, k_int, delay, delay_int)

Ein = 0.75

f23 = enhance_features(frame2, Ein, factor = n, norm = True)
f33 = enhance_features(frame3, Ein, factor = n2, norm = True)

#######################
### Do the Plotting ###
#######################
    
fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [.75, 1, 1], 'height_ratios':[1]})
fig.set_size_inches(10, 4, forward=False)
ax = ax.flatten()

### FIRST PLOT: MM of the First Energy
im = frame.plot(ax = ax[0], cmap = cmap_plot, add_colorbar=False)
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
im2 = f23.T.plot(ax=ax[1], cmap=cmap_plot, add_colorbar=False) #kx, ky, t
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

im3 = f33.T.plot(ax=ax[2], cmap=cmap_plot, add_colorbar=False) #kx, ky, t
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
    fig.savefig((figure_file_name +'.pdf'), format='pdf')


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


#%%

%matplotlib inline

I_res = I/np.max(I)


fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 1], 'height_ratios':[1]})
fig.set_size_inches(10, 4, forward=False)
ax = ax.flatten()

### Plot EDCs at GAMMA vs time

(kx, ky), k_int = (0, 0), 0.2
edc_gamma = I_res.loc[{"kx":slice(kx-k_int/2,kx+k_int/2), "ky":slice(ky-k_int/2,ky+k_int/2)}].sum(dim=("kx","ky"))
edc_gamma = edc_gamma/np.max(edc_gamma)

im = edc_gamma.plot(ax = ax[0], cmap = cmap_LTL, add_colorbar = False)
cbar_ax = fig.add_axes([1, 0.275, 0.025, 0.5])
cbar = fig.colorbar(im, cax=cbar_ax, ticks = [0,1])
cbar.ax.set_yticklabels(['min', 'max'])  # vertically oriented colorbar

ax[0].set_ylim([-1,1])
ax[0].set_xlim([-160,800])
#plt.axhline(0, linestyle = 'dashed', color = 'black')
#plt.axvline(0, linestyle = 'dashed', color = 'black')
plt.xlabel('Delay, fs')
plt.ylabel('Energy, eV')

pts = [100, 200]
colors = ['black', 'red', 'orange', 'purple', 'blue', 'green', 'grey']

for i in np.arange(len(pts)):
    edc = I_res.loc[{"kx":slice(kx-k_int/2,kx+k_int/2), "ky":slice(ky-k_int/2,ky+k_int/2), "delay":slice(pts[i]-5:pts[i]+5)}].sum(dim=("kx","ky","delay"))
    edc = edc/np.max(edc)
    
    e = edc.plot(ax = ax[1], cmap = cmap_LTL, color = colors[i])

#plt.legend(frameon = False)
plt.xlim([-2, 1]) 
plt.ylim([0, 1.5])
plt.ylabel('Norm. Int. + offset, arb. units.')
plt.xlabel('Energy, eV')
plt.axvline(0, color = 'black', linestyle = 'dashed', linewidth = 0.5)
plt.gca().set_aspect(2)

# Fit to Gaussian
#################

##### VBM #####
trunc_e1 = -1
_, _, trunc1, _ = data_handler.get_closest_indices(0, 0, trunc_e1, 0)
trunc_e2 = 0.5
_, _, trunc2, _ = data_handler.get_closest_indices(0, 0, trunc_e2, 0)

p0 = [1, -.3, .5, 0] # Fitting params initial guess [amp, center, width, offset]
bnds = ((0.5, -1, 0.0, 0), (1.5, 0.5, 1, .5))

centers_VBM = np.zeros(len(data_handler.ax_delay))
p_fits_VBM = np.zeros((len(data_handler.ax_delay),4))

for t in np.arange(len(data_handler.ax_delay)):
    try:
        popt, _ = curve_fit(gaussian, data_handler.ax_E[trunc1:trunc2], edcs[trunc1:trunc2,t]/np.max(edcs[trunc1-10:,t]), p0, method=None, bounds = bnds)
    except ValueError:
        popt = [0,0,0,0]
        
    centers_VBM[t] = popt[1]
    p_fits_VBM[t,:] = popt 

# VBM FIT TESTS
t = 40
gauss_test = gaussian(data_handler.ax_E, *p_fits_VBM[t,:])

fig = plt.figure()
plt.plot(data_handler.ax_E, edcs[:,t]/np.max(edcs[trunc1-10:,t]))
plt.plot(data_handler.ax_E, gauss_test, linestyle = 'dashed', color = 'black')
#plt.axvline(trunc_e, linestyle = 'dashed', color = 'black')
plt.xlim([-2,1.5])
plt.xlabel('Energy, eV')
plt.ylabel('Norm. Int, arb. u.')
plt.gca().set_aspect(3)

# PLOT VBM SHIFT DYNAMICS
fig = plt.figure()
plt.plot(data_handler.ax_delay, 1000*(centers_VBM-np.mean(centers_VBM[5:15])), color = 'black', linestyle = 'solid')
plt.xlim([-200, 500])
plt.ylim([-100,100])
plt.xlabel('Delay, fs')
plt.ylabel('Energy Shift, meV')
#plt.axhline(0, linestyle = 'dashed', color = 'black')
#plt.axvline(0, linestyle = 'dashed', color = 'black')

# PLOT VBM PEAK WIDTH DYNAMICS
fig = plt.figure()
plt.plot(data_handler.ax_delay, 1000*p_fits_VBM[:,2], color = 'black', linestyle = 'solid')
plt.xlim([-200, 500])
plt.ylim([520,560])
plt.xlabel('Delay, fs')
plt.ylabel('VBM Peak width, meV')

#%%%

k_int, kx, ky, E, delay = value_manager.get_values()
idx_kx, idx_ky, idx_E, idx_delay = data_handler.get_closest_indices(0, 0, E, delay)
kx_int, ky_int = 2.2, 0.5
dt = data_handler.calculate_dt()

idx_kx_int = round(0.5*kx_int/data_handler.calculate_dk())
idx_ky_int = round(0.5*ky_int/data_handler.calculate_dk())

edcs = I[idx_kx-idx_kx_int:idx_kx+idx_kx_int, idx_ky-idx_ky_int:idx_ky+idx_ky_int, :, :].sum(axis=(0,1))
edcs = edcs/np.max(edcs[:])

# Define Fit Parameters
fit_params = Parameters()
fit_params.add("amp_1", value=1, min=0, max=2, vary=True)
fit_params.add("amp_2", value=.25, min=0.075, max=1, vary=True)
fit_params.add("mean_1", value=1.1, min=.95, max=1.25, vary=True)
fit_params.add("mean_2", value=1.79, min=1.77, max=2.05, vary=True)
fit_params.add("stddev_1", value=0.1, min=0.055, max=0.135, vary=True)
fit_params.add("stddev_2", value=0.06, min=0.035, max=0.12, vary=True)
fit_params.add("offset", value=0.00, min=0, max=0.01, vary=True)

start_e = 0.8
_, _, start, _ = data_handler.get_closest_indices(0, 0, start_e, 0)
stop_e = 2.5
_, _, stop, _ = data_handler.get_closest_indices(0, 0, stop_e, 0)

N = 3
delay_t = [0, 20, 40, 100, 150, 200, 250, 350, 450, 600]

for tt in delay_t:
    _, _, _, t = data_handler.get_closest_indices(0, 0, 0, tt)
    
    edc_test = np.mean(edcs[:,t:t+N], axis = 1)
    #edc_test = edc_test/np.max(edc_test)
    edc_test = edc_test -  np.mean(edcs[:,5:15], axis = 1)
    edc_test = edc_test/np.max(edc_test[start:])
    x = data_handler.ax_E[start:stop]
    
    g1_test, g2_test, offset_test = two_gaussians(data_handler.ax_E, **fit_params)
    test_gauss = g1_test+g2_test+offset_test
    
    output = minimize(objective, fit_params, args = (x, edc_test[start:stop]))
    m1, m2 = output.params.valuesdict()[('mean_1')], output.params.valuesdict()[('mean_2')]
    w1, w2 = output.params.valuesdict()[('stddev_1')], output.params.valuesdict()[('stddev_2')]
    g1_fit, g2_fit, offset_fit  = two_gaussians(data_handler.ax_E, **output.params)
    fit_gauss = g1_fit+g2_fit+offset_fit
    
    ### PLOT TEST CASE
    fig = plt.figure()
    plt.plot(data_handler.ax_E, edc_test, color = 'black', linestyle = 'solid')
    plt.plot(data_handler.ax_E, g1_fit, linestyle = 'dashed', color = 'grey',  label = str(round(m1,2)) + ', '  + str(round(w1,2)) + ' eV')
    plt.plot(data_handler.ax_E, g2_fit, linestyle = 'dashed', color = 'red', label = str(round(m2,2)) + ', '  + str(round(w2,2)) + ' eV')
    plt.plot(data_handler.ax_E, fit_gauss, linestyle = 'dashed', color = 'blue')
    plt.xlim([0.2,2.7])
    plt.ylim([-0.05,1.2])
    plt.xlabel('Energy, eV')
    plt.ylabel('Norm. Int, arb. u.')
    plt.title('t = ' + str(tt) + ' fs, ' + 'N = ' + str(round(N*dt,1)) + ' fs')
    plt.axvline(start_e, linestyle = 'dashed', color = 'grey')
    plt.axvline(stop_e, linestyle = 'dashed', color = 'grey')
    plt.legend(frameon=False)
    plt.gca().set_aspect(1.5)


##### FIT FOR ALL DELAY TIMES

N = 3
centers_excited = np.zeros(len(data_handler.ax_delay))
output_excited = np.zeros((len(data_handler.ax_delay),4))

x = data_handler.ax_E[start:stop]

for t in np.arange(len(data_handler.ax_delay)):
    data = np.mean(edcs[:,t:t+N], axis = 1)
    data = data -  np.mean(edcs[:,5:15], axis = 1)
    data = data/np.max(data[start:])
    output = minimize(objective, fit_params, args = (x, data[start:stop]))
    m1, m2 = output.params.valuesdict()[('mean_1')], output.params.valuesdict()[('mean_2')]
    w1, w2 = output.params.valuesdict()[('stddev_1')], output.params.valuesdict()[('stddev_2')]

    output_excited[t,:] = [m1,m2,w1,w2] 

# PLOT EX and CB SHIFT DYNAMICS
fig = plt.figure()
plt.plot(data_handler.ax_delay[t0-8:], 1000*(output_excited[t0-8:,0]-np.mean(output_excited[5:15,0])), color = 'black', linestyle = 'solid', label = 'X')
plt.plot(data_handler.ax_delay[t0-8:], 1000*(output_excited[t0-8:,1]-np.mean(output_excited[5:15,1])), color = 'red', linestyle = 'solid', label = 'CB')
plt.plot(data_handler.ax_delay[t0-8:], 1000*(centers_VBM[t0-8:]-np.mean(centers_VBM[5:15])), color = 'grey', linestyle = 'solid', label = 'VBM')
plt.xlim([-40, 800])
plt.ylim([-120,150])
plt.xlabel('Delay, fs')
plt.ylabel('Energy Shift, meV')
plt.title('Peak Position Shift, N = ' + str(round(N*dt,1)) + ' fs')
plt.legend(frameon=False)

#plt.axhline(0, linestyle = 'dashed', color = 'black')

# PLOT E_b SHIFT DYNAMICS
eb_ = 1000*( output_excited[:,1] - output_excited[:,0] )
eb_ = eb_ - np.mean(eb_[5:15])
fig = plt.figure()
plt.plot(data_handler.ax_delay[t0-8:], eb_[t0-8:], color = 'purple', linestyle = 'solid', label='Extracted $E_{b}$')
plt.axhline(0, linestyle = 'dashed', color = 'grey')
plt.xlim([-40, 800])
plt.ylim([-120, 200])
plt.title('$E_{b}$ Shift, N = ' + str(round(N*dt,1)) + ' fs')
plt.xlabel('Delay, fs')
plt.ylabel('Energy Shift, meV')
plt.legend(frameon=False)
#plt.axhline(0, linestyle = 'dashed', color = 'black')

# PLOT EX and CB WIDTH DYNAMICS
fig = plt.figure()
plt.plot(data_handler.ax_delay[t0-8:], 1000*(output_excited[t0-8:,2]-np.mean(output_excited[5:15,2])), color = 'black', linestyle = 'solid', label = 'X')
plt.plot(data_handler.ax_delay[t0-8:], 1000*(output_excited[t0-8:,3]-np.mean(output_excited[5:15,3])), color = 'red', linestyle = 'solid', label = 'CB')
plt.plot(data_handler.ax_delay[t0-8:], 1000*(p_fits_VBM[t0-8:,2]-np.mean(p_fits_VBM[:,2][5:15])), color = 'grey', linestyle = 'solid', label = 'VBM')
plt.xlim([-40, 800])
plt.ylim([-50,50])
plt.title('Peak Width Shift, N = ' + str(round(N*dt,1)) + ' fs')
plt.xlabel('Delay, fs')
plt.ylabel('Peak Width Shift, meV')
plt.legend(frameon=False)

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
