# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 11:15:40 2025

@author: lloyd
""" 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as col
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
import xarray as xr

#%% Useful Functions and Definitions for Manipulating Data

# Partition Data into + and - Delays
def get_data_chunks(I, neg_times, t0, ax_delay_offset):
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
    
    if I_res.ndim > 3:    
        frame = I_res.loc[{"E":slice(E-E_int/2, E+E_int/2), "delay":slice(delays-delay_int/2, delays+delay_int/2)}].mean(dim=("E","delay")).T
    
    else:
        frame = I_res.loc[{"E":slice(E-E_int/2, E+E_int/2)}].mean(dim=("E")).T

    return frame

def get_kx_E_frame(I_res, ky, ky_int, delay, delay_int):
    
    if I_res.ndim > 3:    
        frame = I_res.loc[{"ky":slice(ky-ky_int/2, ky+ky_int/2), "delay":slice(delay-delay_int/2, delay+delay_int/2)}].mean(dim="ky").mean(dim="delay")
    
    else:
        frame = I_res.loc[{"ky":slice(ky-ky_int/2, ky+ky_int/2)}].mean(dim="ky")

    return frame

def get_ky_E_frame(I_res, kx, kx_int, delay, delay_int):

    if I_res.ndim > 3:    
        frame = I_res.loc[{"kx":slice(kx-kx_int/2, kx+kx_int/2), "delay":slice(delay-delay_int/2, delay+delay_int/2)}].mean(dim="kx").mean(dim="delay")
    else:
        frame = I_res.loc[{"kx":slice(kx-kx_int/2, kx+kx_int/2)}].mean(dim="kx")

    return frame

def get_waterfall(I_res, kx, kx_int, ky, ky_int):
    
    frame = I_res.loc[{"kx":slice(kx-kx_int/2, kx+kx_int/2), "ky":slice(ky-ky_int/2, ky+ky_int/2)}].mean(dim=("kx","ky"))

    return frame

# Fucntion for Extracting time Traces
def get_time_trace(I_res, E, E_int, k , k_int, norm_trace, subtract_neg, neg_delays):
    
    (kx, ky) = k
    (kx_int, ky_int) = k_int
    d1, d2 = neg_delays[0], neg_delays[1]
    
    trace = I_res.loc[{"E":slice(E-E_int/2, E+E_int/2), "kx":slice(kx-kx_int/2, kx+kx_int/2), "ky":slice(ky-ky_int/2, ky+ky_int/2)}].mean(dim=("kx", "ky", "E"))

    if subtract_neg is True : 
        trace = trace - np.mean(trace.loc[{"delay":slice(d1,d2)}])
    
    if norm_trace is True : 
        trace = trace/np.max(trace)
    
    return trace

def get_edc(I_res, kx, ky, k_int, delay, delay_int):
    
    (kx_int, ky_int) = k_int
    
    if I_res.ndim > 3:    

        edc = I_res.loc[{"kx":slice(kx-kx_int/2, kx+kx_int/2), "ky":slice(ky-ky_int/2, ky+ky_int/2), "delay":slice(delay-delay_int/2,delay+delay_int/2)}].mean(dim=("kx", "ky", "delay"))
    else:
        edc = I_res.loc[{"kx":slice(kx-kx_int/2, kx+kx_int/2), "ky":slice(ky-ky_int/2, ky+ky_int/2)}].mean(dim=("kx", "ky"))

    return edc

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
        frame_neg = get_momentum_map(I, E[i], E_int, -140, 50)
        #frame = frame - frame_neg
        
        f_norm = np.max(frame)
        frame = frame/f_norm
        
        im = frame.plot.imshow(ax = ax[i], clim = None, vmin = 0, vmax = 1, cmap = cmap_plot, add_colorbar=False)
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

def create_custom_diverging_colormap():
    # Create the negative part from seismic (from 0.5 to 1 -> blue to white)
    seismic = plt.get_cmap('Blues')
    seismic_colors = seismic(np.linspace(0, 1, 128))  # -1 to 0

    # Create the positive part from viridis (from 0 to 1)
    viridis = plt.get_cmap('viridis')
    viridis = cmap_LTL
    viridis_colors = viridis(np.linspace(0, 1, 128))  # 0 to 1

    # Combine both
    combined_colors = np.vstack((seismic_colors[::-1], viridis_colors))

    # Create a new colormap
    custom_colormap = LinearSegmentedColormap.from_list('seismic_viridis', combined_colors)

    return custom_colormap

cmap_LTL = custom_colormap(mpl.cm.viridis, 0.2)
cmap_LTL2 = create_custom_diverging_colormap()
