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

#%% #This sets the plots to plot in the IDE window

%matplotlib inline

#%% # Useful Functions and Definitions

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
def get_momentum_map(I_res, E, E_int, delays):
    # Momentum Maps at specified Energies and Delays
        
    frame = I_res[:,:,:,:] # I_sum
    #frame = frame_plot/np.max(frame_plot)
    
    E_ = E
    Ei = (np.abs(ax_E_offset - (E_ - E_int/2))).argmin()
    Ef = (np.abs(ax_E_offset - (E_ + E_int/2))).argmin()
    
    di = (np.abs(ax_delay_offset - (delays[0]))).argmin()
    df = (np.abs(ax_delay_offset - (delays[1]))).argmin() 
    #print(di, df)
    frame = np.transpose(frame[:,:,Ei:Ef,di:df].sum(axis=(2,3)))/(df-di) #average of num of delay points
    
    return frame

# Fucntion for Extracting time Traces
def get_time_traces(i, I, E_trace, E_int, ax_E_offset, t0, k, k_int):
    E_i = np.abs(ax_E_offset - (E_trace-E_int/2)).argmin()    
    E_f = np.abs(ax_E_offset - (E_trace+E_int/2)).argmin()
    kx_i = np.abs(ax_kx - (k[0]-k_int/2)).argmin()    
    kx_f = np.abs(ax_kx - (k[0]+k_int/2)).argmin()
    ky_i = np.abs(ax_kx - (k[1]-k_int/2)).argmin()    
    ky_f = np.abs(ax_kx - (k[1]+k_int/2)).argmin()
    
    trace = I[kx_i:kx_f,ky_i:ky_f,E_i:E_f,:].sum(axis=(0,1,2))

    if subtract_neg is True : 
        trace = trace - np.mean(trace[3:t0-5])
    
    trace = trace/np.max(trace)
    
    return trace

#%%
dkx = (ax_kx[1] - ax_kx[0])
dE = ax_E[1] - ax_E[0]

ax_E_offset = data_handler.ax_E
ax_delay_offset = data_handler.ax_delay
t0 = data_handler.get_t0()

I_sum, I_pos, I_pos_sum, I_neg, I_neg_sum = get_data_chunks([-180,-100], t0, ax_delay_offset) #Get the Neg and Pos delay time arrays

cmap_LTL = plot_manager.custom_colormap(plt.cm.viridis, 0.2) #choose colormap based and percentage of total map for new white transition map

#%% Plot Momentum Maps at Constant Energy

E, E_int = [0, 0.5, 1.6], 0.3 # Energies and Total Energy Integration Window to Plot MMs
delays = [-160, 1200] #Integration range for delays

#######################
figure_file_name = '' 
save_figure = False

%matplotlib inline
fig, ax = plt.subplots(1, len(E), squeeze = False)
ax = ax.flatten()
fig.set_size_inches(8, 5, forward=False)
plt.gcf().set_dpi(300)

for i in np.arange(len(E)):
        
    frame = get_momentum_map(I, E[i], E_int, delays)
    
    extent = [ax_kx[0],ax_kx[-1],ax_ky[0],ax_ky[-1]]
    im = ax[i].imshow((frame), origin='lower', cmap=cmap_plot, clim=None, interpolation='none', extent = extent) #kx, ky, t
    ax[i].set_aspect(1)
    #ax[0].axhline(y,color='black')
    #ax[0].axvline(x,color='bl ack')
    
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

cbar_ax = fig.add_axes([1, 0.325, 0.025, 0.35])
fig.colorbar(im, cax=cbar_ax, ticks = [-1,0,1])
plt.rcParams['svg.fonttype'] = 'none'
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

#%% Plot Dynamics: Extract Traces At Different Energies and Momenta

save_figure = False
figure_file_name = ''

E_trace, E_int = [0.2, 0.6, 1.3], .1 # Energies for Plotting Time Traces ; 1st Energy for MM
k, k_int = [0, 0], 2 # Central (kx, ky) point and k-integration

colors = ['blue', 'purple', 'red'] #colors for plotting the traces

subtract_neg = False #If you want to subtract negative time delay baseline

#######################
### Do the Plotting ###
#######################

fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [.75, 1, 1], 'height_ratios':[1]})
fig.set_size_inches(12 , 4, forward=False)
ax = ax.flatten()
cmap_plot = cmap_LTL

### FIRST PLOT: MM of the First Energy
i = 2
frame = get_momentum_map(I, E_trace[i], E_int, delays)

extent = [ax_kx[0],ax_kx[-1],ax_ky[0],ax_ky[-1]]
im = ax[0].imshow((frame), origin='lower', cmap=cmap_plot, clim=None, interpolation='none', extent = extent) #kx, ky, t
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

# im = ax[0].plot(ax_E_offset, edc_neg, color = 'grey', label = 't < 0 fs')
# im = ax[0].plot(ax_E_offset, edc_pos, color = 'purple', label = 't > 0 fs')
# im = ax[0].plot(ax_E_offset, edc_diff, color = 'green', label = 'Difference', linestyle = 'dashed')

# ax[0].set_ylim(-0.001,0.003)
# ax[0].set_xlabel('Energy, eV', fontsize = 18)
# ax[0].set_ylabel('Norm. Int.', fontsize = 18 )
# ax[0].legend(frameon=False)
# ax[0].set_xticks(np.arange(-1,3.5,0.5))
# for label in ax[0].xaxis.get_ticklabels()[1::2]:
#     label.set_visible(False)
# ax[0].set_xlim(0.5,3)
# #ax[0].axvline(1.35, linestyle = 'dashed', color = 'pink')

### SECOND PLOT: WATERFALL
I_ang_int = I.sum(axis=(0,1))/np.max(I.sum(axis=(0,1)))
energy_limits = [-.5, 2.25]
color_max = 1
waterfall = ax[1].imshow(I_ang_int, clim = [0, .01], origin = 'lower', cmap = cmap_plot, extent=[ax_delay_offset[0], ax_delay_offset[-1], ax_E_offset[0], ax_E_offset[-1]])
#waterfall = ax[1].imshow(diff_ang, clim = clim, origin = 'lower', cmap = cmap_LTL, extent=[ax_delay_offset[0], ax_delay_offset[-1], ax_E_offset[0], ax_E_offset[-1]])
ax[1].set_xlim(ax_delay_offset[1],ax_delay_offset[-1])
ax[1].set_xlabel('Delay, fs', fontsize = 18)
ax[1].set_ylabel('E - E$_{VBM}$, eV', fontsize = 18)
ax[1].set_yticks(np.arange(-1,3.5,0.5))
ax[1].set_ylim(energy_limits)
ax[1].set_title('k-Integrated')

for label in ax[1].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)

hor = ax_delay_offset[-1] - ax_delay_offset[1]
ver =  energy_limits[1] - energy_limits[0]
aspra = hor/ver 
ax[1].set_aspect(aspra/1.5)
ax[1].set_aspect("auto")
fig.colorbar(waterfall, ax=ax[1], shrink = 0.8, ticks = [0, color_max/2, color_max])

### THIRD PLOT: DELAY TRACES (Angle-Integrated)
for i in np.arange(len(E_trace)):
    trace = get_time_traces(i, I, E_trace[i], E_int, ax_E_offset, t0, k, k_int)

    ax[2].plot(ax_delay_offset, trace, color = colors[i], label = str(E_trace[i]) + ' eV')
    ax[1].axhline(E_trace[i]-E_int/2, linestyle = 'dashed', color = colors[i], linewidth = 1.5)
    ax[1].axhline(E_trace[i]+E_int/2, linestyle = 'dashed', color = colors[i], linewidth = 1.5)
    rect = (Rectangle((k[0]-k_int, k[1]-k_int), 2*k_int, 2*k_int, linewidth=1.5,\
                      edgecolor=colors[i], facecolor='None'))
    ax[0].add_patch(rect)
    
ax[2].set_xlim(ax_delay_offset[1],ax_delay_offset[-1])
ax[2].set_ylim(-0.1, 1.1)
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
    