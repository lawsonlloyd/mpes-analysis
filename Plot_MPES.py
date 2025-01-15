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

#%%
%matplotlib inline

#%%
### Transform Data if needed....
#I = np.transpose(I, (0,2,1,3))

dkx = (ax_kx[1] - ax_kx[0])
dE = ax_E[1] - ax_E[0]

ax_E_offset = data_handler.ax_E
ax_delay_offset = data_handler.ax_delay

if I.ndim > 3:
    t0 = data_handler.get_t0()
    
    neg_time = -70
    tnf = (np.abs(ax_delay_offset - neg_time)).argmin()
    I_neg = I[:,:,:,6:tnf+1] #Sum over delay/polarization/theta...
    neg_length = I_neg.shape[3]
    I_neg = I_neg.sum(axis=(3))
        
    I_pos = I[:,:,:,t0+1:]
    pos_length = I_pos.shape[3]
    I_pos = I_pos.sum(axis=(3)) #Sum over delay/polarization/theta...
    
    I_sum = I[:,:,:,:].sum(axis=(3))    

else:
    I_neg = I[:,:,:] #Sum over delay/polarization/theta...
    I_pos = I[:,:,:]
    I_sum = I

cmap_LTL = plot_manager.custom_colormap(plt.cm.viridis, 0.2) #choose colormap based and percentage of total map for new white transition map

#%%
### User Inputs for Plotting MM 

def plot_momentum_map(I_res, E, E_int, fig):
    
    # Plot Momentum Maps at specified Energies
    
    cmap = cmap_LTL 
    
    frame_plot = I_res # I_sum
    frame_plot = frame_plot/np.max(frame_plot)
    
    #### 
    cmap_plot = cmap_LTL
    
    for i in np.arange(len(E)):
        E_ = E[i]
    
        Ei = (np.abs(ax_E_offset - (E_ - E_int/2))).argmin()
        Ef = (np.abs(ax_E_offset - (E_ + E_int/2))).argmin()    
        frame = np.transpose(frame_plot[:,:,Ei:Ef].sum(axis=(2)))
        frame = frame/np.max(frame)
        
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
        #plt.show()

#%%

E = [1.35, 1.55, 1.8]

figure_file_name = 'MMs_negativedelay_RT' 
save_figure = True

fig, ax = plt.subplots(1, len(E), squeeze = False)
ax = ax.flatten()
fig.set_size_inches(8, 5, forward=False)
plt.gcf().set_dpi(300)

plot_momentum_map(I_neg, E, 0.2, fig)

if save_figure is True:
    fig.savefig((figure_file_name +'.svg'), format='svg')


#%%
# Plot Difference MMs of t < 0 and t > 0 fs

%matplotlib inline

save_figure = False
figure_file_name = 'MM_DIFFERENCE'

tMaps, tint  = [1.35, 2], 6

cmapPLOTTING = cmap_LTL #'bone_r' # cmap_LTL

difference_FRAMES = np.zeros((numPlots,I_pos.shape[0],I_pos.shape[1]))

frame_neg = (I_neg[:,:,:])
frame_pos = (I_pos[:,:,:])

#frame_neg = frame_neg/(np.max(frame_neg))
#frame_pos = frame_pos/(np.max(frame_pos))
frame_neg = frame_neg/((neg_length))
frame_pos = frame_pos/((pos_length))
frame_sum = frame_neg + frame_pos

#cts_pos = np.sum(frame_pos[:,:])
#cts_neg = np.sum(frame_neg[:,:])
#cts_total = np.sum(frame_sum)

frame_diff = frame_pos - frame_neg
#frame_diff = frame_pos

frame_maxes = np.zeros(frame_diff.shape[2])
for m in np.arange(frame_diff.shape[2]):
    frame_maxes[m] = np.max(np.abs(frame_diff[:,:,m]))
#frame_diff = frame_diff/(cts_total)
#frame_diff = np.divide(frame_diff, frame_sum)
frame_differences = np.divide(frame_diff,  frame_maxes)

### Plot
numPlots = len(tMaps)

fig, ax = plt.subplots(1, numPlots+1, sharey=False)
plt.gcf().set_dpi(300)
ax = ax.flatten()

sat = [1, 1, 1]
for i in np.arange(numPlots, dtype = int):
    tMap = tMaps[i]
    tMap = (np.abs(ax_E_offset - tMap)).argmin()
    
    frame = frame_differences[:,:,tMap-int(tint/2):tMap+int(tint/2)].sum(axis=(2))
    #frame = frame - np.min(frame)
    frame = frame/np.max(frame)
    #frame = abs(frame)
    difference_FRAMES[i,:,:] = frame    

    extent =  extent=[ax_kx[0],ax_kx[-1],ax_ky[0],ax_ky[-1]]
    im = ax[i].imshow(np.transpose(frame), origin='lower', cmap=cmapPLOTTING, clim=[0,1], interpolation='none', extent=extent) #kx, ky, t
    
    #im = ax[i].imshow(np.transpose(I[:,:,tMap-int(tint/2):tMap+int(tint/2), adc-1:adc+1].sum(axis=(2,3))), origin='lower', cmap='terrain_r', clim=None, interpolation='none', extent=[-2,2,-2,2]) #kx, ky, t
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
    ax[i].set_xlabel('$k_x$, $A^{-1}$', fontsize = 14)
    ax[i].set_ylabel('$k_y$, $A^{-1}$', fontsize = 14)
    ax[i].tick_params(axis='both', labelsize=12)
    ax[i].set_title('$E$ = ' + str((tMaps[i])) + ' eV', fontsize = 14)
    #ax[i].annotate(('E = '+ str(round(tMaps[i],2)) + ' eV'), xy = (-1.85, 1.6), fontsize = 14, weight = 'bold')

delta_MM = difference_FRAMES[0,:,:] - difference_FRAMES[1,:,:]
delta_MM = delta_MM/np.max(np.abs(delta_MM))

i = 2
extent=[ax_kx[0],ax_kx[-1],ax_ky[0],ax_ky[-1]]
im = ax[i].imshow(np.transpose(delta_MM), origin='lower', cmap='seismic', clim=[-1,1], interpolation='none', extent=extent) #kx, ky, t

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
ax[i].set_xlabel('$k_x$, $A^{-1}$', fontsize = 14)
ax[i].set_ylabel('$k_y$, $A^{-1}$', fontsize = 14)
ax[i].tick_params(axis='both', labelsize=12)
ax[i].set_title('$\\Delta$MM ', fontsize = 14)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([1, 0.325, 0.025, 0.35])
fig.colorbar(im, cax=cbar_ax, ticks = [-1,0,1])

#fig.colorbar(im, fraction=0.046, pad=0.04)
fig.tight_layout()
plt.show()

if save_figure is True:
    fig.savefig((figure_file_name +'.svg'), format='svg')

#%%
%matplotlib inline

# Plot Angle Integrated Dynamics

save_figure = True
figure_file_name = 'angle_integrated'

E_trace = [1.35, 2.05, 0.6] # Energies for Plotting
thirdtrace = 0

################################
# Operations to Extract Traces #
################################

### Negative Delays Background Subtraction
e_ = 0.5
e = np.abs(ax_E_offset - e_).argmin()

edc_neg = I_neg.sum(axis=(0,1))
edc_pos = I_pos.sum(axis=(0,1))

edc_neg = edc_neg/neg_length
edc_pos = edc_pos/pos_length

e_n = -.8
e_n_ = np.abs(ax_E_offset - e_n).argmin()
norm_neg = np.max(edc_neg[e_n_:])
norm_pos = np.max(edc_pos)
edc_neg = edc_neg/norm_neg
edc_pos = edc_pos/norm_neg
edc_diff = edc_pos - edc_neg
#edc_diff = .1*edc_diff/np.max(edc_diff)

ang_int = I[:,:,:,:].sum(axis=(0,1)) #Angle Integrated Spectra
n = np.max(ang_int)
ang_int = ang_int/np.max(ang_int)

ang_int_neg = ang_int[:,5:tnf+1].sum(axis=1)
ang_int_neg = np.expand_dims(ang_int_neg, axis=-1) # Add an extra dimension in the last axis.
ang_int_neg = ang_int_neg/((ang_int[:,5:tnf+1]).shape[1])#np.max(ang_int_neg)

diff_ang = ang_int - ang_int_neg
diff_ang = diff_ang/np.max(diff_ang)

###
mask_start = (np.abs(ax_E_offset - 0.95)).argmin()
sat_start = (np.abs(ax_E_offset - 0.5)).argmin()

diff_ang[:,:] *= 1/np.max(diff_ang[mask_start:,:])
satmax = np.max(diff_ang[sat_start:,:])
clim = [0, 1]

# Extract Traces for At Different Energies
E_ = [0, 0, 0]
E_[0] = np.abs(ax_E_offset - E_trace[0]).argmin()    
E_[1] = np.abs(ax_E_offset - E_trace[1]).argmin()
E_[2] = np.abs(ax_E_offset - E_trace[2]).argmin()      

trace_1 = ang_int[E_[0]-2:E_[0]+3,:].sum(axis=0)
trace_2 = ang_int[E_[1]-2:E_[1]+3,:].sum(axis=0)
trace_3 = ang_int[E_[2]-2:E_[2]+3,:].sum(axis=0)

trace_1 = trace_1 - np.mean(trace_1[6:t0-10])
trace_2 = trace_2 - np.mean(trace_2[6:t0-10])
trace_3 = trace_3 - np.mean(trace_3[6:t0-10])

trace_1 = trace_1/np.max(trace_1)
trace_2 = trace_2/np.max(trace_2)
trace_3 = trace_3/np.max(trace_3)

#######################
### Do the Plotting ###
#######################

fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1.5, 1.5], 'height_ratios':[1]})
fig.set_size_inches(12 , 4, forward=False)

ax = ax.flatten()
cmap_plot = cmap_LTL

im = ax[0].plot(ax_E_offset, edc_neg, color = 'grey', label = 't < 0 fs')
im = ax[0].plot(ax_E_offset, edc_pos, color = 'purple', label = 't > 0 fs')
im = ax[0].plot(ax_E_offset, edc_diff, color = 'green', label = 'Difference', linestyle = 'dashed')

#ax[0].axvline(1.55, color = 'grey', linestyle = 'dashed')
#ax[0].axvline(1.15, color = 'black', linestyle = 'dashed')
#ax[0].axvline(2, color = 'black', linestyle = 'dashed')

ax[0].set_ylim(0,0.003)
ax[0].set_xlabel('Energy, eV', fontsize = 18)
ax[0].set_ylabel('Norm. Int.', fontsize = 18 )
ax[0].legend(frameon=False)
ax[0].set_xticks(np.arange(-1,3.5,0.5))
for label in ax[0].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
ax[0].set_xlim(0.5,3)
#ax[0].axvline(1.35, linestyle = 'dashed', color = 'pink')

color_max = 1
waterfall = ax[1].imshow(ang_int, clim = [0, .01], origin = 'lower', cmap = cmap_plot, extent=[ax_delay_offset[0], ax_delay_offset[-1], ax_E_offset[0], ax_E_offset[-1]])
waterfall = ax[1].imshow(diff_ang, clim = clim, origin = 'lower', cmap = cmap_LTL, extent=[ax_delay_offset[0], ax_delay_offset[-1], ax_E_offset[0], ax_E_offset[-1]])
ax[1].set_xlim(-150,820)
ax[1].set_xlabel('Delay, fs', fontsize = 18)
ax[1].set_ylabel('E - E$_{VBM}$, eV', fontsize = 18)
ax[1].set_yticks(np.arange(-1,3.5,0.5))
ax[1].set_ylim(-.1, 3)

for label in ax[1].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)

ax[1].set_aspect(300)

ax[1].axhline(E_trace[0], linestyle = 'dashed', color = 'black')
ax[1].axhline(E_trace[1], linestyle = 'dashed', color = 'red')

fig.colorbar(waterfall, ax=ax[1], shrink = 0.8, ticks = [0, color_max/2, color_max])

ax[2].plot(ax_delay_offset, trace_1, color = 'black', label = str(E_trace[0]) + ' eV')
ax[2].plot(ax_delay_offset, trace_2, color = 'red', label = str(E_trace[1]) + ' eV')

if thirdtrace:
    ax[1].axhline(E_trace[2], linestyle = 'dashed', color = 'purple')
    ax[2].plot(ax_delay_offset, trace_3, color = 'purple', label = str(E_trace[2]) + ' eV')

ax[2].set_xlim(-150,820)
ax[2].set_ylim(-0.1, 1.1)
ax[2].set_ylabel('Norm. Int.', fontsize = 18)
ax[2].set_xlabel('Delay, fs', fontsize = 18)
ax[2].legend(frameon = False)

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

# Plot Dynamics at Distinct Momenta and/or Energy Points

save_figure = True
figure_file_name = 'Dynamics_at_Points_RT'

kx_traces, ky_traces = [0.2, 0.2, -1.05, -1.05, 1.25, 1.25], [0] # kx, ky for plotting
E_traces = [1.35, 2.1, 2.1, 1.35, 1.35, 2.1] # Energies for Plotting
kx_int, ky_int, E_int  = 0.4, .5, 0.2 #Integration Ranges

trace_colors = ['black', 'red', 'pink', 'grey', 'purple', 'green']

cmap_to_plot = cmap_LTL
#cmap_to_plot = 'magma_r'
clim = [-0.1, 1]
delay_lim = [-160, 820]
################################
# Operations to Extract Traces #
################################

## Extract Traces for At Different Energies with Background Subtraction
#traces = np.zeros((4,I.shape[3]))
traces = np.zeros((6,50))

E_int, kx_int, ky_int = E_int/2, kx_int/2, ky_int/2

for t in range(6):
    kxi = (np.abs(ax_kx - (kx_traces[t]-kx_int))).argmin()
    kxf = (np.abs(ax_kx - (kx_traces[t]+kx_int))).argmin()
    kyi = (np.abs(ax_ky - (ky_traces[0]-ky_int))).argmin()
    kyf = (np.abs(ax_ky - (ky_traces[0]+ky_int))).argmin()
    Ei = np.abs(ax_E_offset - (E_traces[t]-E_int)).argmin()  
    Ef = np.abs(ax_E_offset - (E_traces[t]+E_int)).argmin()
    trace = I[kxi:kxf, kyi:kyf, Ei:Ef,:].sum(axis=(0,1,2))

    trace = binArray(trace, 0, 3, 3, np.mean)
    trace = trace - np.mean(trace[1:8-3])
    
    if t == 0:
        norm_factor = np.max(trace)
    
    if t == 1:   
        trace = trace/np.max(trace)
   
    else:
        trace = trace/np.max(trace)
    
    traces[t,:] = trace
    
# MM Frame Difference
MM_frame_diff = frame_differences[:,:,Ei:Ef].sum(axis=(2))
MM_frame_diff = MM_frame_diff/np.max(MM_frame_diff)

# Kx Frame Difference
mask_start = 0.95

I_sum_1 = (I[:,:,:,5:t0-8]) #Sum over delay/theta/ADC for Plotting...
len_1 = I_sum_1.shape[3]
I_sum_1 = I_sum_1.sum(axis=3)/len_1

I_sum_2 = (I[:,:,:,t0+1:-1])
len_2 = I_sum_2.shape[3]              
I_sum_2 = I_sum_2.sum(axis=(3))/len_2 #Sum over delay/theta/ADC for Plotting...

e = np.abs(ax_E_offset - mask_start).argmin()

slice_E_k_1 = I_sum_1[:,kyi:kyf,e:].sum(axis=(1))
norm_1 = np.max(slice_E_k_1)
slice_E_k_1 = slice_E_k_1/norm_1

slice_E_k_2 = I_sum_2[:,kyi:kyf,e:].sum(axis=(1))
slice_E_k_2 = slice_E_k_2/norm_1

I_1_N = slice_E_k_1/np.max(np.abs(slice_E_k_1))
I_2_N = slice_E_k_2/np.max(np.abs(slice_E_k_2))

I_dif = I_2_N - I_1_N

kx_diff = I_2_N - I_1_N
kx_diff = kx_diff/np.max(kx_diff)

energy_cut_kx = kx_diff.sum(axis=0)

#######################
### Do the Plotting ###
#######################

fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 1], 'height_ratios':[1, 1]})
fig.set_size_inches(8, 6, forward=False)
plt.gcf().set_dpi(300)
ax = ax.flatten()
cmap_plot = cmap_LTL

### MM Plot
im = ax[0].imshow(np.transpose(MM_frame_diff), origin='lower', cmap=cmap_to_plot, clim=[0,1], interpolation='none', extent=[ax_kx[0],ax_kx[-1],ax_ky[0],ax_ky[-1]]) #kx, ky, t
#fig.colorbar(ax=ax[0], shrink = 0.8, ticks = [0, 1])
plt.colorbar(im, ax = ax[0], ticks = [])

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
ax[0].set_title('$E$ = ' + str((E_traces[3])) + ' eV', fontsize = 18)
ax[0].tick_params(axis='both', labelsize=16)
ax[0].set_aspect(1)

### kx Difference Plot
i = 2
extent=[ax_kx[0],ax_kx[-1],ax_E_offset[e], ax_E_offset[-1]]
ax[2].imshow(np.transpose(kx_diff), origin='lower', cmap=cmap_to_plot, clim=clim, interpolation='none', vmin = clim[0], vmax=clim[1], extent = extent) #kx, ky, t
plt.colorbar(im, ax = ax[2], ticks = [])
ax[2].set_yticks(np.arange(-0.5,3,0.25))
for label in ax[2].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
ax[2].set_xticks(np.arange(-2,2.1,1))
for label in ax[2].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)          
ax[2].set_xlim(-2,2)
ax[2].set_ylim(1, 2.5)
ax[2].set_ylabel('$E - E_{VBM}$, eV', fontsize = 18)
ax[2].set_xlabel('$k_x, \AA^{-1}$', fontsize = 18)
ax[2].set_title('', fontsize = 18)
ax[2].tick_params(axis='both', labelsize=16)
ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax[2].set_aspect(2.6667)

### ky Difference Plot
# ax[1].imshow(np.transpose(ky_diff), origin='lower', cmap=cmap_to_plot, clim=clim, interpolation='none', extent=[ax_kx[0],ax_kx[-1],ax_E_offset[e], ax_E_offset[-1]]\
#              ,vmin = clim[0], vmax=clim[1]) #kx, ky, t
# ax[1].set_yticks(np.arange(-0.5,2.25,0.25))
# for label in ax[1].yaxis.get_ticklabels()[1::2]:
#     label.set_visible(False)
# ax[1].set_xticks(np.arange(-2,2.1,1))
# for label in ax[1].xaxis.get_ticklabels()[1::2]:
#     label.set_visible(False)          
# ax[1].set_xlim(-2,2)
# ax[1].set_ylim(ylim[0],ylim[1])
# ax[1].set_ylabel('$E - E_{VBM}$, eV', fontsize = 20)
# ax[1].set_xlabel('$k_x, A^{-1}$', fontsize = 20)
# ax[1].set_title('Difference', fontsize = 24)
# ax[1].tick_params(axis='both', labelsize=18)
# ax[1].set_aspect(2.5)

### Time Traces    
for t in range(len(E_traces)):
    rect = (Rectangle((kx_traces[t]-kx_int, E_traces[t]-E_int), 2*kx_int, 2*E_int, linewidth=1.5, \
                      edgecolor=trace_colors[t], facecolor='None'))
    ax[2].add_patch(rect)
    rect2 = (Rectangle((kx_traces[t]-kx_int, ky_traces[0]-ky_int), 2*kx_int, 2*ky_int, linewidth=1, \
                      edgecolor=trace_colors[t], facecolor='None', linestyle = 'dashed'))
    ax[0].add_patch(rect2)

for t in [0,1,2,3,4,5]:
    axis_test = np.linspace(ax_delay_offset[0], ax_delay_offset[-1], 50)
    ax[1].plot(axis_test, traces[t,:], marker = 'o', color = trace_colors[t], \
               label = str(E_traces[t]) + ' eV, ' + str(kx_traces[t]) + ' A^-1')

for t in [2,3]:
    ax[3].plot(axis_test, traces[t,:], color = trace_colors[t], \
               label = str(E_traces[t]) + ' eV, ' + str(kx_traces[t]) + ' A^-1')

for f in [1,3]:

    ax[f].set_ylabel('Norm. Int.', fontsize = 18)
    ax[f].set_xlabel('Delay, fs', fontsize = 18)
    ax[f].set_xticks(np.arange(-600,1200,100))
    for label in ax[f].xaxis.get_ticklabels()[1::2]:
        label.set_visible(False) 
    ax[f].set_xlim(delay_lim[0],delay_lim[1])
    ax[f].set_ylim(-0.1, 1.1)
#    ax[f].set_aspect(500)

    #ax[f].legend(frameon = False)

params = {'lines.linewidth' : 2.5, 'axes.linewidth' : 2, 'axes.labelsize' : 20, 
              'xtick.labelsize' : 16, 'ytick.labelsize' : 16, 'axes.titlesize' : 20, 'legend.fontsize' : 16}
plt.rcParams['svg.fonttype'] = 'none'

#ax[1].set_tick_params(axis='both', labelsize=16)
#plt.gca().set_aspect(200)
    
plt.rcParams.update(params)
fig.tight_layout()

if save_figure is True:
    fig.savefig((figure_file_name + '_test' +'.svg'), format='svg')