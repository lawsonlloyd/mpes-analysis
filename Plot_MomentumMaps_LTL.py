# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 10:44:11 2023

@author: lloyd
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col

#%%

### Transform Data if needed....
#I = np.transpose(I, (0,2,1,3))

t0 = (np.abs(ax_delay_offset - 0)).argmin()

#t0 = (np.abs(ax_delay_offset - 0)).argmin()

I_Summed_neg = I[:,:,:,5:t0-10] #Sum over delay/polarization/theta...
neg_length = I_Summed_neg.shape[3]
I_Summed_neg = I_Summed_neg.sum(axis=(3)) 

I_Summed_pos = I[:,:,:,t0+1:]
pos_length = I_Summed_pos.shape[3]
I_Summed_pos = I_Summed_pos.sum(axis=(3)) #Sum over delay/polarization/theta...

I_Summed_early = I[:,:,:,t0:t0+5].sum(axis=3)
I_Summed_ = I[:,:,:,:].sum(axis=(3)) 

#I_Summed = ndimage.rotate(I_Summed, 12, reshape=False, order=3, mode='constant', cval=0.0, prefilter=True)
mask_start = (np.abs(ax_E_offset - 0.5)).argmin()
logicMask_Full = np.ones((I.shape))
logicMask_Full[:,:,mask_start:] *= 300
I_Enhanced_Full = logicMask_Full * I
#testP = testP - (testP[:,:,:,0:20].sum(axis=3))

#logicMask = np.ones((I_.shape))
#logicMask[:,:,17ff0:] *= 200
#I_ = logicMask * I_

#ax_E = ax_E + 0.05

#%%
### User Inputs
tMaps, tint  = [-.0, 1.1], 2

adcs = [0, 1.5]

#%%
%matplotlib inline

# Plot Momentum Maps at specified Energies
 
### Plot
numPlots = len(tMaps)

fig, ax = plt.subplots(1,2, sharey=True)
plt.gcf().set_dpi(300)

#fig.set_size_inches(12, 6, forward=False)
ax = ax.flatten()

sat = [1, 0.8]
for i in np.arange(numPlots, dtype = int):
    tMap = tMaps[i]
    tMap = (np.abs(ax_E_offset - tMap)).argmin()
    
    adc = adcs[i]
    adc = (np.abs(ax_delay_offset - adc)).argmin()
    frame = np.transpose(I_Summed_[:,:,tMap-int(tint/2):tMap+int(tint/2)].sum(axis=(2)))
    frame = frame/np.max(frame)
    im = ax[i].imshow(frame, origin='lower', cmap='terrain_r', vmax=sat[i], clim=None, interpolation='none', extent=[-2,2,-2,2]) #kx, ky, t
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
    ax[i].set_xlabel('$k_x$', fontsize = 14)
    ax[i].set_ylabel('$k_y$', fontsize = 14)
    ax[i].tick_params(axis='both', labelsize=12)
    ax[i].set_title('$E$ = ' + str((tMaps[i])) + ' eV', fontsize = 16)
    #ax[i].annotate(('E = '+ str(round(tMaps[i],2)) + ' eV'), xy = (-1.85, 1.6), fontsize = 14, weight = 'bold')
    
fig.tight_layout()

#image_format = 'svg' # e.g .png, .svg, etc.
#image_name = 'momentummap.svg'

#fig.savefig(image_name, format=image_format, dpi=600)


#%%
# Plot Difference MMs of t < 0 and t > 0 fs

%matplotlib inline

tMaps, tint  = [1.2, 1.6, 2], 2

### Plot
numPlots = len(tMaps)

fig, ax = plt.subplots(1, numPlots, sharey=True)
plt.gcf().set_dpi(300)
ax = ax.flatten()

sat = [1, 1, 1]
for i in np.arange(numPlots, dtype = int):
    tMap = tMaps[i]
    tMap = (np.abs(ax_E_offset - tMap)).argmin()
    
    frame_neg = np.transpose(I_Summed_neg[:,:,tMap-int(tint/2):tMap+int(tint/2)].sum(axis=(2)))
    frame_pos = np.transpose(I_Summed_pos[:,:,tMap-int(tint/2):tMap+int(tint/2)].sum(axis=(2)))
    
    #frame_neg = frame_neg/(np.max(frame_neg))
    #frame_pos = frame_pos/(np.max(frame_pos))
    frame_neg = frame_neg/((neg_length))
    frame_pos = frame_pos/((pos_length))
    frame_sum = frame_neg + frame_pos
    
    cts_pos = np.sum(frame_pos[:,:])
    cts_neg = np.sum(frame_neg[:,:])
    cts_total = np.sum(frame_sum)
    
    frame_diff = frame_pos - frame_neg
    #frame_diff = frame_diff/(cts_total)
    #frame_diff = np.divide(frame_diff, frame_sum)
    #frame_diff = frame_diff/np.max(np.abs(frame_diff))
    
    im = ax[i].imshow(frame_diff, origin='lower', cmap='seismic', clim=[-1,1], interpolation='none', extent=[-2,2,-2,2]) #kx, ky, t
    
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
    ax[i].set_xlabel('$k_x$', fontsize = 14)
    ax[i].set_ylabel('$k_y$', fontsize = 14)
    ax[i].tick_params(axis='both', labelsize=12)
    ax[i].set_title('$E$ = ' + str((tMaps[i])) + ' eV', fontsize = 16)
    #ax[i].annotate(('E = '+ str(round(tMaps[i],2)) + ' eV'), xy = (-1.85, 1.6), fontsize = 14, weight = 'bold')


fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([1, 0.2, 0.025, 0.6])
fig.colorbar(im, cax=cbar_ax, ticks = [-1,0,1])

#fig.colorbar(im, fraction=0.046, pad=0.04)
fig.tight_layout()
plt.show()


#%%
import numpy             as np
import matplotlib        as mpl
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

# create colormap
# ---------------

# create a colormap that consists of
# - 1/5 : custom colormap, ranging from white to the first color of the colormap
# - 4/5 : existing colormap

# set upper part: 4 * 256/4 entries
upper = mpl.cm.viridis(np.arange(256))

# set lower part: 1 * 256/4 entries
# - initialize all entries to 1 to make sure that the alpha channel (4th column) is 1
lower = np.ones((int(256/9),4))
# - modify the first three columns (RGB):
#   range linearly between white (1,1,1) and the first color of the upper colormap
for i in range(3):
  lower[:,i] = np.linspace(1, upper[0,i], lower.shape[0])

# combine parts of colormap
cmap = np.vstack(( lower, upper ))

# convert to matplotlib colormap
cmap_LTL = mpl.colors.ListedColormap(cmap, name='cmap_LTL', N=cmap.shape[0])
#%%
%matplotlib inline

# Plot Angle Integrated Dynamics

fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1.5, 1.5], 'height_ratios':[1]})
fig.set_size_inches(12 , 4, forward=False)

ax = ax.flatten()

cmap_plot = cmap_LTL
edc_neg = I_Summed_neg.sum(axis=(0,1))
edc_pos = I_Summed_pos.sum(axis=(0,1))

edc_neg = edc_neg/neg_length
edc_pos = edc_pos/pos_length

norm_neg = np.max(edc_neg)
edc_neg = edc_neg/norm_neg
edc_pos = edc_pos/norm_neg
edc_diff = edc_pos - edc_neg
#edc_diff = .1*edc_diff/np.max(edc_diff)

ang_int = I[:,:,:,:].sum(axis=(0,1))
n = np.max(ang_int)
ang_int = ang_int/np.max(ang_int)

ang_int_neg = ang_int[:,5:t0-10].sum(axis=1)
ang_int_neg = np.expand_dims(ang_int_neg, axis=-1) # Add an extra dimension in the last axis.
ang_int_neg = ang_int_neg/np.max(ang_int_neg)

diff_ang = ang_int - ang_int_neg
diff_ang = diff_ang/np.max(diff_ang)
#ang_int = I[55:75,48:58,:,:].sum(axis=(0,1))

E_trace = [1, 1.8, 2.2]
E_ = [0, 0, 0]

E_[0] = np.abs(ax_E_offset - E_trace[0]).argmin()    
E_[1] = np.abs(ax_E_offset - E_trace[1]).argmin()
E_[2] = np.abs(ax_E_offset - E_trace[2]).argmin()      

trace_1 = ang_int[E_[0]-1:E_[0]+1,:].sum(axis=0)
trace_2 = ang_int[E_[1]-1:E_[1]+1,:].sum(axis=0)
trace_3 = ang_int[E_[2]-1:E_[2]+1,:].sum(axis=0)

trace_1 = trace_1 - np.mean(trace_1[3:t0-5])
trace_1 = trace_1/np.max(trace_1)
trace_2 = trace_2 - np.mean(trace_2[3:t0-5])
trace_2 = trace_2/np.max(trace_2)

trace_3 = trace_3 - np.mean(trace_3[3:t0-5])
trace_3 = trace_3/np.max(trace_3)

# Plotting #
###
im = ax[0].plot(ax_E_offset, edc_neg, color = 'grey', label = 't < 0 fs')
im = ax[0].plot(ax_E_offset, edc_pos, color = 'red', label = 't > 0 fs')
im = ax[0].plot(ax_E_offset, edc_diff, color = 'green', label = 'Difference', linestyle = 'dashed')

#ax[0].axvline(1.55, color = 'grey', linestyle = 'dashed')
#ax[0].axvline(1.15, color = 'black', linestyle = 'dashed')
#ax[0].axvline(2, color = 'black', linestyle = 'dashed')

ax[0].set_ylim(0,0.002)
ax[0].set_xlim(-0.5,3)
ax[0].set_xlabel('Energy, eV', fontsize = 18)
ax[0].set_ylabel('Norm. Int.', fontsize = 18 )
ax[0].legend(frameon=False)
ax[0].set_xticks(np.arange(-1,3.5,0.5))
for label in ax[0].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)

color_max = 0.0
waterfall = ax[1].imshow(ang_int, clim = [0, .02], origin = 'lower', cmap = cmap_plot, extent=[ax_delay_offset[0], ax_delay_offset[-1], ax_E_offset[0], ax_E_offset[-1]])
#waterfall = ax[1].imshow(diff_ang, clim = [-.05,.05], origin = 'lower', cmap = 'seismic', extent=[ax_delay_offset[0], ax_delay_offset[-1], ax_E_offset[0], ax_E_offset[-1]])
ax[1].set_xlim(-170,800)
ax[1].set_ylim(-0.5, 3)
ax[1].set_xlabel('Delay, fs', fontsize = 18)
ax[1].set_ylabel('E - E$_{VBM}$, eV', fontsize = 18)
ax[1].set_yticks(np.arange(-1,3.5,0.5))
for label in ax[1].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)

ax[1].set_aspect(250)

ax[1].axhline(E_trace[0], linestyle = 'dashed', color = 'red')
ax[1].axhline(E_trace[1], linestyle = 'dashed', color = 'black')
#ax[1].axhline(E_trace[2], linestyle = 'dashed', color = 'grey')
fig.colorbar(waterfall, ax=ax[1], shrink = 0.8, ticks = [0, color_max/2, color_max])

ax[2].plot(ax_delay_offset, trace_1, color = 'red', label = str(E_trace[0]) + ' eV')
ax[2].plot(ax_delay_offset, trace_2, color = 'black', label = str(E_trace[1]) + ' eV')
#ax[2].plot(ax_delay_offset, trace_3, color = 'grey', label = str(E_trace[2]) + ' eV')

ax[2].set_xlim(-150,800)
ax[2].set_ylim(-0.5, 1.1)
ax[2].set_xlabel('Delay, fs', fontsize = 18)
ax[2].legend(frameon = False)

params = {'lines.linewidth' : 2.5, 'axes.linewidth' : 2, 'axes.labelsize' : 20, 
              'xtick.labelsize' : 16, 'ytick.labelsize' : 16, 'axes.titlesize' : 20, 'legend.fontsize' : 16}
    
plt.rcParams.update(params)

fig.tight_layout()
#ax[1].set_tick_params(axis='both', labelsize=16)
#plt.gca().set_aspect(200)


#%%

# Plot kx, ky cuts Pos/Neg Difference Plots

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

start = 0
stop = t0-15

start2 = t0
stop2 = -1

I_Summed = (I[:,:,:,start:stop].sum(axis=(3))) #Sum over delay/theta/ADC for Plotting...
I_Summed_2 = (I[:,:,:,start2:stop2].sum(axis=(3))) #Sum over delay/theta/ADC for Plotting...

  #I_Summed = ndimage.rotate(I_Summed, 12, reshape=False, order=3, mode='constant', cval=0.0, prefilter=True)
  #I_Rot = I
  #I_Rot = ndimage.rotate(I, 12, reshape=False, order=3, mode='constant', cval=0.0, prefilter=True)
  #I_derivative = 

I_Enhanced = logicMask * I_Summed
I_Enhanced_2 = logicMask * I_Summed_2

#fig.tight_layout()
 
slice_E_k_1 = I_Enhanced[:,y[0]:y[0]+yint,:].sum(axis=(1))
slice_E_k_2 = I_Enhanced_2[:,y[0]:y[0]+yint,:].sum(axis=(1))

slice_E_k_3 = I_Enhanced[x[0]:x[0]+xint,:].sum(axis=(0))
slice_E_k_4 = I_Enhanced_2[x[0]:x[0]+xint,:].sum(axis=(0))

I_1_N = slice_E_k_1/np.max(np.abs(slice_E_k_1))
I_2_N = slice_E_k_2/np.max(np.abs(slice_E_k_2))

I_3_N = slice_E_k_3/np.max(np.abs(slice_E_k_3))
I_4_N = slice_E_k_4/np.max(np.abs(slice_E_k_4))

I_dif = I_2_N - I_1_N
I_dif_ = I_4_N - I_3_N

#line_cut_x_ind = (np.abs(ax_kx - line_cut_x)).argmin()
#line_cut_y_ind = (np.abs(ax_ky - line_cut_y)).argmin()
#line_cut_t_ind = (np.abs(ax_E_offset - E_AOI)).argmin()
#line_cut_1 = I_Enhanced[line_cut_x_ind-1:line_cut_x_ind+1,line_cut_y_ind-1:line_cut_y_ind+1,:].sum(axis=(0,1))

### Panels
###################################33
ylim = [0.75,2.25]

fig, ax = plt.subplots(nrows = 2, ncols=3, gridspec_kw={'width_ratios': [1, 1, 1], 'height_ratios':[1, 1]})

fig.set_size_inches(15, 10, forward=False)
ax = ax.flatten()

map_1 = 'terrain_r'
im_2 = ax[0].imshow(np.transpose(slice_E_k_1), origin='lower', cmap=map_1, clim=None, interpolation='none', extent=[ax_ky[0],ax_ky[-1],ax_E_offset[0], ax_E_offset[-1] ]) #kx, ky, t
im_3 = ax[1].imshow(np.transpose(slice_E_k_2), origin='lower', cmap=map_1, clim=None, interpolation='none', extent=[ax_kx[0],ax_kx[-1],ax_E_offset[0], ax_E_offset[-1] ]) #kx, ky, t
im_4 = ax[2].imshow(np.transpose(I_dif), origin='lower', cmap='seismic', clim=None, interpolation='none', extent=[ax_kx[0],ax_kx[-1],ax_E_offset[0], ax_E_offset[-1] ],vmin = -1, vmax=1) #kx, ky, t
im_4 = ax[3].imshow(np.transpose(slice_E_k_3), origin='lower', cmap=map_1, vmax = None, clim=None, interpolation='none', extent=[ax_ky[0],ax_ky[-1],ax_E_offset[0], ax_E_offset[-1] ]) #kx, ky, t
im_5 = ax[4].imshow(np.transpose(slice_E_k_4), origin='lower', cmap=map_1, clim=None, interpolation='none', extent=[ax_kx[0],ax_kx[-1],ax_E_offset[0], ax_E_offset[-1] ]) #kx, ky, t
im_6 = ax[5].imshow(np.transpose(I_dif_), origin='lower', cmap='seismic', clim=None, interpolation='none', extent=[ax_kx[0],ax_kx[-1],ax_E_offset[0], ax_E_offset[-1] ],vmin = -1, vmax=1) #kx, ky, t
ylim_E = -2.5

for i in range(0,6):
        
    ax[i].set_yticks(np.arange(-0.5,2.25,0.25))
    for label in ax[i].yaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
    
    ax[i].set_xticks(np.arange(-2,2.1,1))
    for label in ax[i].xaxis.get_ticklabels()[1::2]:
        label.set_visible(False)          
    
    ax[i].set_xlim(-2,2)
    ax[i].set_ylim(ylim[0],ylim[1])
    ax[i].set_ylabel('$E$, eV', fontsize = 32)
    ax[i].tick_params(axis='both', labelsize=24)
    ax[i].set_aspect(2.5)

for i in range(0,3):
    ax[i].set_xlabel('$k_x$', fontsize = 32)
    
for i in range(3,6):
    ax[i].set_xlabel('$k_y$', fontsize = 32)

for i in [0,3]:
    ax[i].set_title('t < 0 fs', fontsize = 32)    
    
for i in [1,4]:
    ax[i].set_title('t > 0 fs', fontsize = 32)   
    
for i in [2,5]:
    ax[i].set_title('Difference', fontsize = 32)  
       
fig.tight_layout()

#fig.colorbar(im, ax=fig.get_axes())
#fig.colorbar(im_4,fraction=0.046, pad=0.04)

#fig.savefig(image_name, format=image_format, dpi=600


#%%
# Window and Symmetrize MM for FFT

from scipy import signal
from scipy.fft import fft, fftshift

tMaps, tint  = [1.1], 4

### Plot
numPlots = len(tMaps)

fig, ax = plt.subplots(1, 3, sharey=True)
plt.gcf().set_dpi(300)
ax = ax.flatten()

sat = [1, 1, 1]
for i in np.arange(numPlots, dtype = int):
    tMap = tMaps[i]
    tMap = (np.abs(ax_E_offset - tMap)).argmin()
    
    frame_neg = np.transpose(I_Summed_neg[:,:,tMap-int(tint/2):tMap+int(tint/2)].sum(axis=(2)))
    frame_pos = np.transpose(I_Summed_pos[:,:,tMap-int(tint/2):tMap+int(tint/2)].sum(axis=(2)))
    
    frame_neg = frame_neg/((neg_length))
    frame_pos = frame_pos/((pos_length))
    frame_sum = frame_neg + frame_pos
    
    cts_pos = np.sum(frame_pos[:,:])
    cts_neg = np.sum(frame_neg[:,:])
    cts_total = np.sum(frame_sum)
    
    frame_diff = frame_pos - frame_neg
    frame_diff = np.abs(frame_diff)

    kspace_frame = frame_diff
    kspace_frame = np.abs(np.transpose(I_Summed_early[:,:,tMap-int(tint/2):tMap+int(tint/2)].sum(axis=(2))))
    
    #### Apply Symm and windows
    window = np.zeros((kspace_frame.shape))
    window_2 = np.zeros((kspace_frame.shape))

    k_i = (np.abs(ax_ky - -.3)).argmin()
    k_f = (np.abs(ax_ky - .6)).argmin()
    k_i_2 = (np.abs(ax_kx - -1.25)).argmin()
    k_f_2 = (np.abs(ax_kx - 1.25)).argmin()
    
    for y in range(0,window.shape[1]):
        window[k_i:k_f,y] = signal.windows.tukey(k_f-k_i)
    for x in range(0, window.shape[0]):
        window_2[x,k_i_2:k_f_2] = signal.windows.tukey(k_f_2-k_i_2)
        #window_2[x,k_i_2:k_f_2] = np.ones(k_f_2-k_i_2)
   
    window_3 = window*window_2
    #window[0:k_i,:] *= 0
    #window[k_f:-1,:] *= 0
    #window[:, 0:22] *= 0
    #window[:,80:-1] *= 0
    
    frame_sym = np.zeros(kspace_frame.shape)
    frame_sym[:,:] = kspace_frame[:,:]  + (kspace_frame[:,::-1])    
    frame_sym =  frame_sym[:,:]/2
    
    windowed_frame_symm = frame_sym*window_3
    windowed_frame_nonsymm = kspace_frame*window_3
    #windowed_frame_symm = frame_sym
    
    #windowed_frame_symm = np.zeros(windowed_frame.shape)
    #windowed_frame_symm[:,:] = windowed_frame[:,:]  + (windowed_frame[:,::-1])
    #windowed_frame_symm = windowed_frame_symm/np.abs(windowed_frame_symm)
    
im = ax[0].imshow(kspace_frame, origin='lower', cmap=cmap_LTL, interpolation='none', extent = [ax_kx[0],ax_kx[-1],ax_ky[0],ax_ky[-1]]) #kx, ky, t
im = ax[1].imshow(frame_sym, origin='lower', cmap=cmap_LTL, interpolation='none', extent = [ax_kx[0],ax_kx[-1],ax_ky[0],ax_ky[-1]]) #kx, ky, t
im = ax[2].imshow(windowed_frame_symm, origin='lower', cmap=cmap_LTL, interpolation='none', extent = [ax_kx[0],ax_kx[-1],ax_ky[0],ax_ky[-1]]) #kx, ky, t

    #im = ax[i].imshow(np.transpose(I[:,:,tMap-int(tint/2):tMap+int(tint/2), adc-1:adc+1].sum(axis=(2,3))), origin='lower', cmap='terrain_r', clim=None, interpolation='none', extent=[-2,2,-2,2]) #kx, ky, t
for i in np.arange(3, dtype = int):
    ax[i].axhline(0, color='black', linewidth = 1, linestyle = 'dashed')
    ax[i].axhline(0, color='black', linewidth = 1, linestyle = 'dashed')
    ax[i].axvline(0, color='black', linewidth = 1, linestyle = 'dashed')
    ax[i].axvline(0, color='black', linewidth = 1, linestyle = 'dashed')
    
    ax[i].axvline(-1, color='blue', linewidth = 1, linestyle = 'dashed')
    ax[i].axvline(1, color='blue', linewidth = 1, linestyle = 'dashed')
    
    ax[i].set_aspect(1)
    #ax[0].axhline(y,color='black')
    #ax[0].axvline(x,color='bl ack')
    
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
    ax[i].set_title('$E$ = ' + str((tMaps[0])) + ' eV', fontsize = 16)
    #ax[i].annotate(('E = '+ str(round(tMaps[i],2)) + ' eV'), xy = (-1.85, 1.6), fontsize = 14, weight = 'bold')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([1, 0.2, 0.025, 0.6])
fig.colorbar(im, cax=cbar_ax, ticks = [10,100])

#fig.colorbar(im, fraction=0.046, pad=0.04)
fig.tight_layout()
plt.show()

#%%

# Plot FFT of MMs to obtain real space wavefxn

momentum_frame = windowed_frame_symm
#momentum_frame = windowed_frame_nonsymm

tMaps, tint  = [1.2], 2

k_step = np.abs((ax_kx[1] - ax_kx[0]))
k_length = len(ax_kx)

zplength = 2*k_length
max_r = (1/2)*1/(k_step)
r_axis = np.linspace(-max_r, max_r, num = k_length)
r_axis = np.linspace(-max_r, max_r, num = zplength)


### Plot
numPlots = len(tMaps)

fig, ax = plt.subplots(1, 2, sharey=False)
plt.gcf().set_dpi(300)
ax = ax.flatten()

sat = [1, 1, 1]
for i in np.arange(numPlots, dtype = int):
    tMap = tMaps[i]
    tMap = (np.abs(ax_E_offset - tMap)).argmin()
    
    frame_neg = np.transpose(I_Summed_neg[:,:,tMap-int(tint/2):tMap+int(tint/2)].sum(axis=(2)))
    frame_pos = np.transpose(I_Summed_pos[:,:,tMap-int(tint/2):tMap+int(tint/2)].sum(axis=(2)))
    
    #frame_neg = frame_neg/(np.max(frame_neg))
    #frame_pos = frame_pos/(np.max(frame_pos))
    frame_neg = frame_neg/((neg_length))
    frame_pos = frame_pos/((pos_length))
    frame_sum = frame_neg + frame_pos
    
    cts_pos = np.sum(frame_pos[:,:])
    cts_neg = np.sum(frame_neg[:,:])
    cts_total = np.sum(frame_sum)
    
    frame_diff = frame_pos - frame_neg
    
    #frame_diff = frame_diff/(cts_total)
    #frame_diff = np.divide(frame_diff, frame_sum)
    #frame_diff = frame_diff/np.max(np.abs(frame_diff))
    
    frame_diff = momentum_frame

    frame_diff = np.abs(frame_diff)
    frame_diff = np.sqrt(frame_diff)
    fft_frame = np.fft.fft2(frame_diff, [zplength,zplength])
    fft_frame = np.fft.fftshift(fft_frame, axes = (0,1))
    fft_frame = np.abs(fft_frame)
    #fft_frame = np.square(fft_frame)
    fft_frame = (fft_frame)**2

    x_cut = fft_frame[:,int(zplength/2)]
    y_cut = fft_frame[int(zplength/2),:]
    x_cut = x_cut/np.max(x_cut)
    y_cut = y_cut/np.max(y_cut)

    im = ax[i].imshow(fft_frame, clim = None, origin='lower', cmap=cmap_LTL, interpolation='none', extent = [r_axis[0], r_axis[-1], r_axis[0], r_axis[-1]]) #kx, ky, t
    
    #im = ax[i].imshow(np.transpose(I[:,:,tMap-int(tint/2):tMap+int(tint/2), adc-1:adc+1].sum(axis=(2,3))), origin='lower', cmap='terrain_r', clim=None, interpolation='none', extent=[-2,2,-2,2]) #kx, ky, t

    ax[i].set_aspect(1)
    #ax[0].axhline(y,color='black')
    #ax[0].axvline(x,color='bl ack')
    
    ax[i].set_xticks(np.arange(-8,8.2,1))
    for label in ax[i].xaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
        
    ax[i].set_yticks(np.arange(-8,8.1,1))
    for label in ax[i].yaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
    
    ax[i].set_xlim(-2.5,2.5)
    ax[i].set_ylim(-2.5,2.5)
    #ax[0].set_box_aspect(1)
    ax[i].set_xlabel('$r_a$', fontsize = 14)
    ax[i].set_ylabel('$r_b$', fontsize = 14)
    ax[i].tick_params(axis='both', labelsize=12)
    ax[i].set_title('$E$ = ' + str((tMaps[i])) + ' eV', fontsize = 16)
    #ax[i].annotate(('E = '+ str(round(tMaps[i],2)) + ' eV'), xy = (-1.85, 1.6), fontsize = 14, weight = 'bold')

ax[1].plot(r_axis, x_cut/np.max(1))
ax[1].plot(r_axis, y_cut/np.max(1))
ax[1].set_xlim([0,1.5])
#ax[1].axvline(1.75, color='black')
#ax[1].set_aspect(.003)
ax[1].set_xlabel('r, nm')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([1, 0.2, 0.025, 0.6])
fig.colorbar(im, cax=cbar_ax, ticks = [10,100])

#fig.colorbar(im, fraction=0.046, pad=0.04)
fig.tight_layout()
plt.show()

#%%

### Make a video/gif of the Dynamics!

import imageio
from matplotlib.animation import FuncAnimation
from matplotlib import animation

video_array = I_Enhanced_Full[:,y[0]:y[0]+yint,:,10:].sum(axis=(1)) #Individual Delays
video_axis = ax_delay_offset[10:]
fig = plt.figure()

def update_img(n):
    plt.imshow(np.transpose(video_array[:,:,n:n+15].sum(axis=2)),cmap='terrain_r', origin = 'lower', clim=None, interpolation='none', extent=[ax_ky[0],ax_ky[-1],ax_E_offset[0], ax_E_offset[-1]])
    plt.xlim(-1.6, 1.6)
    plt.xlabel('k, A-1')
    plt.ylabel('E-EF, eV')
    plt.title('T = ' + str(video_axis[n]) + ' fs')
    #fig.set_title('T = ' + str(ax_delay[n]) + ' fs',(10,10), fontsize = 16, weight = 'bold', color='white')

v_len = video_array.shape[2]

ani = FuncAnimation(fig, update_img, frames = v_len)
writergif = animation.PillowWriter(fps=5)

#ani.save('trARPES_test.gif', writer=writergif)

#%%
%matplotlib inline

fig = plt.figure()

fig.set_size_inches(10, 10, forward=False)

ang_int = I[:,:,:,:].sum(axis=(0,1))
n = np.max(ang_int)

plt.imshow(ang_int/n, vmax=0.0025, origin = 'lower', cmap = 'magma_r', extent=[ax_delay_offset[0], ax_delay_offset[-1], ax_E_offset[0], ax_E_offset[-1]])

plt.xlim(-170,800)
plt.ylim(-0.25, 3)
plt.xlabel('Delay, fs', fontsize = 18)
plt.ylabel('E - E$_{VBM}$, eV', fontsize = 18)
plt.tick_params(axis='both', labelsize=16)
plt.gca().set_aspect(200)

#%%
%matplotlib inline
    
fig = plt.figure()

fig.set_size_inches(10, 10, forward=False)

slice_E_k_1 = I_Enhanced[:,y[0]:y[0]+yint,:].sum(axis=(1))
slice_E_k_2 = I_Enhanced_2[:,y[0]:y[0]+yint,:].sum(axis=(1))

I_1_N = slice_E_k_1/np.max(np.abs(slice_E_k_1))
I_2_N = slice_E_k_2/np.max(np.abs(slice_E_k_2))

I_dif = I_2_N - I_1_N
im_2 = plt.imshow(np.transpose(I_dif), origin='lower', cmap='seismic', clim=None, interpolation='none', extent=[ax_ky[0],ax_ky[-1],ax_E_offset[0], ax_E_offset[-1]], vmin = -1, vmax = 1) #kx, ky, t
    
ylim_E = -2.5

ax[0].set_yticks(np.arange(-0.5,2.25,0.25))
for label in ax[0].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
    
plt.xlim(-2,2)
plt.ylim(0.5,2)
plt.xlabel('$k_x$', fontsize = 18)
plt.ylabel('E-E_$VBM$, eV', fontsize = 18)
plt.tick_params(axis='both', labelsize=16)
plt.colorbar()
#ax[1].set_title('Slice', fontsize = 14)
#ax[2].set_title('Slice', fontsize = 14)
#plt.aspect(2)
fig.tight_layout()

image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'CBM_difference.svg'

fig.savefig(image_name, format=image_format, dpi=600)