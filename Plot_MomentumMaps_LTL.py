# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 10:44:11 2023

@author: lloyd
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
from matplotlib.patches import Rectangle

#%%
### Transform Data if needed....
#I = np.transpose(I, (0,2,1,3))

t0 = (np.abs(ax_delay_offset - 0)).argmin()
dt = ax_delay_offset[10]-ax_delay_offset[9]

#t0 = (np.abs(ax_delay_offset - 0)).argmin()

I_Summed_neg = I[:,:,:,5:t0-10] #Sum over delay/polarization/theta...
neg_length = I_Summed_neg.shape[3]
I_Summed_neg = I_Summed_neg.sum(axis=(3)) 

I_Summed_pos = I[:,:,:,t0+1:]
pos_length = I_Summed_pos.shape[3]
I_Summed_pos = I_Summed_pos.sum(axis=(3)) #Sum over delay/polarization/theta...

I_Summed_early = I[:,:,:,t0-2:t0+5].sum(axis=3)
I_Summed_ = I[:,:,:,:].sum(axis=(3))

dE = (ax_E_offset[1] - ax_E_offset[0])
dkx = (ax_kx[1] - ax_kx[0])
dky = (ax_ky[1] - ax_ky[0]) 

#%%

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
### User Inputs for Plotting MM 

tMaps, tint  = [1.3, 1.6], 5

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

sat = [1, 1]
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

tMaps, tint  = [1.25, 2], 4

cmapPLOTTING = 'bone_r'#'bone_r' # cmap_LTL

difference_FRAMES = np.zeros((numPlots,I_Summed_pos.shape[0],I_Summed_pos.shape[1]))
### Plot
numPlots = len(tMaps)

fig, ax = plt.subplots(1, numPlots+1, sharey=False)
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
    frame_diff = frame_diff/np.max(np.abs(frame_diff))
    difference_FRAMES[i,:,:] = frame_diff    
        
    im = ax[i].imshow(frame_diff, origin='lower', cmap=cmapPLOTTING, clim=[0,1], interpolation='none', extent=[-2,2,-2,2]) #kx, ky, t
    
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
im = ax[i].imshow(delta_MM, origin='lower', cmap='seismic', clim=[-1,1], interpolation='none', extent=[-2,2,-2,2]) #kx, ky, t

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
#upper = mpl.cm.jet(np.arange(256))

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
cmap_LTL = mpl.colors.ListedColormap(cmap, name='viridis_LTL', N=cmap.shape[0])

#%%
%matplotlib inline

# Plot Angle Integrated Dynamics

E_trace = [1.15, 1.9, 2.2] # Energies for Plotting

################################
# Operations to Extract Traces #
################################

### Negative Delays Background Subtraction
edc_neg = I_Summed_neg.sum(axis=(0,1))
edc_pos = I_Summed_pos.sum(axis=(0,1))

edc_neg = edc_neg/neg_length
edc_pos = edc_pos/pos_length

norm_neg = np.max(edc_neg)
edc_neg = edc_neg/norm_neg
edc_pos = edc_pos/norm_neg
edc_diff = edc_pos - edc_neg
#edc_diff = .1*edc_diff/np.max(edc_diff)

ang_int = I[:,:,:,:].sum(axis=(0,1)) #Angle Integrated Spectra
n = np.max(ang_int)
ang_int = ang_int/np.max(ang_int)

ang_int_neg = ang_int[:,5:t0-10].sum(axis=1)
ang_int_neg = np.expand_dims(ang_int_neg, axis=-1) # Add an extra dimension in the last axis.
ang_int_neg = ang_int_neg/np.max(ang_int_neg)

diff_ang = ang_int - ang_int_neg
diff_ang = diff_ang/np.max(diff_ang)
#ang_int = I[55:75,48:58,:,:].sum(axis=(0,1))

# Extract Traces for At Different Energies
E_ = [0, 0, 0]
E_[0] = np.abs(ax_E_offset - E_trace[0]).argmin()    
E_[1] = np.abs(ax_E_offset - E_trace[1]).argmin()
E_[2] = np.abs(ax_E_offset - E_trace[2]).argmin()      

trace_1 = ang_int[E_[0]-1:E_[0]+1,:].sum(axis=0)
trace_2 = ang_int[E_[1]-1:E_[1]+1,:].sum(axis=0)
trace_3 = ang_int[E_[2]-1:E_[2]+1,:].sum(axis=0)

trace_1 = trace_1 - np.mean(trace_1[3:t0-5])
trace_2 = trace_2 - np.mean(trace_2[3:t0-5])
trace_3 = trace_3 - np.mean(trace_3[3:t0-5])

trace_2 = trace_2/np.max(trace_2)
trace_1 = trace_1/np.max(trace_1)
trace_3 = trace_3/np.max(trace_3)

#######################
### Do the Plotting ###
#######################

fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1.5, 1.5], 'height_ratios':[1]})
fig.set_size_inches(12 , 4, forward=False)

ax = ax.flatten()
cmap_plot = cmap_LTL

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
#waterfall = ax[1].imshow(ang_int, clim = [0, .02], origin = 'lower', cmap = cmap_plot, extent=[ax_delay_offset[0], ax_delay_offset[-1], ax_E_offset[0], ax_E_offset[-1]])
waterfall = ax[1].imshow(diff_ang, clim = [-.05,.05], origin = 'lower', cmap = 'seismic', extent=[ax_delay_offset[0], ax_delay_offset[-1], ax_E_offset[0], ax_E_offset[-1]])
ax[1].set_xlim(-200,1050)
ax[1].set_ylim(-0.5, 3)
ax[1].set_xlabel('Delay, fs', fontsize = 18)
ax[1].set_ylabel('E - E$_{VBM}$, eV', fontsize = 18)
ax[1].set_yticks(np.arange(-1,3.5,0.5))

for label in ax[1].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)

ax[1].set_aspect(300)

ax[1].axhline(E_trace[0], linestyle = 'dashed', color = 'red')
ax[1].axhline(E_trace[1], linestyle = 'dashed', color = 'black')
#ax[1].axhline(E_trace[2], linestyle = 'dashed', color = 'grey')
fig.colorbar(waterfall, ax=ax[1], shrink = 0.8, ticks = [0, color_max/2, color_max])

ax[2].plot(ax_delay_offset, trace_1, color = 'red', label = str(E_trace[0]) + ' eV')
ax[2].plot(ax_delay_offset, trace_2, color = 'black', label = str(E_trace[1]) + ' eV')
#ax[2].plot(ax_delay_offset, trace_3, color = 'grey', label = str(E_trace[2]) + ' eV')

ax[2].set_xlim(-300,1050)
ax[2].set_ylim(-0.1, 1.1)
ax[2].set_ylabel('Norm. Int.', fontsize = 18)
ax[2].set_xlabel('Delay, fs', fontsize = 18)
ax[2].legend(frameon = False)

params = {'lines.linewidth' : 2.5, 'axes.linewidth' : 2, 'axes.labelsize' : 20, 
              'xtick.labelsize' : 16, 'ytick.labelsize' : 16, 'axes.titlesize' : 20, 'legend.fontsize' : 16}
    
plt.rcParams.update(params)

fig.tight_layout()
#ax[1].set_tick_params(axis='both', labelsize=16)
#plt.gca().set_aspect(200)

#%%
# Plot Dynamics at Distinct Momenta and/or Energy Points

kx_traces, ky_traces = [-1.7, -1.0, -1.0, -1.7], [-0.08] # kx, ky for plotting
E_traces = [1.15, 1.15, 1.9, 1.9] # Energies for Plotting
kx_int, ky_int, E_int  = 0.4, 0.6, 0.2 #Integration Ranges

trace_colors = ['black', 'green', 'purple', 'blue']

################################
# Operations to Extract Traces #
################################

## Extract Traces for At Different Energies with Background Subtraction
traces = np.zeros((4,I.shape[3]))
E_int, kx_int, ky_int = E_int/2, kx_int/2, ky_int/2

for t in range(4):
    xi = (np.abs(ax_kx - (kx_traces[t]-kx_int))).argmin()
    xf = (np.abs(ax_kx - (kx_traces[t]+kx_int))).argmin()
    yi = (np.abs(ax_ky - (ky_traces[0]-ky_int))).argmin()
    yf = (np.abs(ax_ky - (ky_traces[0]+ky_int))).argmin()
    Ei = np.abs(ax_E_offset - (E_traces[t]-E_int)).argmin()  
    Ef = np.abs(ax_E_offset - (E_traces[t]+E_int)).argmin()  
    trace = I[xi:xf, yi:yf, Ei:Ef,:].sum(axis=(0,1,2))
    trace = trace/np.max(trace)
    trace = trace - np.mean(trace[3:t0-5])
    traces[t,:] = trace/np.max(trace)

kx_diff = I_2_N - I_1_N
#######################
### Do the Plotting ###
#######################

fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1.5, 1.5], 'height_ratios':[1]})
fig.set_size_inches(12 , 4, forward=False)

ax = ax.flatten()
cmap_plot = cmap_LTL

### kx Difference Plot
ax[0].imshow(np.transpose(kx_diff), origin='lower', cmap='seismic', clim=[-1,1], interpolation='none', extent=[ax_kx[0],ax_kx[-1],ax_E_offset[e], ax_E_offset[-1] ],vmin = -1, vmax=1) #kx, ky, t
ax[0].set_yticks(np.arange(-0.5,2.25,0.25))
for label in ax[0].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
ax[0].set_xticks(np.arange(-2,2.1,1))
for label in ax[0].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)          
ax[0].set_xlim(-2,2)
ax[0].set_ylim(ylim[0],ylim[1])
ax[0].set_ylabel('$E - E_{VBM}$, eV', fontsize = 20)
ax[0].set_xlabel('$k_x, A^{-1}$', fontsize = 20)
ax[0].set_title('Difference', fontsize = 24)
ax[0].tick_params(axis='both', labelsize=18)
ax[0].set_aspect(2.5)

### Time Traces    
for t in range(4):
    rect = (Rectangle((kx_traces[t]-kx_int, E_traces[t]-E_int), 2*kx_int, 2*E_int, linewidth=1.5, \
                      edgecolor=trace_colors[t], facecolor='None'))
    ax[0].add_patch(rect)

for t in [0,1]:
    ax[1].plot(ax_delay_offset, traces[t,:], color = trace_colors[t], \
               label = str(E_traces[t]) + ' eV, ' + str(kx_traces[t]) + ' A^-1')

for t in [2,3]:
    ax[2].plot(ax_delay_offset, traces[t,:], color = trace_colors[t], \
               label = str(E_traces[t]) + ' eV, ' + str(kx_traces[t]) + ' A^-1')

for f in range(1,3):
    ax[f].set_xlim(-300,1050)
    ax[f].set_ylim(-0.2, 1.1)
    ax[f].set_ylabel('Norm. Int.', fontsize = 18)
    ax[f].set_xlabel('Delay, fs', fontsize = 18)
    #ax[f].legend(frameon = False)

params = {'lines.linewidth' : 2.5, 'axes.linewidth' : 2, 'axes.labelsize' : 20, 
              'xtick.labelsize' : 16, 'ytick.labelsize' : 16, 'axes.titlesize' : 20, 'legend.fontsize' : 16}

plt.rcParams.update(params)
fig.tight_layout()

#%%
%matplotlib inline
# Plot kx, ky cuts Pos/Neg Difference Plots

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

y = [48,46]
x = [75]

start = 5
stop = t0-17

start2 = t0
stop2 = -1

I_Summed = (I[:,:,:,start:stop]) #Sum over delay/theta/ADC for Plotting...
len_1 = I_Summed.shape[3]
I_Summed = I_Summed.sum(axis=3)/len_1

I_Summed_2 = (I[:,:,:,start2:stop2])
len_2 = I_Summed_2.shape[3]              
I_Summed_2 = I_Summed_2.sum(axis=(3))/len_2 #Sum over delay/theta/ADC for Plotting...

  #I_Summed = ndimage.rotate(I_Summed, 12, reshape=False, order=3, mode='constant', cval=0.0, prefilter=True)
  #I_Rot = I
  #I_Rot = ndimage.rotate(I, 12, reshape=False, order=3, mode='constant', cval=0.0, prefilter=True)
  #I_derivative = 

I_Enhanced = logicMask * I_Summed
I_Enhanced_2 = logicMask * I_Summed_2

#fig.tight_layout()
e = 100
slice_E_k_1 = I_Enhanced[:,y[0]:y[0]+yint,e:].sum(axis=(1))
norm_1 = np.max(slice_E_k_1)
slice_E_k_1 = slice_E_k_1/norm_1

slice_E_k_2 = I_Enhanced_2[:,y[0]:y[0]+yint,e:].sum(axis=(1))
slice_E_k_2 = slice_E_k_2/norm_1

slice_E_k_3 = I_Enhanced[x[0]:x[0]+xint,:,e:].sum(axis=(0))
norm_3 = np.max(slice_E_k_3)
slice_E_k_3 = slice_E_k_3/norm_3

slice_E_k_4 = I_Enhanced_2[x[0]:x[0]+xint,:,e:].sum(axis=(0))
slice_E_k_4 = slice_E_k_4/norm_3

I_1_N = slice_E_k_1/np.max(np.abs(slice_E_k_1))
I_2_N = slice_E_k_2/np.max(np.abs(slice_E_k_2))

I_3_N = slice_E_k_3/np.max(np.abs(slice_E_k_3))
I_4_N = slice_E_k_4/np.max(np.abs(slice_E_k_4))

I_dif = I_2_N - I_1_N
I_dif_ = I_4_N - I_3_N

#I_dif = I_dif/np.max(I_dif)
#I_dif_ = I_dif_/np.max(I_dif_)

#line_cut_x_ind = (np.abs(ax_kx - line_cut_x)).argmin()
#line_cut_y_ind = (np.abs(ax_ky - line_cut_y)).argmin()
#line_cut_t_ind = (np.abs(ax_E_offset - E_AOI)).argmin()
#line_cut_1 = I_Enhanced[line_cut_x_ind-1:line_cut_x_ind+1,line_cut_y_ind-1:line_cut_y_ind+1,:].sum(axis=(0,1))

### Panels
###################################33

plt.figure()
plt.imshow(np.transpose(I_Enhanced[:,:,110:170].sum(axis=2)))
plt.axhline(y[0], color = 'black')
plt.axhline(y[0]+yint, color = 'black')
plt.axvline(x[0])
plt.axvline(x[0]+xint)

ylim = [0.75,2.25]

###
fig, ax = plt.subplots(nrows = 2, ncols=3, gridspec_kw={'width_ratios': [1, 1, 1], 'height_ratios':[1, 1]})
fig.set_size_inches(15, 10, forward=False)
ax = ax.flatten()

map_1 = 'terrain_r'
im_2 = ax[0].imshow(np.transpose(slice_E_k_1), origin='lower', cmap=map_1, clim=None, interpolation='none', extent=[ax_ky[0],ax_ky[-1],ax_E_offset[e], ax_E_offset[-1] ]) #kx, ky, t
im_3 = ax[1].imshow(np.transpose(slice_E_k_2), origin='lower', cmap=map_1, clim=None, interpolation='none', extent=[ax_kx[0],ax_kx[-1],ax_E_offset[e], ax_E_offset[-1] ]) #kx, ky, t
im_4 = ax[2].imshow(np.transpose(I_dif), origin='lower', cmap='seismic', clim=[-1,1], interpolation='none', extent=[ax_kx[0],ax_kx[-1],ax_E_offset[e], ax_E_offset[-1] ],vmin = -1, vmax=1) #kx, ky, t
im_4 = ax[3].imshow(np.transpose(slice_E_k_3), origin='lower', cmap=map_1, vmax = None, clim=None, interpolation='none', extent=[ax_ky[0],ax_ky[-1],ax_E_offset[e], ax_E_offset[-1] ]) #kx, ky, t
im_5 = ax[4].imshow(np.transpose(slice_E_k_4), origin='lower', cmap=map_1, clim=None, interpolation='none', extent=[ax_ky[0],ax_ky[-1],ax_E_offset[e], ax_E_offset[-1] ]) #kx, ky, t
im_6 = ax[5].imshow(np.transpose(I_dif_), origin='lower', cmap='seismic', clim=None, interpolation='none', extent=[ax_ky[0],ax_ky[-1],ax_E_offset[e], ax_E_offset[-1] ],vmin = -1, vmax=1) #kx, ky, t
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
    ax[i].set_ylabel('$E - E_{VBM}$, eV', fontsize = 32)
    ax[i].tick_params(axis='both', labelsize=24)
    ax[i].set_aspect(2.5)

for i in range(0,3):
    ax[i].set_xlabel('$k_x, A^{-1}$', fontsize = 32)
    
for i in range(3,6):
    ax[i].set_xlabel('$k_y, A^{-1}$', fontsize = 32)

for i in [0,3]:
    ax[i].set_title('Int. t < 0 fs', fontsize = 32)    
    
for i in [1,4]:
    ax[i].set_title('Int. t > 0 fs', fontsize = 32)   
    
for i in [2,5]:
    ax[i].set_title('Difference', fontsize = 32)  
       
fig.tight_layout()

#fig.colorbar(im, ax=fig.get_axes())
#fig.colorbar(im_4,fraction=0.046, pad=0.04)

#fig.savefig(image_name, format=image_format, dpi=600

#%%
%matplotlib inline
###
# Window and Symmetrize MM for FFT
###

from scipy import signal
from scipy.fft import fft, fftshift

tMaps, tint  = [1.8], 6
k_i, k_f = -0.08, 0.42
k_i_2, k_f_2 = -1.35, 1.35

### Plot
numPlots = len(tMaps)

fig, ax = plt.subplots(1, 3, sharey=True)
plt.gcf().set_dpi(300)
ax = ax.flatten()

sat = [1, 1, 1]
for i in np.arange(numPlots):
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
    frame_diff = (frame_diff)

    frame_early = np.abs(np.transpose(I_Summed_early[:,:,tMap-int(tint/2):tMap+int(tint/2)].sum(axis=(2))))
    
    ###                   ###
    ### Do the Operations ###
    ###                   ###  
    kspace_frame = np.abs(frame_diff)
    kspace_frame = frame_pos #All Pos delays
    #kspace_frame = frame_early
    
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
    
    #fig, ax = plt.subplots(1, 3)
    #ax.flatten()
    #ax[0].imshow(kspace_frame, cmap='terrain_r')
    #ax[0].axhline(48)
    #ax[0].axvline(55)

    #i = 48
    #j = 55
    #kx_cut = kspace_frame[:,j-1:j+1].sum(axis=1)
    #kx_cut = kx_cut/np.max(kx_cut)
    #ky_cut = kspace_frame[i-1:i+1,:].sum(axis = 0)
    #ky_cut = ky_cut/np.max(ky_cut)

    #ax[1].plot(gaussian_kx, linestyle = 'dashed', color = 'green')
    #ax[1].plot(kx_cut)
    #ax[2].plot(gaussian_ky, linestyle = 'dashed', color = 'green')
    #ax[2].plot(ky_cut)
    #ax[1].set_xlim(30,70)
    
    #kx_cut_deconv = signal.deconvolve(kx_cut, gaussian_kx)
    
    #### Apply Symm and windows
    kspace_frame = kspace_frame/np.max(kspace_frame)
    window = np.zeros((kspace_frame.shape))
    window_2 = np.zeros((kspace_frame.shape))
    
    win_1 = np.zeros(kspace_frame.shape[0])
    win_2 = np.zeros(kspace_frame.shape[1])

    k_i = (np.abs(ax_ky - k_i)).argmin()
    k_f = (np.abs(ax_ky - k_f)).argmin()
    k_i_2 = (np.abs(ax_kx - k_i_2)).argmin()
    k_f_2 = (np.abs(ax_kx - k_f_2)).argmin()
    
    tuk_1 = signal.windows.boxcar(k_f-k_i)
    tuk_2 = signal.windows.boxcar(k_f_2-k_i_2)
   
    win_1[k_i:k_f] = tuk_1
    win_2[k_i_2:k_f_2] = tuk_2
   
    window_4 = np.outer(win_1, win_2)
   
    for yy in range(0,window.shape[1]):
        window[k_i:k_f,yy] = signal.windows.tukey(k_f-k_i)
        window[k_i:k_f,yy] = np.ones(k_f-k_i)

    for xx in range(0, window.shape[0]):
        window_2[xx,k_i_2:k_f_2] = signal.windows.tukey(k_f_2-k_i_2)
        window_2[xx,k_i_2:k_f_2] = np.ones(k_f_2-k_i_2)
   
    window_3 = window*window_2
    #window_3 = np.outer(signal.windows.tukey(k_f_2-k_i_2), signal.windows.tukey(k_f-k_i))
    #window[0:k_i,:] *= 0
    #window[k_f:-1,:] *= 0
    #window[:, 0:22] *= 0
    #window[:,80:-1] *= 0
    
    ### Symmetrize Data
    frame_sym = np.zeros(kspace_frame.shape)
    frame_sym[:,:] = kspace_frame[:,:]  + (kspace_frame[:,::-1])    
    frame_sym =  frame_sym[:,:]/2
    
    windowed_frame_symm = frame_sym*window_4
    windowed_frame_nonsymm = kspace_frame*window_4
    #windowed_frame_symm = frame_sym
    
    ####### WSe2
    
    circle_mask = False
    
    #Circular Mask
    mask = np.zeros((len(ax_kx), len(ax_ky)))
    row = int((len(ax_kx)/2))
    col = row
    k_outer = np.abs((ax_kx - 1.75)).argmin()
    k_inner = np.abs((ax_kx - 1.75)).argmin()
    radius = 48#52 # 52
    radius_2 = 34#38 # 38
    rr, cc = disk((row, col), radius)
    mask[rr, cc] = 1
    rr, cc = disk((row, col), radius_2)
    mask[rr, cc] = 0
    
    wl = 6
    k_points_y = [113, 73, 34, 34, 74, 113] 
    k_points_x = [53, 30, 52, 100, 122, 100]
    
    #k_points_y = [78, 100, 83, 39, 21, 41] 
    #k_points_x = [24, 61, 95, 95, 61, 23]
   
    mask = np.zeros((len(ax_kx), len(ax_ky)))
    
    for k in range(0,6):
        x = k_points_x[k] 
        y = k_points_y[k] 
        window_new[x-mi:x+mi,y-mi:y+mi] = window2d
        
        #Circular Mask
        row = x
        col = y
        k_outer = np.abs((ax_kx - 1.75)).argmin()
        k_inner = np.abs((ax_kx - 1.75)).argmin()
        radius = 8#52 # 52
        rr, cc = disk((row, col), radius)
        mask[rr, cc] = 1
    ######## WSe2
    
    #windowed_frame_symm = np.zeros(windowed_frame.shape)
    #windowed_frame_symm[:,:] = windowed_frame[:,:]  + (windowed_frame[:,::-1])
    #windowed_frame_symm = windowed_frame_symm/np.abs(windowed_frame_symm)
    
im = ax[0].imshow(kspace_frame, origin='lower', cmap=cmap_LTL, interpolation='none', extent = [ax_kx[0],ax_kx[-1],ax_ky[0],ax_ky[-1]]) #kx, ky, t
im = ax[1].imshow(frame_sym, origin='lower', cmap=cmap_LTL, interpolation='none', extent = [ax_kx[0],ax_kx[-1],ax_ky[0],ax_ky[-1]]) #kx, ky, t
im = ax[2].imshow(windowed_frame_symm, origin='lower', cmap=cmap_LTL, interpolation='none', extent = [ax_kx[0],ax_kx[-1],ax_ky[0],ax_ky[-1]]) #kx, ky, t

    #im = ax[i].imshow(np.transpose(I[:,:,tMap-int(tint/2):tMap+int(tint/2), adc-1:adc+1].sum(axis=(2,3))), origin='lower', cmap='terrain_r', clim=None, interpolation='none', extent=[-2,2,-2,2]) #kx, ky, t
for i in np.arange(3):
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

xk = 48
yk = 48

a = frame_sym[:,xk-5:xk+5].sum(axis=1)
aa = np.max(abs(a))
aaa = a/aa

b = windowed_frame_symm[:,xk-5:xk+5].sum(axis=1)
bb = np.max(abs(b))
bbb = b/aa
c = window_4[:,xk-5:xk+5].sum(axis = 1)
cc = np.max(abs(c))
ccc = c/cc

#####
a = frame_sym[yk-5:yk+5,:].sum(axis=0)
aa = np.max(abs(a))
aaa_2 = a/aa

b = windowed_frame_symm[yk-5:yk+5,:].sum(axis=0)
bb = np.max(abs(b))
bbb_2 = b/aa
#lawson was here.
c = window_4[yk-5:yk+5,:].sum(axis=0)
cc = np.max(abs(c))
ccc_2 = c/cc

cut = [aaa, bbb, ccc]
cut_2 = [aaa_2, bbb_2, ccc_2]

titles = ['Frame Sym', 'Windowed', 'Tukey Win']

fig, ax = plt.subplots(2, 1, sharey=False)
ax = ax.flatten()
for i in range(0,3):
    ax[0].plot(cut[i])
    ax[0].set_title(titles[i])
    ax[1].plot(cut_2[i])
    ax[1].set_title(titles[i])    
fig.tight_layout()

#fig, ax = plt.subplots(1, 3, sharey=False)
#ax = ax.flatten()
for i in range(0,3):
    fr = cut[i]
    fr = np.sqrt(fr)
    fft_ = np.fft.fft(fr)
    fft_ = np.fft.fftshift(fft_, axes = 0)
    fft_ = np.abs(fft_)**2

    #ax[i].plot(np.real(fft_))
    #ax[i].plot(np.imag(fft_))
    #ax[i].set_title(titles[i])
    #ax[i].plot(fftshift(fft(np.abs(cut[i]))))
#fig.tight_layout()

################
fig, ax = plt.subplots(2, 3, sharey=False)
ax = ax.flatten()

cut = [frame_sym, windowed_frame_symm, window_4]

for i in range(0,3):
    
    ax[i].imshow(cut[i])
    ax[i].set_title(titles[i], fontsize = 12)

for i in range(0,3):

    fr = cut[i]
    #fr = np.sqrt(fr)
    fft_ = np.fft.fft2(fr)
    fft_ = np.fft.fftshift(fft_,  axes = (0,1))
    fft_ = np.abs(fft_)**2
    
    ax[i+3].imshow(fft_)
    ax[i+3].set_title('FT ' + titles[i], fontsize = 12)
    ax[i+3].set_xlim(40,60)
    ax[i+3].set_ylim(40,60)

    #ax[i].plot(fftshift(fft(np.abs(cut[i]))))

fig.tight_layout()
plt.show()


#%%

# Plot FFT of MMs to obtain real space wavefxn

momentum_frame = windowed_frame_symm
momentum_frame = windowed_frame_nonsymm
#momentum_frame = window_4

k_step = np.abs((ax_kx[1] - ax_kx[0]))
k_length = len(ax_kx)

k_step_y = np.abs((ax_ky[1] - ax_ky[0]))
k_length_y = len(ax_ky)

zplength = 5*k_length+1
max_r = (1/2)*1/(k_step)
r_axis = np.linspace(-max_r, max_r, num = k_length)
r_axis = np.linspace(-max_r, max_r, num = zplength)

### Plot
numPlots = len(tMaps)

fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1], 'height_ratios':[1]})
plt.gcf().set_dpi(300)
ax = ax.flatten()

sat = [1, 1, 1]
for i in np.arange(numPlots, dtype = int):
    
    ### Do the FFT operations to get --> |Psi(x,y)|^2
    #momentum_frame = momentum_frame - np.mean(momentum_frame[30:45,30:45])
    momentum_frame = np.abs(momentum_frame)/np.max(momentum_frame)
    momentum_frame = np.sqrt(momentum_frame)
    fft_frame = np.fft.fft2(momentum_frame, [zplength, zplength])
    fft_frame = np.fft.fftshift(fft_frame, axes = (0,1))
    fft_frame = np.abs(fft_frame)**2

    ### Take x and y cuts and extract bohr radius
    x_cut = fft_frame[:,int(zplength/2)-1]
    y_cut = fft_frame[int(zplength/2)-1,:]
    x_cut = x_cut/np.max(x_cut)
    y_cut = y_cut/np.max(y_cut)
    
    x_brad = (np.abs(x_cut[int(zplength/2):-2] - 0.5)).argmin()
    y_brad = (np.abs(y_cut[int(zplength/2):-2] - 0.5)).argmin()
    x_brad = int(zplength/2) + x_brad
    y_brad = int(zplength/2) + y_brad
    x_brad = r_axis[x_brad]
    y_brad = r_axis[y_brad]

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
    
    ax[i].set_xlim(-2,2)
    ax[i].set_ylim(-2,2)
    #ax[0].set_box_aspect(1)
    ax[i].set_xlabel('$r_a$, nm', fontsize = 16)
    ax[i].set_ylabel('$r_b$, nm', fontsize = 16)
    ax[i].tick_params(axis='both', labelsize=14)
    ax[i].set_title('$E$ = ' + str((tMaps[i])) + ' eV', fontsize = 16)
    #ax[i].annotate(('E = '+ str(round(tMaps[i],2)) + ' eV'), xy = (-1.85, 1.6), fontsize = 14, weight = 'bold')

ax[1].plot(r_axis, x_cut/np.max(1), color = 'black', label = '$r_b$')
ax[1].plot(r_axis, y_cut/np.max(1), color = 'red', label = '$r_a$')
#ax[1].axhline(0.5, linestyle = 'dashed', color = 'blue')
#ax[1].axvline(0.0, linestyle = 'dashed', color = 'blue')
ax[1].axvline(x_brad, linestyle = 'dashed', color = 'black', linewidth = 2)
ax[1].axvline(y_brad, linestyle = 'dashed', color = 'red', linewidth = 2)
ax[1].set_xlim([0, 2])
ax[1].set_ylim([-0.025, 1.025])
ax[1].set_xlabel('$r$, nm', fontsize = 16)
ax[1].set_ylabel('Norm. Int.', fontsize = 16)
ax[1].set_title('$r^*_a/r^*_b$ = ' + str(round(x_brad/y_brad,2)), fontsize = 16)
ax[1].tick_params(axis='both', labelsize=14)
ax[1].legend(frameon = False)
plt.text(1, 0.58, '$r^*_b$ = ' + str(round(x_brad,2)) + ' nm', fontsize = 11, color = 'black', fontweight = 4)
plt.text(1, 0.47, '$r^*_a$ = ' + str(round(y_brad,2)) + ' nm', fontsize = 11, color = 'red', fontweight = 4)

ax[1].set_yticks(np.arange(-0,1.5,0.5))
#for label in ax[1].yaxis.get_ticklabels()[1::2]:
 #   label.set_visible(False)
#ax[1].axvline(1.75, color='black')
ax[1].set_aspect(2)
ax[1].set_xlabel('r, nm')
#fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([1, 0.2, 0.025, 0.6])
#fig.colorbar(im, cax=cbar_ax, ticks = [10,100])
#fig.colorbar(im, fraction=0.046, pad=0.04)
fig.tight_layout()
plt.show()

#%%

I_Neg = I[:,:,:,6:t0-15]
neg_len = I_Neg.shape[3]
I_Neg = I_Neg.sum(axis=3)/neg_len

I_Difference_Full = np.zeros(I.shape)
for d in np.arange(0,len(ax_delay_offset)):
    I_Difference_Full[:,:,:,d] = I[:,:,:,d] # - I_Neg
    
norm_diff_full = np.max(abs(I_Difference_Full))
I_Difference_Full = I_Difference_Full/norm_diff_full

scale_factor = np.max(abs(I_Difference_Full[:,:,0:100,:]))/np.max(abs(I_Difference_Full[:,:,100:,:]))

logic_mask = np.ones((I_Difference_Full.shape))
logic_mask[:,:,mask_start:,:] *= scale_factor
I_Difference_Full = logic_mask * I_Difference_Full
    
#%%


#%%
## Make a video/gif of the Dynamics!

#y = [y]
import imageio
from matplotlib.animation import FuncAnimation
from matplotlib import animation

I_Summed_neg = I[:,:,:,6:t0-15]
neg_len = I_Summed_neg.shape[3]
I_Summed_neg = I_Summed_neg.sum(axis=3)/neg_len

I_Summed_neg_cb = I_Summed_neg[:,y[0]:y[0]+yint,100:177].sum(axis=(1)) #Individual Delays
I_Summed_neg = I_Summed_neg[:,y[0]:y[0]+yint,0:177].sum(axis=(1)) #Individual Delays

I_Neg = I[:,:,:,6:t0-15]
neg_len = I_Neg.shape[3]
I_Neg = I_Neg.sum(axis=3)/neg_len

video_array = I[:,y[0]:y[0]+yint,0:177,20:].sum(axis=(1)) #Individual Delays
#video_array = I_Enhanced_Full[x[0]:x[0]+yint,:,0:177,10:].sum(axis=(1)) #Individual Delays
#I_Summed_neg = I_Summed_neg[:,y[0]:y[0]+yint,100:177].sum(axis=(1)) #Individual Delays

video_axis = ax_delay_offset[20:]
fig = plt.figure()

def update_img(n):

    vid = I_Difference_Full[:,y[0]:y[0]+yint,0:177,20:].sum(axis=(1))
   # vid = vid/np.max(abs(vid))
    
    fr_cb = I[:,y[0]:y[0]+yint,100:177,20:].sum(axis=(1))
    fr_cb = (fr_cb[:,:,n:n+5])
    fr_len = fr_cb.shape[2]
    fr_cb = fr_cb.sum(axis=2)/fr_len
    fr_difference_cb = fr_cb #- I_Summed_neg_cb   
    CB_Norm = np.max(abs(fr_difference_cb))
    #fr_difference = fr_difference/CB_Norm

    fr = I[:,y[0]:y[0]+yint,0:177,20:].sum(axis=(1))
    fr = (fr[:,:,n:n+5]).sum(axis=2)/fr_len
    fr_difference = fr #- I_Summed_neg   
    total_Norm = np.max(abs(fr_difference))
    fr_difference = fr_difference/total_Norm
    
    scale_factor = total_Norm/CB_Norm
    
    #mask_start = (np.abs(ax_E_offset - 0.5)).argmin()
    logic_mask = np.ones((fr_difference.shape))
    logic_mask[:,mask_start:] *= scale_factor
    fr_difference_scaled = logic_mask * fr_difference
    
    plt.imshow(np.transpose(vid[:,:,n:n+2].sum(axis=2))/2,cmap='terrain_r', origin = 'lower', clim=None, interpolation='none', extent=[ax_ky[0],ax_ky[-1],ax_E_offset[0], ax_E_offset[177]])
    plt.xlim(-1.6, 1.6)
    plt.xlabel('$k, A^{-1}$', fontsize = 18)
    plt.ylabel('$E-E_{VBM}$, eV', fontsize = 18)
    plt.title('T = ' + str(round(video_axis[n])) + ' fs', fontsize = 22)
    #fig.set_title('T = ' + str(ax_delay[n]) + ' fs',(10,10), fontsize = 16, weight = 'bold', color='white')
    
v_len = video_array.shape[2]

ani = FuncAnimation(fig, update_img, frames = v_len)
writergif = animation.PillowWriter(fps=8)

ani.save('trARPES_test_New.gif', writer=writergif)
#%%

### Make a video/gif of the Dynamics!

#y = [y]
import imageio
from matplotlib.animation import FuncAnimation
from matplotlib import animation

I_Summed_neg = I[:,:,:,6:t0-15]
neg_len = I_Summed_neg.shape[3]
I_Summed_neg = I_Summed_neg.sum(axis=3)/neg_len

I_Summed_neg_cb = I_Summed_neg[:,y[0]:y[0]+yint,100:177].sum(axis=(1)) #Individual Delays
I_Summed_neg = I_Summed_neg[:,y[0]:y[0]+yint,0:177].sum(axis=(1)) #Individual Delays

I_Neg = I[:,:,:,6:t0-15]
neg_len = I_Neg.shape[3]
I_Neg = I_Neg.sum(axis=3)/neg_len

video_array = I[:,y[0]:y[0]+yint,0:177,20:].sum(axis=(1)) #Individual Delays
#video_array = I_Enhanced_Full[x[0]:x[0]+yint,:,0:177,10:].sum(axis=(1)) #Individual Delays
#I_Summed_neg = I_Summed_neg[:,y[0]:y[0]+yint,100:177].sum(axis=(1)) #Individual Delays

video_axis = ax_delay_offset[20:]
fig = plt.figure()

def update_img(n):

    vid = I_Difference_Full[:,y[0]:y[0]+yint,0:177,20:].sum(axis=(1))
   # vid = vid/np.max(abs(vid))
    
    fr_cb = I[:,y[0]:y[0]+yint,100:177,20:].sum(axis=(1))
    fr_cb = (fr_cb[:,:,n:n+5])
    fr_len = fr_cb.shape[2]
    fr_cb = fr_cb.sum(axis=2)/fr_len
    fr_difference_cb = fr_cb - I_Summed_neg_cb   
    CB_Norm = np.max(abs(fr_difference_cb))
    #fr_difference = fr_difference/CB_Norm

    fr = I[:,y[0]:y[0]+yint,0:177,20:].sum(axis=(1))
    fr = (fr[:,:,n:n+5]).sum(axis=2)/fr_len
    fr_difference = fr - I_Summed_neg   
    total_Norm = np.max(abs(fr_difference))
    fr_difference = fr_difference/total_Norm
    
    scale_factor = total_Norm/CB_Norm
    
    #mask_start = (np.abs(ax_E_offset - 0.5)).argmin()
    logic_mask = np.ones((fr_difference.shape))
    logic_mask[:,mask_start:] *= scale_factor
    fr_difference_scaled = logic_mask * fr_difference
    
    plt.imshow(np.transpose(vid[:,:,n:n+2].sum(axis=2))/2,cmap='terrain_r', origin = 'lower', clim=None, interpolation='none', extent=[ax_ky[0],ax_ky[-1],ax_E_offset[0], ax_E_offset[177]])
    plt.xlim(-1.6, 1.6)
    plt.xlabel('$k, A^{-1}$')
    plt.ylabel('$E-E_{VBM}$, eV')
    plt.title('T = ' + str(round(video_axis[n])) + ' fs')
    #fig.set_title('T = ' + str(ax_delay[n]) + ' fs',(10,10), fontsize = 16, weight = 'bold', color='white')
    
v_len = video_array.shape[2]

ani = FuncAnimation(fig, update_img, frames = v_len)
writergif = animation.PillowWriter(fps=8)

ani.save('trARPES_test_New.gif', writer=writergif)


#%%

### Make a video/gif of the Dynamics -- with Pos-Neg Difference!

import imageio
from matplotlib.animation import FuncAnimation
from matplotlib import animation

tstart = -200
tstart = np.abs(ax_delay_offset-tstart).argmin()
video_axis = ax_delay_offset[tstart:]

fig = plt.figure()

I_Summed_neg = I[:,:,:,5:t0-10]
neg_len = I_Summed_neg.shape[3]
I_Summed_neg = I_Summed_neg.sum(axis=3)/neg_len

I_truncated = (I[:,:,:,tstart:]) #Sum over delay/theta/ADC for Plotting...

def update_img(n):
    print(n+5)
    #tstart = -200
    #tstart_neg = np.abs(ax_delay_offset-).argmin()
    
    #n delay average
    #start2 = n
    #stop2 = n+5
    I_Summed_2 = I_truncated[:,:,:,n+n+5]/(5)
    #frame_len = I_Summed_2.shape[3]
    #I_Summed_2 = I_Summed_2.sum(axis=3)/frame_len
    
    I_Enhanced_neg = 0.005*logicMask * I_Summed_neg
    I_Enhanced_2 = 0.005*logicMask * I_Summed_2
     
    slice_E_k_1 = I_Enhanced_neg[:,y[0]:y[0]+yint,:].sum(axis=(1))
    f1_norm = np.max(slice_E_k_1)
    slice_E_k_1 = slice_E_k_1/f1_norm
    
    slice_E_k_2 = I_Enhanced_2[:,y[0]:y[0]+yint,:].sum(axis=(1))
    slice_E_k_2 = slice_E_k_2/f1_norm

    slice_E_k_3 = I_Enhanced_neg[x[0]:x[0]+xint,:].sum(axis=(0))
    slice_E_k_4 = I_Enhanced_2[x[0]:x[0]+xint,:].sum(axis=(0))
    
    I_1_N = slice_E_k_1/np.max(np.abs(slice_E_k_1))
    I_2_N = slice_E_k_2/np.max(np.abs(slice_E_k_1)) #sliceE_K_2
    
    I_3_N = slice_E_k_3/np.max(np.abs(slice_E_k_3))
    I_4_N = slice_E_k_4/np.max(np.abs(slice_E_k_4))
    
    I_dif = I_2_N - I_1_N
    I_dif_ = I_4_N - I_3_N
    
    #I_dif = I_dif/np.max(I_dif)
    #y = [y]
    
    #tstart = 5
    video_array = I_dif[0:177,:] #Individual Delays
    
    plt.imshow(np.transpose(I_dif[:,:]),cmap='seismic', origin = 'lower', clim=[-1,1], interpolation='none', extent=[ax_ky[0],ax_ky[-1],ax_E_offset[0], ax_E_offset[177]])
    plt.xlim(-1.6, 1.6)
    plt.xlabel('$k, A^{-1}$')
    plt.ylabel('$E-E_{VBM}$, eV')
    plt.title('T = ' + str(round(video_axis[n])) + ' fs')
    #fig.set_title('T = ' + str(ax_delay[n]) + ' fs',(10,10), fontsize = 16, weight = 'bold', color='white')

v_len = len(ax_delay_offset[tstart:-10]) #video_array.shape[1]

ani = FuncAnimation(fig, update_img, frames = 100)
writergif = animation.PillowWriter(fps=10)
ani.save('trARPES_test_DIFF2.gif', writer=writergif)

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