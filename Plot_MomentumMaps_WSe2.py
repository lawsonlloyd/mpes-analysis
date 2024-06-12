# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 10:44:11 2023

@author: lloyd
"""
#%%
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

I_Summed_pos = I[:,:,:,t0+-1:t0+3]
pos_length = I_Summed_pos.shape[3]
I_Summed_pos = I_Summed_pos.sum(axis=(3)) #Sum over delay/polarization/theta...

I_Summed_ = I[:,:,:,:].sum(axis=(3)) # Summ over all delay/ADC bins

#I_Summed = ndimage.rotate(I_Summed, 12, reshape=False, order=3, mode='constant', cval=0.0, prefilter=True)
mask_start = (np.abs(ax_E_offset - 1.0)).argmin()
logicMask_Full = np.ones((I.shape))
logicMask_Full[:,:,mask_start:] *= 200
I_Enhanced_Full = logicMask_Full * I

logicMask = np.ones((I_Summed.shape))
logicMask[:,:,mask_start:] *= 50
I_Enhanced = logicMask * I_Summed
#testP = testP - (testP[:,:,:,0:20].sum(axis=3))

#logicMask = np.ones((I_.shape))
#logicMask[:,:,17ff0:] *= 200
#I_ = logicMask * I_

#ax_E = ax_E + 0.05

#%%
%matplotlib inline
# Plot EDCs

fig, ax = plt.subplots(1,3, sharey=False)
plt.gcf().set_dpi(300)

edc = I_Summed_[x[0]-2:x[0]+2,y[0]-2:y[0]+2,:].sum(axis=(0,1))
edc = edc/np.max(edc)

ax[0].plot(ax_E_offset, edc, color = 'red')
ax[0].axvline(0, color = 'black', linestyle = 'dashed', linewidth = 1)
ax[0].axvline(1.55, color = 'black', linestyle = 'dashed', linewidth = 1)
ax[0].axvline(-.9, color = 'black', linestyle = 'dashed', linewidth = 1)

ax[0].set_xlim([-2, 2])

ax[1].plot(ax_E_offset, edc, color = 'red')
ax[1].axvline(0, color = 'black', linestyle = 'dashed', linewidth = 1)
ax[1].axvline(1.55, color = 'black', linestyle = 'dashed', linewidth = 1)
ax[1].set_xlim([-0.1, 2])
ax[1].set_ylim([0.0, 0.05])

ax[2].imshow(np.transpose(I_Enhanced[x[0]-2:x[0]+2,:,:].sum(axis=(0))), aspect = 'auto', extent=[ax_kx[0], ax_kx[-1], ax_E_offset[-1], ax_E_offset[0]])
ax[2].imshow(np.transpose(I_Enhanced[:, y[0]-2:y[0]+2,:].sum(axis=(1))), aspect = 'auto', extent=[ax_kx[0], ax_kx[-1], ax_E_offset[-1], ax_E_offset[0]])
ax[2].set_extent=([ax_E_offset[0], ax_E_offset[-1], ax_kx[0], ax_kx[-1]])
ax[2].invert_yaxis()
ax[2].axhline(0, linestyle = 'dashed', color = 'black', linewidth = 1)
ax[2].axhline(1.55, linestyle = 'dashed', color = 'black', linewidth = 1)
ax[2].set_ylim([-1,2])

fig.tight_layout()
plt.show()

#%%
### User Inputs
tMaps, tint  = [0, 1.5], 2

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
%matplotlib auto
plt.imshow(frame_diff)

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
# Window and Symmetrize MM for FFT
%matplotlib inline

from scipy import signal
from scipy.fft import fft, fftshift
import numpy as np
from skimage.draw import disk

tMaps, tint  = [1.55], 3
window_k_width = .3
circle_mask = False

### Plot
numPlots = len(tMaps)

fig, ax = plt.subplots(1, 3, sharey=True)
plt.gcf().set_dpi(300)
ax = ax.flatten()

sat = [1, 1, 1]
for i in np.arange(numPlots, dtype = int):
    tMap = tMaps[i]
    tMap = (np.abs(ax_E_offset - tMap)).argmin()
    
    frame_diff = np.abs(I[:,:,tMap-2:tMap+2,1:3].sum(axis=(2,3)))
    #frame_diff = np.abs(I_Summed_pos[:,:,tMap-2:tMap+2].sum(axis=2))
    #2D Tukey Window
    window_new = np.zeros((frame_diff.shape))
    k_step = np.abs((ax_kx[1] - ax_kx[0]))
    mi = int(window_k_width/k_step) # Half of window length along one direction
    window1d = np.abs(signal.windows.tukey(2*mi))
    window2d = np.sqrt(np.outer(window1d,window1d))

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
    #window_new = mask

#    window2d = mask
    
    #Old Window
    # window = np.zeros((frame_diff.shape))
    # window_2 = np.zeros((frame_diff.shape))
    # k_i = (np.abs(ax_kx - -.3)).argmin()
    # k_f = (np.abs(ax_kx - .7)).argmin()
    # for y in range(0,window.shape[1]):
    #     window[k_i:k_f,y] = signal.windows.tukey(k_f-k_i)
    # for x in range(0, window.shape[0]):
    #     window_2[x,20:83] = signal.windows.tukey(63)
    # window = window*window_2
    
    wl = 6
    k_points_y = [113, 73, 34, 34, 74, 113] 
    k_points_x = [53, 30, 52, 100, 122, 100]
    
    k_points_y = [78, 100, 83, 39, 21, 41] 
    k_points_x = [24, 61, 95, 95, 61, 23]
   
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
    
    window_new = mask
        #rr, cc = disk((row, col), radius_2)
        #mask[rr, cc] = 0
        #window_new = mask
    
    
    frame_sym = np.zeros(frame_diff.shape)
    frame_sym[:,:] = frame_diff[:,:]  + (frame_diff[:,::-1])    
    frame_sym =  frame_sym[:,:]/2
    
    if circle_mask is True:
        windowed_frame_symm = frame_diff*mask*window_new
    elif circle_mask is False:
        windowed_frame_symm = frame_diff*window_new

    #windowed_frame_symm = frame_sym*window_new
    
    #windowed_frame_symm = np.zeros(windowed_frame.shape)
    #windowed_frame_symm[:,:] = windowed_frame[:,:]  + (windowed_frame[:,::-1])
    #windowed_frame_symm = windowed_frame_symm/np.abs(windowed_frame_symm)
    
im = ax[0].imshow(frame_diff, origin='lower', cmap=cmap_LTL, interpolation='none', extent = [ax_kx[0],ax_kx[-1],ax_ky[0],ax_ky[-1]]) #kx, ky, t
im = ax[1].imshow(frame_sym, origin='lower', cmap=cmap_LTL, interpolation='none', extent = [ax_kx[0],ax_kx[-1],ax_ky[0],ax_ky[-1]]) #kx, ky, t
im = ax[2].imshow(windowed_frame_symm, origin='lower', cmap=cmap_LTL, interpolation='none', extent = [ax_kx[0],ax_kx[-1],ax_ky[0],ax_ky[-1]]) #kx, ky, t

    #im = ax[i].imshow(np.transpose(I[:,:,tMap-int(tint/2):tMap+int(tint/2), adc-1:adc+1].sum(axis=(2,3))), origin='lower', cmap='terrain_r', clim=None, interpolation='none', extent=[-2,2,-2,2]) #kx, ky, t
for i in np.arange(3, dtype = int):
    ax[i].axhline(0, color='black', linewidth = 1, linestyle = 'dashed')
    ax[i].axhline(0, color='black', linewidth = 1, linestyle = 'dashed')
    ax[i].axvline(0, color='black', linewidth = 1, linestyle = 'dashed')
    ax[i].axvline(0, color='black', linewidth = 1, linestyle = 'dashed')
    
    ax[i].axvline(-1.1, color='blue', linewidth = 1, linestyle = 'dashed')
    ax[i].axvline(1.1, color='blue', linewidth = 1, linestyle = 'dashed')
    
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
    ax[i].set_xlabel('$r_x$', fontsize = 14)
    ax[i].set_ylabel('$r_y$', fontsize = 14)
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

j = 52
i = 114

a = frame_sym[:,i-5:i+5].sum(axis=1)
aa = np.max(abs(a))
aaa = a/aa

b = windowed_frame_symm[:,i-5:i+5].sum(axis=1)
bb = np.max(abs(b))
bbb = b/bb
#lawson was here.
c = window_new[:,i-5:i+5].sum(axis = 1)
cc = np.max(abs(c))
ccc = c/cc

cut = [aaa, bbb, ccc]
titles = ['Frame Sym', 'Windowed', 'Tukey Win']

plt.figure()
for i in range(0,3):
    plt.plot(cut[i])
    ax[i].set_title(titles[i])
fig.tight_layout()

fig, ax = plt.subplots(1, 3, sharey=False)
ax = ax.flatten()

for i in range(0,3):
    fr = cut[i]
    fr = np.sqrt(fr)
    fft_ = np.fft.fft(fr)
    fft_ = np.fft.fftshift(fft_, axes = 0)
    fft_ = np.abs(fft_)**2

    ax[i].plot(np.real(fft_))
    ax[i].plot(np.imag(fft_))
    ax[i].set_title(titles[i])
    #ax[i].plot(fftshift(fft(np.abs(cut[i]))))
fig.tight_layout()

fig, ax = plt.subplots(2, 3, sharey=False)
ax = ax.flatten()

cut = [frame_sym, windowed_frame_symm, window_new]

for i in range(0,3):
    
    ax[i].imshow(cut[i])
    ax[i].set_title(titles[i], fontsize = 12)

for i in range(0,3):

    fr = cut[i]
    fr = np.sqrt(fr)
    fft_ = np.fft.fft2(fr)
    fft_ = np.fft.fftshift(fft_,  axes = (0,1))
    fft_ = np.abs(fft_)**2
    
    ax[i+3].imshow(fft_)
    ax[i+3].set_title('FT ' + titles[i], fontsize = 12)
    ax[i+3].set_xlim(50,100)
    ax[i+3].set_ylim(50,100)

    #ax[i].plot(fftshift(fft(np.abs(cut[i]))))

fig.tight_layout()
plt.show()

#%%
# Plot FFT of MMs to obtain real space wavefxn

momentum_frame = windowed_frame_symm

tMaps, tint  = [1.5], 2

k_step = np.abs((ax_kx[1] - ax_kx[0]))
k_length = len(ax_kx)

zplength = 3*k_length
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
    
    ax[i].set_xlim(-3,3)
    ax[i].set_ylim(-3,3)
    #ax[0].set_box_aspect(1)
    ax[i].set_xlabel('$r_a$', fontsize = 14)
    ax[i].set_ylabel('$r_b$', fontsize = 14)
    ax[i].tick_params(axis='both', labelsize=12)
    ax[i].set_title('$E$ = ' + str((tMaps[i])) + ' eV', fontsize = 16)
    #ax[i].annotate(('E = '+ str(round(tMaps[i],2)) + ' eV'), xy = (-1.85, 1.6), fontsize = 14, weight = 'bold')

ax[1].plot(r_axis, x_cut/np.max(1))
ax[1].plot(r_axis, y_cut/np.max(1))
ax[1].set_xlim([0,4])
ax[1].axvline(1.75, color='black')
#ax[1].set_aspect(.003)
ax[1].set_xlabel('r,')
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