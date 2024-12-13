#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 20:38:09 2024

@author: lawsonlloyd
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
from skimage.draw import disk
from scipy.optimize import curve_fit
from scipy import signal
from scipy.fft import fft, fftshift

#%%

def transform_data(data_handler, I, neg_time):
    if I.ndim > 3:
        t0 = data_handler.get_t0()
        
        neg_time = neg_time
        tnf = (np.abs(ax_delay_offset - neg_time)).argmin()
        I_neg = I[:,:,:,5:tnf+1]
        neg_length = I_neg.shape[3]
        I_neg = I_neg.sum(axis=(3))/neg_length
            
        I_pos = I[:,:,:,t0:]
        pos_length = I_pos.shape[3]
        I_pos = I_pos.sum(axis=(3))/pos_length
        
        I_sum_delay = I[:,:,:,:].sum(axis=(3))    
    else:
        I_neg = I[:,:,:] #Sum over delay/polarization/theta...
        I_pos = I[:,:,:]
        I_sum_delay = I

    return I_neg, I_pos, I_sum_delay, t0


def make_MM_frames(I_neg, I_pos, E, E_int):
    E_int = E_int/2
    E_i = (np.abs(ax_E_offset - (E-Eint))).argmin()
    E_f = (np.abs(ax_E_offset - (E+Eint))).argmin()
    
    frame_neg = np.transpose(I_neg[:,:,E_i:E_f+1].sum(axis=(2)))
    frame_pos = np.transpose(I_pos[:,:,E_i:E_f+1].sum(axis=(2)))
    frame_sum = frame_neg + frame_pos
    
    cts_pos, cts_neg, cts_total = np.sum(frame_pos[:,:]),  np.sum(frame_neg[:,:]),  np.sum(frame_sum)
    frame_diff = frame_pos - frame_neg

    return frame_pos, frame_neg, frame_diff


def window_MM(kspace_frame, k_i_x, k_f_x, k_i_y, k_f_y):    

    #mn = np.mean(kspace_frame[:,25:35])
    #kspace_frame = kspace_frame - mn
    #kspace_frame[kspace_frame<0] = 0
    
    ### Deconvolve k-space momentum broadening, Gaussian with FWHM 0.063A-1
    fwhm = 0.063
    fwhm_pixel = fwhm/dkx
    sigma = fwhm_pixel/2.355
    
    gaussian_kxky = np.outer(gaussian_kx, gaussian_ky)
    #kx_cut_deconv = signal.deconvolve(kx_cut, gaussian_kx)
    
    #### Apply Symm and windows
    kspace_frame = kspace_frame/np.max(kspace_frame)
    window = np.zeros((kspace_frame.shape))
    window_2 = np.zeros((kspace_frame.shape))
    
    win_1 = np.zeros(kspace_frame.shape[0])
    win_2 = np.zeros(kspace_frame.shape[1])
    win_1_box = np.zeros(kspace_frame.shape[0])
    win_2_box = np.zeros(kspace_frame.shape[1])

    k_i = (np.abs(ax_ky - k_i_y)).argmin()
    k_f = (np.abs(ax_ky - k_f_y)).argmin()
    k_i_2 = (np.abs(ax_kx - k_i_x)).argmin()
    k_f_2 = (np.abs(ax_kx - k_f_x)).argmin()
    
    tuk_1 = signal.windows.tukey(k_f-k_i)
    box_1 = signal.windows.boxcar(k_f-k_i)

    tuk_2 = signal.windows.tukey(k_f_2-k_i_2)
    box_2 = signal.windows.boxcar(k_f_2-k_i_2)

    win_1[k_i:k_f] = tuk_1
    win_1_box[k_i:k_f] = box_1

    win_2[k_i_2:k_f_2] = tuk_2
    win_2_box[k_i_2:k_f_2] = box_2

    window_4 = np.outer(win_1, win_2)
    window_5 = np.outer(win_1_box, win_2_box) # Square Window
    window_6 = np.outer(win_1, win_2_box)
    #window_6 = np.outer(win_1_box, win_2)
    
    for yy in range(0,window.shape[1]):
        window[k_i:k_f,yy] = signal.windows.tukey(k_f-k_i)
        window[k_i:k_f,yy] = np.ones(k_f-k_i)

    for xx in range(0, window.shape[0]):
        window_2[xx,k_i_2:k_f_2] = signal.windows.tukey(k_f_2-k_i_2)
        window_2[xx,k_i_2:k_f_2] = np.ones(k_f_2-k_i_2)
   
    ### Symmetrize Data
    frame_sym = np.zeros(kspace_frame.shape)
    frame_sym[:,:] = kspace_frame[:,:]  + (kspace_frame[:,::-1])    
    frame_sym =  frame_sym[:,:]/2
    
    windowed_frame_symm = frame_sym*window_6
    windowed_frame_nonsymm = kspace_frame*window_6
    #windowed_frame_symm = frame_sym

    return windowed_frame_nonsymm

#%%

I_neg, I_pos, I_sum_delay, t0 = transform_data(data_handler, I, -60) # Get negative delay and positive delay data sets

ax_E_offset = data_handler.ax_E
ax_delay_offset = data_handler.ax_delay
    
dkx = (ax_kx[1] - ax_kx[0])
dE = ax_delay_offset[1] - ax_delay_offset[0]

cmap_LTL = plot_manager.custom_colormap(plt.cm.viridis, 0.2) #choose colormap based and percentage of total map for new white transition map
    
#%%
%matplotlib inline

###
# Window and Symmetrize MM for FFT
###

save_figure = True
figure_file_name = '2DFFT_Windowing' 
    
E, E_int  = 1.35, 0.160
k_i_y, k_f_y = -.3, .3 #ky
k_i_x, k_f_x = 0, 1.15 #kx

fig, ax = plt.subplots(1, 3, sharey=True)
plt.gcf().set_dpi(300)
ax = ax.flatten()
            
im = ax[0].imshow(kspace_frame, origin='lower', cmap=cmap_LTL, interpolation='none', extent = [ax_kx[0],ax_kx[-1],ax_ky[0],ax_ky[-1]]) #kx, ky, t
im = ax[1].imshow(frame_sym, origin='lower', cmap=cmap_LTL, interpolation='none', extent = [ax_kx[0],ax_kx[-1],ax_ky[0],ax_ky[-1]]) #kx, ky, t
im = ax[2].imshow(windowed_frame_symm, origin='lower', cmap=cmap_LTL, interpolation='none', extent = [ax_kx[0],ax_kx[-1],ax_ky[0],ax_ky[-1]]) #kx, ky, t

for i in np.arange(3):
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
    ax[i].set_xlabel('$k_x$', fontsize = 14)
    ax[i].set_ylabel('$k_y$', fontsize = 14)
    ax[i].tick_params(axis='both', labelsize=12)
    ax[i].set_title('$E$ = ' + str(E) + ' eV', fontsize = 16)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([1, 0.2, 0.025, 0.6])
fig.colorbar(im, cax=cbar_ax, ticks = [10,100])

#fig.colorbar(im, fraction=0.046, pad=0.04)
fig.tight_layout()
plt.show()

if save_figure is True:
    fig.savefig((figure_file_name +'.svg'), format='svg')

#%%

%matplotlib inline


save_figure = True
figure_file_name = 'MM_FFT' 
    

#####                                              #####
##### Plot FFT of MMs to obtain real space wavefxn #####
#####                                              #####

#momentum_frame = windowed_frame_symm
momentum_frame = windowed_frame_nonsymm

#momentum_frame = window_4
#momentum_frame = window_6

#momentum_frame = momentum_frame - np.mean(momentum_frame)

##########################
##########################
### Define real-space axis

k_step = np.abs((ax_kx[1] - ax_kx[0]))
k_length = len(ax_kx)

k_step_y = np.abs((ax_ky[1] - ax_ky[0]))
k_length_y = len(ax_ky)

zplength = 2024 #5*k_length+1
max_r = (1/2)*1/(k_step)

#r_axis = np.linspace(-max_r, max_r, num = k_length)
r_axis = np.linspace(-max_r, max_r, num = zplength)
#r_axis = r_axis/(10)

# Shuo Method ?
N = 1 #(zplength)
Fs = 1/((2*np.max(ax_kx))/len(ax_kx))
r_axis = np.arange(0,zplength)*Fs/zplength
r_axis = r_axis - (np.max(r_axis)/2)
r_axis = r_axis/(1)

### Do the FFT operations to get --> |Psi(x,y)|^2 ###
momentum_frame_ = np.abs(momentum_frame)/np.max(momentum_frame)
momentum_frame_sqrt = np.sqrt(momentum_frame_)
fft_frame = np.fft.fft2(momentum_frame_sqrt, [zplength, zplength])
fft_frame = np.fft.fftshift(fft_frame, axes = (0,1))

fft_frame_rsq = np.abs(fft_frame) 
fft_frame_s = np.square(np.abs(fft_frame)) #frame squared

### Take x and y cuts and extract bohr radius
ky_cut = momentum_frame_[:,int(len(ax_ky)/2)-1-4:int(len(ax_ky)/2)-1+4].sum(axis=1)
ky_cut = ky_cut/np.max(ky_cut)
kx_cut = momentum_frame_[int(len(ax_kx)/2)-1-4:int(len(ax_kx)/2)-1+4,:].sum(axis=0)
kx_cut = kx_cut/np.max(kx_cut)

y_cut = fft_frame_s[:,int(zplength/2)-1] # real space Psi*^2 cut
x_cut = fft_frame_s[int(zplength/2)-1,:]
x_cut = x_cut/np.max(x_cut)
y_cut = y_cut/np.max(y_cut)

r2_cut_y = fft_frame_rsq[:,int(zplength/2)-1] #real space Psi cut
r2_cut_y = np.square(np.abs(r2_cut_y*r_axis)) #|r*Psi(r)|^2
r2_cut_y = r2_cut_y/np.max(r2_cut_y)

r2_cut_x = fft_frame_rsq[int(zplength/2)-1,:]
r2_cut_x = np.square(np.abs(r2_cut_x[0:1090]*r_axis[0:1090]))
r2_cut_x = r2_cut_x/np.max(r2_cut_x)

rdist_brad_x = np.argmax(r2_cut_x[int(zplength/2)-10:int(zplength/2)+90])
rdist_brad_y = np.argmax(r2_cut_y[int(zplength/2)-10:int(zplength/2)+150])

rdist_brad_x = r_axis[int(zplength/2)-10 + rdist_brad_x]
rdist_brad_y = r_axis[int(zplength/2)-10 + rdist_brad_y]

x_brad = (np.abs(x_cut[int(zplength/2)-10:int(zplength/2)+200] - 0.5)).argmin()
y_brad = (np.abs(y_cut[int(zplength/2)-10:] - 0.5)).argmin()
x_brad = int(zplength/2)-10 + x_brad
y_brad = int(zplength/2)-10 + y_brad
x_brad = r_axis[x_brad]
y_brad = r_axis[y_brad]

############
### Plot ###
############

fig, ax = plt.subplots(2, 2)
fig.set_size_inches(6,8)
plt.gcf().set_dpi(300)
ax = ax.flatten()

im00 = ax[0].imshow(kspace_frame/np.max(kspace_frame), clim = None, origin = 'lower', vmax = 1, cmap=cmap_LTL, interpolation = 'none', extent = [ax_kx[0], ax_kx[-1], ax_ky[0], ax_ky[-1]])
im0 = ax[1].imshow(momentum_frame/np.max(momentum_frame), clim = None, origin = 'lower', vmax = 1, cmap=cmap_LTL, interpolation = 'none', extent = [ax_kx[0], ax_kx[-1], ax_ky[0], ax_ky[-1]])
im = ax[2].imshow(fft_frame_s, clim = None, origin='lower', cmap=cmap_LTL, interpolation='none', extent = [r_axis[0], r_axis[-1], r_axis[0], r_axis[-1]]) #kx, ky, t
#single_k_circle = plt.Circle((single_ky, single_kx), single_rad, color='red', linestyle = 'dashed', linewidth = 1.5, clip_on=False, fill=False)
#ax[1].add_patch(single_k_circle)
ax[0].set_aspect(1)
ax[1].set_aspect(1)
ax[2].set_aspect(1)

#ax[0].axhline(y,color='black')
#ax[0].axvline(x,color='bl ack')

ax[0].set_xticks(np.arange(-3,3.2,1))
for label in ax[0].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
   # label.set_xticklabels(tick_labels.astype(int))
    
ax[0].set_yticks(np.arange(-3,3.2,1))
for label in ax[0].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)

ax[1].set_xticks(np.arange(-3,3.2,1))
for label in ax[1].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
   # label.set_xticklabels(tick_labels.astype(int))
    
ax[1].set_yticks(np.arange(-3,3.2,1))
for label in ax[1].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)

ax[2].set_xticks(np.arange(-8,8.2,1))
for label in ax[2].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
    
ax[2].set_yticks(np.arange(-8,8.1,1))
for label in ax[2].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
    
ax[3].set_xticks(np.arange(0,5.2,1))
#for label in ax[3].xaxis.get_ticklabels()[1::2]:
    #label.set_visible(False)    
    
ax[0].set_xlim(-2,2)
ax[0].set_ylim(-2,2)
#ax[0].set_box_aspect(1)
ax[0].set_xlabel('$k_x$, $\AA^{-1}$', fontsize = 16)
ax[0].set_ylabel('$k_y$,  $\AA^{-1}$', fontsize = 16)
ax[0].tick_params(axis='both', labelsize=10)
ax[0].set_title('$E$ = ' + str(E) + ' eV, ' + '$\Delta$E = ' + str(tint*dE) + ' eV', fontsize = 14)
ax[0].set_title('$E$ = ' + str(E) + ' eV ', fontsize = 14)
fig.suptitle('E = ' + str(E) + ' eV, $\Delta$E = ' + str(tint_E) + ' meV,  $\Delta$$k_{rad}$ = ' + str(window_k_width) + ' $\AA^{-1}$', fontsize = 18)

ax[1].set_xlim(-2,2)
ax[1].set_ylim(-2,2)
#ax[0].set_box_aspect(1)
ax[1].set_xlabel('$k_x$, $\AA^{-1}$', fontsize = 16)
ax[1].set_ylabel('$k_y$,  $\AA^{-1}$', fontsize = 16)
ax[1].tick_params(axis='both', labelsize=10)
ax[1].set_title('Windowed', fontsize = 15)
 
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

#ax[3].annotate('$r^*_x$', xy = (rdist_brad+.15, 0.5), fontsize = 14, color = 'blue', weight = 'bold')
ax[3].set_xlim([0, 2])
ax[3].set_ylim([-0.025, 1.025])
ax[3].set_xlabel('$r$, nm', fontsize = 16)
ax[3].set_ylabel('Norm. Int.', fontsize = 16)
ax[3].set_title(('$r^*_x$ = ' + str(round(x_brad,2)) + ' nm' + \
                 ', $r^*_y$ = ' + str(round(y_brad,1))) + ' nm', fontsize = 14)
#ax[3].set_title(('$r^*_x$ = ' + str(round(x_brad,2)) + ' nm' + ', $r^*_y$ = ' + str(round(y_brad,2))) + ' nm', fontsize = 14)
ax[3].tick_params(axis='both', labelsize=10)
ax[3].set_yticks(np.arange(-0,1.5,0.5))
ax[3].set_aspect(2)
ax[3].set_xlabel('$r$, nm')
ax[3].legend(frameon=False, fontsize = 12)
fig.subplots_adjust(right=0.58, top = 1.1)
fig.tight_layout()
plt.show()

new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'}

plt.rcParams.update(new_rc_params)

if save_figure is True:
    fig.savefig((figure_file_name +'.svg'), format='svg')
    
print(rdist_brad_x)
print(rdist_brad_y)
