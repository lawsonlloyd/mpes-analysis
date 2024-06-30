# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 10:44:11 2023

@author: lloyd
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
from scipy import signal
from scipy.fft import fft, fftshift
import numpy as np
from skimage.draw import disk


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
mask_start = (np.abs(ax_E_offset - 0.75)).argmin()
logicMask_Full = np.ones((I.shape))
logicMask_Full[:,:,mask_start:] *= 25
I_Enhanced_Full = logicMask_Full * I

logicMask = np.ones((I_Summed.shape))
logicMask[:,:,mask_start:] *= 10
I_Enhanced = logicMask * I_Summed
#testP = testP - (testP[:,:,:,0:20].sum(axis=3))

#logicMask = np.ones((I_.shape))
#logicMask[:,:,17ff0:] *= 200
#I_ = logicMask * I_

#ax_E = ax_E + 0.05

#%%
%matplotlib inline
# Plot EDCs

kspace_frame_full = np.abs(I[:,:,:,t0-5:t0+5].sum(axis=(3))) #Scan 160, XUV Pol, Bulk LTL

tMapE = 1.7
kx = -0.05
ky = 0.025

#tMapE = 1.6
#kx = 0.68
#ky = 1.1

exciton = 1.5
vbm_K = -0.5
cmap_to_use = cmap_LTL

######
######

cb_factor = 20
tMap = (np.abs(ax_E_offset - tMapE)).argmin()
x = (np.abs(ax_kx - kx)).argmin()
y = (np.abs(ax_ky - ky)).argmin()

window_full = np.zeros(I_Enhanced.shape)

#LTL Bulk WSe2, XUV POL Integrated 
k_points_y = [115, 73, 34, 34, 74, 115]
k_points_x = [53, 30, 52, 100, 122, 98]

#Shuo ML WSe2
k_points_y = [78, 100, 83, 39, 21, 41] 
k_points_x = [24, 61, 95, 95, 61, 23]

#Shuo Dataset WSe2 Matlab
k_points_y = [58, 84, 84, 56, 28, 29] 
k_points_x = [27, 43, 76, 92, 75, 42]

# Window @K Points
for k in range(0,6):
    xc = k_points_x[k] 
    yc = k_points_y[k] 
    
    #Circular Mask around each k Point
    row = xc
    col = yc
    #k_outer = np.abs((ax_kx - 1.75)).argmin()
    #k_inner = np.abs((ax_kx - 1.75)).argmin()
    window_k_width = .1 #0.2
    radius = round(window_k_width/dkx) #8#52 # 52 #pixels
    rr, cc = disk((row, col), radius)
    window_full[rr, cc, :] = 1 #/np.max(kspace_frame_full[rr,cc])

kspace_frame_full_windowed = kspace_frame_full*window_full

edc = (kspace_frame_full_windowed[:,:,:]).sum(axis=(0,1))
edc_mask = np.ones((edc.shape))
edc_mask[mask_start:] *= cb_factor
edc = edc*edc_mask
#edc = I_Enhanced[x-3:x+3,y-3:y+3,:].sum(axis=(0,1))
edc = edc/np.max(edc[-120:])

mm = kspace_frame_full_windowed[:,:,tMap-int(tint/2):tMap+int(tint/2)].sum(axis=(2))
mm = (I[:,:,tMap-int(tint/2):tMap+int(tint/2),:].sum(axis=(2,3)))
mm = mm/np.max(mm)

fig, ax = plt.subplots(2,2,sharey=False)
ax = ax.flatten()
plt.gcf().set_dpi(300)

im = ax[0].imshow((mm), origin='lower', cmap=cmap_to_use, clim=None, extent=[ax_kx[0], ax_kx[-1], ax_ky[-1], ax_ky[0]]) #kx, ky, t
ax[0].set_aspect(1)
ax[0].axvline(ax_kx[x],color='green', linestyle = 'dashed', linewidth = 1)
ax[0].axhline(ax_ky[y],color='purple', linestyle = 'dashed', linewidth = 1)
ax[0].set_xlim(-2,2)
ax[0].set_ylim(-2,2)
ax[0].set_xticks(np.arange(-2,2.2,1))
for label in ax[0].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)    
ax[0].set_yticks(np.arange(-2,2.1,1))
for label in ax[0].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
ax[0].set_xlabel('$k_x$,  $\AA^{-1}$', fontsize = 14)
ax[0].set_ylabel('$k_y$,  $\AA^{-1}$', fontsize = 14)
ax[0].tick_params(axis='both', labelsize=12)
ax[0].set_title('$E$ = ' + str((tMapE)) + ' eV', fontsize = 16)
#ax[i].annotate(('E = '+ str(round(tMaps[i],2)) + ' eV'), xy = (-1.85, 1.6), fontsize = 14, weight = 'bold')

ax[1].plot(ax_E_offset, edc, color = 'red')
ax[1].axvline(0, color = 'black', linestyle = 'dashed', linewidth = 1)
ax[1].axvline(exciton, color = 'black', linestyle = 'dashed', linewidth = 1)
#ax[1].axvline(vbm_K, color = 'black', linestyle = 'dashed', linewidth = 1)
ax[1].set_xticks(np.arange(-2,2.2,1)) 
ax[1].set_yticks(np.arange(-2,2.1,1))
ax[1].set_xlim([-1, 2.5])
ax[1].set_ylim([0,1.2])
ax[1].set_xlabel('E, eV',  fontsize = 14)
ax[1].set_ylabel('Norm. Int.',  fontsize = 14)
ax[1].set_title('EDC @ K Points')
ax[1].annotate('x100', xy = (exciton+.1, 0.8), fontsize = 10, color = 'red')

ax[2].imshow(np.transpose(I_Enhanced[y-2:y+2,:,:].sum(axis=(0))), cmap = cmap_to_use, extent=[ax_kx[0], ax_kx[-1], ax_E_offset[-1], ax_E_offset[0]])
ax[2].set_extent=([ax_E_offset[0], ax_E_offset[-1], ax_kx[0], ax_kx[-1]])
ax[2].axvline(ax_ky[y], linestyle = 'dashed', color = 'green', linewidth = 1)
#ax[2].axvline(kx, linestyle = 'dashed', color = 'purple', linewidth = 1)
ax[2].axhline(0, linestyle = 'dashed', color = 'black', linewidth = 1)
#ax[2].axhline(vbm_K, linestyle = 'dashed', color = 'black', linewidth = 1)
ax[2].axhline(exciton, linestyle = 'dashed', color = 'black', linewidth = 1)
ax[2].invert_yaxis()
ax[2].set_xlabel('$k_x$,  $\AA^{-1}$', fontsize = 14)
ax[2].set_ylabel('E, eV', fontsize = 14)
ax[2].set_xticks(np.arange(-2,2.2,1))
for label in ax[2].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)    
ax[2].set_yticks(np.arange(-2,2.1,1))
for label in ax[2].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
ax[2].set_ylim([-2,2.25])
ax[2].set_xlim([-2,2])
ax[2].set_aspect(0.5)

ax[3].imshow(np.transpose(I_Enhanced[:,x-2:x+2,:].sum(axis=(1))), cmap = cmap_to_use, extent=[ax_kx[0], ax_kx[-1], ax_E_offset[-1], ax_E_offset[0]])
ax[3].set_extent=([ax_E_offset[0], ax_E_offset[-1], ax_ky[0], ax_ky[-1]])
ax[3].axvline(ax_kx[x], linestyle = 'dashed', color = 'purple', linewidth = 1)
ax[3].axhline(0, linestyle = 'dashed', color = 'black', linewidth = 1)
#ax[3].axhline(vbm_K, linestyle = 'dashed', color = 'black', linewidth = 1)
ax[3].axhline(exciton, linestyle = 'dashed', color = 'black', linewidth = 1)
ax[3].invert_yaxis()
ax[3].set_ylim([-1.25,2.25])
ax[3].set_aspect(.5)
ax[3].set_xlabel('$k_y$,  $\AA^{-1}$', fontsize = 14)
ax[3].set_ylabel('E, eV', fontsize = 14)
ax[3].set_xticks(np.arange(-2,2.2,1))
for label in ax[3].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)    
ax[3].set_yticks(np.arange(-2,2.1,1))
for label in ax[3].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
ax[3].set_ylim([-2,2.25])
ax[3].set_xlim([-2,2])
ax[3].set_aspect(0.5)

for a in ax[3:4]:
    a.tick_params(color='green', labelcolor='black')
    for spine in a.spines.values():
        spine.set_edgecolor('green')
        
for a in ax[2:3]:
    a.tick_params(color='purple', labelcolor='black')
    for spine in a.spines.values():
        spine.set_edgecolor('purple')
                
#ax[2].set_title('$k_{x}$')
#ax[3].set_title('$k_{y}$')
fig.tight_layout()
plt.show()

#%%
### User Inputs

#%%
%matplotlib inline

# Plot Momentum Maps at specified Energies
tMaps, tint  = [0, 1.5, 1.6, 1.7], 2

### Plot
numPlots = len(tMaps)

fig, ax = plt.subplots(1,numPlots, sharey=True)
ax = ax.flatten()
plt.gcf().set_dpi(300)

#fig.set_size_inches(12, 6, forward=False)

for i in np.arange(numPlots):
    tMap = tMaps[i]
    tMap = (np.abs(ax_E_offset - tMap)).argmin()
    #frame = np.transpose(I[:,:,tMap-int(tint/2):tMap+int(tint/2),4:6].sum(axis=(2,3)))
    frame = np.transpose(I[:,:,tMap-int(tint/2):tMap+int(tint/2),:].sum(axis=(2,3)))
    frame = frame/np.max(frame)
    im = ax[i].imshow(frame, origin='lower', cmap='terrain_r', clim=None, interpolation='none', extent=[ax_kx[0], ax_kx[-1], ax_ky[-1], ax_ky[0]]) #kx, ky, t
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

#%%
# Window and Symmetrize MM for FFT
%matplotlib inline

######### User Inputs #########
tMaps, tint_E  = [1.7], 0.2
window_k_width = .325
circle_mask = False

#LTL Bulk WSe2, XUV POL Integrated 
k_points_y = [114, 73, 34, 34, 74, 115]
k_points_x = [53, 30, 52, 100, 122, 98]

#Shuo ML WSe2
#k_points_y = [78, 100, 83, 39, 21, 41] 
#k_points_x = [24, 61, 95, 95, 61, 23]

#Shuo Dataset WSe2
#k_points_y = [58, 84, 84, 56, 28, 29] 
#k_points_x = [27, 43, 76, 92, 75, 42]

single_k = 0

##############################################
########## Perform the Operations ############
##############################################
tMap = tMaps[0]
tMap = (np.abs(ax_E_offset - tMap)).argmin()

dE = (ax_E_offset[1] - ax_E_offset[0])
dkx = np.abs((ax_kx[1] - ax_kx[0]))
dky = np.abs((ax_ky[1] - ax_ky[0]))

Eint = round(0.5*tint_E/dE)
tint = Eint
xint = round(xint_k/dkx)
yint = round(yint_k/dky) 

kspace_frame = np.abs(I[:,:,tMap-Eint:tMap+Eint,:].sum(axis=(2,3))) #Scan 160, XUV Pol, Bulk LTL
#kspace_frame = np.abs(I[:,:,tMap-Eint:tMap+Eint,2:4].sum(axis=(2,3))) #Scan ??, Delay, ML Shuo

#kspace_frame = kspace_frame - np.mean(kspace_frame)
kspace_frame = kspace_frame/np.max(kspace_frame)

window_new = np.zeros((kspace_frame.shape))
window_new_single = np.zeros((kspace_frame.shape))
window_circle_mask = np.zeros((kspace_frame.shape))

#2D Tukey Window
radius = round(window_k_width/dkx) #8#52 # 52 #pixels
window1d = np.abs(signal.windows.blackman(3*radius))
window2d = np.sqrt(np.outer(window1d,window1d))
window2d = window2d/np.max(window2d)

for k in range(0,6):
    x = k_points_x[k] 
    y = k_points_y[k] 
    
    #Circular Mask around each k Point
    row = x
    col = y
    #k_outer = np.abs((ax_kx - 1.75)).argmin()
    #k_inner = np.abs((ax_kx - 1.75)).argmin()
    radius = round(window_k_width/dkx) #8#52 # 52 #pixels
    rr, cc = disk((row, col), radius)
    window_new[rr, cc] = 1/np.max(kspace_frame[rr,cc])
   
    rad = radius
    window1d = np.abs(signal.windows.blackman(2*radius))
    window2d = np.sqrt(np.outer(window1d,window1d))
    window2d = window2d/np.max(kspace_frame[x-radius:x+radius,y-radius:y+radius])
    
    #window_new[x-radius:x+radius,y-radius:y+radius] = window2d
    
    if k == single_k:
        window_new_single[rr, cc] = 1/np.max(kspace_frame[rr,cc])
        single_kx = ax_kx[x]
        single_ky = ax_ky[y]
        single_rad = window_k_width
        #window_new_single = np.zeros((kspace_frame.shape))
       # window_new_single[x-radius:x+radius,y-radius:y+radius] = window2d
        
        rad = radius
        window1d = np.abs(signal.windows.tukey(2*radius))
        window2d = np.sqrt(np.outer(window1d,window1d))
        window2d = window2d/np.max(kspace_frame[x-radius:x+radius,y-radius:y+radius])
        
        roi_cut = kspace_frame[x-1:x+1,:].sum(axis=0)
        win_cut = window_new_single[x-1:x+1,:].sum(axis=0)
        window_2d_full = np.zeros((kspace_frame.shape))
        window_2d_full[x-radius:x+radius,y-radius:y+radius] = window2d
        
frame_sym = np.zeros(kspace_frame.shape)
frame_sym[:,:] = kspace_frame[:,:]  + (kspace_frame[:,::-1])    
frame_sym =  frame_sym[:,:]/2

if circle_mask is True:
    row = int(len(ax_kx)/2)
    col = int(len(ax_kx)/2)
    #k_inner = np.abs((ax_kx - 0.75)).argmin()
    #k_outer = np.abs((ax_kx - 1.7)).argmin()
    k_inner = 0.98
    k_outer = 1.38
    radius = round(window_k_width/dkx) #8#52 # 52 #pixels
    rr, cc = disk((row, col), round(k_outer/dkx))
    window_circle_mask[rr,cc] = 1
    rr, cc = disk((row, col), round(k_inner/dkx))
    window_circle_mask[rr,cc] = 0
    
    windowed_frame_symm = frame_sym*window_circle_mask
    windowed_frame_symm_single = frame_sym*window_circle_mask*window_new_single

    windowed_frame_nonsymm = kspace_frame*window_circle_mask
    windowed_frame_nonsymm_single = kspace_frame*window_circle_mask*window_new_single
    
elif circle_mask is False:
    windowed_frame_symm = frame_sym*window_new
    windowed_frame_symm_single = frame_sym*window_new_single
    
    windowed_frame_nonsymm = kspace_frame*window_new*window_2d_full
    windowed_frame_nonsymm_single = kspace_frame*window_new_single*window_2d_full


#%%
### Quality Control ###

window_test = np.zeros((kspace_frame.shape))
win = np.zeros((kspace_frame.shape))
window_k_width_test = 0.25
x = k_points_x[0] 
y = k_points_y[0] 
row = x
col = y
radius = round(window_k_width_test/dkx) #8#52 # 52 #pixels
rr, cc = disk((row, col), radius)
win[rr, cc] = 1/np.max(kspace_frame[rr,cc])

window_k_width_test = 0.25
radius = round(window_k_width_test/dkx) #8#52 # 52 #pixels

window1d = np.abs(signal.windows.tukey(2*radius))
window2d = np.sqrt(np.outer(window1d,window1d))
window2d = window2d/np.max(kspace_frame[x-radius:x+radius,y-radius:y+radius])

roi_cut = kspace_frame[x-1:x+1,:].sum(axis=0)
win_cut = window_new_single[x-1:x+1,:].sum(axis=0)
window_2d_full = np.zeros((kspace_frame.shape))
window_2d_full[x-radius:x+radius,y-radius:y+radius] = window2d
        
window_test[x-radius:x+radius, y-radius:y+radius] = window2d
window_test = win*window_2d_full #win

roi_cut = (kspace_frame*window_test)[x-1:x+1,:].sum(axis=0)
roi_cut = roi_cut - np.mean(roi_cut[y-30:y-10])
roi_cut = roi_cut/np.max(roi_cut)

win_cut = window_test[x-1:x+1,:].sum(axis=0)
windowed_full = (kspace_frame*window_test)*window_2d_full #windowed_frame_nonsymm_single #*window_test
windowed_cut = roi_cut*win_cut

fft_windowed_cut = np.abs(np.fft.fftshift(np.fft.fft(np.abs(windowed_cut)/np.max(windowed_cut))))
fft_roi_cut = np.abs(np.fft.fftshift(np.fft.fft(np.abs(roi_cut)/np.max(roi_cut))))
fft_win_cut = np.abs(np.fft.fftshift(np.fft.fft(np.abs(win_cut)/np.max(win_cut))))

plt.subplot(2, 2, 1)
plt.imshow(kspace_frame, cmap = cmap_LTL)

plt.subplot(2, 2, 2)
plt.imshow(windowed_full, cmap = cmap_LTL)
plt.axhline(x)

plt.subplot(2, 2, 3)
plt.plot(roi_cut/np.max(roi_cut), 'k')
plt.plot(windowed_cut/np.max(windowed_cut), 'r--')
plt.plot(win_cut/np.max(win_cut),'r')
plt.xlim([y-20,y+20])

plt.subplot(2, 2, 4)
plt.plot(fft_roi_cut/np.max(fft_roi_cut), 'k')
plt.plot(fft_windowed_cut/np.max(fft_windowed_cut), 'r--')
plt.plot(fft_win_cut/np.max(fft_win_cut),'r')
plt.tight_layout()

### ^^^ Quality Control ^^^ ###
#%%

#####                                              #####
##### Plot FFT of MMs to obtain real space wavefxn #####
#####                                              #####

### Take Momentum Map of Interest from above...

#momentum_frame = windowed_frame_symm
#momentum_frame_single = windowed_frame_symm_single

momentum_frame = windowed_frame_nonsymm
momentum_frame_single = windowed_frame_nonsymm_single

#momentum_frame = window_new_single
#momentum_frame_single = window_new_single

### Define real-space axis
k_step = np.abs((ax_kx[1] - ax_kx[0]))
k_length = len(ax_kx)

k_step_y = np.abs((ax_ky[1] - ax_ky[0]))
k_length_y = len(ax_ky)

zplength = 512 #5*k_length+1
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
momentum_frame_ = np.sqrt(momentum_frame_)
fft_frame = np.fft.ifft2(momentum_frame, [zplength, zplength])
fft_frame = np.fft.fftshift(fft_frame, axes = (0,1))
fft_frame = np.abs(fft_frame)
fft_frame = np.square(fft_frame)

momentum_frame_single = np.abs(momentum_frame_single)/np.max(momentum_frame_single)
momentum_frame_single = np.sqrt(momentum_frame_single)
fft_frame_s = np.fft.fft2(momentum_frame_single, [zplength, zplength])
fft_frame_s = np.fft.fftshift(fft_frame_s, axes = (0,1))
fft_frame_rsq = (fft_frame_s) 
fft_frame_s = np.abs(fft_frame_s)
fft_frame_s = np.square(fft_frame_s)

### Take x and y cuts and extract bohr radius
x_cut = fft_frame_s[:,int(zplength/2)-1]
y_cut = fft_frame_s[int(zplength/2)-1,:]
x_cut = x_cut/np.max(x_cut)
y_cut = y_cut/np.max(y_cut)

r2_cut = fft_frame_rsq[:,int(zplength/2)-1]
r2_cut = np.abs(r2_cut*r_axis)**2
r2_cut = r2_cut/np.max(r2_cut)

rdist_brad = np.argmax(r2_cut)
rdist_brad = r_axis[rdist_brad]

x_brad = (np.abs(x_cut[int(zplength/2)-10:] - 0.5)).argmin()
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
im = ax[2].imshow(fft_frame, clim = None, origin='lower', cmap=cmap_LTL, interpolation='none', extent = [r_axis[0], r_axis[-1], r_axis[0], r_axis[-1]]) #kx, ky, t
single_k_circle = plt.Circle((single_ky, single_kx), single_rad, color='red', linestyle = 'dashed', linewidth = 1.5, clip_on=False, fill=False)
ax[1].add_patch(single_k_circle)

#im = ax[i].imshow(np.transpose(I[:,:,tMap-int(tint/2):tMap+int(tint/2), adc-1:adc+1].sum(axis=(2,3))), origin='lower', cmap='terrain_r', clim=None, interpolation='none', extent=[-2,2,-2,2]) #kx, ky, t
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
ax[0].set_title('$E$ = ' + str((tMaps[0])) + ' eV, ' + '$\Delta$E = ' + str(tint_E) + ' eV', fontsize = 14)
#fig.suptitle('E = ' + str(tMaps[0]) + ' eV, $\Delta$E = ' + str(tint_E) + ' meV,  $\Delta$$k_{rad}$ = ' + str(window_k_width) + ' $\AA^{-1}$', fontsize = 18)

ax[1].set_xlim(-2,2)
ax[1].set_ylim(-2,2)
#ax[0].set_box_aspect(1)
ax[1].set_xlabel('$k_x$, $\AA^{-1}$', fontsize = 16)
ax[1].set_ylabel('$k_y$,  $\AA^{-1}$', fontsize = 16)
ax[1].tick_params(axis='both', labelsize=10)
#ax[1].set_title('$E$ = ' + str((tMaps[i])) + ' eV', fontsize = 16)
ax[1].set_title('Window: $\Delta$$k_{rad}$ = ' + str(window_k_width) + ' $\AA^{-1}$', fontsize = 15)
 
ax[2].set_xlim(-4,4)
ax[2].set_ylim(-4,4)
#ax[0].set_box_aspect(1)
ax[2].set_xlabel('$r_x$, nm', fontsize = 16)
ax[2].set_ylabel('$r_y$, nm', fontsize = 16)
ax[2].tick_params(axis='both', labelsize=10)
ax[2].set_title('2D FFT', fontsize = 15)
#ax[2].set_title('$E$ = ' + str((tMaps[i])) + ' eV', fontsize = 16)
#ax[i].annotate(('E = '+ str(round(tMaps[i],2)) + ' eV'), xy = (-1.85, 1.6), fontsize = 14, weight = 'bold')

#ax[2].plot(r_axis, x_cut/np.max(1), color = 'black', label = '$r_b$')
ax[3].plot(r_axis, x_cut/np.max(1), color = 'red', label = '$r_a$')
ax[3].plot(r_axis, r2_cut, color = 'blue')
#ax[1].axhline(0.5, linestyle = 'dashed', color = 'blue')
#ax[1].axvline(0.0, linestyle = 'dashed', color = 'blue')
#ax[2].axvline(x_brad, linestyle = 'dashed', color = 'black', linewidth = 2)
ax[3].axvline(rdist_brad, linestyle = 'dashed', color = 'blue', linewidth = 2)
ax[3].annotate('$r^*_x$', xy = (rdist_brad+.15, 0.5), fontsize = 14, color = 'blue', weight = 'bold')
ax[3].set_xlim([0, 3])
ax[3].set_ylim([-0.025, 1.025])
ax[3].set_xlabel('$r$, nm', fontsize = 16)
ax[3].set_ylabel('Norm. Int.', fontsize = 16)
ax[3].set_title('$r^*_x$ = ' + str(round(rdist_brad,2)), fontsize = 16)
ax[3].tick_params(axis='both', labelsize=10)
ax[3].set_yticks(np.arange(-0,1.5,0.5))
ax[3].set_aspect(3)
ax[3].set_xlabel('$r$, nm')
fig.subplots_adjust(right=0.58, top = 1.1)
fig.tight_layout()
plt.show()

#ax[2].legend(frameon = False)
#ax[3].annotate('$r^*_x$', xy = (x_brad+.2, 0.8), fontsize = 14, color = 'red', weight = 'bold')
#plt.text(0.1, 0.5, '$r^*_b$ = ' + str(round(x_brad,2)) + ' nm', fontsize = 10, color = 'black', fontweight = 4)
#plt.text(0.1, 0.4, '$r^*_a$ = ' + str(round(y_brad,2)) + ' nm', fontsize = 10, color = 'red', fontweight = 4)

#for label in ax[1].yaxis.get_ticklabels()[1::2]:
 #   label.set_visible(False)
#ax[1].axvline(1.75, color='black')

#fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([1, 0.2, 0.025, 0.6])
#fig.colorbar(im, cax=cbar_ax, ticks = [10,100])
#fig.colorbar(im, fraction=0.046, pad=0.04)

#%%


############
### Plot ###
############

# =============================================================================
# panel_titles = ['Data', 'Symmetrized', 'Windowed']
# numPlots = len(tMaps)
# fig, ax = plt.subplots(1, 3, sharey=False)
# plt.gcf().set_dpi(300)
# ax = ax.flatten()
# 
# im = ax[0].imshow(kspace_frame, origin='lower', cmap=cmap_LTL, interpolation='none', extent = [ax_kx[0],ax_kx[-1],ax_ky[0],ax_ky[-1]]) #kx, ky, t
# im = ax[1].imshow(frame_sym, origin='lower', cmap=cmap_LTL, interpolation='none', extent = [ax_kx[0],ax_kx[-1],ax_ky[0],ax_ky[-1]]) #kx, ky, t
# im = ax[2].imshow(windowed_frame_symm, origin='lower', cmap=cmap_LTL, interpolation='none', extent = [ax_kx[0],ax_kx[-1],ax_ky[0],ax_ky[-1]]) #kx, ky, t
# single_k_circle = plt.Circle((single_ky, single_kx), single_rad, color='red', linestyle = 'dashed', linewidth = 1.5, clip_on=False, fill=False)
# ax[2].add_patch(single_k_circle)
# 
# #im = ax[i].imshow(np.transpose(I[:,:,tMap-int(tint/2):tMap+int(tint/2), adc-1:adc+1].sum(axis=(2,3))), origin='lower', cmap='terrain_r', clim=None, interpolation='none', extent=[-2,2,-2,2]) #kx, ky, t
# for i in np.arange(3):
#     #ax[i].axhline(0, color='black', linewidth = 1, linestyle = 'dashed')
#     #ax[i].axvline(0, color='black', linewidth = 1, linestyle = 'dashed')
#     #ax[i].axvline(-1.1, color='blue', linewidth = 1, linestyle = 'dashed')
#     #ax[i].axvline(1.1, color='blue', linewidth = 1, linestyle = 'dashed')
#     ax[i].set_aspect(1)
#     
#     ax[i].set_xticks(np.arange(-2,2.2,1))
#     for label in ax[i].xaxis.get_ticklabels()[1::2]:
#         label.set_visible(False)
#         
#     ax[i].set_yticks(np.arange(-2,2.1,1))
#     for label in ax[i].yaxis.get_ticklabels()[1::2]:
#         label.set_visible(False)
#         
#     ax[i].set_xlim(-2,2)
#     ax[i].set_ylim(-2,2)
#     #ax[0].set_box_aspect(1)
#     ax[i].set_xlabel('$k_x$', fontsize = 14)
#     ax[i].set_ylabel('$k_y$', fontsize = 14)
#     ax[i].tick_params(axis='both', labelsize=12)
#     ax[i].set_title(panel_titles[i], fontsize = 14)
#     #ax[i].annotate(('E = '+ str(round(tMaps[i],2)) + ' eV'), xy = (-1.85, 1.6), fontsize = 14, weight = 'bold')
# #cbar_ax = fig.add_axes([1, 0.2, 0.025, 0.6])
# #fig.colorbar(im, cax=cbar_ax, ticks = [10,100])
# #fig.subplots_adjust(right=0.8, top = 0.3)
# fig.tight_layout()
# fig.suptitle('E = ' + str(tMaps[0]) + ' eV, $\Delta$E = ' + str(tint_E) + ' meV,  $\Delta$$k_{rad}$ = ' + str(window_k_width) + ' $\AA^{-1}$', fontsize = 18)
# fig.subplots_adjust(top = 1.2)
# plt.show()
# =============================================================================

#%%
p#%#%%
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
for i in np.arange(numPlots):
    
    frame_diff = momentum_frame
    frame_diff = np.abs(frame_diff)
    frame_diff = np.sqrt(frame_diff)
    fft_frame = np.fft.fft2(frame_diff, [zplength,zplength])
    fft_frame = np.fft.fftshift(fft_frame, axes = (0,1))
    fft_frame = np.abs(fft_frame)
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
