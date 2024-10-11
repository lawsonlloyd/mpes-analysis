# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 10:44:11 2023

@author: lloyd
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
from skimage.draw import disk
from scipy.optimize import curve_fit

#%%
### Transform Data if needed....
#I = np.transpose(I, (0,2,1,3))

if I.ndim > 3:
    t0 = data_handler.get_t0()
    
    I_neg = I[:,:,:,5:t0-7] #Sum over delay/polarization/theta...
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

dkx = (ax_kx[1] - ax_kx[0])

ax_E_offset = data_handler.ax_E
ax_delay_offset = data_handler.ax_delay

#%% # Make a plot based of the GUI slices

%matplotlib inline

pick_k_slice = 0

ax_E = data_handler.ax_E
ax_ky = data_handler.ax_ky

kint, kx, ky, E, delay = value_manager.get_values()
kx_i, ky_i, E_i, delay_i = data_handler.get_closest_indices(kx, pick_k_slice, E, delay)

cut = data_handler.get_ky_map()

cut_edc = cut[:, ky_i-2:ky_i+2].sum(axis=1)
cut_edc = cut_edc/np.max(cut_edc)

fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1], 'height_ratios':[1]})
fig.set_size_inches(8 , 4, forward=False)
ax = ax.flatten()

cmap ='gray'
ax[0].imshow(cut, origin='lower', extent =[ax_ky[0], ax_ky[-1], ax_E[0], ax_E[-1]], cmap=cmap)
ax[0].set_xlim([-1.6, 1.6])
ax[0].set_ylim([-6, 0.5])
ax[0].set_xlabel('ky')
ax[0].set_ylabel('E, eV')
ax[0].set_aspect(.75)
ax[0].set_title('Cut at kx = ' + str(round(kx,4)))
ax[0].axvline(pick_k_slice,color = 'red', linestyle = 'dashed')

ax[1].plot(ax_E, cut_edc)
ax[1].set_xlim([-6, 1])
ax[1].set_ylabel('Int')
ax[1].set_xlabel('E, eV')
ax[1].set_aspect(6)
ax[1].set_title('Cut at kx = ' + str(round(kx,3)))


#%% Plot the EDCs at specified kx, ky to determine VBM Zero Energy Reference

kx, ky, E = -1.75, 0.0, .25
kx_int, ky_int, E_int = .16 , .16, .2

mask_start = (np.abs(ax_E_offset - 0.75)).argmin()

O = .25 #Scan 163
O = -0.2 #Scan 162
#O = +0.072 #Scan 188
#O = 0.27 #Scan 062
#O = 7.28 #Scan 383

xi = (np.abs(ax_kx - (kx-kx_int))).argmin()
xf = (np.abs(ax_kx - (kx+kx_int))).argmin()
yi = (np.abs(ax_ky - (ky-ky_int))).argmin()
yf = (np.abs(ax_ky - (ky+ky_int))).argmin()
Ei = (np.abs(ax_E_offset - (E-E_int))).argmin()
Ef = (np.abs(ax_E_offset - (E+E_int))).argmin()

mask = np.ones((I.shape[2]))
mask[mask_start:] *=100
edc_neg = (I_neg[xi:xf,yi:yf,:].sum(axis=(0,1)))*mask
edc_pos = (I_pos[xi:xf,yi:yf,:].sum(axis=(0,1)))*mask
edc_neg = edc_neg/np.max(edc_neg)
edc_pos = edc_pos/np.max(edc_pos)

###
fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1], 'height_ratios':[1]})
fig.set_size_inches(12 , 4, forward=False)
ax = ax.flatten()

tMap = (np.abs(ax_E+O - 0)).argmin()
frame = (I_neg[:,:,Ei:Ef].sum(axis=(2)))
frame = frame/np.max(frame)
extent = [ax_kx[0],ax_kx[-1],ax_ky[0],ax_ky[-1]]
im0 = ax[0].imshow(np.transpose((frame)), origin='lower', cmap=cmap_LTL, vmax=1, clim=None, interpolation='none', extent=extent) #kx, ky, t
#im = ax[i].imshow(np.transpose(I[:,:,tMap-int(tint/2):tMap+int(tint/2), adc-1:adc+1].sum(axis=(2,3))), origin='lower', cmap='terrain_r', clim=None, interpolation='none', extent=[-2,2,-2,2]) #kx, ky, t
R = (Rectangle((kx-kx_int, ky-ky_int), 2*kx_int, 2*ky_int, linewidth=1, \
                   edgecolor='black', facecolor='None', linestyle = 'dashed'))
ax[0].axhline(ky,color='black', linestyle = 'dashed', linewidth = 1.5)
ax[0].axvline(kx,color='black', linestyle = 'dashed', linewidth = 1.5)
ax[0].add_patch(R)
ax[0].set_aspect(1)
ax[0].set_xlim(-2,2)
ax[0].set_ylim(-2,2)
ax[0].set_xticks(np.arange(-2,2.2,0.5))
for label in ax[0].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
ax[0].set_yticks(np.arange(-2,2.1,0.5))
for label in ax[0].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)

#ax[0].set_box_aspect(1)
ax[0].set_xlabel('$k_x$', fontsize = 14)
ax[0].set_ylabel('$k_y$', fontsize = 14)
ax[0].tick_params(axis='both', labelsize=12)
ax[0].set_title('$E$ = ' + str(0) + ' eV', fontsize = 16)

###
im = ax[1].plot(ax_E+O, edc_neg, color = 'grey', label = 't < 0 fs')
im = ax[1].plot(ax_E+O, 1*edc_pos, color = 'red', label = 't > 0 fs')
#im = ax[1].plot(ax_E_offset+O, 1*edc_diff, color = 'green', label = 'Difference', linestyle = 'dashed')

ax[1].axvline(-0, color = 'black', linestyle = 'dashed')\
#ax[0].axvline(1.55, color = 'grey', linestyle = 'dashed')
ax[1].axvline(1.35, color = 'black', linestyle = 'dashed')

ax[1].set_xlabel('Energy, eV', fontsize = 18)
ax[1].set_ylabel('Norm. Int.', fontsize = 18 )
ax[1].legend(frameon=False)
ax[1].set_xticks(np.arange(-1,3.5,0.5))
for label in ax[1].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
ax[1].set_ylim(0,1.1)
ax[1].set_xlim(-0.5,2.5)    
ax[1].set_title('$k_{x}$ = ' + str(kx) + ' $A^{-1}$' + ', $k_{y}$= ' + str(ky) + ' $A^{-1}$', fontsize = 16)

#%%

from lmfit import Parameters, minimize, report_fit

def gaussian(x, amp_1, mean_1, stddev_1, offset):
    
    g1 = amp_1 * np.exp(-((x - mean_1) / 4 / stddev_1)**2)+offset
    
    return g1

def two_gaussians(x, amp_1, amp_2, mean_1, mean_2, stddev_1, stddev_2, offset):
    
    g1 = amp_1 * np.exp(-((x - mean_1) / 4 / stddev_1)**2)
    g2 = amp_2 * np.exp(-((x - mean_2) / 4 / stddev_2)**2)
    
    return g1, g2, offset

def objective(params, x, data):
    
    g1, g2, offset = two_gaussians(x, **params)
    fit = g1+g2+offset
    resid = np.abs(data-fit)**2
    
    return resid

#%% # Plot dynamic EDCs

%matplotlib inline

k_int, kx, ky, E, delay = value_manager.get_values()
idx_kx, idx_ky, idx_E, d_i = data_handler.get_closest_indices(kx, ky, E, delay)
idx_k_int = round(0.5*k_int/data_handler.calculate_dk())

edcs = I[idx_kx-idx_k_int:idx_kx+idx_k_int, idx_ky-idx_k_int:idx_ky+idx_k_int, :, :].sum(axis=(0,1))
edcs = edcs/np.max(edcs[48:])
#edcs = edcs - np.mean(edcs[5:15])

plt.imshow((edcs), cmap = cmap_LTL, extent = [data_handler.ax_delay[0], data_handler.ax_delay[-1], data_handler.ax_E[0], data_handler.ax_E[-1]], aspect = 'auto', origin = 'lower', vmin = 0, vmax = 1)
plt.ylim([-0.5,1])
plt.xlim([-160,800])
#plt.axhline(0, linestyle = 'dashed', color = 'black')
#plt.axvline(0, linestyle = 'dashed', color = 'black')
plt.xlabel('Delay, fs')
plt.ylabel('Energy, eV')

pts = [-100, 0, 50, 100, 200, 500, 700]
colors = ['black', 'red', 'orange', 'purple', 'blue', 'green', 'grey']

fig = plt.figure()

for i in np.arange(len(pts)):
    d = pts[i]
    _, _, _, dd = data_handler.get_closest_indices(0, 0, 0, d)
    edc = I[idx_kx-idx_k_int:idx_kx+idx_k_int, idx_ky-idx_k_int:idx_ky+idx_k_int, :, dd-2:dd+2].sum(axis=(0,1,3))
    edc = edc/np.max(edc[48:]) + 0.05*i
    plt.plot(data_handler.ax_E, edc, color = colors[i], label = (str(round(data_handler.ax_delay[dd])) +' fs'))

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
trunc_e1 = -0.3
_, _, trunc1, _ = data_handler.get_closest_indices(0, 0, trunc_e, 0)
trunc_e2 = 0.9
_, _, trunc2, _ = data_handler.get_closest_indices(0, 0, trunc_e2, 0)

p0 = [1, .1, .2, 0] # Fitting params initial guess [amp, center, width, offset]
bnds = ((0.5, -0.5, 0.0, 0), (1.5, 0.5, .2, .2))

centers_VBM = np.zeros(len(data_handler.ax_delay))
p_fits_VBM = np.zeros((len(data_handler.ax_delay),4))

for t in np.arange(len(data_handler.ax_delay)):
    popt, _ = curve_fit(gaussian, data_handler.ax_E[trunc1:trunc2], edcs[trunc1:trunc2,t]/np.max(edcs[trunc1-10:,t]), p0, method=None, bounds = bnds)
    centers_VBM[t] = popt[1]
    p_fits_VBM[t,:] = popt 

# VBM FIT TESTS
t = 50
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
plt.xlim([-160, 800])
plt.xlabel('Delay, fs')
plt.ylabel('Energy Shift, meV')
#plt.axhline(0, linestyle = 'dashed', color = 'black')
#plt.axvline(0, linestyle = 'dashed', color = 'black')

# PLOT VBM PEAK WIDTH DYNAMICS
fig = plt.figure()
plt.plot(data_handler.ax_delay, p_fits_VBM[:,2], color = 'black', linestyle = 'solid')
#plt.ylim([-0.5,1])
plt.xlim([-160, 800])
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

#%%

cmap_LTL = plot_manager.custom_colormap(plt.cm.viridis, 0.2) #choose colormap based and percentage of total map for new white transition map

#%%
%matplotlib inline

### User Inputs for Plotting MM 

E, E_int  = [0.1, 1.1], 6

# Plot Momentum Maps at specified Energies

cmap = cmap_LTL 

frame_plot = I_sum

#### 
fig, ax = plt.subplots(1,2, sharey=True)
plt.gcf().set_dpi(300)
ax = ax.flatten()

sat = [1, 1]
for i in np.arange(len(E)):
    E_ = E[i]

    Ei = (np.abs(ax_E_offset - (E_-E_int))).argmin()
    Ef = (np.abs(ax_E_offset - (E_+E_int))).argmin()    
    
    frame = np.transpose(frame_plot[:,:,Ei:Ef].sum(axis=(2)))
    frame = frame/np.max(frame)
    extent = [ax_kx[0],ax_kx[-1],ax_ky[0],ax_ky[-1]]
    im = ax[i].imshow((frame), origin='lower', cmap=cmap, vmax=sat[i], clim=None, interpolation='none', extent = extent) #kx, ky, t

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
    ax[i].set_title('$E$ = ' + str((E[i])) + ' eV', fontsize = 16)
    #ax[i].annotate(('E = '+ str(round(tMaps[i],2)) + ' eV'), xy = (-1.85, 1.6), fontsize = 14, weight = 'bold')
    cbar_ax = fig.add_axes([1, 0.325, 0.025, 0.35])
    fig.colorbar(im, cax=cbar_ax, ticks = [-1,0,1])

fig.tight_layout()

#%%
# Plot Difference MMs of t < 0 and t > 0 fs

%matplotlib inline

tMaps, tint  = [1.35, 2], 6

cmapPLOTTING = cmap_LTL #'bone_r' # cmap_LTL

difference_FRAMES = np.zeros((numPlots,I_pos.shape[0],I_pos.shape[1]))

frame_neg = (I_neg[:,:,:])
frame_pos = (I_pos[:,:,:])

#frame_neg = frame_neg/(np.max(frame_neg))
#frame_pos = frame_pos/(np.max(frame_pos))
frame_neg = frame_neg/((neg_length))
frame_pos = frame_pos#/((pos_length))
frame_sum = frame_neg + frame_pos

#cts_pos = np.sum(frame_pos[:,:])
#cts_neg = np.sum(frame_neg[:,:])
#cts_total = np.sum(frame_sum)

frame_diff = frame_pos - frame_neg

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
    frame = frame - np.min(frame)
    frame = frame/np.max(frame)
    frame = abs(frame)
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

#%%
%matplotlib inline

# Plot Angle Integrated Dynamics

E_trace = [1.3, 2.2, 0.6] # Energies for Plotting
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
ang_int_neg = ang_int_neg/(t0-10-5)#np.max(ang_int_neg)

diff_ang = ang_int - ang_int_neg
diff_ang = diff_ang/np.max(diff_ang)

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

trace_1 = trace_1 - np.mean(trace_1[3:t0-5])
trace_2 = trace_2 - np.mean(trace_2[3:t0-5])
trace_3 = trace_3 - np.mean(trace_3[3:t0-5])

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
im = ax[0].plot(ax_E_offset, edc_pos, color = 'red', label = 't > 0 fs')
im = ax[0].plot(ax_E_offset, edc_diff, color = 'green', label = 'Difference', linestyle = 'dashed')

#ax[0].axvline(1.55, color = 'grey', linestyle = 'dashed')
#ax[0].axvline(1.15, color = 'black', linestyle = 'dashed')
#ax[0].axvline(2, color = 'black', linestyle = 'dashed')

ax[0].set_ylim(0,0.002)
ax[0].set_xlabel('Energy, eV', fontsize = 18)
ax[0].set_ylabel('Norm. Int.', fontsize = 18 )
ax[0].legend(frameon=False)
ax[0].set_xticks(np.arange(-1,3.5,0.5))
for label in ax[0].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
ax[0].set_xlim(-0.1,3)

color_max = 1
#waterfall = ax[1].imshow(ang_int, clim = [0, .02], origin = 'lower', cmap = cmap_plot, extent=[ax_delay_offset[0], ax_delay_offset[-1], ax_E_offset[0], ax_E_offset[-1]])
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
    
plt.rcParams.update(params)

fig.tight_layout()
#ax[1].set_tick_params(axis='both', labelsize=16)
#plt.gca().set_aspect(200)


#%%
%matplotlib inline
# Plot kx, ky cuts Pos/Neg Difference Plots

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

kx, ky = 0, 0.0
kx_int, ky_int = .25 , 0.5

mask_start = 0.9

############################################
kxi = (np.abs(ax_kx - (kx-kx_int))).argmin()
kxf = (np.abs(ax_kx - (kx+kx_int))).argmin()
kyi = (np.abs(ax_ky - (ky-ky_int))).argmin()
kyf = (np.abs(ax_ky - (ky+ky_int))).argmin()

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

slice_E_k_3 = I_sum_1[kxi:kxf,:,e:].sum(axis=(0))
norm_3 = np.max(slice_E_k_3)
slice_E_k_3 = slice_E_k_3/norm_3

slice_E_k_4 = I_sum_2[kxi:kxf,:,e:].sum(axis=(0))
slice_E_k_4 = slice_E_k_4/norm_3

I_1_N = slice_E_k_1/np.max(np.abs(slice_E_k_1))
I_2_N = slice_E_k_2/np.max(np.abs(slice_E_k_2))

I_3_N = slice_E_k_3/np.max(np.abs(slice_E_k_3))
I_4_N = slice_E_k_4/np.max(np.abs(slice_E_k_4))

I_dif = I_2_N - I_1_N
I_dif_ = I_4_N - I_3_N

I_dif = I_dif/np.max(I_dif_)
I_dif_ = I_dif_/np.max(I_dif)

### Panels
###################################33

ylim = [1,2.5]
vmin, vmax = 0, 1

###
fig, ax = plt.subplots(nrows = 2, ncols=3, gridspec_kw={'width_ratios': [1, 1, 1], 'height_ratios':[1, 1]})
fig.set_size_inches(15, 10, forward=False)
ax = ax.flatten()

map_1 = 'terrain_r'
map_2 = cmap_LTL
im_2 = ax[0].imshow(np.transpose(slice_E_k_1), origin='lower', cmap=map_1, clim=None, interpolation='none', extent=[ax_ky[0],ax_ky[-1],ax_E_offset[e], ax_E_offset[-1] ]) #kx, ky, t
im_3 = ax[1].imshow(np.transpose(slice_E_k_2), origin='lower', cmap=map_1, clim=None, interpolation='none', extent=[ax_kx[0],ax_kx[-1],ax_E_offset[e], ax_E_offset[-1] ]) #kx, ky, t
im_4 = ax[2].imshow(np.transpose(I_dif), origin='lower', cmap=cmap_LTL, clim=[-1,1], interpolation='none', extent=[ax_kx[0],ax_kx[-1],ax_E_offset[e], ax_E_offset[-1] ],vmin = vmin, vmax=vmax) #kx, ky, t
im_4 = ax[3].imshow(np.transpose(slice_E_k_3), origin='lower', cmap=map_1, vmax = None, clim=None, interpolation='none', extent=[ax_ky[0],ax_ky[-1],ax_E_offset[e], ax_E_offset[-1] ]) #kx, ky, t
im_5 = ax[4].imshow(np.transpose(slice_E_k_4), origin='lower', cmap=map_1, clim=None, interpolation='none', extent=[ax_ky[0],ax_ky[-1],ax_E_offset[e], ax_E_offset[-1] ]) #kx, ky, t
im_6 = ax[5].imshow(np.transpose(I_dif_), origin='lower', cmap=cmap_LTL, clim=None, interpolation='none', extent=[ax_ky[0],ax_ky[-1],ax_E_offset[e], ax_E_offset[-1] ],vmin = vmin, vmax=vmax) #kx, ky, t
ylim_E = -2.5

for i in range(0,6):
        
    ax[i].set_yticks(np.arange(-0.5,3.5,0.25))
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

#%%

# Plot Dynamics at Distinct Momenta and/or Energy Points

kx_traces, ky_traces = [0.2, 0.2, -1.1, -1.1], [.8] # kx, ky for plotting
E_traces = [1.35, 2.15, 1.35, 2.15] # Energies for Plotting
kx_int, ky_int, E_int  = .4, .4, 0.2 #Integration Ranges

trace_colors = ['black', 'red', 'grey', 'pink']

cmap_to_plot = cmap_LTL
#cmap_to_plot = 'magma_r'
clim = [-0.1, 1]
delay_lim = [-160, 820]
################################
# Operations to Extract Traces #
################################

## Extract Traces for At Different Energies with Background Subtraction
traces = np.zeros((4,I.shape[3]))
E_int, kx_int, ky_int = E_int/2, kx_int/2, ky_int/2

for t in range(4):
    kxi = (np.abs(ax_kx - (kx_traces[t]-kx_int))).argmin()
    kxf = (np.abs(ax_kx - (kx_traces[t]+kx_int))).argmin()
    kyi = (np.abs(ax_ky - (ky_traces[0]-ky_int))).argmin()
    kyf = (np.abs(ax_ky - (ky_traces[0]+ky_int))).argmin()
    Ei = np.abs(ax_E_offset - (E_traces[t]-E_int)).argmin()  
    Ef = np.abs(ax_E_offset - (E_traces[t]+E_int)).argmin()  
    trace = I[kxi:kxf, kyi:kyf, Ei:Ef,:].sum(axis=(0,1,2))
    trace = trace/np.max(trace)
    trace = trace - np.mean(trace[3:t0-5])
    traces[t,:] = trace/np.max(trace)

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

#######################
### Do the Plotting ###
#######################

fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 1], 'height_ratios':[1, 1]})
fig.set_size_inches(8, 6, forward=False)

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
ax[0].set_xlabel('$k_x$, $A^{-1}$', fontsize = 20)
ax[0].set_ylabel('$k_y$, $A^{-1}$', fontsize = 20)
ax[0].tick_params(axis='both', labelsize=18)
ax[0].set_title('$E$ = ' + str((E_traces[3])) + ' eV', fontsize = 20)
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
ax[2].set_ylabel('$E - E_{VBM}$, eV', fontsize = 20)
ax[2].set_xlabel('$k_x, A^{-1}$', fontsize = 20)
ax[2].set_title('', fontsize = 20)
ax[2].tick_params(axis='both', labelsize=18)
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
for t in range(4):
    rect = (Rectangle((kx_traces[t]-kx_int, E_traces[t]-E_int), 2*kx_int, 2*E_int, linewidth=1.5, \
                      edgecolor=trace_colors[t], facecolor='None'))
    ax[2].add_patch(rect)
    rect2 = (Rectangle((kx_traces[t]-kx_int, ky_traces[0]-ky_int), 2*kx_int, 2*ky_int, linewidth=1, \
                      edgecolor=trace_colors[t], facecolor='None', linestyle = 'dashed'))
    ax[0].add_patch(rect2)

for t in [0,1]:
    ax[1].plot(ax_delay_offset, traces[t,:], color = trace_colors[t], \
               label = str(E_traces[t]) + ' eV, ' + str(kx_traces[t]) + ' A^-1')

for t in [2,3]:
    ax[3].plot(ax_delay_offset, traces[t,:], color = trace_colors[t], \
               label = str(E_traces[t]) + ' eV, ' + str(kx_traces[t]) + ' A^-1')

for f in [1,3]:

    ax[f].set_ylabel('Norm. Int.', fontsize = 18)
    ax[f].set_xlabel('Delay, fs', fontsize = 18)
    ax[f].set_xticks(np.arange(-600,1200,100))
    for label in ax[f].xaxis.get_ticklabels()[1::2]:
        label.set_visible(False) 
    ax[f].set_xlim(delay_lim[0],delay_lim[1])
    ax[f].set_ylim(-0.2, 1.1)
#    ax[f].set_aspect(500)

    #ax[f].legend(frameon = False)

params = {'lines.linewidth' : 2.5, 'axes.linewidth' : 2, 'axes.labelsize' : 20, 
              'xtick.labelsize' : 16, 'ytick.labelsize' : 16, 'axes.titlesize' : 20, 'legend.fontsize' : 16}

plt.rcParams.update(params)
fig.tight_layout()


#%%
%matplotlib inline

###
# Window and Symmetrize MM for FFT
###

from scipy import signal
from scipy.fft import fft, fftshift

tMaps, tint  = [1.35], 6
k_i, k_f = -.4, .4 #ky
k_i_2, k_f_2 = -2, -0.8 #kx

#window_choice = 

### Plot
numPlots = len(tMaps)

fig, ax = plt.subplots(1, 3, sharey=True)
plt.gcf().set_dpi(300)
ax = ax.flatten()

sat = [1, 1, 1]
for i in np.arange(numPlots):
    tMap = tMaps[i]
    tMap = (np.abs(ax_E_offset - tMap)).argmin()
    
    frame_neg = np.transpose(I_neg[:,:,tMap-int(tint/2):tMap+int(tint/2)].sum(axis=(2)))
    frame_pos = np.transpose(I_pos[:,:,tMap-int(tint/2):tMap+int(tint/2)].sum(axis=(2)))
    
    frame_neg = frame_neg/((neg_length))
    frame_pos = frame_pos/((pos_length))
    frame_sum = frame_neg + frame_pos
    
    cts_pos = np.sum(frame_pos[:,:])
    cts_neg = np.sum(frame_neg[:,:])
    cts_total = np.sum(frame_sum)
    
    frame_diff = frame_pos - frame_neg
    frame_diff = (frame_diff)
    
    ###                   ###
    ### Do the Operations ###
    ###                   ###  
    
    kspace_frame = frame_diff
    
    mn = np.mean(kspace_frame[:,25:35])
    kspace_frame = kspace_frame - mn
    kspace_frame[kspace_frame<0] = 0
    
    #kspace_frame = frame_pos #All Pos delays
    
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
    win_1_box = np.zeros(kspace_frame.shape[0])
    win_2_box = np.zeros(kspace_frame.shape[1])

    k_i = (np.abs(ax_ky - k_i)).argmin()
    k_f = (np.abs(ax_ky - k_f)).argmin()
    k_i_2 = (np.abs(ax_kx - k_i_2)).argmin()
    k_f_2 = (np.abs(ax_kx - k_f_2)).argmin()
    
    tuk_1 = signal.windows.tukey(k_f-k_i)
    box_1 = signal.windows.boxcar(k_f-k_i)

    tuk_2 = signal.windows.tukey(k_f_2-k_i_2)
    box_2 = signal.windows.boxcar(k_f_2-k_i_2)

    win_1[k_i:k_f] = tuk_1
    win_1_box[k_i:k_f] = box_1

    win_2[k_i_2:k_f_2] = tuk_2
    win_2_box[k_i_2:k_f_2] = box_2

    window_4 = np.outer(win_1, win_2)
    window_5 = np.outer(win_1_box, win_2_box)
    window_6 = np.outer(win_1_box, win_2)
    
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
    
    windowed_frame_symm = frame_sym*window_5
    windowed_frame_nonsymm = kspace_frame*window_5
    #windowed_frame_symm = frame_sym
    
    
im = ax[0].imshow(kspace_frame, origin='lower', cmap=cmap_LTL, interpolation='none', extent = [ax_kx[0],ax_kx[-1],ax_ky[0],ax_ky[-1]]) #kx, ky, t
im = ax[1].imshow(frame_sym, origin='lower', cmap=cmap_LTL, interpolation='none', extent = [ax_kx[0],ax_kx[-1],ax_ky[0],ax_ky[-1]]) #kx, ky, t
im = ax[2].imshow(windowed_frame_symm, origin='lower', cmap=cmap_LTL, interpolation='none', extent = [ax_kx[0],ax_kx[-1],ax_ky[0],ax_ky[-1]]) #kx, ky, t

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

%matplotlib inline

#####                                              #####
##### Plot FFT of MMs to obtain real space wavefxn #####
#####                                              #####

momentum_frame = windowed_frame_symm
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
fft_frame_s = np.square(np.abs(fft_frame))

### Take x and y cuts and extract bohr radius
kx_cut = momentum_frame_[:,int(len(ax_kx)/2)-1-4:int(len(ax_kx)/2)-1+4].sum(axis=1)
kx_cut = kx_cut/np.max(kx_cut)
ky_cut = momentum_frame_[int(len(ax_ky)/2)-1-4:int(len(ax_ky)/2)-1+4,:].sum(axis=0)
ky_cut = ky_cut/np.max(ky_cut)

x_cut = fft_frame_s[:,int(zplength/2)-1]
y_cut = fft_frame_s[int(zplength/2)-1,:]
x_cut = x_cut/np.max(x_cut)
y_cut = y_cut/np.max(y_cut)

r2_cut_x = fft_frame_rsq[:,int(zplength/2)-1]
r2_cut_x = np.square(np.abs(r2_cut_x*r_axis))
r2_cut_x = r2_cut_x/np.max(r2_cut_x)

r2_cut_y = fft_frame_rsq[int(zplength/2)-1,:]
r2_cut_y = np.square(np.abs(r2_cut_y*r_axis))
r2_cut_y = r2_cut_y/np.max(r2_cut_y)

rdist_brad_x = np.argmax(r2_cut_x)
rdist_brad_y = np.argmax(r2_cut_y)

rdist_brad_x = r_axis[rdist_brad_x]
rdist_brad_y = r_axis[rdist_brad_y]

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
im = ax[2].imshow(fft_frame_s, clim = None, origin='lower', cmap=cmap_LTL, interpolation='none', extent = [r_axis[0], r_axis[-1], r_axis[0], r_axis[-1]]) #kx, ky, t
#single_k_circle = plt.Circle((single_ky, single_kx), single_rad, color='red', linestyle = 'dashed', linewidth = 1.5, clip_on=False, fill=False)
#ax[1].add_patch(single_k_circle)

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
ax[0].set_title('$E$ = ' + str((tMaps[0])) + ' eV ', fontsize = 14)

#fig.suptitle('E = ' + str(tMaps[0]) + ' eV, $\Delta$E = ' + str(tint_E) + ' meV,  $\Delta$$k_{rad}$ = ' + str(window_k_width) + ' $\AA^{-1}$', fontsize = 18)

ax[1].set_xlim(-2,2)
ax[1].set_ylim(-2,2)
#ax[0].set_box_aspect(1)
ax[1].set_xlabel('$k_x$, $\AA^{-1}$', fontsize = 16)
ax[1].set_ylabel('$k_y$,  $\AA^{-1}$', fontsize = 16)
ax[1].tick_params(axis='both', labelsize=10)
#ax[1].set_title('$E$ = ' + str((tMaps[i])) + ' eV', fontsize = 16)
ax[1].set_title('Windoed', fontsize = 15)
 
ax[2].set_xlim(-2,2)
ax[2].set_ylim(-2,2)
#ax[0].set_box_aspect(1)
ax[2].set_xlabel('$r_x$, nm', fontsize = 16)
ax[2].set_ylabel('$r_y$, nm', fontsize = 16)
ax[2].tick_params(axis='both', labelsize=10)
ax[2].set_title('2D FFT', fontsize = 15)
#ax[2].set_title('$E$ = ' + str((tMaps[i])) + ' eV', fontsize = 16)
#ax[i].annotate(('E = '+ str(round(tMaps[i],2)) + ' eV'), xy = (-1.85, 1.6), fontsize = 14, weight = 'bold')

#ax[2].plot(r_axis, x_cut/np.max(1), color = 'black', label = '$r_b$')
ax[3].plot(r_axis, x_cut/np.max(x_cut), color = 'black', label = '$r_x$')
#ax[3].plot(r_axis, r2_cut_x, color = 'black', linestyle = 'dashed')

ax[3].plot(r_axis, y_cut/np.max(y_cut), color = 'red', label = '$r_y$')
#ax[3].plot(r_axis, r2_cut_y, color = 'red', linestyle = 'dashed')

ax[3].axvline(x_brad, linestyle = 'dashed', color = 'black', linewidth = 1.5)
ax[3].axvline(y_brad, linestyle = 'dashed', color = 'red', linewidth = 1.5)

#ax[3].axvline(rdist_brad_x, linestyle = 'dashed', color = 'black', linewidth = 1.5)
#ax[3].axvline(rdist_brad_y, linestyle = 'dashed', color = 'red', linewidth = 1.5)

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
#momentum_frame = windowed_frame_nonsymm

#momentum_frame = window_4
#momentum_frame = window_5

#######################################
k_step = np.abs((ax_kx[1] - ax_kx[0]))
k_length = len(ax_kx)

k_step_y = np.abs((ax_ky[1] - ax_ky[0]))
k_length_y = len(ax_ky)

zplength = 5*k_length+1
max_r = (1/2)*1/(k_step)
r_axis = np.linspace(-max_r, max_r, num = k_length)
r_axis = np.linspace(-max_r, max_r, num = zplength)


####################################################
### Plot
fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 1], 'height_ratios':[1]})
fig.set_size_inches(10, 6, forward=False)
plt.gcf().set_dpi(300)
ax = ax.flatten()

sat = [1, 1, 1]

i = 0
im = ax[i].imshow(momentum_frame, clim = None, origin='lower', cmap=cmap_LTL, interpolation='none', extent = [ax_kx[0], ax_kx[-1], ax_kx[0], ax_kx[-1]]) #kx, ky, t

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
ax[i].set_title('$E$ = ' + str((tMaps[i-1])) + ' eV', fontsize = 16)

for i in np.arange(1):
    i = i + 1
    
    ### Do the FFT operations to get --> |Psi(x,y)|^2
    #momentum_frame = momentum_frame - np.mean(momentum_frame[30:45,30:45])
    momentum_frame = np.abs(momentum_frame)/np.max(momentum_frame)
    momentum_frame = np.sqrt(momentum_frame)
    fft_frame = np.fft.fft2(momentum_frame, [zplength, zplength])
    fft_frame = np.fft.fftshift(fft_frame, axes = (0,1))
    fft_frame = np.square(fft_frame)
    fft_frame = np.abs(fft_frame)
    fft_frame = fft_frame/np.max(fft_frame)
    
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
    ax[i].set_title('$E$ = ' + str((tMaps[i-1])) + ' eV', fontsize = 16)
    #ax[i].annotate(('E = '+ str(round(tMaps[i],2)) + ' eV'), xy = (-1.85, 1.6), fontsize = 14, weight = 'bold')

ax[2].plot(r_axis, x_cut/np.max(1), color = 'black', label = '$r_b$')
ax[2].plot(r_axis, y_cut/np.max(1), color = 'red', label = '$r_a$')
#ax[1].axhline(0.5, linestyle = 'dashed', color = 'blue')
#ax[1].axvline(0.0, linestyle = 'dashed', color = 'blue')
ax[2].axvline(x_brad, linestyle = 'dashed', color = 'black', linewidth = 2)
ax[2].axvline(y_brad, linestyle = 'dashed', color = 'red', linewidth = 2)
ax[2].set_xlim([0, 2])
ax[2].set_ylim([-0.025, 1.025])
ax[2].set_xlabel('$r$, nm', fontsize = 16)
ax[2].set_ylabel('Norm. Int.', fontsize = 16)
ax[2].set_title('$r^*_a/r^*_b$ = ' + str(round(x_brad/y_brad,2)), fontsize = 16)
ax[2].tick_params(axis='both', labelsize=14)
ax[2].legend(frameon = False)
plt.text(1, 0.58, '$r^*_b$ = ' + str(round(x_brad,2)) + ' nm', fontsize = 11, color = 'black', fontweight = 4)
plt.text(1, 0.47, '$r^*_a$ = ' + str(round(y_brad,2)) + ' nm', fontsize = 11, color = 'red', fontweight = 4)
ax[2].set_yticks(np.arange(-0,1.5,0.5))
ax[2].set_xlabel('r, nm')
ax[2].set_aspect(2)
fig.tight_layout()
plt.show()

#for label in ax[1].yaxis.get_ticklabels()[1::2]:
 #   label.set_visible(False)
#ax[1].axvline(1.75, color='black')
#fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([1, 0.2, 0.025, 0.6])
#fig.colorbar(im, cax=cbar_ax, ticks = [10,100])
#fig.colorbar(im, fraction=0.046, pad=0.04)


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
    
#%% ## Make a video/gif of the Dynamics!

#y = [y]
import imageio
from matplotlib.animation import FuncAnimation
from matplotlib import animation

I_neg = I[:,:,:,6:t0-15]
neg_len = I_neg.shape[3]
I_neg = I_neg.sum(axis=3)/neg_len

I_neg_cb = I_neg[:,y[0]:y[0]+yint,100:177].sum(axis=(1)) #Individual Delays
I_neg = I_neg[:,y[0]:y[0]+yint,0:177].sum(axis=(1)) #Individual Delays

I_Neg = I[:,:,:,6:t0-15]
neg_len = I_Neg.shape[3]
I_Neg = I_Neg.sum(axis=3)/neg_len

video_array = I[:,y[0]:y[0]+yint,0:177,20:].sum(axis=(1)) #Individual Delays
#video_array = I_Enhanced_Full[x[0]:x[0]+yint,:,0:177,10:].sum(axis=(1)) #Individual Delays
#I_neg = I_neg[:,y[0]:y[0]+yint,100:177].sum(axis=(1)) #Individual Delays

video_axis = ax_delay_offset[20:]
fig = plt.figure()

def update_img(n):

    vid = I_Difference_Full[:,y[0]:y[0]+yint,0:177,20:].sum(axis=(1))
   # vid = vid/np.max(abs(vid))
    
    fr_cb = I[:,y[0]:y[0]+yint,100:177,20:].sum(axis=(1))
    fr_cb = (fr_cb[:,:,n:n+5])
    fr_len = fr_cb.shape[2]
    fr_cb = fr_cb.sum(axis=2)/fr_len
    fr_difference_cb = fr_cb #- I_neg_cb   
    CB_Norm = np.max(abs(fr_difference_cb))
    #fr_difference = fr_difference/CB_Norm

    fr = I[:,y[0]:y[0]+yint,0:177,20:].sum(axis=(1))
    fr = (fr[:,:,n:n+5]).sum(axis=2)/fr_len
    fr_difference = fr #- I_neg   
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

I_neg = I[:,:,:,6:t0-15]
neg_len = I_neg.shape[3]
I_neg = I_neg.sum(axis=3)/neg_len

I_neg_cb = I_neg[:,y[0]:y[0]+yint,100:177].sum(axis=(1)) #Individual Delays
I_neg = I_neg[:,y[0]:y[0]+yint,0:177].sum(axis=(1)) #Individual Delays

I_Neg = I[:,:,:,6:t0-15]
neg_len = I_Neg.shape[3]
I_Neg = I_Neg.sum(axis=3)/neg_len

video_array = I[:,y[0]:y[0]+yint,0:177,20:].sum(axis=(1)) #Individual Delays
#video_array = I_Enhanced_Full[x[0]:x[0]+yint,:,0:177,10:].sum(axis=(1)) #Individual Delays
#I_neg = I_neg[:,y[0]:y[0]+yint,100:177].sum(axis=(1)) #Individual Delays

video_axis = ax_delay_offset[20:]
fig = plt.figure()

def update_img(n):

    vid = I_Difference_Full[:,y[0]:y[0]+yint,0:177,20:].sum(axis=(1))
   # vid = vid/np.max(abs(vid))
    
    fr_cb = I[:,y[0]:y[0]+yint,100:177,20:].sum(axis=(1))
    fr_cb = (fr_cb[:,:,n:n+5])
    fr_len = fr_cb.shape[2]
    fr_cb = fr_cb.sum(axis=2)/fr_len
    fr_difference_cb = fr_cb - I_neg_cb   
    CB_Norm = np.max(abs(fr_difference_cb))
    #fr_difference = fr_difference/CB_Norm

    fr = I[:,y[0]:y[0]+yint,0:177,20:].sum(axis=(1))
    fr = (fr[:,:,n:n+5]).sum(axis=2)/fr_len
    fr_difference = fr - I_neg   
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


#%% Make a video/gif of the Dynamics -- with Pos-Neg Difference!

import imageio
from matplotlib.animation import FuncAnimation
from matplotlib import animation

tstart = -200
tstart = np.abs(ax_delay_offset-tstart).argmin()
video_axis = ax_delay_offset[tstart:]

fig = plt.figure()

I_neg = I[:,:,:,5:t0-10]
neg_len = I_neg.shape[3]
I_neg = I_neg.sum(axis=3)/neg_len

I_truncated = (I[:,:,:,tstart:]) #Sum over delay/theta/ADC for Plotting...

def update_img(n):
    print(n+5)
    #tstart = -200
    #tstart_neg = np.abs(ax_delay_offset-).argmin()
    
    #n delay average
    #start2 = n
    #stop2 = n+5
    I_sum_2 = I_truncated[:,:,:,n+n+5]/(5)
    #frame_len = I_sum_2.shape[3]
    #I_sum_2 = I_sum_2.sum(axis=3)/frame_len
    
    I_Enhanced_neg = 0.005*logicMask * I_neg
    I_Enhanced_2 = 0.005*logicMask * I_sum_2
     
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