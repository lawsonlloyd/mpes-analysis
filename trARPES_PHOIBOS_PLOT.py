# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 14:39:37 2025

@author: lloyd
"""

#%%

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
from skimage.draw import disk
from scipy.optimize import curve_fit
from scipy.special import erf
import csv
from Loader import DataLoader
import xarray as xr

import phoibos

#%% Load Data

filename = '2024 Bulk CrSBr Phoibos.csv'

scan_info = {}
data_path_info = 'R:\Lawson\mpes-analysis'
data_path = 'R:\Lawson\Data\phoibos'
#data_path = '/Users/lawsonlloyd/Desktop/Data/phoibos'

scan = 13069
energy_offset, delay_offset, force_offset = 19.72, -45, True

scan_info = phoibos.get_scan_info(data_path_info, filename, {})
res = phoibos.load_data(data_path, scan, scan_info, energy_offset, delay_offset, force_offset)

#%% Fitting Functions etc

def gaussian(x, amp_1, mean_1, stddev_1, offset):
    
    g1 = amp_1 * np.exp(-0.5*((x - mean_1) / stddev_1)**2)+offset
    
    return g1

def two_gaussians(x, amp_1, amp_2, mean_1, mean_2, stddev_1, stddev_2, offset):
    
    g1 = amp_1 * np.exp(-0.5*((x - mean_1) / stddev_1)**2)
    g2 = amp_2 * np.exp(-0.5*((x - mean_2) / stddev_2)**2)
    
    g = g1 + g2 + np.abs(offset)
    return g

def two_gaussians_report(x, amp_1, amp_2, mean_1, mean_2, stddev_1, stddev_2, offset):
    
    g1 = amp_1 * np.exp(-0.5*((x - mean_1) / stddev_1)**2)
    g2 = amp_2 * np.exp(-0.5*((x - mean_2) / stddev_2)**2)
    
    g = g1 + g2 + offset
    return g, g1, g2, offset

def objective(params, x, data):
    
    g1, g2, offset = two_gaussians(x, **params)
    fit = g1+g2+offset
    resid = np.abs(data-fit)**2
    
    return resid
#%% # PLOT DATA PANEL: Initial Overview

%matplotlib inline

E, E_int = [1.325, 2.075], 0.1
E, E_int = [1.37, 2.1], 0.1
<<<<<<< HEAD
E, E_int = [1.3, 2.05], 0.1
=======
E, E_int = [1.27, 2.0], 0.1
>>>>>>> a38650bd1f52a19cd63dc0e6efcf247c2b4793ff

k, k_int = 0, 20
d1, d2 = -1000, -400
d3, d4 = 500, 3000

colormap = 'terrain_r'
E_inset = 0.9

#WL = scan_info[str(scan)].get("Wavelength")
#per = (scan_info[str(scan)].get("Percent"))
#Temp = float(scan_info[str(scan)].get("Temperature"))

### Plot ###
colors = ('black','red')
fig, axx = plt.subplots(2, 2)
fig.set_size_inches(8, 8, forward=False)
plt.gcf().set_dpi(200)
axx = axx.flatten()

im1 = res.loc[{'Delay':slice(-1000,50000)}].sum(axis=2).T.plot.imshow(ax = axx[0], cmap = colormap)
axx[0].set_title(f"Scan{scan}: {WL} nm, {per}%, T = {Temp}")

waterfall = res.loc[{'Energy':slice(E[0]-1,E[1]+1), 'Angle':slice(-12,12)}].sum(dim='Angle')
waterfall =  waterfall/np.max(waterfall.loc[{"Energy":slice(E_inset,3)}])

im2 = waterfall.plot.imshow(ax = axx[1], x='Delay', vmax = 1, cmap = colormap)
axx[1].axvline(0, color = 'grey', linestyle = 'dashed')

edc_1 = res.loc[{'Angle':slice(k-k_int/2, k+k_int/2), 'Delay':slice(d1, d2)}].mean(axis=(0,2))
edc_2 = res.loc[{'Angle':slice(k-k_int/2, k+k_int/2), 'Delay':slice(d3, d4)}].mean(axis=(0,2)) 
edc_norm = np.max(edc_1)

edc_1 = edc_1/edc_norm
edc_2 = edc_2/edc_norm
edc_diff = edc_2 - edc_1
edc_diff = edc_diff/np.max(edc_diff.loc[{'Energy':slice(1.1,3.1)}])

im3 = edc_1.plot(ax = axx[2], label = f"t < {d2} fs", color = 'grey')
im3 = edc_2.plot(ax = axx[2], label = f"t = {d3} fs to {d4} fs", color = 'green')
im3d = edc_diff.loc[{'Energy':slice(0.6,2.5)}].plot(ax = axx[2], label = 'Diff.', color = 'purple')
#im3 = res.loc[{'Angle':slice(-12,12), 'Energy':slice(1,2.5), 'Delay':slice(-1000,5000)}].sum(axis=(0,2)).plot(ax = axx[2])
axx[2].axvline(0, color = 'grey', linestyle = 'dashed')
axx[2].legend(frameon=False)
#axx[2].set_yscale('log')
#axx[2].set_ylim(0,0.015)
axx[2].set_xlim(-1,2.75)
axx[2].set_ylim(0,1.2)
axx[2].set_ylabel('Int.')
axx[2].set_title('EDCs: Norm. to Neg.')

rect = (Rectangle((k-k_int/2, -2), k_int, 5 , linewidth=.5,\
                     edgecolor='violet', facecolor='purple', alpha = 0.25))
axx[0].add_patch(rect) #Add rectangle to plot

for i in np.arange(len(E)):
    #trace = res.loc[{'Angle':slice(-12,12), 'Energy':slice(E[i]-E_int/2, E[i]+E_int/2)}].sum(axis=(0,1))
    trace = phoibos.get_time_trace(res, E[i], E_int, k, k_int, False, False)
    trace.plot(ax = axx[3], color = colors[i])
    rect = (Rectangle((-1000, E[i]-E_int/2), res.Delay.values[-1]*1.5, E_int , linewidth=.5,\
                         edgecolor=colors[i], facecolor=colors[i], alpha = 0.3))
    axx[1].add_patch(rect) #Add rectangle to plot
    axx[2].axvline(E[i], color = colors[i], linestyle = 'dashed')

axx[3].axvline(0, color = 'grey', linestyle = 'dashed')
axx[3].set_xlim(trace.Delay.values[0]-50, trace.Delay.values[-1])
axx[1].set_xlim(trace.Delay.values[0]-50, trace.Delay.values[-1])

params = {'lines.linewidth' : 2, 'axes.linewidth' : 1.5, 'axes.labelsize' : 16, 
          'xtick.labelsize' : 14, 'ytick.labelsize' : 14, 'axes.titlesize' : 16, 'legend.fontsize' : 12}
plt.rcParams.update(params)

fig.tight_layout()
plt.show()

#fig.savefig('Scan' + str(scan_to_plot) + '.svg')

#%% Check E0 offset

energy = edc_1.Energy.values
edc = edc_1.values
energy_tr = edc_1.loc[{'Energy':slice(-0.5,.5)}].Energy.values
edc_tr = edc_1.loc[{'Energy':slice(-0.5,.5)}].values

popt, pcov = curve_fit(gaussian, energy_tr, edc_tr, p0=(.8,.2,.1,.1), bounds=((0, -1, 0, 0), (2, 2, 2, 1)) )

fit = gaussian(energy, *popt)

plt.plot(energy, edc, color = 'grey')
plt.plot(energy, fit, color = 'green', linestyle = 'dashed')
plt.title(f"E0 = {round(popt[1],3)}")
plt.xlim(-1,2)

#%% Define t0 from Exciton Rise

#E, E_int = [1.37, 2.125], 0.1
E, E_int = 1.3, 0.1
A, A_int = 0, 24
subtract_neg = True
norm_trace = True

fig, ax = plt.subplots()
fig.set_size_inches(6, 4, forward=False)

t0 = 0
tau = 55
def rise_erf(t, t0, tau):
    r = 0.5 * (1 + erf((t - t0) / tau))
    return r

trace = phoibos.get_time_trace(res, E, E_int, A, A_int, subtract_neg, norm_trace)

rise = rise_erf(res.Delay, 30, 45)

p0 = [-30, 45]
popt, pcov = curve_fit(rise_erf, trace.loc[{"Delay":slice(-180,200)}].Delay.values ,
                                trace.loc[{"Delay":slice(-180,200)}].values,
                                p0, method="lm")

rise_fit = rise_erf(np.linspace(-200,200,50), *popt)

ax.plot(res.Delay, trace, 'ko')
ax.plot(np.linspace(-200,200,50), rise_fit, 'pink')
#ax[1].plot(I_res.delay, rise, 'red')
ax.axvline(0, color = 'grey', linestyle = 'dashed')
ax.set_xlim([-250, 500]) 
ax.set_ylim(-.1,1.25)
ax.set_title(f"t0 offset = {round(popt[0],3)}")
fig.tight_layout()

print(round(popt[0],3))

#%% # PLOT THREE PANEL DIFFERENCE SPECTRA

save_figure = False
figure_file_name = 'DIFFERENCE_PANELS3'

delays = [10, 100000]
E, E_int = [1.3, 2.05], 0.1
#E, E_int = [1.77, 2.125], 0.1

#E, E_int = [1.3, 2.0], 0.1

A, A_int = 0, 20

colormap = 'terrain_r'
subtract_neg = True
norm_trace = False

###
#WL = scan_info[str(scan)].get("Wavelength")
#per = (scan_info[str(scan)].get("Percent"))
#Temp = float(scan_info[str(scan)].get("Temperature"))

res_neg = res.loc[{'Delay':slice(-1000,-300)}]
res_pos = res.loc[{'Delay':slice(0,5000)}]

res_neg_mean = res_neg.mean(axis=2)
res_pos_mean = res_pos.mean(axis=2)

#res_diff_E_Ang = res_pos_mean - res_neg_mean

res_diff_E_Ang = res.loc[{'Delay':slice(delays[0],delays[1])}].mean(axis=2) - res_neg_mean
#res_diff_E_Ang = res.loc[{'Delay':slice(-100,0)}].mean(axis=2) - res_neg_mean
#res_diff_E_Ang = res.loc[{'Delay':slice(250,350)}].mean(axis=2) - res_neg_mean

res_diff_E_Ang = res_diff_E_Ang/np.max(np.abs(res_diff_E_Ang))

E_inset = 0.75

res_diff = res - res_neg.mean(axis=2)
res_diff_sum_Angle = res_diff.loc[{'Angle':slice(-A-A_int/2,A+A_int/2)}].sum(axis=0)
res_diff_sum_Angle = res_diff_sum_Angle/np.max(res_diff_sum_Angle)

res_diff_sum_Angle_Normed = phoibos.enhance_features(res_diff_E_Ang, E_inset, _ , True)
res_diff_sum_Angle = phoibos.enhance_features(res_diff_sum_Angle, E_inset, _ , True)

trace_1 = phoibos.get_time_trace(res, E[0], E_int, A, A_int, subtract_neg, norm_trace)
trace_2 = phoibos.get_time_trace(res, E[1], E_int, A, A_int, subtract_neg, norm_trace)

trace_2 = trace_2/trace_1.max()
trace_1 = trace_1/trace_1.max()

############
### PLOT ###
############

fig, axx = plt.subplots(1, 3)
fig.set_size_inches(12, 4, forward=False)
plt.gcf().set_dpi(200)
axx = axx.flatten()

im1 = res_diff_sum_Angle_Normed.T.plot.imshow(ax = axx[0], cmap = 'seismic', vmin = -1, vmax = 1)

im2 = res_diff_sum_Angle.plot.imshow(ax = axx[1], cmap = 'seismic', vmin = -1, vmax = 1)

im3 = trace_1.plot(ax = axx[2], color = 'black')
im3 = trace_2.plot(ax = axx[2], color = 'red')
#im_dyn = axx[2].plot(trace_1.Delay.loc[{"Delay":slice(0,50000)}].values, \
 #                  0.6*np.exp(-trace_1.Delay.loc[{"Delay":slice(0,50000)}].values/18000) +

  #                 0.3*np.exp(-trace_1.Delay.loc[{"Delay":slice(0,50000)}].values/2000))
axx[0].axhline(E[0],  color = 'black')
axx[0].axhline(E[1],  color = 'red')

axx[0].axhline(E_inset,  color = 'grey', linestyle = 'dashed')
axx[0].set_title(f"I({delays[0]}:{delays[1]} fs) - I(<-300 fs)")
axx[0].set_ylim(-1,3)

axx[1].axhline(E[0],  color = 'black')
axx[1].axhline(E[1],  color = 'red')
axx[1].set_ylim(-0,3.1)
axx[1].set_xlim(-200, 20000)
#axx[1].axvline(-50,  color = 'grey', linestyle = 'dashed')
axx[1].axhline(E_inset,  color = 'grey', linestyle = 'dashed')
#axx[2].axvline(-400,  color = 'grey', linestyle = 'dashed')
axx[1].set_title(f"Scan{scan}. Angle-Integr.")

axx[2].set_xlim(res.Delay[0], res.Delay[-1])
axx[2].set_title(f"{WL} nm, {per}%, T = {Temp}")
axx[2].set_xlim(-500,20000)

fig.tight_layout()
#plt.show()

if save_figure is True:
    fig.savefig((figure_file_name +'.svg'), format='svg')
    
#%% # PLOT THREE PANEL DIFFERENCE arpes at different Delays

fig, axx = plt.subplots(1, 3)
fig.set_size_inches(12, 4, forward=False)
plt.gcf().set_dpi(200)
axx = axx.flatten()

ts = [0, 250, 1250]
E_inset = 0.8

for i in np.arange(len(ts)):
    
    panel = phoibos.make_diff_ARPES(res, [ts[i]-50, ts[i]+50], E_inset)
    im1 = panel.T.plot.imshow(ax = axx[i], cmap = 'seismic', vmin = -1, vmax = 1)
    
    axx[i].axhline(E_inset,  color = 'grey', linestyle = 'dashed')
    axx[i].set_title(f"I(t = {ts[i]} fs) - I(t<-300 fs)")
    axx[i].set_ylim(-1,3)
    #axx[1].set_title(f"Scan{scan}. Angle-Integr.")
    #axx[2].set_title(f"{WL} nm, {per}%, T = {Temp}")

fig.tight_layout()
plt.show()

#%% Plot Neg, Pos, and Difference Panels

%matplotlib inline

E, E_int = [1.375, 2.125], 0.1

colormap = 'Purples' #'bone_r'

fig, axx = plt.subplots(1, 3)
fig.set_size_inches(8, 3, forward=False)
plt.gcf().set_dpi(200)
axx = axx.flatten()

E_inset = .8

neg_enh = phoibos.enhance_features(res_neg_mean, E_inset, 1, True)
pos_enh = phoibos.enhance_features(res_pos_mean, E_inset, 1, True)
diff_enh = phoibos.enhance_features(res_diff_E_Ang, E_inset, 1, True) 

im1 = neg_enh.T.plot(ax = axx[0], cmap = colormap)
im2 = pos_enh.T.plot(ax = axx[1], cmap = colormap)
diff_enh.T.plot(ax = axx[2], cmap = 'seismic', vmin = -1, vmax = 1)

axx[2].axhline(E_inset, color = 'black', linewidth = 1)
axx[2].set_title(f"Scan{scan}")

fig.tight_layout()
plt.show()

#%% Determine E = 0 from VBM Fit

energy = edc_1.Energy.values
edc = edc_1.values
energy_tr = edc_1.loc[{'Energy':slice(-0.5,.5)}].Energy.values
edc_tr = edc_1.loc[{'Energy':slice(-0.5,.5)}].values

popt, pcov = curve_fit(gaussian, energy_tr, edc_tr, p0=(.8,.2,.1,.1), bounds=((0, -1, 0, 0), (2, 2, 2, 1)) )

fit = gaussian(energy, *popt)

plt.plot(energy, edc, color = 'grey')
plt.plot(energy, fit, color = 'green', linestyle = 'dashed')
plt.title(f"E0 = {round(popt[1],3)}")
plt.xlim(-1,2)


E, E_int = [1.35, 2.1], 0.1
k, k_int = -4, 30
d1, d2 = -1000, -300
d3, d4 = 500, 3000

colormap = 'terrain_r'

WL = scan_info[str(scan)].get("Wavelength")
per = (scan_info[str(scan)].get("Percent"))
Temp = float(scan_info[str(scan)].get("Temperature"))

### Plot ###
colors = ('black','red')
fig, axx = plt.subplots(2, 2)
fig.set_size_inches(8, 8, forward=False)
plt.gcf().set_dpi(200)
axx = axx.flatten()

im1 = res.loc[{'Delay':slice(-1000,5000)}].sum(axis=2).T.plot.imshow(ax = axx[0], cmap = colormap)
axx[0].set_title(f"Scan{scan}: {WL} nm, {per}%, T = {Temp}")

waterfall = res.loc[{'Energy':slice(E[0]-1,E[1]+1), 'Angle':slice(-12,12)}].sum(axis=0)
waterfall =  waterfall/np.max(waterfall.loc[{"Energy":slice(1,3)}])

im2 = waterfall.plot.imshow(ax = axx[1], vmax = 1, cmap = colormap)
axx[1].axvline(0, color = 'grey', linestyle = 'dashed')

edc_1 = res.loc[{'Angle':slice(k-k_int/2, k+k_int/2), 'Delay':slice(d1, d2)}].mean(axis=(0,2))
edc_2 = res.loc[{'Angle':slice(k-k_int/2, k+k_int/2), 'Delay':slice(d3, d4)}].mean(axis=(0,2)) 
edc_norm = np.max(edc_1)

edc_1 = edc_1/edc_norm
edc_2 = edc_2/edc_norm
edc_diff = edc_2 - edc_1
edc_diff = edc_diff/np.max(edc_diff.loc[{'Energy':slice(1.1,3)}])

im3 = edc_1.plot(ax = axx[2], label = f"t < {d2} fs", color = 'grey')
im3 = edc_2.plot(ax = axx[2], label = f"t = {d3} fs to {d4} fs", color = 'green')
im3d = edc_diff.loc[{'Energy':slice(0.6,2.5)}].plot(ax = axx[2], label = 'Diff.', color = 'purple')
#im3 = res.loc[{'Angle':slice(-12,12), 'Energy':slice(1,2.5), 'Delay':slice(-1000,5000)}].sum(axis=(0,2)).plot(ax = axx[2])
axx[2].axvline(0, color = 'grey', linestyle = 'dashed')
axx[2].legend(frameon=False)
#axx[2].set_yscale('log')
#axx[2].set_ylim(0,0.015)
axx[2].set_xlim(-1,2.75)
axx[2].set_ylim(0,1.2)
axx[2].set_ylabel('Int.')
axx[2].set_title('EDCs: Norm. to Neg.')

rect = (Rectangle((k-k_int/2, -2), k_int, 5 , linewidth=.5,\
                     edgecolor='violet', facecolor='purple', alpha = 0.3))
axx[0].add_patch(rect) #Add rectangle to plot

for i in np.arange(len(E)):
    trace = res.loc[{'Angle':slice(-12,12), 'Energy':slice(E[i]-E_int/2, E[i]+E_int/2)}].sum(axis=(0,1))
    trace.plot(ax = axx[3], color = colors[i])
    rect = (Rectangle((-1000, E[i]-E_int/2), 5000, E_int , linewidth=.5,\
                         edgecolor=colors[i], facecolor=colors[i], alpha = 0.3))
    axx[1].add_patch(rect) #Add rectangle to plot
    axx[2].axvline(E[i], color = colors[i], linestyle = 'dashed')

axx[3].axvline(0, color = 'grey', linestyle = 'dashed')

params = {'lines.linewidth' : 2, 'axes.linewidth' : 1.5, 'axes.labelsize' : 16, 
          'xtick.labelsize' : 14, 'ytick.labelsize' : 14, 'axes.titlesize' : 16, 'legend.fontsize' : 12}
plt.rcParams.update(params)

fig.tight_layout()
plt.show()

#%% Define t0 from Exciton Rise

from scipy.special import erf

scans = [9219, 9217, 9218, 9216, 9220, 9228]
offsets_t0 = [-162.1, -152.7, -183.2, -118.4, -113.2, -125.0]

scans =      [9227, 9219, 9217, 9218, 9216, 9220, 9228, 9231, 9525, 9517, 9526] # Scans to analyze and fit below: 910 nm + 400 nm
offsets_t0 = [-191.7, -157.4, -152.7, -183.1, -118.5, -113.7, -125.2, -147.6, -77, -151.1, -200.6]

scans = 9370
#scans = [9227, 9219, 9217, 9218, 9216, 9220, 9228, 9231,    9525, 9517, 9526]
#offsets_t0 = [-191.7, -162.4, -152.7, -183.1, -118.5, -113.7, -125.2, -147.6, -77, -151.1, -200.6]

t0_offsets = []

fig, ax = plt.subplots(4, 3)
fig.set_size_inches(8, 12, forward=False)
ax = ax.flatten()

i = 0    
for scan_i in scans:
    
    res = phoibos.load_data(data_path, scan_i, scan_info, 19.72, offsets_t0[i], False)

    ### Plot EDCs at GAMMA vs time
    
    (kx, ky), k_int = (0 , 0), 24
    trace = res.loc[{'Angle':slice(-12,12), 'Energy':slice(E1-E_int/2, E1+E_int/2)}].sum(axis=(0,1))
    trace = trace - trace[3:8].mean()
    trace = trace/np.max(trace)
#    trace.plot(ax = ax[i], color = 'red')
    
    trace_ex = res.loc[{"Angle":slice(kx-k_int/2,kx+k_int/2), 'Energy':slice(E1-E_int/2, E1+E_int/2)}].sum(dim=("Angle","Energy"))
    trace_ex = trace_ex - trace_ex[2:6].mean()
    trace_ex = trace_ex/np.max(trace_ex.loc[{"Delay":slice(-200,250)}])
    
    t0 = 0
    tau = 55
    def rise_erf(t, t0, tau):
        r = 0.5 * (1 + erf((t - t0) / tau))
        return r
    
    rise = rise_erf(res.Delay, 30, 45)
    
    p0 = [-30, 45]
    popt, pcov = curve_fit(rise_erf, trace_ex.loc[{"Delay":slice(-180,200)}].Delay.values ,
                                    trace_ex.loc[{"Delay":slice(-180,200)}].values,
                                    p0, method="lm")
    
    rise_fit = rise_erf(np.linspace(-200,200,50), *popt)
    
    ax[i].plot(res.Delay, trace_ex, 'ko')
    ax[i].plot(np.linspace(-200,200,50), rise_fit, 'pink')
    #ax[1].plot(I_res.delay, rise, 'red')
    
    ax[i].axvline(0, color = 'grey', linestyle = 'dashed')
    
    ax[i].set_xlim([-250, 500]) 
    ax[i].set_ylim(-.1,1.25)
    ax[i].set_title(f"t0 offset = {round(popt[0],3)}")
    print(round(popt[0],3))
    t0_offsets.append(popt[0])
    i += 1

fig.tight_layout()