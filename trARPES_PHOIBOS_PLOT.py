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
import csv
from Loader import DataLoader
import xarray as xr

#%%

filename = '2024 Bulk CrSBr Phoibos.csv'

scan_info = {}
data_path = 'R:\Lawson\Data'

with open(data_path + '//' + filename) as f:
    
    reader = csv.DictReader(f
                            )
    for row in reader:
        key = row.pop('Scan')
        if key in scan_info:
            # implement your duplicate row handling here
            pass
        scan_info[key] = row

#%%

phoibos = True

data_path = 'path_to_your_data'
filename = 'your_file_name.h5'

data_path = 'R:\Lawson\Data\phoibos'
#data_path = '/Users/lawsonlloyd/Desktop/Data/'

scan = 9220
filename = f"Scan{scan}.h5"

data_loader = DataLoader(data_path + '//' + filename)
res = data_loader.load_phoibos()

energy_offset = + 19.65
res = res.assign_coords(Energy=(res.Energy-energy_offset))

#%% # PLOT DATA PANEL: Initial Overview

%matplotlib inline

E1, E2, E3 = 1.37, 2.1, 0.1
A1, A2 = -10 , 10
d1, d2 = -1000, -400
d3, d4 = 0, 200

colormap = 'terrain_r'

WL = scan_info[str(scan)].get("Wavelength")
per = (scan_info[str(scan)].get("Percent"))
Temp = float(scan_info[str(scan)].get("Temperature"))

fig, axx = plt.subplots(2, 2)
fig.set_size_inches(8, 8, forward=False)
plt.gcf().set_dpi(200)
axx = axx.flatten()

im1 = res.loc[{'Delay':slice(-1000,5000)}].sum(axis=2).T.plot.imshow(ax = axx[0], cmap = colormap)
axx[0].set_title(f"Scan{scan}: {WL} nm, {per}%, T = {Temp}")

im2 = res.loc[{'Energy':slice(E1-1,E2+0.8), 'Angle':slice(-12,12)}].sum(axis=0).plot.imshow(ax = axx[1], vmax = .3e7, cmap = colormap)
axx[1].axvline(0, color = 'grey', linestyle = 'dashed')
axx[1].axhline(E1, color = 'black')
axx[1].axhline(E2, color = 'red')

edc_1 = res.loc[{'Angle':slice(A1, A2), 'Delay':slice(d1,-d2)}].mean(axis=(0,2))
edc_2 = res.loc[{'Angle':slice(A1,A2), 'Delay':slice(d3,d4)}].mean(axis=(0,2)) 
edc_norm = np.max(edc_1)

edc_1 = edc_1/edc_norm
edc_2 = edc_2/edc_norm
edc_diff = edc_2 - edc_1
#edc_diff = edc_diff/np.max(edc_diff.loc[{'Energy':slice(0.6,3)}])

im3 = edc_1.plot(ax = axx[2], label = f"t < {d2} fs", color = 'grey')
im3 = edc_2.plot(ax = axx[2], label = f"t = {d3} fs to {d4} fs", color = 'green')
im3d = edc_diff.loc[{'Energy':slice(0.6,2.5)}].plot(ax = axx[2], label = 'Diff.', color = 'purple')
#im3 = res.loc[{'Angle':slice(-12,12), 'Energy':slice(1,2.5), 'Delay':slice(-1000,5000)}].sum(axis=(0,2)).plot(ax = axx[2])
axx[2].axvline(0, color = 'grey', linestyle = 'dashed')
axx[2].legend(frameon=False)
#axx[2].set_yscale('log')
axx[2].set_ylim(0,0.0075)
axx[2].set_xlim(-1,2.75)
axx[2].set_ylabel('Int.')
axx[2].set_title('EDCs: Norm. to Neg.')

im4 = res.loc[{'Angle':slice(-12,12), 'Energy':slice(E1-Eint/2, E1+Eint/2)}].sum(axis=(0,1)).plot(ax = axx[3], color = 'black')
im4 = res.loc[{'Angle':slice(-12,12), 'Energy':slice(E2-Eint/2, E2+Eint/2)}].sum(axis=(0,1)).plot(ax = axx[3], color = 'red')
axx[3].axvline(0, color = 'grey', linestyle = 'dashed')

params = {'lines.linewidth' : 2, 'axes.linewidth' : 1.5, 'axes.labelsize' : 16, 
          'xtick.labelsize' : 14, 'ytick.labelsize' : 14, 'axes.titlesize' : 16, 'legend.fontsize' : 12}
plt.rcParams.update(params)

fig.tight_layout()
plt.show()

#fig.savefig('Scan' + str(scan_to_plot) + '.svg')

#%% # PLOT THREE PANEL DIFFERENCE

delays = [0,3000]

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
d1 = (res_diff_E_Ang.loc[{'Energy':slice(-1,E_inset)}])
d2 = (res_diff_E_Ang.loc[{'Energy':slice(E_inset,3)}])
d2 = d2/np.max(abs(d2))
d3 = xr.concat([d1, d2], dim = "Energy")
res_diff_sum_Angle_Normed = d3

res_diff = res - res_neg.mean(axis=2)
res_diff_sum_Angle = res_diff.loc[{'Angle':slice(-12,12)}].sum(axis=0)
res_diff_sum_Angle = res_diff_sum_Angle/np.max(res_diff_sum_Angle)

trace_1 = res.loc[{'Angle':slice(-12,12), 'Energy':slice(E1-Eint/2, E1+Eint/2)}].sum(axis=(0,1))
trace_2 = res.loc[{'Angle':slice(-12,12), 'Energy':slice(E2-Eint/2, E2+Eint/2)}].sum(axis=(0,1))
trace_1 = trace_1 - trace_1.loc[{'Delay':slice(-1000,-350)}].mean()
trace_2 = trace_2 - trace_2.loc[{'Delay':slice(-1000,-350)}].mean()

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

axx[0].axhline(E_inset,  color = 'grey', linestyle = 'dashed')
axx[1].axhline(E1,  color = 'black')
axx[1].axhline(E2,  color = 'red')
axx[0].set_title(f"I({delays[0]}:{delays[1]} fs) - I(<-300 fs)")
axx[0].set_ylim(-1,3)
axx[1].set_ylim(-1,3)
axx[1].axvline(-50,  color = 'grey', linestyle = 'dashed')

#axx[2].axvline(-400,  color = 'grey', linestyle = 'dashed')
axx[1].set_title(f"Scan{scan}. Angle-Integr.")
axx[2].set_xlim(res.Delay[0], res.Delay[-1])
#axx[2].set_xlim(res.Delay[0], 25000)
axx[2].set_title(f"{WL} nm, {per}%, T = {Temp}")
axx[2].set_xlim(-500,3000)

fig.tight_layout()
plt.show()

#%% # PLOT THREE PANEL DIFFERENCE

def make_diff_ARPES(res, delays, E_inset):

    res_neg = res.loc[{'Delay':slice(-1000,-300)}]
    res_pos = res.loc[{'Delay':slice(0,5000)}]
    
    res_neg_mean = res_neg.mean(axis=2)
    res_pos_mean = res_pos.mean(axis=2)
    
    #res_diff_E_Ang = res_pos_mean - res_neg_mean
    res_diff_E_Ang = res.loc[{'Delay':slice(delays[0],delays[1])}].mean(axis=2) - res_neg_mean
    res_diff_E_Ang = res_diff_E_Ang/np.max(np.abs(res_diff_E_Ang))
    
    d1 = (res_diff_E_Ang.loc[{'Energy':slice(-1,E_inset)}])
    d2 = (res_diff_E_Ang.loc[{'Energy':slice(E_inset,3)}])
    d2 = d2/np.max(abs(d2))
    d3 = xr.concat([d1, d2], dim = "Energy")
    res_diff_sum_Angle_Normed = d3

    return res_diff_sum_Angle_Normed

############
### PLOT ###
############

fig, axx = plt.subplots(1, 3)
fig.set_size_inches(12, 4, forward=False)
plt.gcf().set_dpi(200)
axx = axx.flatten()

ts = [0, 250, 1250]

for i in np.arange(len(ts)):
    
    panel = make_diff_ARPES(res, [ts[i]-50, ts[i]+50], 0.75)
    im1 = panel.T.plot.imshow(ax = axx[i], cmap = 'seismic', vmin = -1, vmax = 1)
    
    axx[i].axhline(E_inset,  color = 'grey', linestyle = 'dashed')
    axx[i].set_title(f"I(t = {ts[i]} fs) - I(t<-300 fs)")
    axx[i].set_ylim(-1,3)
    #axx[1].set_title(f"Scan{scan}. Angle-Integr.")
    #axx[2].set_title(f"{WL} nm, {per}%, T = {Temp}")

fig.tight_layout()
plt.show()

#%% # PLOT THREE PANEL DIFFERENCE

%matplotlib inline

E1, E2, Eint = 1.375, 2.125, 0.1

colormap = 'terrain_r'

fig, axx = plt.subplots(1, 3)
fig.set_size_inches(8, 3, forward=False)
plt.gcf().set_dpi(200)
axx = axx.flatten()

E_inset = 0.75
im1 = res_neg_mean.T.plot(ax = axx[0], cmap = colormap)
im2 = res_pos_mean.T.plot(ax = axx[1], cmap = colormap)
#im3 = (res_diff_angle).T.plot(ax = axx[2], cmap = 'seismic', vmin = -1, vmax = 1)
d1 = (res_diff_angle.loc[{'Energy':slice(-1,E_inset)}])
d2 = (res_diff_angle.loc[{'Energy':slice(E_inset,3)}])
d2 = d2/np.max(np.abs(d2))
d3 = xr.concat([d1, d2], dim = "Energy")
d3.T.plot(ax = axx[2], cmap = 'seismic', vmin = -1, vmax = 1)
axx[2].axhline(E_inset, color = 'black', linewidth = 1)
axx[2].set_title(f"Scan{scan}")

fig.tight_layout()
plt.show()

############
############
#%%


#%%

def func(x, a, b, tau1, tau2):
    return a*np.exp(-x/tau1)+b*np.exp(-x/tau2)

delays_trunc = trace_1.loc[{"Delay":slice(0,20000)}].Delay.values
trace_trunc =  trace_1.loc[{"Delay":slice(0,20000)}].values

delays = trace_1.Delay.values
trace =  trace_1.values

popt, pcov = curve_fit(func, delays_trunc, trace_trunc, p0=(1,1,2000,15000))

fit = func(delays_trunc, *popt)

fig = plt.figure()
plt.plot(delays, trace, 'o', color = 'grey')
plt.plot(delays_trunc, fit, color = 'blue')
plt.title(f"Biexp: tau_1 = {round(popt[2])}, tau_2 = {round(popt[3],0)}")
plt.xlim(-600,20000)
plt.ylabel('Int.')
plt.xlabel('Delay, fs')
#print(popt)

save_figure = False
figure_file_name = f"Long_Delays_{scan}"
#plt.rcParams['svg.fonttype'] = 'none'
new_rc_params = {'text.usetex': False, "svg.fonttype": 'none'}
plt.rcParams.update(new_rc_params)

if save_figure is True:
    fig.savefig((figure_file_name +'.svg'), format='svg')


#%%

def gaussian(x, amp_1, mean_1, stddev_1, offset):
    
    g1 = np.exp(-0.5*((x - mean_1) / stddev_1)**2)+np.abs(offset)
    g1 = g1/np.max(g1)
    
    g1 = g1*amp_1
    
    return g1

energy = edc_1.Energy.values
edc = edc_1.values
energy_tr = edc_1.loc[{'Energy':slice(-0.12,.5)}].Energy.values
edc_tr = edc_1.loc[{'Energy':slice(-0.12,.5)}].values

popt, pcov = curve_fit(gaussian, energy_tr, edc_tr, p0=(.8,.2,.1,.1))

fit = gaussian(energy, *popt)

plt.plot(energy, edc, color = 'grey')
plt.plot(energy, fit, color = 'green', linestyle = 'dashed')
plt.title(f"E0 = {round(popt[1],3)}")
plt.xlim(-1,2)
