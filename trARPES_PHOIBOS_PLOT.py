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
        
            
def load_data(scan, energy_offset, delay_offset):
    filename = f"Scan{scan}.h5"
    
    data_loader = DataLoader(data_path + '//' + filename)
    res = data_loader.load_phoibos()
    
    res = res.assign_coords(Energy=(res.Energy-energy_offset))
    res = res.assign_coords(Delay=(res.Delay-delay_offset))
    
    return res

#%%

# Fucntion for Extracting time Traces
def get_time_trace(I_res, E, E_int, k , k_int, subtract_neg, norm_trace):
        
    
    trace = res.loc[{"Energy":slice(E-E_int/2, E+E_int/2), "Angle":slice(k-k_int/2, k+k_int/2)}].sum(dim=("Energy","Angle"))

    if subtract_neg is True : 
        trace = trace - np.mean(trace.loc[{"Delay":slice(-1000,-300)}])
    
    if norm_trace is True : 
        trace = trace/np.max(trace)
    
    return trace

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

def enhance_features(I_res, Ein, factor, norm):
    
    I1 = I_res.loc[{"Energy":slice(-3.5,Ein)}]
    I2 = I_res.loc[{"Energy":slice(Ein,3.5)}]

    if norm is True:
        I1 = I1/np.max(I1)
        I2 = I2/np.max(I2)
    else:
        I1 = I1/factor[0]
        I2 = I2/factor[1]
        
    I3 = xr.concat([I1, I2], dim = "Energy")
    
    return I3

#%%

phoibos = True

data_path = 'path_to_your_data'
filename = 'your_file_name.h5'

data_path = 'R:\Lawson\Data\phoibos'
#data_path = '/Users/lawsonlloyd/Desktop/Data/'

scan = 9216
energy_offset = + 19.72
delay_offset = -80

res = load_data(scan, energy_offset, delay_offset)

#%% # PLOT DATA PANEL: Initial Overview

%matplotlib inline

E1, E2, E3 = 1.37, 2.1, 0.1
A1, A2 = -10, -8
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
axx[0].axvline(A1, linestyle = 'dashed', color = 'grey')
axx[0].axvline(A2, linestyle = 'dashed', color = 'grey')

im2 = res.loc[{'Energy':slice(E1-1,E2+0.8), 'Angle':slice(-12,12)}].sum(axis=0).plot.imshow(ax = axx[1], vmax = .3e7, cmap = colormap)
axx[1].axvline(0, color = 'grey', linestyle = 'dashed')
axx[1].axhline(E1, color = 'black')
axx[1].axhline(E2, color = 'red')

edc_1 = res.loc[{'Angle':slice(A1, A2), 'Delay':slice(d1, d2)}].mean(axis=(0,2))
edc_2 = res.loc[{'Angle':slice(A1, A2), 'Delay':slice(d3, d4)}].mean(axis=(0,2)) 
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
#axx[2].set_ylim(0,0.015)
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


#%%

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
#%% # PLOT THREE PANEL DIFFERENCE SPECTRA

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

#%% # PLOT THREE PANEL DIFFERENCE arpes at different Delays

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

#%% # Plot Panels

%matplotlib inline

E1, E2, Eint = 1.375, 2.125, 0.1

colormap = 'Purples' #'bone_r'

fig, axx = plt.subplots(1, 3)
fig.set_size_inches(8, 3, forward=False)
plt.gcf().set_dpi(200)
axx = axx.flatten()

E_inset = 0.75

neg_enh = enhance_features(res_neg_mean, E_inset, 1, True)
pos_enh = enhance_features(res_pos_mean, E_inset, 1, True)
diff_enh = enhance_features(res_diff_sum_Angle_Normed, E_inset, 1, True) 

im1 = neg_enh.T.plot(ax = axx[0], cmap = colormap)
im2 = pos_enh.T.plot(ax = axx[1], cmap = colormap)
diff_enh.T.plot(ax = axx[2], cmap = 'seismic', vmin = -1, vmax = 1)

axx[2].axhline(E_inset, color = 'black', linewidth = 1)
axx[2].set_title(f"Scan{scan}")

fig.tight_layout()
plt.show()
#%%

#%% # PLOT Fluence Delay TRACES All Together

save_figure = True
figure_file_name = 'Combined'

fig, axx = plt.subplots(1, 2)
fig.set_size_inches(12, 4, forward=False)
axx = axx.flatten()

scans = [9219, 9217, 9218, 9216, 9220, 9228]
power = 1.05*np.asarray([153, 111, 91, 66, 47, 32, 15, 10, 8, 5])
fluence = [.2, .35, .8, 1.74, 2.4, 2.9]

k, k_int = (0), 24
E1, E2, E3, E_int = 1.75, 2.1, 0.1, 1
subtract_neg = True
norm_trace = False

cn = 100
p_min = .1
p_max = 3.5

colors1 = plt.cm.Purples(np.linspace(p_min, 3.5, cn)) 
cm2 = plt.cm.ScalarMappable(norm = plt.Normalize(vmin=p_min,vmax=p_max), cmap=plt.cm.Purples)

colors2 = plt.cm.Reds(np.linspace(p_min, 3.5, cn))
cm = plt.cm.ScalarMappable(norm = plt.Normalize(vmin=p_min,vmax=p_max), cmap=plt.cm.Reds)

fluence_cbar = np.linspace(p_min, p_max, cn)

i = 0
for scan_i in scans:
    
    res = load_data(scan_i, energy_offset, delay_offset)
    trace_1 = get_time_trace(res, E1, E_int, k , k_int, subtract_neg, norm_trace)
    trace_2 = get_time_trace(res, E2, E_int, k , k_int, subtract_neg, norm_trace)
    trace_2 = trace_2/np.max(trace_1)
    trace_1 = trace_1/np.max(trace_1)

    j_fluence = (np.abs(fluence_cbar-fluence[i])).argmin()

    t1 = trace_1.plot(ax = axx[0], color = colors1[j_fluence])
    t2 = trace_2.plot(ax = axx[1], color = colors2[j_fluence])
    
    i += 1
    
axx[0].set_xlim([-500,3000])
axx[1].set_xlim([-500,3000])
axx[0].set_ylabel('Int., a.u.')
cbar = plt.colorbar(cm, ax=axx[1])
cbar.set_label('Fluence', rotation=90, fontsize=22)
cbar.ax.tick_params(labelsize=20)

cbar = plt.colorbar(cm2, ax=axx[0])
cbar.set_label('Fluence', rotation=90, fontsize=22)
cbar.ax.tick_params(labelsize=20)

fig.tight_layout()

if save_figure is True:
    fig.savefig((figure_file_name +'.svg'), format='svg')
    
#%% # PLOT Fluence Delay TRACES All Together

save_figure = False
figure_file_name = 'phoibosfluencetraces'

fig, axx = plt.subplots(1, 2)
fig.set_size_inches(12, 4, forward=False)
axx = axx.flatten()

scans = [9219, 9217, 9218, 9216, 9220, 9228]
power = 1.05*np.asarray([153, 111, 91, 66, 47, 32, 15, 10, 8, 5])
fluence = [.2, .35, .8, 1.74, 2.4, 2.9]

k, k_int = (0), 24
E1, E2, E3, E_int = 1.37, 2.1, 0.1, 0.1
subtract_neg = True
norm_trace = False

cn = 100
p_min = .1
p_max = 3.5

colors2 = plt.cm.Reds(np.linspace(p_min, 3.5, cn))
cm = plt.cm.ScalarMappable(norm = plt.Normalize(vmin=p_min,vmax=p_max), cmap=plt.cm.Reds)

colors1 = plt.cm.bone_r(np.linspace(p_min, 3.5, cn)) 
cm2 = plt.cm.ScalarMappable(norm = plt.Normalize(vmin=p_min,vmax=p_max), cmap=plt.cm.bone_r)

fluence_cbar = np.linspace(p_min, p_max, cn)

i = 0
for scan_i in scans:
    
    res = load_data(scan_i, energy_offset, delay_offset)
    trace_1 = get_time_trace(res, E1, E_int, k , k_int, subtract_neg, norm_trace)
    trace_2 = get_time_trace(res, E2, E_int, k , k_int, subtract_neg, norm_trace)
    trace_2 = trace_2/np.max(trace_1)
    trace_1 = trace_1/np.max(trace_1)

    j_fluence = (np.abs(fluence_cbar-fluence[i])).argmin()

    t1 = trace_1.plot(ax = axx[0], color = colors1[j_fluence])
    t2 = trace_2.plot(ax = axx[1], color = colors2[j_fluence])
    
    i += 1
    
axx[0].set_xlim([-500,3000])
axx[1].set_xlim([-500,3000])
axx[0].set_ylabel('Int., a.u.')
cbar = plt.colorbar(cm, ax=axx[1])
cbar.set_label('Fluence', rotation=90, fontsize=22)
cbar.ax.tick_params(labelsize=20)

cbar = plt.colorbar(cm2, ax=axx[0])
cbar.set_label('Fluence', rotation=90, fontsize=22)
cbar.ax.tick_params(labelsize=20)

fig.tight_layout()

if save_figure is True:
    fig.savefig((figure_file_name +'.svg'), format='svg')
    
    
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



#%% Do the EDC Fits: Functions

###################
##### Fit EDCs ####
###################

##### VBM #########

def fit_vbm_dynamics(res, k, k_int):
    e1 = -.2
    e2 = 0.6
    p0 = [1, 0, .2, 0] # Fitting params initial guess [amp, center, width, offset]
    bnds = ((0.5, -1, 0.0, 0), (1.5, 0.5, 1, .5))
    
    centers_VBM = np.zeros(len(res.Delay))
    p_fits_VBM = np.zeros((len(res.Delay),4))
    p_err_VBM = np.zeros((len(res.Delay),2))
    
    (kx), k_int = k, k_int
    edc_gamma = res.loc[{"Angle":slice(kx-k_int/2,kx+k_int/2)}].sum(dim=("Angle"))
    edc_gamma = edc_gamma/np.max(edc_gamma)
    
    n = len(res.Delay)
    for t in np.arange(n):
        edc_i = edc_gamma.loc[{"Energy":slice(e1,e2)}][:,t].values
        edc_i = edc_i/np.max(edc_i)
        
        try:
            popt, pcov = curve_fit(gaussian, edc_gamma.loc[{"Energy":slice(e1,e2)}].Energy.values, edc_i, p0, method=None, bounds = bnds)
        except ValueError:
            print('oops')
            popt = [0,0,0,0]
            
        centers_VBM[t] = popt[1]
        p_fits_VBM[t,:] = popt
        perr = np.sqrt(np.diag(pcov))
        p_err_VBM[t,:] = perr[1:2+1]
        
    return centers_VBM, p_fits_VBM, p_err_VBM

##### CBM AND EXCITON #####

def fit_ex_cbm_dynamics(res, delay_int):
    delay_int = 50
    e1 = 1.1
    e2 = 3
    p0 = [1, 0.3,  1.3, 2.1,  0.2, 0.2, 0] # Fitting params initial guess [amp, center, width, offset]
    bnds = ((0.5, 0.1, 1.0, 1.5, 0.1, 0.1, 0), (1.5, 0.7, 1.5, 2.3, 0.9, 0.9, .3))
    
    centers_CBM = np.zeros(len(res.Delay))
    centers_EX = np.zeros(len(res.Delay))
    Ebs = np.zeros(len(res.Delay))
    
    p_fits_excited = np.zeros((len(res.Delay),7))
    p_err_excited = np.zeros((len(res.Delay),7))
    p_err_eb = np.zeros((len(res.Delay)))
    
    n = len(res.Delay.values)
    for t in range(n):
    
        kx_frame = res.loc[{"Delay":slice(res.Delay.values[t]-delay_int/2, res.Delay.values[t]+delay_int/2)}].mean(dim="Delay")
        kx_frame = kx_frame - res.loc[{"Delay":slice(-1000,-150)}].mean(dim="Delay")
    
        kx_edc_i = kx_frame.loc[{"Angle":slice(-12,12)}].sum(dim="Angle")
        kx_edc_i = kx_edc_i/np.max(kx_edc_i.loc[{"Energy":slice(0.8,3)}])
        
        try:
            popt, pcov = curve_fit(two_gaussians, kx_edc_i.loc[{"Energy":slice(e1,e2)}].Energy.values, kx_edc_i.loc[{"Energy":slice(e1,e2)}].values, p0, method=None, bounds = bnds)
        except ValueError:
            print('Oops!')
            popt = [0,0,0,0]
       
        centers_EX[t] = popt[2]
        centers_CBM[t] = popt[3]
        Eb = round(popt[3] - popt[2],3)
        Ebs[t] = Eb
        perr = np.sqrt(np.diag(pcov))
        p_fits_excited[t,:] = popt
        
        p_err_excited[t,:] = perr 
        p_err_eb[t] = np.sqrt(perr[3]**2+perr[2]**2)
        
    return centers_EX, centers_CBM, Ebs, p_fits_excited, p_err_excited, p_err_eb

#%% # Do the Fitting for VBM, EXCITON, AND CBM

res = load_data(scan, energy_offset, delay_offset)

centers_VBM, p_fits_VBM, p_err_VBM = fit_vbm_dynamics(res, kx, 4)

centers_EX, centers_CBM, Ebs, p_fits_excited, p_err_excited, p_err_eb = fit_ex_cbm_dynamics(res, delay_int)

#%% Plot and Fit EDCs of the VBM

%matplotlib inline

save_figure = False
figure_file_name = 'EDC_phoibos'

#I_res = I.groupby_bins('delay', 50)
#I_res = I_res.rename({"delay_bins":"delay"})
#I_res = I_res/np.max(I_res)
res_n = res/np.max(res)

fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1.5, 1], 'height_ratios':[1]})
fig.set_size_inches(10, 4, forward=False)
ax = ax.flatten()

### Plot EDCs at GAMMA vs time
(kx), k_int = (-3), 6
edc_gamma = res.loc[{"Angle":slice(kx-k_int/2,kx+k_int/2)}].sum(dim=("Angle"))
edc_gamma = edc_gamma/np.max(edc_gamma)

im = edc_gamma.plot.imshow(ax = ax[0], cmap = cmap_LTL, add_colorbar = False)
# cbar_ax = fig.add_axes([.51, 0.275, 0.025, 0.5])
cbar = fig.colorbar(im, ax = ax[0], ticks = [0,1])
cbar.ax.set_yticklabels(['min', 'max'])  # vertically oriented colorbar

ax[0].set_ylim([-1,1])
ax[0].set_xlim([edc_gamma.Delay[0], edc_gamma.Delay[-1]])
#plt.axhline(0, linestyle = 'dashed', color = 'black')
#plt.axvline(0, linestyle = 'dashed', color = 'black')
ax[0].set_xlabel('Delay, fs')
ax[0].set_ylabel('E - E$_{VBM}$, eV')

pts = [-120, 0, 50, 100, 500]
delay_int = 50
colors = ['black', 'red', 'orange', 'purple', 'blue', 'green', 'grey']
n = len(pts)
colors = mpl.cm.inferno(np.linspace(0,.8,n))
                    
for i in range(n):
    edc = res.loc[{"Angle":slice(kx-k_int/2,kx+k_int/2), "Delay":slice(pts[i]-delay_int/2,pts[i]+delay_int/2)}].sum(dim=("Angle","Delay"))
    edc = edc/np.max(edc)
    
    e = edc.plot(ax = ax[1], color = colors[i], label = f"{pts[i]} fs")

#plt.legend(frameon = False)
ax[1].set_xlim([-1.5, 1]) 
#ax[1].set_ylim([0, 1.1])
ax[1].set_xlabel('E - E$_{VBM}$, eV')
ax[1].set_ylabel('Norm. Int.')
#ax[1].axvline(0, color = 'black', linestyle = 'dashed', linewidth = 0.5)
ax[1].legend(frameon=False, loc = 'upper left', fontsize = 11)
#ax[1].set_yscale('log')
#plt.ax[1].gca().set_aspect(2)

# #ax[1].plot(edc_gamma.Energy.values, edc_gamma[:,t].values/edc_gamma.loc[{"Energy":slice(e1,e2)}][:,t].values.max(), color = 'pink')
# ax[1].plot(edc_gamma.Energy.values, gauss_test, linestyle = 'dashed', color = 'grey')
# #plt.axvline(trunc_e, linestyle = 'dashed', color = 'black')
# ax[1].set_xlim([-2,1.5])
# ax[1].set_xlabel('Energy, eV')
# ax[1].set_ylabel('Norm. Int, arb. u.')
# #plt.gca().set_aspect(3)

# # PLOT VBM SHIFT DYNAMICS
# #fig = plt.figure()
# ax[2].plot(edc_gamma.Delay.values, 1000*(p_fits_VBM[:,1] - p_fits_VBM[0:10,1].mean()), color = 'grey')
# ax[2].set_xlim([edc_gamma.Delay.values[1], edc_gamma.Delay.values[-1]])
# ax[2].set_ylim([-30,20])
# ax[2].set_xlabel('Delay, fs')
# ax[2].set_ylabel('Energy Shift, meV')
# #plt.axhline(0, linestyle = 'dashed', color = 'black')
# #plt.axvline(0, linestyle = 'dashed', color = 'black')

# # PLOT VBM PEAK WIDTH DYNAMICS
# ax2 = ax[2].twinx()
# ax2.plot(edc_gamma.Delay.values, 1000*(p_fits_VBM[:,2] - 0*p_fits_VBM[0:10,2].mean()), color = 'pink')
# #ax2.set_ylim([-75,50])
# ax2.set_ylabel('Energy Width Shift, meV')

fig.tight_layout()

if save_figure is True:
    fig.savefig((figure_file_name +'.svg'), format='svg', dpi = 300)

#%% Plot VBM Fit Results

figure_file_name = 'EDC_phoibos_fits1'
save_figure = True

fig, ax = plt.subplots(1, 2)
fig.set_size_inches(10, 4, forward=False)
ax = ax.flatten()

# VBM FIT TESTS FOR ONE POINT
t = 15
gauss_test = gaussian(edc_gamma.Energy.values, *p_fits_VBM[t,:])
ax[0].plot(edc_gamma.Energy.values, edc_gamma[:,t].values/edc_gamma.loc[{"Energy":slice(-0.2,0.5)}][:,t].values.max(), color = 'black')
ax[0].plot(edc_gamma.Energy.values, gauss_test, linestyle = 'dashed', color = 'grey')
#plt.axvline(trunc_e, linestyle = 'dashed', color = 'black')
ax[0].set_xlim([-1,1])
ax[0].set_xlabel('E - E$_{VBM}$, eV')
ax[0].set_ylabel('Norm. Int.')
ax[0].axvline(e1, linestyle = 'dashed', color = 'pink')
ax[0].axvline(e2, linestyle = 'dashed', color = 'pink')

# PLOT VBM SHIFT DYNAMICS

t = 39 # Show only after 50 (?) fs
y_vb, y_vb_err = 1000*(p_fits_VBM[:,1] - p_fits_VBM[0:10,1].mean()), 1000*p_err_VBM[:,0]
y_vb_w, y_vb_w_err = 1000*(p_fits_VBM[:,2]),  1000*p_err_VBM[:,1]

ax[1].plot(res.Delay.values, y_vb, color = 'navy', label = '$\Delta E_{VBM}$')
ax[1].fill_between(res.Delay.values, y_vb - y_vb_err, y_vb + y_vb_err, color = 'navy', alpha = 0.5)

ax[1].set_xlim([edc_gamma.Delay.values[1], edc_gamma.Delay.values[-1]])
ax[1].set_ylim([-40,75])
ax[1].set_xlabel('Delay, fs')
ax[1].set_ylabel('$\Delta E_{VBM}$, meV', color = 'navy')
ax[1].legend(frameon=False)

#PLOT VBM PEAK WIDTH DYNAMICS
ax2 = ax[1].twinx()
ax2.plot(res.Delay.values, y_vb_w, color = 'maroon', label = '$\sigma_{VBM}$')
ax2.fill_between(res.Delay.values, y_vb_w - y_vb_w_err, y_vb_w + y_vb_w_err, color = 'maroon', alpha = 0.5)
ax2.set_ylim([160,280])
ax2.legend(frameon=False, loc = 'upper left')
ax2.set_ylabel('${VBM}$ Peak Width, meV', color = 'maroon')

fig.tight_layout()

if save_figure is True:
    fig.savefig((figure_file_name +'.svg'), format='svg')

#%% TEST: CBM EDC Fitting: Extract Binding Energy

E_trace, E_int = [1.35, 2.1], .12 # Energies for Plotting Time Traces ; 1st Energy for MM
delay, delay_int = 50, 50

kx_frame = res.loc[{"Delay":slice(delay-delay_int/2, delay+delay_int/2)}].mean(dim="Delay")
kx_frame = kx_frame - res.loc[{"Delay":slice(-1000,-150)}].mean(dim="Delay")

kx_edc = kx_frame.loc[{"Angle":slice(-12,12)}].sum(dim="Angle")
kx_edc = kx_edc/np.max(kx_edc.loc[{"Energy":slice(0.8,3)}])

##### X and CBM ####
e1 = 1.15
e2 = 3
p0 = [1, 0.3,  1.35, 2.1,  0.2, 0.2, 0] # Fitting params initial guess [amp, center, width, offset]
bnds = ((0.5, 0.1, 1.0, 1.5, 0.1, 0.1, 0), (1.5, 0.7, 1.5, 2.3, 0.9, 0.9, .3))

popt_2, _ = curve_fit(two_gaussians, kx_edc.loc[{"Energy":slice(e1,e2)}].Energy.values, kx_edc.loc[{"Energy":slice(e1,e2)}].values, p0, method=None, bounds = bnds)
g, g1, g2, offset = two_gaussians_report(kx_edc.loc[{"Energy":slice(0,3)}].Energy.values, *popt_2)
Eb = round(popt_2[3] - popt_2[2],2)

fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1], 'height_ratios':[1]})
fig.set_size_inches(10, 4, forward=False)
ax = ax.flatten()

kx_edc.plot(ax=ax[0], color = 'black')
ax[0].plot(kx_edc.loc[{"Energy":slice(0,3)}].Energy.values, g1, color='black',linestyle = 'dashed')
ax[0].plot(kx_edc.loc[{"Energy":slice(0,3)}].Energy.values, g2, color='red',linestyle = 'dashed')
ax[0].plot(kx_edc.loc[{"Energy":slice(0,3)}].Energy.values, g, color='grey',linestyle = 'solid')
ax[0].set_title(f"$\Delta t = {delay}$ fs : $E_B = {1000*Eb}$ meV")
ax[0].set_xlim(0.5,3)
ax[0].set_ylim(0, 1.1)

delay, delay_int = 250, 50

kx_frame = res.loc[{"Delay":slice(delay-delay_int/2, delay+delay_int/2)}].mean(dim="Delay")
kx_frame = kx_frame - res.loc[{"Delay":slice(-1000,-150)}].mean(dim="Delay")

kx_edc = kx_frame.loc[{"Angle":slice(-12,12)}].sum(dim="Angle")
kx_edc = kx_edc/np.max(kx_edc.loc[{"Energy":slice(0.8,3)}])

##### X and CBM ####
e1 = 1.15
e2 = 3
p0 = [1, 0.3,  1.35, 2.1,  0.2, 0.2, 0] # Fitting params initial guess [amp, center, width, offset]
bnds = ((0.5, 0.1, 1.0, 1.5, 0.1, 0.1, 0), (1.5, 0.7, 1.5, 2.3, 0.9, 0.9, .3))

popt_2, _ = curve_fit(two_gaussians, kx_edc.loc[{"Energy":slice(e1,e2)}].Energy.values, kx_edc.loc[{"Energy":slice(e1,e2)}].values, p0, method=None, bounds = bnds)
g, g1, g2, offset = two_gaussians_report(kx_edc.loc[{"Energy":slice(0,3)}].Energy.values, *popt_2)
Eb = round(popt_2[3] - popt_2[2],2)

kx_edc.plot(ax=ax[1], color = 'black')
ax[1].plot(kx_edc.loc[{"Energy":slice(0,3)}].Energy.values, g1, color='black',linestyle = 'dashed')
ax[1].plot(kx_edc.loc[{"Energy":slice(0,3)}].Energy.values, g2, color='red',linestyle = 'dashed')
ax[1].plot(kx_edc.loc[{"Energy":slice(0,3)}].Energy.values, g, color='grey',linestyle = 'solid')
ax[1].set_title(f"$\Delta t = {delay}$ fs : $E_B = {1000*Eb}$ meV")
ax[1].set_xlim(0.5,3)
ax[1].set_ylim(0, 1.1)

#%% Plot Excited State EDC Fits and Binding Energy

figure_file_name = 'EDC_fits_phoibos_excted2'
save_figure = True

fig, ax = plt.subplots(1, 3,  gridspec_kw={'width_ratios': [1, 1.25, 1.25], 'height_ratios':[1]})
fig.set_size_inches(13, 4, forward=False)
ax = ax.flatten()

kx_edc.plot(ax=ax[0], color = 'black')
ax[0].plot(kx_edc.loc[{"Energy":slice(0,3)}].Energy.values, g1, color='black',linestyle = 'dashed')
ax[0].plot(kx_edc.loc[{"Energy":slice(0,3)}].Energy.values, g2, color='red',linestyle = 'dashed')
ax[0].plot(kx_edc.loc[{"Energy":slice(0,3)}].Energy.values, g, color='grey',linestyle = 'solid')
ax[0].set_title(f"$\Delta t = {delay}$ fs : $E_B = {1000*Eb}$ meV")
ax[0].set_xlim(0.5,3)
ax[0].set_ylim(0, 1.1)
ax[0].set_ylabel('Norm. Int.')
ax[0].set_xlabel('$E - E_{VBM}$, eV', color = 'black')

# PLOT CBM and EX SHIFT DYNAMICS
#fig = plt.figure()
t = 11 #Show only after 50 (?) fs
tt = 11
y_ex, y_ex_err = 1*(centers_EX[t:] - 0*centers_EX[-12].mean()), 1*p_err_excited[t:,2]
y_cb, y_cb_err = 1*(centers_CBM[tt:]- 0*centers_CBM[-12].mean()),  1*p_err_excited[tt:,3]

ax[1].plot(res.Delay.values[t:], y_ex, color = 'black', label = 'EX')
ax[1].fill_between(res.Delay.values[t:], y_ex - y_ex_err, y_ex + y_ex_err, color = 'grey', alpha = 0.5)
ax[1].set_xlim([0, edc_gamma.Delay.values[-1]])
#ax[1].set_ylim([1.1,2.3])
ax[1].set_ylim([1.15, 1.4])
ax[1].set_xlabel('Delay, fs')

ax2 = ax[1].twinx()
ax2.plot(res.Delay.values[tt:], y_cb, color = 'red', label = 'CBM')
ax2.fill_between(res.Delay.values[tt:], y_cb - y_cb_err, y_cb + y_cb_err, color = 'pink', alpha = 0.5)
ax2.set_ylim([1.95,2.2])
#ax[1].Energyrrorbar(I.Delay.values[t:], 1*(centers_EX[t:]), yerr = p_err_excited[t:,2], marker = 'o', color = 'black', label = 'EX')
#ax[1].Energyrrorbar(I.Delay.values[t:], 1*(centers_CBM[t:]), yerr = p_err_excited[t:,3], marker = 'o', color = 'red', label = 'CBM')
#ax[1].set_ylim([1.1,2.3])
ax[1].set_ylabel('$E_{EX}$, eV', color = 'black')
ax2.set_ylabel('$E_{CBM}$, eV', color = 'red')
ax[1].set_title(f"From {round(res.Delay.values[t])} fs")
ax[1].legend(frameon=False, loc = 'upper right')
ax2.legend(frameon=False, loc = 'upper left' )

ax[2].plot(res.Delay.values[t:], 1000*Ebs[t:], color = 'purple', label = '$E_{B}$')
ax[2].fill_between(res.Delay.values[t:], 1000*Ebs[t:] - 1000*p_err_eb[t:], 1000*Ebs[t:] + 1000*p_err_eb[t:], color = 'violet', alpha = 0.5)
ax[2].set_xlim([0, edc_gamma.Delay.values[-1]])
ax[2].set_ylim([625,825])
ax[2].set_xlabel('Delay, fs')
ax[2].set_ylabel('$E_{B}$, meV', color = 'black')
ax[2].legend(frameon=False)

# # PLOT VBM PEAK WIDTH DYNAMICS
# ax2 = ax[1].twinx()
# ax2.plot(edc_gamma.Delay.values, 1000*(p_fits_VBM[:,2] - 0*p_fits_VBM[0:10,2].mean()), color = 'maroon')
# #ax2.set_ylim([-75,50])
# ax2.set_ylabel('${VBM}$ Peak Width, meV', color = 'maroon')

fig.tight_layout()

if save_figure is True:
    fig.savefig((figure_file_name +'.svg'), format='svg')


#%%

def plot_band_dynamics(ax):

    # PLOT VBM SHIFT DYNAMICS
    y_vb, y_vb_err = 1000*(p_fits_VBM[:,1] - p_fits_VBM[0:10,1].mean())+offset[i], 1000*(p_err_VBM[:,0])
    y_vb_w, y_vb_w_err = 1000*(p_fits_VBM[:,2]),  1000*p_err_VBM[:,1]
    
    colors = ['black', 'blue', 'purple', 'green', 'orange', 'red']
    #colors = mpl.cm.jet(np.linspace(0,.9,len(scans)))

    ax[0].plot(res.Delay.values, y_vb, color = colors[i], label = '$\Delta E_{VBM}$')
    ax[0].fill_between(res.Delay.values, y_vb - y_vb_err, y_vb + y_vb_err, color = colors[i], alpha = 0.5)
    ax[0].axhline(offset[i], color = 'grey', linestyle = 'dashed')
    ax[0].set_xlim([-20, edc_gamma.Delay.values[-1]])
    ax[0].set_ylim([-20,115])
    ax[0].set_xlabel('Delay, fs')
    ax[0].set_ylabel('$\Delta E_{VBM}$, meV', color = 'navy')
    #ax[0].legend(frameon=False)
    
    # PLOT CBM and EX SHIFT DYNAMICS
    t = 1
    tt = 1
    y_ex, y_ex_err = 1*(centers_EX[t:] - 1*centers_EX[-4:].mean())+0.01*offset[i], 1*p_err_excited[t:,2]
    y_cb, y_cb_err = 1*(centers_CBM[tt:]- 1*centers_CBM[-4:].mean())+0.01*offset[i],  1*p_err_excited[tt:,3]
    
    #colors = mpl.cm.jet(np.linspace(0,.9,len(scans)))

    ax[1].plot(res.Delay.values[t:], y_ex, color = colors[i], label = 'EX')
    ax[1].fill_between(res.Delay.values[t:], y_ex - y_ex_err, y_ex + y_ex_err, color = colors[i], alpha = 0.5)
    ax[1].axhline(0.01*offset[i], color = 'grey', linestyle = 'dashed')
    ax[1].set_xlim([-20, edc_gamma.Delay.values[-1]])
    #ax[1].set_ylim([1.1,2.3])
    ax[1].set_ylim([-0.2,1.2])
    ax[1].set_xlabel('Delay, fs')
    ax[1].set_ylabel('$E_{EX}$, eV', color = 'black')

    #colors = mpl.cm.jet(np.linspace(0,.9,len(scans)))
#    ax2 = ax[1].twinx()
    ax[2].plot(res.Delay.values[tt:], y_cb, color = colors[i], label = 'CBM')
    ax[2].fill_between(res.Delay.values[tt:], y_cb - y_cb_err, y_cb + y_cb_err, color = colors[i], alpha = 0.5)
    ax[2].axhline(0.01*offset[i], color = 'grey', linestyle = 'dashed')
    ax[2].set_ylim([-0.2,1.2])
    ax[2].set_xlim([-20, edc_gamma.Delay.values[-1]])
    ax[2].set_ylabel('$E_{CBM}$, eV', color = 'red')
    #ax[1].set_title(f"From {round(res.Delay.values[t])} fs")
    #ax[1].legend(frameon=False, loc = 'upper right')
    #ax2.legend(frameon=False, loc = 'upper left' )
    
   # colors = mpl.cm.jet(np.linspace(0,.9,len(scans)))
    y_eb =  1000*Ebs[:] - 1000*Ebs[-4:].mean() + 7.5*offset[i]
    ax[3].plot(res.Delay.values[:], y_eb, color = colors[i], label = '$E_{B}$')
    ax[3].fill_between(res.Delay.values[:], y_eb - 1000*p_err_eb[:], y_eb + 1000*p_err_eb[:], color = colors[i], alpha = 0.5)
    ax[3].axhline(7.5*offset[i], color = 'grey', linestyle = 'dashed')
    ax[3].set_xlim([-20, edc_gamma.Delay.values[-1]])
    ax[3].set_ylim([-150,850])
    ax[3].set_xlabel('Delay, fs')
    ax[3].set_ylabel('$E_{B}$, meV', color = 'black')
    #ax[2].legend(frameon=False)
#    plt.show()

#%%

save_figure = True
figure_file_name = 'phoibos_power_fits'

fig, ax = plt.subplots(2, 2)
fig.set_size_inches(12, 8, forward=False)
ax = ax.flatten()

scans = [9219, 9217, 9218, 9216, 9220, 9228]
offset = np.linspace(0,100,6)
i = 0
(kx), k_int = (-3), 4

for scan_i in scans:
    res = load_data(scan_i, energy_offset, delay_offset)
    
    centers_VBM, p_fits_VBM, p_err_VBM = fit_vbm_dynamics(res, kx, 4)
    centers_EX, centers_CBM, Ebs, p_fits_excited, p_err_excited, p_err_eb = fit_ex_cbm_dynamics(res, delay_int)
    
    plot_band_dynamics(ax)
    i += 1
    
fig.tight_layout()

if save_figure is True:
    fig.savefig((figure_file_name +'.svg'), format='svg')


#%%

def fit_vbm_int(res, k, k_int):
    e1 = -.2
    e2 = 0.6
    p0 = [1, 0, .2, 0] # Fitting params initial guess [amp, center, width, offset]
    bnds = ((0.5, -1, 0.0, 0), (1.5, 0.5, 1, .5))
    
    (kx), k_int = k, k_int
    edc_gamma = res.loc[{"Angle":slice(kx-k_int/2,kx+k_int/2), "Delay":slice(0,3000)}].sum(dim=("Angle", "Delay"))
    edc_gamma = edc_gamma/np.max(edc_gamma)
    
    edc_i = edc_gamma.loc[{"Energy":slice(e1,e2)}].values
    edc_i = edc_i/np.max(edc_i)
    
    try:
        popt, pcov = curve_fit(gaussian, edc_gamma.loc[{"Energy":slice(e1,e2)}].Energy.values, edc_i, p0, method=None, bounds = bnds)
    except ValueError:
        print('oops')
        popt = [0,0,0,0]
        
    centers_VBM_i = popt[1]
    p_fits_VBM_i = popt
    perr = np.sqrt(np.diag(pcov))
    p_err_VBM_i = perr[1:2+1]
        
    return centers_VBM_i, p_fits_VBM_i, p_err_VBM_i

##### CBM AND EXCITON #####

def fit_ex_cbm_int(res):
    delay_int = 50
    e1 = 1.1
    e2 = 3
    p0 = [1, 0.3,  1.3, 2.1,  0.2, 0.2, 0] # Fitting params initial guess [amp, center, width, offset]
    bnds = ((0.5, 0.1, 1.0, 1.5, 0.1, 0.1, 0), (1.5, 0.7, 1.5, 2.3, 0.9, 0.9, .3))
    
    centers_CBM = np.zeros(len(res.Delay))
    centers_EX = np.zeros(len(res.Delay))
    Ebs = np.zeros(len(res.Delay))
    
    p_fits_excited = np.zeros((len(res.Delay),7))
    p_err_excited = np.zeros((len(res.Delay),7))
    p_err_eb = np.zeros((len(res.Delay)))

    #kx_frame = res.loc[{"Delay":slice(res.Delay.values[t]-delay_int/2, res.Delay.values[t]+delay_int/2)}].mean(dim="Delay")
    kx_frame = res - res.loc[{"Delay":slice(-1000,-150)}].mean(dim="Delay")

    kx_frame = kx_frame.loc[{"Delay":slice(0,3000)}].mean(dim="Delay")
    kx_edc_i = kx_frame.loc[{"Angle":slice(-12,12)}].sum(dim="Angle")
    kx_edc_i = kx_edc_i/np.max(kx_edc_i.loc[{"Energy":slice(0.8,3)}])
    
    try:
        popt, pcov = curve_fit(two_gaussians, kx_edc_i.loc[{"Energy":slice(e1,e2)}].Energy.values, kx_edc_i.loc[{"Energy":slice(e1,e2)}].values, p0, method=None, bounds = bnds)
    except ValueError:
        print('Oops!')
        popt = [0,0,0,0]
   
    centers_EX_i = popt[2]
    centers_CBM_i = popt[3]
    Eb = round(popt[3] - popt[2],3)
    Ebs_i = Eb
    perr = np.sqrt(np.diag(pcov))
    p_fits_excited_i = popt
    
    p_err_excited_i = perr[2:3+1] 
    p_err_eb_i = np.sqrt(perr[3]**2+perr[2]**2)
        
    return centers_EX_i, centers_CBM_i, Ebs_i, p_fits_excited_i, p_err_excited_i, p_err_eb_i


#%%
save_figure = True
figure_file_name = 'phoibos_power_fits2'

scans = [9219, 9217, 9218, 9216, 9220, 9228]
offset = np.linspace(0,100,6)
i = 0
(kx), k_int = (-3), 4

centers_VBM, p_fits_VBM, p_err_VBM = [], [], []
centers_EX, centers_CBM, Ebs, p_fits_excited, p_err_excited, p_err_eb = [], [], [], [], [], []

for scan_i in scans:
    res = load_data(scan_i, energy_offset, delay_offset)
    
    centers_VBM_i, p_fits_VBM_i, p_err_VBM_i = fit_vbm_int(res, kx, 4)
    centers_EX_i, centers_CBM_i, Ebs_i, p_fits_excited_i, p_err_excited_i, p_err_eb_i = fit_ex_cbm_int(res)
    
    i += 1
    
    centers_VBM.append(centers_VBM_i)
    p_fits_VBM.append(p_fits_VBM_i)
    p_err_VBM.append(p_err_VBM_i)
    
    centers_EX.append(centers_EX_i)
    centers_CBM.append(centers_CBM_i)
    Ebs.append(Ebs_i)
    p_fits_excited.append(p_fits_excited_i)
    p_err_excited.append(p_err_excited_i)
    p_err_eb.append(p_err_eb_i)
    
p_err_eb = np.asarray(p_err_eb)
p_err_excited = np.asarray(p_err_excited)
p_err_VBM = np.asarray(p_err_VBM)

#%%
fig, ax = plt.subplots(1, 2)
fig.set_size_inches(12, 4, forward=False)
ax = ax.flatten()

y1, y_vb_err = 1000*(centers_VBM - centers_VBM[0]), 1000*p_err_VBM[:,0]
y2, y_ex_err = 1000*(centers_EX - centers_EX[0]), 1000*p_err_excited[:,0]
y3, y_cb_err = 1000*(centers_CBM - centers_CBM[0]), 1000*p_err_excited[:,1]
y4, y_eb_err = 1000*(Ebs - Ebs[0]), 1000*p_err_eb

colors = ['grey', 'black', 'red']
i = 0
#ax[0].plot(y1, color = 'grey')
ax[0].errorbar(x = range(0,6), y = y1, yerr = y_vb_err, marker = 'o', color = 'grey', label = 'VBM')
ax[0].errorbar(x = range(0,6), y = y2, yerr = y_ex_err, marker = 'o', color = 'black', label = 'ex')
ax[0].errorbar(x = range(0,6), y = y3, yerr = y_cb_err, marker = 'o', color = 'red', label = 'CBM')
ax[0].axhline(0, color = 'grey', linestyle = 'dashed')
#ax[0].fill_between(y1 - y_vb_err, y1 + y_vb_err, color = colors[i], alpha = 0.5)

ax[1].errorbar(x = range(0,6), y = y4, yerr = y_eb_err, marker = 'o', color = 'purple', label = '$E_{b}$')
ax[1].axhline(0, color = 'grey', linestyle = 'dashed')

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
