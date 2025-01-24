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

#%%

phoibos = True

data_path = 'path_to_your_data'
filename = 'your_file_name.h5'

data_path = 'R:\Lawson\Data\phoibos'
#data_path = '/Users/lawsonlloyd/Desktop/Data/'

scan = 9539
filename = f"Scan{scan}.h5"

data_loader = DataLoader(data_path + '//' + filename)
res = data_loader.load_phoibos()

energy_offset = + 19.6
res = res.assign_coords(Energy=(res.Energy-energy_offset))

#%%

filename = '2024 Bulk CrSBr Phoibos.csv'

scan_info = {}
data_path = 'R:\Lawson\Data'

reader = csv.DictReader(open(data_path + '//' + filename))

for row in reader:
    key = row.pop('Scan')
    if key in scan_info:
        # implement your duplicate row handling here
        pass
    scan_info[key] = row
    
#print(scan_info)

#%% # PLOT DATA PANEL: Initial Overview

%matplotlib inline

E1, E2, Eint = 1.375, 2.125, 0.1

colormap = 'terrain_r'

WL = scan_info[str(scan)].get("Wavelength")
per = float(scan_info[str(scan)].get("Percent"))
Temp = float(scan_info[str(scan)].get("Temperature"))

fig, axx = plt.subplots(2, 2)
fig.set_size_inches(8, 8, forward=False)
plt.gcf().set_dpi(200)
axx = axx.flatten()

im1 = res.loc[{'Delay':slice(-1000,5000)}].sum(axis=2).T.plot(ax = axx[0], cmap = colormap)
axx[0].set_title(f"Scan{scan}: {WL} nm, {per}%, T = {Temp}")

im2 = res.loc[{'Energy':slice(E1-1,E2+0.8), 'Angle':slice(-12,12)}].sum(axis=0).plot(ax = axx[1], vmax = .3e7, cmap = colormap)
axx[1].axvline(0, color = 'black')

im3 = res.loc[{'Angle':slice(-12,-10), 'Delay':slice(-1000,5000)}].sum(axis=(0,2)).plot(ax = axx[2])
#im3 = res.loc[{'Angle':slice(-12,12), 'Energy':slice(1,2.5), 'Delay':slice(-1000,5000)}].sum(axis=(0,2)).plot(ax = axx[2])
axx[2].axvline(0)

im4 = res.loc[{'Angle':slice(-12,12), 'Energy':slice(E1-Eint, E1+Eint)}].sum(axis=(0,1)).plot(ax = axx[3], color = 'black')
im4 = res.loc[{'Angle':slice(-12,12), 'Energy':slice(E2-Eint, E2+Eint)}].sum(axis=(0,1)).plot(ax = axx[3], color = 'red')
axx[1].axhline(E1, color = 'black')
axx[1].axhline(E2, color = 'red')

params = {'lines.linewidth' : 2, 'axes.linewidth' : 1.5, 'axes.labelsize' : 16, 
          'xtick.labelsize' : 14, 'ytick.labelsize' : 14, 'axes.titlesize' : 16, 'legend.fontsize' : 12}
plt.rcParams.update(params)

fig.tight_layout()
plt.show()

#fig.savefig('Scan' + str(scan_to_plot) + '.svg')

#%% # PLOT THREE PANEL DIFFERENCE

%matplotlib inline

E1, E2, Eint = 1.375, 2.125, 0.1

colormap = 'terrain_r'

fig, axx = plt.subplots(1, 3)
fig.set_size_inches(8, 3, forward=False)
plt.gcf().set_dpi(200)
axx = axx.flatten()

res_neg = res.loc[{'Delay':slice(-1000,-300)}]
res_pos = res.loc[{'Delay':slice(0,5000)}]

neg_length = res_neg.shape[2]
pos_length = res_pos.shape[2]

res_neg = res_neg.sum(axis=2)/neg_length
res_pos = res_pos.sum(axis=2)/pos_length

res_diff = res_pos - res_neg
res_diff_angle = res_diff/np.max(res_diff)

E_inset = 0.75
im1 = res_neg.T.plot(ax = axx[0], cmap = colormap)
im2 = res_pos.T.plot(ax = axx[1], cmap = colormap)
#im3 = (res_diff_angle).T.plot(ax = axx[2], cmap = 'seismic', vmin = -1, vmax = 1)
d1 = (res_diff_angle.loc[{'Energy':slice(-1,E_inset)}])
d2 = (10*res_diff_angle.loc[{'Energy':slice(E_inset,3)}])
d3 = xr.concat([d1, d2], dim = "Energy")
d3.T.plot(ax = axx[2], cmap = 'seismic', vmin = -1, vmax = 1)
axx[2].axhline(E_inset, color = 'black', linewidth = 1)
axx[2].set_title(f"Scan{scan}")

fig.tight_layout()
plt.show()

############
############
res_neg = res.loc[{'Delay':slice(-1000,-300)}]
res_pos = res.loc[{'Delay':slice(0,5000)}]

neg_length = res_neg.shape[2]
pos_length = res_pos.shape[2]

#res_neg = res_neg/neg_length
res_pos = res_pos/pos_length

res_diff = res - res_neg.mean(axis=2)

res_diff = res_diff.loc[{'Angle':slice(-10,10)}].sum(axis=0)
res_diff_delay = res_diff/np.max(res_diff)

fig, axx = plt.subplots(1, 2)
fig.set_size_inches(8, 3, forward=False)
plt.gcf().set_dpi(200)
axx = axx.flatten()

#im1 = res_diff_angle.T.plot(ax = axx[0], cmap = 'seismic', vmin = -1, vmax = 1)
im2 = res_diff_delay.plot(ax = axx[0], cmap = 'seismic', vmin = -1, vmax = 1)

trace_1 = res.loc[{'Angle':slice(-12,12), 'Energy':slice(E1-Eint, E1+Eint)}].sum(axis=(0,1))
trace_2 = res.loc[{'Angle':slice(-12,12), 'Energy':slice(E2-Eint, E2+Eint)}].sum(axis=(0,1))
trace_1 = trace_1 - trace_1.loc[{'Delay':slice(-1000,-300)}].mean()
trace_2 = trace_2 - trace_2.loc[{'Delay':slice(-1000,-300)}].mean()

trace_2 = trace_2/trace_1.max()
trace_1 = trace_1/trace_1.max()

im4 = trace_1.plot(ax = axx[1], color = 'black')
im4 = trace_2.plot(ax = axx[1], color = 'red')
axx[0].axhline(E1,  color = 'black')
axx[0].axhline(E2,  color = 'red')
axx[0].set_ylim(-0.25,2.5)
axx[1].set_xlim(res.Delay[0], res.Delay[-1])
axx[0].set_title(f"Scan{scan}")
axx[1].set_title(f"{WL} nm, {per}%, T = {Temp}")

fig.tight_layout()
plt.show()

#%%
