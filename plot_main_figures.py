#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 18:30:21 2025

@author: lawsonlloyd
"""
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

import phoibos
import mpes

#%% Useful Script Portions for Main Text Figures

#%% Figure 1: Momentum Maps

E, E_int = [1.25, 2.05], 0.2
E, E_int = [1.33, 2.14], 0.2
titles = ['Exciton', 'CBM']
temp = 120 

delays, delay_int = [500, 500], 1000 #Integration range for delays

#######################

%matplotlib inline

figure_file_name = f'MMs_RT_posdelays' 
save_figure = True
image_format = 'pdf'

#cmap_plot = viridis_white
I_frame, cmap_plot, scale = I_diff, cmap_LTL2, [-1, 1]
I_frame, cmap_plot, scale = I_res, cmap_LTL, [0, 1]

#cmap_plot = 'turbo'

fig, ax = plt.subplots(1, 2, squeeze = False, sharey=False)
fig.set_size_inches(6, 5, forward=False)
plt.gcf().set_dpi(400)
ax = ax.flatten()

norm_frame_ex = I_res.loc[{"E":slice(1.25-E_int/2,1.25+E_int/2), "delay":slice(75,125)}].mean(dim=("E")).max().values
norm_frame_cbm = I_res.loc[{"E":slice(2.05-E_int/2,2.05+E_int/2), "delay":slice(75,125)}].mean(dim=("E")).max().values
norm_all = I_res.loc[{"E":slice(1,3)}].max()
    
frame_norms = []
for i in np.arange(np.max([len(E), len(delays)])):
    if len(delays) == 1:
        time_delay = delays[0]
    else:
        time_delay = delays[i]

    if len(E) == 1:
        energy = E[0]
    else:
        energy = E[i]
        
    frame = mpes.get_momentum_map(I_frame, energy, E_int, time_delay, delay_int)
#    frame = I_res.loc[{"E":slice(E[i]-E_int/2,E[i]+E_int/2), "delay":slice(100,140)}].mean(dim=("E","delay")).T

    if i in [0, 1, 2]:
        frame_norm = norm_frame_ex
    elif i in [3, 4, 5]:
        frame_norm = norm_frame_cbm

    frame = frame/frame.max()
    frame_norms.append(np.max(frame.values))
    
    im = frame.plot.imshow(ax = ax[i], vmin = scale[0], vmax = scale[1], cmap = cmap_plot, add_colorbar=False)
    ax[i].set_aspect(1)
    ax[i].set_xlim(-2,2)
    ax[i].set_ylim(-2,2)
    
    ax[i].set_xticks(np.arange(-2,2.2,1))
    for label in ax[i].xaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
       
    ax[i].set_yticks(np.arange(-2,2.1,1))
    for label in ax[i].yaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
    ax[i].set_xlabel('$k_x$, $\AA^{-1}$', fontsize = 16)
    ax[i].set_ylabel('$k_y$, $\AA^{-1}$', fontsize = 16)

    ax[i].set_title(f"$\Delta$t = {time_delay} fs", fontsize = 16)
    ax[i].set_title(titles[i], fontsize = 16)

    ax[i].tick_params(axis='both', labelsize=14)
    ax[i].text(-1.85, 1.55,  f"E = {energy:.2f} eV", size=12)
    ax[i].text(-1.85, -1.85,  f'T = {temp} K', size=12)

#    ax[i].set_ylabel("")  # Removes the y-axis label

#ax[0].set_ylabel('$k_y$, $\AA^{-1}$', fontsize = 18)
#ax[3].set_ylabel('$k_y$, $\AA^{-1}$', fontsize = 18)

cbar_ax = fig.add_axes([1, 0.33, 0.025, 0.35])
cbar = fig.colorbar(im, cax=cbar_ax, ticks = [0,1])
cbar.ax.set_yticklabels(['min', 'max'])  # vertically oriented colorbar

fig.text(.03, 0.75, "(a)", fontsize = 18, fontweight = 'regular')
# fig.text(.03, 0.5, "(d)", fontsize = 18, fontweight = 'regular')
fig.text(.525, 0.75, "(b)", fontsize = 18, fontweight = 'regular')
# fig.text(.36, 0.5, "(e)", fontsize = 18, fontweight = 'regular')
# fig.text(.69, 0.975, "(c)", fontsize = 18, fontweight = 'regular')
# fig.text(.69, 0.5, "(f)", fontsize = 18, fontweight = 'regular')

plt.rcParams['svg.fonttype'] = 'none'
fig.tight_layout()
 
#fig = mpes_helper.plot_momentum_maps(I, E, E_int, delays, delay_int, cmap_plot)

#add_bz(fig, x, y)

# y, x = np.pi/4.76, np.pi/3.52

# for r in np.arange(1,2,2):
    
#     rect = (Rectangle((0-r*x, 0-y), 2*x, 2*y, linewidth=1.5,\
#                          edgecolor='k', facecolor='None'))
#     #rect2 = (Rectangle((0-r*x, 0+y), 2*x, 2*y, linewidth=1.5,\
#                           #   edgecolor='k', facecolor='None'))
#     fig.axes[0].add_patch(rect)

# fig.axes[0].plot(0,0, 'ok', markersize = 4)
# fig.axes[0].plot(0,2*y, 'ok', markersize = 4)
# fig.axes[0].plot(0,-2*y, 'ok', markersize = 4)
# fig.axes[0].plot(2*x, 0,  'ok', markersize = 4)
# fig.axes[0].plot(-2*x, 0, 'ok', markersize = 4)

# fig.axes[0].plot(x, 0, 'k', marker = '_', markersize = 6)
# fig.axes[0].plot(0, y, 'k', marker = '|', markersize = 8)

#fig.axes[0].set_xlim(-3,3)

if save_figure is True:
    fig.savefig(figure_file_name + '.'+ image_format, bbox_inches='tight', format=image_format) 

#%% Figure 2: kx Panel + Dynamics

save_figure = False
figure_file_name = 'Figure 2'
image_format = 'svg'

E, E_int = [1.25, 2.05], .2 # Energies for Plotting Time Traces ; 1st Energy for MM

(kx, ky), (kx_int, ky_int) = (0, 0), (0.5, .5) # Central (kx, ky) point and k-integration
delay, delay_int = 500, 1000 #kx frame

Ein = .9 #Enhance excited states above this Energy, eV
energy_limits = [0.9, 2.75] # Energy Y Limits for Plotting Panels 3 and 4
plot_symmetry_points = True
colors = ['black', 'red', 'purple'] #colors for plotting the traces

subtract_neg, neg_delays = True, [-110,-70] #If you want to subtract negative time delay baseline
norm_trace = False

#######################
### Do the Plotting ###
#######################

i = 0
frame = mpes.get_momentum_map(I, E_MM, E_MM_int, 500, 2000)

fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1.5], 'height_ratios':[1]})
fig.set_size_inches(10, 4, forward=False)
ax = ax.flatten()

### kx Frame
kx_norm_frame = mpes.get_kx_E_frame(I_res, ky, ky_int, 100, delay_int)
kx_norm_frame = I_res.loc[{"ky":slice(ky-ky_int/2, ky+ky_int/2), "delay":slice(100-50/2, 100+50/2)}].mean(dim="ky").mean(dim="delay")
#kx_norm_frame = I_res.loc[{"ky":slice(ky-ky_int/2, ky+ky_int/2)}].mean(dim="ky")

kx_frame = mpes.get_kx_E_frame(I_res, ky, ky_int, delay, delay_int)

kx_frame = mpes.get_kx_E_frame(I_diff, ky, ky_int, delay, delay_int)
scale = [-1, 1]
cmap_plot = cmocean.cm.balance
kx_frame = kx_frame/np.max(kx_frame.loc[{"E":slice(Ein,3)}])

im2 = kx_frame.T.plot.imshow(ax=ax[0], cmap=cmap_plot, add_colorbar=False, vmin=scale[0], vmax=scale[1]) #kx, ky, t
ax[0].set_aspect(1)
ax[0].set_xticks(np.arange(-2,2.2,1))
for label in ax[0].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
ax[0].set_yticks(np.arange(-2,4.1,.25))
for label in ax[0].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax[0].set_xlabel('$k_x$, $\AA^{-1}$', fontsize = 18)
ax[0].set_ylabel('$E - E_{VBM}, eV$', fontsize = 18)
#    ax[0].set_title(f'$k_y$ = {ky} $\pm$ {ky_int/2} $\AA^{{-1}}$', fontsize = 18)
ax[0].set_title( f"$\Delta$t = {delay} fs", fontsize = 18)
ax[0].tick_params(axis='both', labelsize=16)
ax[0].set_xlim(-2,2)
ax[0].set_ylim(energy_limits[0],energy_limits[1])
#    ax[0].text(-1.9, energy_limits[1]-0.3,  f"$\Delta$t = {delay} $\pm$ {delay_int/2:.0f} fs", size=14)
ax[0].axhline(Ein, linestyle = 'dashed', color = 'black', linewidth = 1)

if plot_symmetry_points is True:
    ax[0].axvline(0, linestyle = 'dashed', color = 'pink', linewidth = 1.5)
    ax[0].axvline(X, linestyle = 'dashed', color = 'pink', linewidth = 1.5)
    ax[0].axvline(-X, linestyle = 'dashed', color = 'pink', linewidth = 1.5)
    ax[0].axvline(2*X, linestyle = 'dashed', color = 'pink', linewidth = 1.5)
    ax[0].axvline(-2*X, linestyle = 'dashed', color = 'pink', linewidth = 1.5)
#ax[0].fill_between(I.E.values, I.kx.values - kx_int/2, I.kx.values + kx_int/2, color = 'pink', alpha = 0.5)
ax[0].set_aspect("auto")
    
### Delay Traces
trace_norms = []
for i in np.arange(len(E)):
    
    trace = mpes.get_time_trace(I, E[i], E_int, (kx, ky), (kx_int, ky_int), norm_trace, subtract_neg, neg_delays)
    trace_norms.append(np.max(trace))

    if i == 0:
        trace = trace/np.max(trace)
    elif i == 1:
        trace = trace/trace_norms[0]
    
    print(trace.max())
    trace.plot(ax = ax[1], color = colors[i], label = str(E[i]) + ' eV')
    
    rect2 = (Rectangle((kx-kx_int/2, E[i]-E_int/2), kx_int, E_int , linewidth=.5,\
                         edgecolor=colors[i], facecolor=colors[i], alpha = 0.5))
#    if kx_int < 4:
    ax[0].add_patch(rect2) #Add rectangle to plot

ax[1].set_xlim(I.delay[1], I.delay[-1])

if norm_trace is True:
    ax[1].set_ylim(-0.1, 1.1)
else:
    ax[1].set_ylim(-0.1, 1.1*np.max(1))
    
ax[1].set_ylabel('Norm. Int.', fontsize = 18)
ax[1].set_xlabel('Delay, fs', fontsize = 18)
ax[1].legend(frameon = False)
ax[1].set_title('Delay Traces')
ax[1].set_xticks(np.arange(-400,1400,200))
for label in ax[1].xaxis.get_ticklabels()[1::2]:
    label.set_visible(True)
ax[1].set_yticks(np.arange(0,1.2,.25))
for label in ax[1].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
ax[1].set_xlim(np.max([-150,I.delay.values[1]]),I.delay.values[-1])   
ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
fig.text(.03, 0.975, "(a)", fontsize = 18, fontweight = 'regular')
fig.text(.44, 0.975, "(b)", fontsize = 18, fontweight = 'regular')

params = {'lines.linewidth' : 2.5, 'axes.linewidth' : 2, 'axes.labelsize' : 20, 
              'xtick.labelsize' : 16, 'ytick.labelsize' : 16, 'axes.titlesize' : 20, 'legend.fontsize' : 16}
plt.rcParams['svg.fonttype'] = 'none'
new_rc_params = {'text.usetex': False, "svg.fonttype": 'none'}
plt.rcParams.update(new_rc_params)
plt.rcParams.update(params)

fig.tight_layout()
#ax[1].set_tick_params(axis='both', labelsize=16)
#plt.gca().set_aspect(200)

if save_figure is True:
    fig.savefig(figure_file_name + '.'+ image_format, bbox_inches='tight', format=image_format) 
    
#%%

edc = I_res.loc[{"delay":slice(delay-delay_int/2,delay+delay_int/2), "ky":slice(ky-ky_int/2,ky+ky_int/2), "kx":slice(-2,2)}].sum(dim=("ky","kx","delay"))
edc = edc/np.max(edc.loc[{"E":slice(energy_limits[0],energy_limits[1])}])

edc.plot()
plt.xlim(0.75,2.75)
plt.ylim(0,1.2)


#%% Figure 3: Fluence and Wavelength Dependence.

%matplotlib inline

save_figure = False
figure_file_name = 'Figure 3_v2'
image_format = 'svg'
data_path = 'R:\Lawson\Data\phoibos'

# Scans to plot
# Standard 915 nm Excitation
scans = [9219, 9217, 9218, 9216, 9220, 9228]

# Combined
scans = [9219, 9218, 9228, 9370, 9412, 9526] #915 nm (top 3) ; 700 nm, 640 nm, 400 nm

#power = 1.05*np.asarray([153, 111, 91, 66, 47, 32, 15, 10, 8, 5])
fluence = [.2, .35, .8, 1.74, 2.4, 2.9]
fluence = [.2, .8, 2.9, 4.5, 3.6, 0]
          
# Specify energy and Angle ranges
E, E_int = [1.325, 2.075], 0.1
E, E_int = [1.32, 2.05], 0.1

k, k_int = 0, 20
subtract_neg = True
norm_trace = False

# Plot
fig, ax = plt.subplots(2, 3)
fig.set_size_inches(12, 6, forward=False)
plt.gcf().set_dpi(300)
ax = ax.flatten()

for i in np.arange(len(scans)):
    scan_i = scans[i]
    res = phoibos.load_data(data_path, scan_i, scan_info, _ , _ , False)
    WL = scan_info[str(scan_i)].get("Wavelength")

#    res = phoibos.load_data(scan_i, energy_offset, offsets_t0[i])
    trace_1 = phoibos.get_time_trace(res, E[0], E_int, k, k_int, subtract_neg, norm_trace)
    trace_2 = phoibos.get_time_trace(res, E[1], E_int, k, k_int, subtract_neg, norm_trace)
    norm = np.max([trace_1,trace_2])
    #trace_2 = trace_2/np.max(trace_1)
    #trace_1 = trace_1/np.max(trace_1)
    trace_2 = trace_2/norm
    trace_1 = trace_1/norm
    
    t1 = trace_1.plot(ax = ax[i], color = 'black', linewidth = 3)
    t2 = trace_2.plot(ax = ax[i], color = 'crimson', linewidth = 3)
    #ax[i].text(1000, .9, f"$n_{{eh}} = {fluence[i]:.2f} x 10^{{13}} cm^{{-2}}$", fontsize = 14, fontweight = 'regular')

    # Set major ticks at every 500
    xticks = np.arange(-1000, 3500, 500)
    ax[i].set_xticks(xticks)
    
    # Hide every other label by replacing with an empty string
    xtick_labels = [str(tick/1000) if i % 2 == 0 else "" for i, tick in enumerate(xticks)]
    ax[i].set_xticklabels(xtick_labels)
    ax[i].set_yticks(np.arange(-0.5,1.25,0.25))
    for label in ax[i].yaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
    ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax[i].set_xlim([-500,3000])
    ax[i].set_ylim([-0.1,1.1])
    ax[i].set_xlabel('Delay, ps')
    ax[i].set_ylabel('Nom. Int.')
    ax[i].set_title(f'$\lambda_{{ex}} $ = {WL} nm', fontsize = 22)
    #ax[0].set_title('Exciton')
    #ax[1].set_title('CBM')

fig.text(.01, 1, "(a)", fontsize = 20, fontweight = 'regular')
fig.text(.335, 1, "(b)", fontsize = 20, fontweight = 'regular')
fig.text(.665, 1, "(c)", fontsize = 20, fontweight = 'regular')
fig.text(.01, .52, "(d)", fontsize = 20, fontweight = 'regular')
fig.text(.335, .52, "(e)", fontsize = 20, fontweight = 'regular')
fig.text(.665, .52, "(f)", fontsize = 20, fontweight = 'regular')

# Add colorbar for the discrete color mapping
# sm = cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])  # Required for colorbar to work
# cbar_ax = fig.add_axes([1, 0.17, 0.02, 0.75])
# cbar = fig.colorbar(sm, cax=cbar_ax, ticks=midpoints)
# cbar.set_label("$n_{eh}$ ($x$10$^{13}$ cm$^{-2})$")
# cbar.ax.set_yticklabels([f"{f:.2f}" for f in fluence])  # Format tick labels

params = {'lines.linewidth' : 3.5, 'axes.linewidth' : 2, 'axes.labelsize' : 20, 
              'xtick.labelsize' : 16, 'ytick.labelsize' : 16, 'axes.titlesize' : 20, 'legend.fontsize' : 16}
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update(params)

fig.tight_layout()

if save_figure is True:
    fig.savefig(figure_file_name + '.'+ image_format, bbox_inches='tight', format=image_format)

#%%

