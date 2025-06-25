#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 18:12:50 2025

@author: lawsonlloyd
"""

save_figure = False
figure_file_name = '400nm'
image_format = 'pdf'

# Standard 400 nm Excitation
scans = [9525, 9517, 9526]
scans = [9517]
#scans = [9526]
power = [20, 36.3, 45]
fluence = [0.3, 0.54, 0.68]

####
E, E_int = [1.3, 2.05], 0.1
k, k_int = (0), 24
subtract_neg = True
norm_trace = False

fig, axx = plt.subplots(1,2)
fig.set_size_inches(10, 4, forward=False)
axx = axx.flatten()

fluence = np.array(fluence)

custom_colors = ['lightsteelblue', 'royalblue', 'mediumblue'] #colors for plotting the traces


i = 0
for scan_i in scans:
    
    res = phoibos.load_data(data_path, scan_i, scan_info, energy_offset, delay_offset, force_offset)
    trace_1 = phoibos.get_time_trace(res, E[0], E_int, k, k_int, subtract_neg, norm_trace)
    trace_2 = phoibos.get_time_trace(res, E[1], E_int, k, k_int, subtract_neg, norm_trace)
    norm = trace_2.max()
    trace_2 = trace_2/norm
    trace_1 = trace_1/norm

    im1 = res_diff_sum_Angle_Normed.T.plot.imshow(ax = axx[0], cmap = cmap_LTL, vmin = 0, vmax = 1, add_colorbar = False)

    t1 = trace_1.plot(ax = axx[1], color = 'black', linewidth = 3)
    t2 = trace_2.plot(ax = axx[1], color = 'red', linewidth = 3, label = f'{fluence[i]} mJ / cm$^{{2}}$')
    
    #test = exp_rise_monoexp_decay(trace_1.Delay.values, 1.02, 258, 6700)
    #axx[0].plot(trace_1.Delay.values, test, color = 'maroon')

    #test2 = exp_rise_monoexp_decay(trace_1.Delay.values, .8, 400, 6000)
    #test2 = exciton_model(trace_1.Delay.values, 1., 450, 8000)
    #axx[0].plot(trace_1.Delay.values, 0.85*test2/np.max(test2), color = 'green', linestyle = 'dashed')
    
    #test3 = exp_rise_biexp_decay(trace_1.Delay.values, 1, 350, .92, 240, 2500)
    #test4= exp_rise_biexp_decay(trace_1.Delay.values, 1, 250, .9, 300, 4000)
    #test5 = (np.exp(-trace_1.Delay.values/400))*(1-np.exp(-trace_1.Delay.values/300))
    
    #axx[1].plot(trace_1.Delay.values, test3/np.max(test3), color = 'pink')
    #axx[1].plot(trace_1.Delay.values, test4/np.max(test4), color = 'red', linestyle= 'dashed')
    #axx[1].plot(trace_1.Delay.values, test5/np.max(test5), color = 'green', linestyle= 'dashed')

    #t1 = trace_1.plot(ax = axx[1], color = 'royalblue', linewidth = 3)
    #t2 = trace_2.plot(ax = axx[1], color = 'blue', linewidth = 3, label = f'{fluence[i]} mJ / cm$^{{2}}$')

    i += 1

axx[1].set_xticks(np.arange(-1000,3500,500))
for label in axx[1].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)

axx[1].set_yticks(np.arange(-0.5,1.25,0.25))
for label in axx[1].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
    
axx[0].set_xticks(np.arange(-30,30,5))
for label in axx[0].xaxis.get_ticklabels():
    label.set_visible(False)

axx[0].set_yticks(np.arange(-0.5,3.25,0.25))
for label in axx[0].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
    
axx[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
axx[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

axx[0].set_xlim([-12,12])
axx[1].set_xlim([-500,3000])
axx[0].set_ylim([0.7,2.7])
axx[1].set_ylim([-0.1,1.1])

axx[0].set_xlabel('$k_{II}$')
axx[1].set_xlabel('Delay, fs')
axx[0].set_ylabel('Norm. Int.')
axx[1].set_ylabel('Norm. Int.')

axx[0].set_title('Exciton')
axx[1].set_title('CBM')
axx[1].legend(frameon=False)
fig.text(.01, 0.975, "(a)", fontsize = 20, fontweight = 'regular')
fig.text(.5, 0.975, "(b)", fontsize = 20, fontweight = 'regular')

params = {'lines.linewidth' : 3.5, 'axes.linewidth' : 2, 'axes.labelsize' : 20, 
              'xtick.labelsize' : 16, 'ytick.labelsize' : 16, 'axes.titlesize' : 20, 'legend.fontsize' : 16}
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update(params)

fig.tight_layout()

if save_figure is True:
    fig.savefig(figure_file_name + '.'+ image_format, bbox_inches='tight', format=image_format)
