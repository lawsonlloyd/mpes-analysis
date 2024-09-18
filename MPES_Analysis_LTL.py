#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 16:45:17 2023

@author: lawsonlloyd
"""

#%%

import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from scipy import ndimage, misc
from LoadData import LoadData
from generate_custom_colormap import custom_colormap

#%% Load File in your path...

#fn = 'your_data_file.h5'
I, ax_kx, ax_ky, ax_E, ax_delay = LoadData(fn)

E_offset = 0.2
delay_offset = 0

#%%
### Transform Data, axes if needed...

E_offset = +0#ML Shuo Data
delay_offset = 0

E_offset = +0.8 # 0.65 Scan 160
delay_offset = 0

E_offset = 0.25 #Scan 163
delay_offset = 100

E_offset = -0.3 #Scan 162
delay_offset = 85

E_offset = +7.25
delay_offset = 85

#E_offset = -0.2 #Scan 188
#delay_offset = +60

#E_offset = 0.0 #Scan 062

#E_offset = +0.5 #Scan 138
#delay_offset = 0

#%%
### User Inputs

points = [(-1.8, 0.05),(1, 0.1)] #starting points for the ROI to plot dynamics and traces

tMap_E, tint_E = [1.55, 1.2], 0.08 #Energy and E integration for traces
xint_k, yint_k = 0.5, 0.5 #Total momentum integration range for kx, ky

ylim_E = -3.25 #Energy minimum for k cut plots

mask_start = 0.8

#%%

%matplotlib auto

#########################
### Interactive Plot! ###
#########################

ax_E_offset = ax_E + E_offset 
ax_delay_offset = ax_delay + delay_offset

t0 = (np.abs(ax_delay_offset - 0)).argmin()
mask_start = (np.abs(ax_E_offset - mask_start)).argmin() #Enhanced CB

cmap_to_use = 'terrain_r'
cmap_to_use = custom_colormap('Lawson', 'viridis')
#cmap_to_use = 'bone_r'

global new_x, new_y

points = np.array(points)
tMap_E = np.array(tMap_E)

x = [0,0]
y = [0,0]
t = [0,0]

xf = [0,0]
yf = [0,0]
xi = [0,0]
yi = [0,0]

x[0] = (np.abs(ax_kx - (points[0,0]-(xint_k/2)))).argmin()
y[0] = (np.abs(ax_ky - (points[0,1]-(yint_k/2)))).argmin()

x[1] = (np.abs(ax_kx - (points[1,0]-(xint_k/2)))).argmin()
y[1] = (np.abs(ax_ky - (points[1,1]-(yint_k/2)))).argmin()

t[0] = (np.abs(ax_E_offset - (tMap_E[0]-(tint_E/2)))).argmin()
t[1] = (np.abs(ax_E_offset - tMap_E[1])).argmin()
E = t[0]

t0 = (np.abs(ax_delay_offset - 0)).argmin()

if I.ndim > 3:
    dt = ax_delay_offset[1] - ax_delay_offset[0]
else:
    dt = 1
    
dE = (ax_E_offset[1] - ax_E_offset[0])
dkx = (ax_kx[1] - ax_kx[0])
dky = (ax_ky[1] - ax_ky[0])

Eint = int(np.round(tint_E/dE))
tint = int(Eint)
xint = np.round(xint_k/dkx)
yint = np.round(yint_k/dky) 
xint, yint = int(xint), int(yint)

##########################################
############### Plotting #################
##########################################

#fig, ax = plt.subplots(nrows = 2, ncols=3, gridspec_kw={'width_ratios': [1, 1.25, 1.25], 'height_ratios':[1.25, 1]})
fig, ax = plt.subplots(nrows = 2, ncols=2, gridspec_kw={'width_ratios': [1, 1], 'height_ratios':[1, 1]})

fig.set_size_inches(15, 10, forward=False)
ax = ax.flatten()
#fig.tight_layout()

### First Panel
extentImage = [ax_kx[0], ax_kx[-1], ax_ky[0], ax_ky[-1]]

firstPanel = I[:,:,E:(E+Eint),:].sum(axis=(2,3))

im_1 = ax[0].imshow(np.transpose(firstPanel), origin='lower', cmap=cmap_to_use, clim=None, interpolation='none', extent=extentImage) #kx, ky, t
line_horizontal = ax[0].axhline(points[0,1], color='black', linestyle = 'dashed')
line_vertical = ax[0].axvline(points[0,0], color='black', linestyle = 'dashed')

line_horizontal_2 = ax[0].axhline(points[1,1], color='purple', linestyle = 'dashed')
line_vertical_2 = ax[0].axvline(points[1,0], color='purple', linestyle = 'dashed')

# Initial square properties

square_color = 'black'
x_center, y_center = (points[0])
new_x, new_y = x_center, y_center
half_length = xint_k / 2

# Define the coordinates of the square

def make_square(center, half_length_x, half_length_y) :
    
    x_center = center[0]
    y_center = center[1]
    
    square_x = [x_center - half_length_x, x_center + half_length_x, x_center + half_length_x, x_center - half_length_x, x_center - half_length_x]
    square_y = [y_center - half_length_y, y_center - half_length_y, y_center + half_length_y, y_center + half_length_y, y_center - half_length_y]
    
    return square_x, square_y

square_x, square_y = make_square(points[0], half_length, half_length)
square, = ax[0].plot(square_x, square_y, color='black', linewidth = 1, linestyle='dashed')

square_x, square_y = make_square(points[1], half_length, half_length)
square_2, = ax[0].plot(square_x, square_y, color='purple', linewidth = 1, linestyle='dashed')
    
#plot_square(center, side_length, square_color)

ax[0].set_xlim(-2.1, 2.1)
ax[0].set_ylim(-2.1, 2.1)
ax[0].set_xlabel('$k_x$', fontsize = 18)
ax[0].set_ylabel('$k_y$', fontsize = 18)
ax[0].set_title('Momentum Map', fontsize = 20)
ax[0].tick_params(axis='both', labelsize=16)
#ax[0].annotate(('E = '+ str(round(ax_E_offset[t[0]],2)) + ' eV'), xy = (-1.9, 1.8), fontsize = 16, weight = 'bold')
ax[0].set_aspect(1)

#rect = ax[0].add_patch(Rectangle((points[0,0]-xint_k/2, points[0,1]-yint_k/2), xint_k, yint_k, edgecolor='red',facecolor='none',lw=2))
#ax[0].add_patch(Rectangle((points[1,0]-xint_k/2, points[1,1]-yint_k/2), xint_k, yint_k, edgecolor='blue',facecolor='none',lw=2))
###
    
slice_E_k_1 = I[:,y[0]:y[0]+yint,:,:].sum(axis=(1,3))
slice_E_k_2 = I[x[0]:x[0]+xint,:,:,:].sum(axis=(0,3))

slice_E_k_1 = slice_E_k_1/np.max(slice_E_k_1)
slice_E_k_2 = slice_E_k_2/np.max(slice_E_k_2)
slice_E_k_1[:,mask_start:] *= 1/np.max(slice_E_k_1[:,mask_start:])
slice_E_k_2[:,mask_start:] *= 1/np.max(slice_E_k_2[:,mask_start:])

#line_cut_x_ind = (np.abs(ax_kx - line_cut_x)).argmin()
#line_cut_y_ind = (np.abs(ax_ky - line_cut_y)).argmin()
#line_cut_t_ind = (np.abs(ax_E_offset - E_AOI)).argmin()

### Second and Third Panels
x_i = int(2)
y_i = int(3)
im_2 = ax[x_i].imshow(np.transpose(slice_E_k_1), origin='lower', cmap=cmap_to_use, clim=None, interpolation='none', extent=[ax_ky[0],ax_ky[-1],ax_E_offset[0], ax_E_offset[-1] ]) #kx, ky, t
im_3 = ax[y_i].imshow(np.transpose(slice_E_k_2), origin='lower', cmap=cmap_to_use, clim=None, interpolation='none', extent=[ax_kx[0],ax_kx[-1],ax_E_offset[0], ax_E_offset[-1] ]) #kx, ky, t
#lhor = ax[1].axhline(tMap_E[0],color='red')
line_horizontal_E = ax[x_i].axhline(tMap_E[0],color='black', linestyle = 'dashed')
line_horizontal_E_2 = ax[y_i].axhline(tMap_E[0],color='black', linestyle = 'dashed')

line_horizontal_E_3 = ax[x_i].axhline(tMap_E[1],color='purple', linestyle = 'dashed')
line_horizontal_E_4 = ax[y_i].axhline(tMap_E[1],color='purple', linestyle = 'dashed')

#lhor_E2 = ax[1].axhline(tMap_E[1],color='blue')
#lhor2_E2 = ax[2].axhline(tMap_E[1],color='blue')
kx_ver_1 = ax[x_i].axvline(points[0,0], color='black', linestyle = 'dashed')
ky_ver_2 = ax[y_i].axvline(points[0,1], color='black', linestyle = 'dashed')

kx_ver_1_2 = ax[x_i].axvline(points[1,0], color='purple', linestyle = 'dashed')
ky_ver_2_2 = ax[y_i].axvline(points[1,1], color='purple', linestyle = 'dashed')

half_length_E = tint_E / 2

square_x, square_y = make_square([points[0,0], tMap_E[0]], half_length, half_length_E)
#square_3, = ax[1].plot(square_x, square_y, color='black', linewidth = 1, linestyle='--')

square_x, square_y = make_square([points[1,0], tMap_E[1]], half_length, half_length_E)

ax[x_i].set_xticks(np.arange(-3,3.1,0.5))
for label in ax[x_i].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
    
ax[x_i].set_yticks(np.arange(-5,3.25,0.5))
for label in ax[x_i].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)

ax[y_i].set_xticks(np.arange(-3,3.1,0.5))
for label in ax[y_i].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)    

ax[y_i].set_yticks(np.arange(-5,3.25,0.5))
for label in ax[y_i].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)

ax[x_i].set_xlim(-2,2)
ax[y_i].set_xlim(-2,2)
ax[x_i].set_ylim(ylim_E, 3.1)
ax[y_i].set_ylim(ylim_E, 3.1)
ax[x_i].set_xlabel('$k_x$', fontsize = 18)
ax[y_i].set_xlabel('$k_y$', fontsize = 18)
ax[x_i].set_ylabel('E - E$_{VBM}$, eV', fontsize = 18)
ax[y_i].set_ylabel('E - E$_{VBM}$, eV', fontsize = 18)
ax[x_i].tick_params(axis='both', labelsize=14)
ax[y_i].tick_params(axis='both', labelsize=14)
ax[x_i].set_aspect(0.8)
ax[y_i].set_aspect(0.8)

t_i = int(1)

if np.ndim(I) > 3:
    #trace1 = I_Rot[].sum()
    trace1 = I[x[0]:x[0]+xint,y[0]:y[0]+yint,t[0]:t[0]+tint,:].sum(axis=(0,1,2))
    trace2 = I[x[1]:x[1]+xint,y[1]:y[1]+yint,t[1]:t[1]+tint,:].sum(axis=(0,1,2))
    
    trace1 = trace1 - np.mean(trace1[2:12])
    trace1 = trace1/np.max(trace1)
    trace2 = trace2 - np.mean(trace2[2:12])
    trace2 = trace2/np.max(trace2)
    
    im4, = ax[t_i].plot(ax_delay_offset, trace1, color = 'black', label = 'E = ' + str(tMap_E[0]) + ' eV')
    im4_2, = ax[t_i].plot(ax_delay_offset, trace2, color = 'purple', label = 'E = ' + str(tMap_E[1]) + ' eV')
        
    #ax[3].plot(ax_delay_offset, trace2, color ='blue',label = 'E = ' + str(tMap_E[1]) + ' eV')
    ax[t_i].set_yticks([-0.25, 0, 0.25, 0.5, 0.75, 1, 1.25])
    ax[t_i].set_yticklabels(['', '0', '', '0.5', '', '1', 1.25])
    
    ax[t_i].set_ylim(-0.25, 1.1)
    #ax[4].set_ylim(-1.1,1.1)
    ax[t_i].set_title('Dynamics', fontsize = 20)
    ax[t_i].set_ylabel('Int.', fontsize = 18)
    ax[t_i].tick_params(axis='both', labelsize=16)
    ax[t_i].set_xlabel('Delay, fs', fontsize = 18)
    ax[t_i].set_aspect('auto')
    ax[t_i].set_xticks(np.arange(-500,1500,250))
    for label in ax[t_i].xaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
    ax[t_i].set_xlim(ax_delay_offset[1],ax_delay_offset[-2])

    #ax[t_i].legend(frameon = False)

    #ax[4].set_yticks() 
    # setting ticks for y-axis 
    
    #ax[2].axis('off')
    #ax[5].axis('off')
else:
    ax[1].axis('off')
    #ax[5].axis('off')
    #ax[4].axis('off')

#######################################################
### Define and Implement and Interactive Components ###
#######################################################

# Defining the Slider button
# xposition, yposition, width and height
E_slide = plt.axes([0.045, 0.6, 0.03, 0.25])
E_slide_2 = plt.axes([0.075, 0.6, 0.03, 0.25])
delay_slide = plt.axes([0.8, 0.475, 0.15, 0.03])
delay_integration = plt.axes([0.8, 0.44, 0.15, 0.03])

delay_button = plt.axes([0.75, 0.38, 0.15, 0.05])
diff_box = CheckButtons(delay_button, ['Pos-Neg Delay Diff'])

# Properties of the slider
E_factor = Slider(E_slide, 'E, eV', -4, 3.5, valinit=tMap_E[0], valstep=0.025, color = 'black', orientation = 'vertical')
E_factor_2 = Slider(E_slide_2, 'E, eV', -4, 3.5, valinit=tMap_E[1], valstep=0.025, color = 'purple', orientation = 'vertical')
delay_factor_ = Slider(delay_slide, 'delay, fs ', -200, 1000, valinit=-100, valstep=10, color = 'green', orientation = 'horizontal')               
delay_int_factor = Slider(delay_integration, 'Int., fs ', 10, 1000, valinit=50, valstep=10, color = 'green', orientation = 'horizontal')               

#int_factor = Slider(int_slide, 'Int',
                 # 0, 1, valinit= 0.2, valstep=0.04)
 
#fig.tight_layout()
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
plt.subplots_adjust(hspace = 0.2, wspace = 0.0)

def my_button(label):
    print('okay')

def my_button_2(label):
    print('what')
        
# Recalculate and Updating the plot
    
def update(val):
    
    current_v_E = E_factor.val
    E = (np.abs(ax_E_offset - current_v_E)).argmin()
    
    current_v_E_2 = E_factor_2.val
    E_2 = (np.abs(ax_E_offset - current_v_E_2)).argmin()
    #E = (np.abs(ax_E_offset - (new_y_E-(tint/2)))).argmin() #central slice for plotting
    current_v_d = delay_factor_.val
    delay_point = (np.abs(ax_delay_offset - current_v_d)).argmin()
    
    current_v_d_int = delay_int_factor.val
    dtp = round((current_v_d_int/dt)/2)
    
    tint_ = int(tint/2)
    new_to_plot = I[:,:,E-tint_:E+tint_].sum(axis=(2,3)) #Time delay integrated
    new_to_plot2 = I[:,y[0]:y[0]+yint,:].sum(axis=(1,3)) #Integrates all
    new_to_plot3 = I[x[0]:x[0]+xint,:].sum(axis=(0,3))
    
    if diff_box.get_status()[0] is True:
        #new_to_plot = I[:,:,E_new:(E_new+tint),dt:dt+1].sum(axis=(2,3))
        tp = delay_point
        neg_frames_map = I[:,:,E-tint_:E+tint_,4:t0-10].sum(axis=(2,3))/(t0-10-4)
        neg_frames_x = I[x[0]:x[0]+xint,:,:,4:t0-10].sum(axis=(0,3))/(t0-10-4)
        neg_frames_y = I[:,y[0]:y[0]+yint,:,4:t0-10].sum(axis=(1,3))/(t0-10-4)
        
        new_to_plot = I[:,:,E-tint_:E+tint_,tp-dtp:tp+dtp].sum(axis=(2,3))/(2*dtp+1) - neg_frames_map
        new_to_plot2 = I[:,y[0]:y[0]+yint,:,tp-dtp:tp+dtp].sum(axis=(1,3))/(2*dtp) - neg_frames_y
        new_to_plot3 = I[x[0]:x[0]+xint,:,:,tp-dtp:tp+dtp].sum(axis=(0,3))/(2*dtp) - neg_frames_x

        new_to_plot = new_to_plot/np.max(np.abs(new_to_plot))
        new_to_plot2 = new_to_plot2/np.max(np.abs(new_to_plot2))
        new_to_plot3 = new_to_plot3/np.max(np.abs(new_to_plot3))
        
        new_to_plot2[:,mask_start:] *= 1
        new_to_plot3[:,mask_start:] *= 1
            
    if np.ndim(I) > 3:

        newtrace = I[x[0]:x[0]+xint,y[0]:y[0]+yint,E-(tint_):E+(tint_),:].sum(axis=(0,1,2))
        newtrace = newtrace - np.mean(newtrace[2:t0-10])
        newtrace = newtrace/np.max(newtrace)
        
        newtrace_2 = I[x[1]:x[1]+xint,y[1]:y[1]+yint,E_2-tint_:E_2+tint_,:].sum(axis=(0,1,2))
        newtrace_2 = newtrace_2 - np.mean(newtrace_2[2:t0-10])
        newtrace_2 = newtrace_2/np.max(newtrace_2)
        
        im4.set_ydata(newtrace)
        im4_2.set_ydata(newtrace_2)
    
    im_1.set_data(np.transpose(new_to_plot))
    im_2.set_data(np.transpose(new_to_plot2))
    im_3.set_data(np.transpose(new_to_plot3))

    line_horizontal_E.set_ydata(current_v_E)
    line_horizontal_E_2.set_ydata(current_v_E)
    
    line_horizontal_E_3.set_ydata(current_v_E_2)
    line_horizontal_E_4.set_ydata(current_v_E_2)
    
    c_max = np.max(new_to_plot)
    im_1.set_clim(vmin=0, vmax = c_max*1.0)
    if diff_box.get_status()[0] is True:
        im_1.set_clim(vmin=-1*c_max, vmax = 1.0*c_max)
        im_2.set_clim(vmin=-1*c_max, vmax = c_max*1.0)
        im_3.set_clim(vmin=-1, vmax = c_max*1.0)
        im_1.set_cmap('seismic')
        im_2.set_cmap('seismic')
        im_3.set_cmap('seismic')
    
def update_square(x_center, y_center, half_length, square):
    #global x_center, y_center
    
    # Update square position based on line intersections
    square_x = [x_center - half_length, x_center + half_length, x_center + half_length, x_center - half_length, x_center - half_length]
    square_y = [y_center - half_length, y_center - half_length, y_center + half_length, y_center + half_length, y_center - half_length]

    square.set_data(square_x, square_y)
    
def update_square_2(x_center, y_center):
    #global x_center, y_center
    
    # Update square position based on line intersections
    square_x = [x_center - half_length, x_center + half_length, x_center + half_length, x_center - half_length, x_center - half_length]
    square_y = [y_center - half_length, y_center - half_length, y_center + half_length, y_center + half_length, y_center - half_length]

    square_2.set_data(square_x, square_y)
    
def prioritize_line_selection(event, line1, line2):
    dist_1 = np.linalg.norm(np.array(line1.get_xydata()).T - np.array([event.xdata, event.ydata]), axis = 1)
    dist_2 = np.linalg.norm(np.array(line2.get_xydata()).T - np.array([event.xdata, event.ydata]), axis = 1)
    
    if min(dist_1) < min(dist_2):
        return line1
    else:
        return line2
    
def on_press(event):
    global press_horizontal, press_vertical, offset_y_horizontal, offset_x_vertical, press_horizontal_E, \
    press_horizontal_2, press_vertical_2, offset_y_horizontal_2, offset_x_vertical_2

    #line_horizontal_SEL = prioritize_line_selection(event, line_horizontal, line_horizontal_2)
    
    #First Point
    if line_horizontal.contains(event)[0]:
        press_horizontal = True
        offset_y_horizontal = line_horizontal.get_ydata()[0] - event.ydata

    if line_vertical.contains(event)[0]:
        press_vertical = True
        offset_x_vertical = line_vertical.get_xdata()[0] - event.xdata
    
    #2nd Point
    if line_horizontal_2.contains(event)[0]:
        press_horizontal_2 = True
        offset_y_horizontal_2 = line_horizontal_2.get_ydata()[0] - event.ydata

    if line_vertical_2.contains(event)[0]:
        press_vertical_2 = True
        offset_x_vertical_2 = line_vertical_2.get_xdata()[0] - event.xdata
    
    #Energy
    if line_horizontal_E.contains(event)[0]:
            press_horizontal_E = True
            offset_y_horizontal_E = line_horizontal_E.get_ydata()[0] - event.ydata

def on_release(event):
    global press_horizontal, press_vertical, press_horizontal_E, press_horizontal_2, press_vertical_2

    #if press_horizontal or press_vertical:
     #   update_text()

    press_horizontal = False
    press_vertical = False
    press_horizontal_E = False
    press_horizontal_2 = False
    press_vertical_2 = False
    
def on_motion(event):
    global press_horizontal, press_vertical, press_horizontal_E,  press_horizontal_2, press_vertical_2, \
           new_x, new_y, new_x_2, new_y_2

    #First
    if press_horizontal:
        new_y = event.ydata + offset_y_horizontal
        line_horizontal.set_ydata([new_y, new_y])
        y[0] = (np.abs(ax_ky - (new_y-(yint_k/2)))).argmin() #central slice for plotting
        ky_ver_2.set_xdata([new_y, new_y])
        update_square(new_x, new_y, half_length, square)
        #update_square(new_x, new_y, half_length, square_3)
        
    if press_vertical:
        new_x = event.xdata + offset_x_vertical
        line_vertical.set_xdata([new_x, new_x])
        x[0] = (np.abs(ax_kx - (new_x-(xint_k/2)))).argmin() #central slice for plotting
        kx_ver_1.set_xdata([new_x, new_x])
        update_square(new_x, new_y, half_length, square)
        #update_square(new_x, new_y, half_length, square_3)
        
    #Second
    if press_horizontal_2:
        new_y_2 = event.ydata + offset_y_horizontal_2
        line_horizontal_2.set_ydata([new_y_2, new_y_2])
        y[1] = (np.abs(ax_ky - (new_y_2-(yint_k/2)))).argmin() #central slice for plotting
        ky_ver_2_2.set_xdata([new_y_2, new_y_2])
        update_square(new_x_2, new_y_2, half_length, square_2)
        
    if press_vertical_2:
        new_x_2 = event.xdata + offset_x_vertical_2
        line_vertical_2.set_xdata([new_x_2, new_x_2])
        x[1] = (np.abs(ax_kx - (new_x_2-(xint_k/2)))).argmin() #central slice for plotting
        kx_ver_1_2.set_xdata([new_x_2, new_x_2])
        update_square(new_x_2, new_y_2, half_length, square_2)
        
    if press_horizontal_E:
        new_y_E = event.ydata + offset_y_horizontal_E
        #line_horizontal_E.set_ydata([new_y_E, new_y_E])
        line_horizontal_E_2.set_ydata([new_y_E, new_y_E])
        #lhor.set_xdata([new_y_E, new_y_E])        
        
    if press_horizontal or press_vertical or press_horizontal_E or press_vertical_2 or press_horizontal_2:
        #update_text()
        fig.canvas.draw()
    
    current_v_E = E_factor.val
    E = (np.abs(ax_E_offset - current_v_E)).argmin()
    
    current_v_E_2 = E_factor_2.val
    E_2 = (np.abs(ax_E_offset - current_v_E_2)).argmin()
    #E = (np.abs(ax_E_offset - (new_y_E-(tint/2)))).argmin() #central slice for plotting
    current_v_d = delay_factor_.val
    delay_point = (np.abs(ax_delay_offset - current_v_d)).argmin()
    
    current_v_d_int = delay_int_factor.val
    dtp = round((current_v_d_int/dt)/2)
    
    tint_ = int(tint/2)
    new_to_plot = I[:,:,E-tint_:E+tint_,:].sum(axis=(2,3)) #Time delay integrated
    new_to_plot2 = I[:,y[0]:y[0]+yint,:,:].sum(axis=(1,3)) #Integrates all
    new_to_plot3 = I[x[0]:x[0]+xint,:,:,:].sum(axis=(0,3))
    
    if diff_box.get_status()[0] is True:
        #new_to_plot = I[:,:,E_new:(E_new+tint),dt:dt+1].sum(axis=(2,3))
        tp = delay_point
        neg_frames_map = I[:,:,E-tint_:E+tint_,4:t0-10].sum(axis=(2,3))/(t0-10-4)
        neg_frames_x = I[x[0]:x[0]+xint,:,:,4:t0-10].sum(axis=(0,3))/(t0-10-4)
        neg_frames_y = I[:,y[0]:y[0]+yint,:,4:t0-10].sum(axis=(1,3))/(t0-10-4)
        
        new_to_plot = I[:,:,E-tint_:E+tint_,tp-dtp:tp+dtp].sum(axis=(2,3))/(2*dtp) - neg_frames_map
        new_to_plot2 = I[:,y[0]:y[0]+yint,:,tp-dtp:tp+dtp].sum(axis=(1,3))/(2*dtp) - neg_frames_y
        new_to_plot3 = I[x[0]:x[0]+xint,:,:,tp-dtp:tp+dtp].sum(axis=(0,3))/(2*dtp) - neg_frames_x

        new_to_plot = new_to_plot/np.max(np.abs(new_to_plot))
        new_to_plot2 = new_to_plot2/np.max(np.abs(new_to_plot2))
        new_to_plot3 = new_to_plot3/np.max(np.abs(new_to_plot3))
        
        new_to_plot2[:,mask_start:] *= 1
        new_to_plot3[:,mask_start:] *= 1
        
    if np.ndim(I) > 3:

        newtrace = I[x[0]:x[0]+xint,y[0]:y[0]+yint,E-(tint_):E+(tint_),:].sum(axis=(0,1,2))
        newtrace = newtrace - np.mean(newtrace[2:t0-10])
        newtrace = newtrace/np.max(newtrace)
        
        newtrace_2 = I[x[1]:x[1]+xint,y[1]:y[1]+yint,E_2-tint_:E_2+tint_,:].sum(axis=(0,1,2))
        newtrace_2 = newtrace_2 - np.mean(newtrace_2[2:t0-10])
        newtrace_2 = newtrace_2/np.max(newtrace_2)
                
        im4.set_ydata(newtrace)
        im4_2.set_ydata(newtrace_2)
        
    im_1.set_data(np.transpose(new_to_plot))
    im_2.set_data(np.transpose(new_to_plot2))
    im_3.set_data(np.transpose(new_to_plot3))
    
    c_max = np.max(new_to_plot)
    c_max2 = np.max(new_to_plot2)
    c_max3 = np.max(new_to_plot3)

    im_1.set_clim(vmin=0, vmax = c_max*1.0)
    im_2.set_clim(vmin=0, vmax = c_max2*1.0)
    im_3.set_clim(vmin=0, vmax = c_max3*1.0)
    
    if diff_box.get_status()[0] is True:
            im_1.set_cmap('seismic')
            im_2.set_cmap('seismic')
            im_3.set_cmap('seismic')
            im_1.set_clim(vmin=-1*c_max, vmax = c_max*1.0)
            im_2.set_clim(vmin=-1, vmax = c_max*1.0)
            im_3.set_clim(vmin=-1, vmax = c_max*1.0)
    #current_v_E = E_factor.val
    #E = (np.abs(ax_E_offset - current_v_E)).argmin()

        #im.set_clim(vmin=-1 , vmax = 1.0)
        #im2.set_clim(vmin=-1*c_max1 , vmax = 1*c_max1*1.0)
        #im3.set_clim(vmin=-1*1*c_max2, vmax = 1*c_max2*1.0)      
# Connect the event handlers
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
        
# Calling the function "update" when the value of the slider is changed
E_factor.on_changed(update)
E_factor_2.on_changed(update)
delay_factor_.on_changed(update)
plt.subplots_adjust(hspace = 0.3)
plt.show()