# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:36:40 2024

@author: lloyd
"""

#Manager.py

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class DataHandler:
    def __init__(self, I, ax_kx, ax_ky, ax_E, ax_delay, *offsets):
        if offsets:
            E_offset = offsets[0]
            delay_offset = offsets[1]
        else:
            E_offset = 0
            delay_offset = 0
        self.I = I
        self.ax_kx = ax_kx
        self.ax_ky = ax_ky
        self.ax_E = ax_E + E_offset
        self.ax_delay = ax_delay + delay_offset  
    
    def calculate_dt(self):
        if self.I.ndim > 3:
            dt = self.ax_delay[1] - self.ax_delay[0]
            return dt
        else:
            return 1
    
    def get_t0(self):
        if self.I.ndim > 3:
            t0 = (np.abs(self.ax_delay - 0)).argmin()
            return t0
        else:
            return 1
        
    def get_closest_indices(self, kx, ky, E, delay):
        kx_idx = (np.abs(self.ax_kx - kx)).argmin()
        ky_idx = (np.abs(self.ax_ky - ky)).argmin()
        E_idx = (np.abs(self.ax_E - E)).argmin()
        delay_idx = (np.abs(self.ax_delay - delay)).argmin()
        return kx_idx, ky_idx, E_idx, delay_idx

class ValueHandler:
    def __init__(self):
        self.k_int, self.kx, self.ky, self.E, self.delay = 0.4, 0, 0, 0, 0

    def update_k_int_value(self, k_int):
        self.k_int = k_int
        
    def update_kx_value(self, kx):
        self.kx = kx        
    
    def update_ky_value(self, ky):
        self.ky = ky   
        
    def update_E_value(self, E):
        self.E = E
   
    def update_delay_value(self, delay):
        self.delay = delay 
        
    def get_values(self):
        return self.k_int, self.kx, self.ky, self.E, self.delay

class PlotHandler:
    def __init__(self, data_handler, value_manager):
        self.data_handler = data_handler
        self.value_manager = value_manager
        self.fig, self.ax = self.create_fig_axes()
        self.cmap = self.custom_colormap('viridis')
        self.im_1, self.im_2, self.im_3, self.im_4 = None, None, None, None
        self.time_trace_1 = None
        self.t0 = self.data_handler.get_t0()

        # Initial setup for plots
        self.initialize_plots()
        
    def create_fig_axes(self):
        # Create the figure and axes for subplots
        fig, ax = plt.subplots(nrows=2, ncols=2, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1]})
        fig.set_size_inches(15, 10)
        return fig, ax.flatten()

    def initialize_plots(self,):
        
        # Define intial kx, ky, Energy, and Delay Points for Images
        k_int, kx, ky, E, delay = self.value_manager.get_values()
        idx_kx, idx_ky, idx_E, idx_delay = self.data_handler.get_closest_indices(kx, ky, E, delay)
    
        # Initial Momentum Map kx, ky Image  (top left)
        frame_temp = np.transpose(self.data_handler.I[:, :, idx_E-2:idx_E+3, :].sum(axis=(2,3)))
        self.im_1 = self.ax[0].imshow(frame_temp/np.max(frame_temp),\
                                      extent = [self.data_handler.ax_kx[0], self.data_handler.ax_kx[-1],  self.data_handler.ax_ky[0], self.data_handler.ax_ky[-1]],\
                                      cmap=self.cmap, aspect='auto', origin='lower')
            
        self.ax[0].set_title("E = " + str(self.data_handler.ax_E[idx_E]) + ' eV')
        self.ax[0].set_aspect(1)
        
        self.ax[0].set_xticks(np.arange(-3,3.1,0.5))
        for label in self.ax[0].xaxis.get_ticklabels()[1::2]:
            label.set_visible(False)
        
        self.ax[0].set_yticks(np.arange(-3,3.1,0.5))
        for label in self.ax[0].yaxis.get_ticklabels()[1::2]:
                label.set_visible(False)    
        self.ax[0].set_xlim([-2,2])
        self.ax[0].set_ylim([-2,2])
        self.ax[0].set_xlabel('$k_x$', fontsize = 14)
        self.ax[0].set_ylabel('$k_y$', fontsize = 14)
        
        # Initial kx vs E Image (bottom left)
        frame_temp = np.transpose(self.data_handler.I[:, idx_ky-2:idx_ky+3, :, :].sum(axis=(1,3)))
        self.im_2 = self.ax[2].imshow(frame_temp/np.max(frame_temp),\
                                      extent = [self.data_handler.ax_kx[0], self.data_handler.ax_kx[-1],  self.data_handler.ax_E[0], self.data_handler.ax_E[-1]],\
                                      cmap=self.cmap, aspect='auto', origin='lower')
        self.ax[2].set_title("Energy Cut")
        self.ax[2].set_xlabel('$k_x$', fontsize = 14)
        self.ax[2].set_ylabel("E, eV")
        self.ax[2].set_aspect(.5)
        self.ax[2].set_xticks(np.arange(-3,3.1,0.5))
        for label in self.ax[2].xaxis.get_ticklabels()[1::2]:
            label.set_visible(False)
        self.ax[2].set_yticks(np.arange(-5,3.1,0.5))
        for label in self.ax[2].yaxis.get_ticklabels()[1::2]:
            label.set_visible(False)   
        self.ax[2].set_xlim(-2,2)
        self.ax[2].set_ylim(-3,3)

        # Initial ky vs E Image (bottom left)
        frame_temp = np.transpose(self.data_handler.I[idx_kx-2:idx_kx+3, :, :, :].sum(axis=(0,3)))
        self.im_3 = self.ax[3].imshow(frame_temp/np.max(frame_temp),\
                                     extent = [self.data_handler.ax_ky[0], self.data_handler.ax_ky[-1],  self.data_handler.ax_E[0], self.data_handler.ax_E[-1]],\
                                     cmap=self.cmap, aspect='auto', origin='lower')
        self.ax[3].set_title("Energy Cut")
        self.ax[3].set_xlabel('$k_y$', fontsize = 14)
        self.ax[3].set_ylabel("E, eV")
        self.ax[3].set_aspect(.5)
        self.ax[3].set_xticks(np.arange(-3,3.1,0.5))
        for label in self.ax[3].xaxis.get_ticklabels()[1::2]:
            label.set_visible(False)
        self.ax[3].set_yticks(np.arange(-5,3.1,0.5))
        for label in self.ax[3].yaxis.get_ticklabels()[1::2]:
            label.set_visible(False)   
        self.ax[3].set_xlim(-2,2)
        self.ax[3].set_ylim(-3,3)
        
        # Initial Dynamics Time Trace (top right)
        time_trace = self.data_handler.I[idx_kx-2:idx_kx+3, idx_ky-2:idx_ky+3, idx_E-2:idx_E+3, :].sum(axis=(0,1,2))
        time_trace = time_trace - np.mean(time_trace[5:self.t0-5])
        
        self.time_trace_1, = self.ax[1].plot(self.data_handler.ax_delay, time_trace/np.max(time_trace), color = 'black')
        self.ax[1].set_xticks(np.arange(-400,1250,200))
        for label in self.ax[1].xaxis.get_ticklabels()[1::2]:
            label.set_visible(False)
        self.ax[1].set_ylim([-0.1, 1.1])
        self.ax[1].set_xlim([self.data_handler.ax_delay[5], self.data_handler.ax_delay[-5]])
        self.ax[1].set_title("Dynamics")
        self.ax[1].set_xlabel("Delay, fs")
        self.ax[1].set_ylabel("Intensity")

        # Add interactive horizontal and vertical lines (for cuts)
        self.horizontal_line_0 = self.ax[0].axhline(y=self.data_handler.ax_kx[idx_kx], color='black', linestyle='--', linewidth = 1.5)
        self.vertical_line_0 = self.ax[0].axvline(x=self.data_handler.ax_ky[idx_ky], color='black', linestyle='--', linewidth = 1.5)
    
        self.horizontal_line_1 = self.ax[2].axhline(y=self.data_handler.ax_E[idx_E], color='black', linestyle='--', linewidth = 1.5)
        self.vertical_line_1 = self.ax[2].axvline(x=self.data_handler.ax_kx[idx_kx], color='black', linestyle='--', linewidth = 1.5)
                
        self.horizontal_line_2 = self.ax[3].axhline(y=self.data_handler.ax_E[idx_E], color='black', linestyle='--', linewidth = 1.5)
        self.vertical_line_2 = self.ax[3].axvline(x=self.data_handler.ax_ky[idx_ky], color='black', linestyle='--', linewidth = 1.5)

        # Add Squares
        square_x, square_y = self.make_square(kx, ky, k_int)
        self.square_0, = self.ax[0].plot(square_x, square_y, color='black', linewidth = 1, linestyle='dashed')
        
        self.fig.tight_layout()
        
    def make_square(self, kx_center, ky_center, k_width):
        """Calculate and return square corner coordinates."""
        half_k_width = k_width/2
        square_x = [
            kx_center - half_k_width, 
            kx_center + half_k_width, 
            kx_center + half_k_width, 
            kx_center - half_k_width, 
            kx_center - half_k_width
        ]
        
        square_y = [
            ky_center - half_k_width, 
            ky_center - half_k_width, 
            ky_center + half_k_width, 
            ky_center + half_k_width, 
            ky_center - half_k_width
        ]
        
        return square_x, square_y
        
    def update_image(self):
        k_int, kx, ky, E, delay = self.value_manager.get_values()
        idx_kx, idx_ky, idx_E, idx_delay = self.data_handler.get_closest_indices(kx, ky, E, delay)
        frame_temp = np.transpose(self.data_handler.I[:, :, idx_E-2:idx_E+3,:].sum(axis = (2,3)))
        self.im_1.set_data(frame_temp/np.max(frame_temp))  # Update image for new E
        self.ax[0].set_title("E = " + str(round(self.data_handler.ax_E[idx_E],1)) + ' eV')
        
    def update_kx_image(self):
        k_int, kx, ky, E, delay = self.value_manager.get_values()
        idx_kx, idx_ky, idx_E, idx_delay = self.data_handler.get_closest_indices(kx, ky, E, delay)
        frame_temp = np.transpose(self.data_handler.I[:, idx_ky-2:idx_ky+3, :, :].sum(axis = (1,3)))
        self.im_2.set_data(frame_temp/np.max(frame_temp))  # Update image for new E

    def update_ky_image(self):
        k_int, kx, ky, E, delay = self.value_manager.get_values()
        idx_kx, idx_ky, idx_E, idx_delay = self.data_handler.get_closest_indices(kx, ky, E, delay)
        frame_temp = np.transpose(self.data_handler.I[idx_kx-2:idx_kx+3, :, :, :].sum(axis = (0,3)))
        self.im_3.set_data(frame_temp/np.max(frame_temp))  # Update image for new E

    def update_time_traces(self):
        """Update the time traces when the square is moved or resized."""
        k_int, kx, ky, E, delay = self.value_manager.get_values()
        idx_kx, idx_ky = [0, 0], [0, 0]
        idx_kx[0], idx_ky[0], idx_E, _ = self.data_handler.get_closest_indices(kx-k_int/2, ky-k_int/2, E, delay)
        idx_kx[1], idx_ky[1], _, _ = self.data_handler.get_closest_indices(kx+k_int/2, ky+k_int/2, E, delay)

        time_trace = self.data_handler.I[idx_kx[0]:idx_kx[1], idx_ky[0]:idx_ky[1], idx_E-2:idx_E+3, :].sum(axis=(0,1,2))
        time_trace = time_trace - np.mean(time_trace[5:self.t0-5])
        # Update the time trace plots
        self.time_trace_1.set_ydata(time_trace/np.max(time_trace))        

    def update_square(self):
        """Update square position on plot."""
        k_int, kx, ky, E, delay = self.value_manager.get_values()
        
        square_x, square_y = self.make_square(kx, ky, k_int)
        self.square_0.set_data(square_x, square_y)
        
    def custom_colormap(self, CMAP):
        # create a colormap that consists of
        # - 1/5 : custom colormap, ranging from white to the first color of the colormap
        # - 4/5 : existing colormap
        # set upper part: 4 * 256/4 entries
        
        if CMAP == 'viridis':
            upper = mpl.cm.viridis(np.arange(256))
        else:
            upper = mpl.cm.magma(np.arange(256))
    
        upper = upper[56:,:]
        #upper = mpl.cm.jet(np.arange(256))
        #upper = mpl.cm.magma_r(np.arange(256))
        
        # set lower part: 1 * 256/4 entries
        # - initialize all entries to 1 to make sure that the alpha channel (4th column) is 1
        lower = np.ones((int(200/3),4))
        # - modify the first three columns (RGB):
        #   range linearly between white (1,1,1) and the first color of the upper colormap
        for i in range(3):
          lower[:,i] = np.linspace(1, upper[0,i], lower.shape[0])
        
        # combine parts of colormap
        cmap = np.vstack(( lower, upper ))
        
        # convert to matplotlib colormap
        custom_cmap = mpl.colors.ListedColormap(cmap, name='custom', N=cmap.shape[0])
        
        return custom_cmap

class EventHandler:
    def __init__(self, slider_manager, plot_manager, value_manager):
        self.slider_manager = slider_manager
        self.plot_manager = plot_manager
        self.value_manager = value_manager
        self.press_horizontal = False
        self.press_vertical = False
        
    def on_press(self, event):
        """Handle mouse press events and update square or lines."""
        if self.plot_manager.horizontal_line_0.contains(event)[0]:
            self.press_horizontal = True    
        if self.plot_manager.vertical_line_0.contains(event)[0]:
            self.press_vertical = True 
            #self.plot_manager.update_square(self.plot_manager.square)  # Redraw square and update traces
    
    def on_motion(self, event):
        if self.press_horizontal:    
            new_ky = event.ydata
            self.value_manager.update_ky_value(new_ky)
            self.plot_manager.horizontal_line_0.set_ydata(y = new_ky)
            self.plot_manager.vertical_line_2.set_xdata(x = new_ky)
            self.plot_manager.update_kx_image()
            self.plot_manager.update_time_traces()
            self.plot_manager.update_square()  # Update square position
            self.plot_manager.fig.canvas.draw()

        if self.press_vertical:    
            new_kx = event.xdata
            self.value_manager.update_kx_value(new_kx)
            self.plot_manager.vertical_line_0.set_xdata(x = new_kx)
            self.plot_manager.vertical_line_1.set_xdata(x = new_kx)
            self.plot_manager.update_ky_image()
            self.plot_manager.update_time_traces()
            self.plot_manager.update_square()  # Update square position
            self.plot_manager.fig.canvas.draw()

    def on_release(self, event):
        """Handle mouse release events."""
        self.press_horizontal = False
        self.press_vertical = False

class SliderManager:
    def __init__(self, plot_manager, value_manager):
        self.plot_manager = plot_manager
        self.value_manager = value_manager
        self.E_slider, self.k_int_slider = self.create_sliders()

    def create_sliders(self):
        """Create the sliders for energy and delay."""
        E_slider = Slider(plt.axes([0.025, 0.6, 0.03, 0.25]), 'E, eV', -4, 3.5, valinit=0, valstep = 0.05, color = 'black', orientation = 'vertical')
        k_int_slider = Slider(plt.axes([0.055, 0.6, 0.03, 0.25]), '$\Delta k$, $A^{-1}$', 0, 4, valinit=.5, valstep = 0.1, color = 'red', orientation = 'vertical')
        
        return E_slider, k_int_slider

    def on_slider_update(self, val):
        """Update plots based on slider values."""
        E = self.E_slider.val
        k_int = self.k_int_slider.val
        self.value_manager.update_E_value(E)
        self.value_manager.update_k_int_value(k_int)
        
        self.plot_manager.horizontal_line_1.set_ydata(y = E)
        self.plot_manager.horizontal_line_2.set_ydata(y = E)
        
        self.plot_manager.update_image()
        self.plot_manager.update_square()
        self.plot_manager.update_time_traces()
        self.plot_manager.fig.canvas.draw()
