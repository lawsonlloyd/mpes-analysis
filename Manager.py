# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:36:40 2024

@author: lloyd
"""

#Manager.py

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons

class DataHandler:
    def __init__(self, I, ax_kx, ax_ky, ax_E, ax_delay):
        self.I = I
        self.ax_kx = ax_kx
        self.ax_ky = ax_ky
        self.ax_E = ax_E
        self.ax_delay = ax_delay  
    
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

class Square:
    def __init__(self, x_center, y_center, half_length):
        self.x_center = x_center
        self.y_center = y_center
        self.half_length = half_length

    def get_coordinates(self):
        """Calculate and return square corner coordinates."""
        square_x = [
            self.x_center - self.half_length, self.x_center + self.half_length,
            self.x_center + self.half_length, self.x_center - self.half_length,
            self.x_center - self.half_length
        ]
        square_y = [
            self.y_center - self.half_length, self.y_center - self.half_length,
            self.y_center + self.half_length, self.y_center + self.half_length,
            self.y_center - self.half_length
        ]
        return square_x, square_y

    def update_position(self, new_x_center, new_y_center):
        self.x_center = new_x_center
        self.y_center = new_y_center

class PlotHandler:
    def __init__(self, data_handler, slider_manager):
        self.data_handler = data_handler
        self.slider_manager = slider_manager
        self.fig, self.ax = self.create_fig_axes()
        self.cmap = self.custom_colormap('viridis')
        self.im_1, self.im_2, self.im_3, self.im_4 = None, None, None, None
        self.time_trace_1, self.time_trace_2 = None, None
        self.horizontal_line_0, self.vertical_line_0 = None, None
        self.kx, self.ky, self.E, self.delay = 0, 0, 1.5, 100
        # Square instance to manage the interactive region
        self.square = Square(0, 0, 0.5)  # Default square with center at (0, 0)
        
        # Initial setup for plots
        self.initialize_plots()

    def get_current_values(self):
        
        self.kx = self.vertical_line_0.get_data()[0]
        self.ky = self.horizontal_line_0.get_data()[1]
        #self.E = self.slider_manager.E_slider.val
        #self.delay = self.slider_manager.E_slider.val
        
    def create_fig_axes(self):
        # Create the figure and axes for subplots
        fig, ax = plt.subplots(nrows=2, ncols=2, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1]})
        fig.set_size_inches(15, 10)
        return fig, ax.flatten()

    def initialize_plots(self):
        
        # Define intial kx, ky, Energy, and Delay Points for Images
        idx_kx, idx_ky, idx_E, idx_delay = self.data_handler.get_closest_indices(self.kx, self.ky, self.E, self.delay)
    
        # Initial Momentum Map kx, ky Image  (top left)
        frame_temp = np.transpose(self.data_handler.I[:, :, idx_E-2:idx_E+3, :].sum(axis=(2,3)))
        self.im_1 = self.ax[0].imshow(frame_temp/np.max(frame_temp),\
                                      extent = [self.data_handler.ax_kx[0], self.data_handler.ax_kx[-1],  self.data_handler.ax_ky[0], self.data_handler.ax_ky[-1]],\
                                      cmap=self.cmap, aspect='auto', origin='lower')
            
        self.ax[0].set_title("E = " + str(self.data_handler.ax_E[idx_E]) + ' eV')
        self.ax[0].set_aspect(1)
        self.ax[0].set_xlim([self.data_handler.ax_kx[0], self.data_handler.ax_kx[-1]])
        self.ax[0].set_ylim([self.data_handler.ax_ky[0], self.data_handler.ax_ky[-1]])

        # Initial kx vs E Image (bottom left)
        self.im_2 = self.ax[2].imshow(np.transpose(self.data_handler.I[:, idx_ky-2:idx_ky+3, :, :].sum(axis=(1,3))),\
                                      extent = [self.data_handler.ax_kx[0], self.data_handler.ax_kx[-1],  self.data_handler.ax_E[0], self.data_handler.ax_E[-1]],\
                                      cmap=self.cmap, aspect='auto', origin='lower')
        self.ax[2].set_title("Energy Cut")
        self.ax[2].set_xlabel("k")
        self.ax[2].set_ylabel("E, eV")
        
        # Initial ky vs E Image (bottom left)
        self.im_3 = self.ax[3].imshow(np.transpose(self.data_handler.I[idx_kx-2:idx_kx+3, :, :, :].sum(axis=(0,3))),\
                                     extent = [self.data_handler.ax_ky[0], self.data_handler.ax_ky[-1],  self.data_handler.ax_E[0], self.data_handler.ax_E[-1]],\
                                     cmap=self.cmap, aspect='auto', origin='lower')
        self.ax[3].set_title("Energy Cut")
        self.ax[3].set_xlabel("k")
        self.ax[3].set_ylabel("E, eV")
        
        # Initial Dynamics Time Trace (top right)
        time_trace = self.data_handler.I[idx_kx-2:idx_kx+3, idx_ky-2:idx_ky+3,  idx_E-2:idx_E+3, :].sum(axis=(0,1,2))
        time_trace - np.mean(time_trace[6:15])
        time_trace = time_trace/np.max(time_trace)
        self.time_trace_1, = self.ax[1].plot(self.data_handler.ax_delay, time_trace)
        self.ax[1].set_title("Dynamics")
        self.ax[1].set_xlabel("Delay,fs")
        self.ax[1].set_ylabel("Intensity")

        # Add interactive horizontal and vertical lines (for cuts)
        self.horizontal_line_0 = self.ax[0].axhline(y=self.data_handler.ax_kx[idx_kx], color='black', linestyle='--', linewidth = 1.5)
        self.vertical_line_0 = self.ax[0].axvline(x=self.data_handler.ax_ky[idx_ky], color='black', linestyle='--', linewidth = 1.5)
    
        self.horizontal_line_1 = self.ax[2].axhline(y=self.data_handler.ax_E[idx_E], color='black', linestyle='--', linewidth = 1.5)
        self.vertical_line_1 = self.ax[2].axvline(x=self.data_handler.ax_kx[idx_kx], color='black', linestyle='--', linewidth = 1.5)
                
        self.horizontal_line_2 = self.ax[3].axhline(y=self.data_handler.ax_E[idx_E], color='black', linestyle='--', linewidth = 1.5)
        self.vertical_line_2 = self.ax[3].axvline(x=self.data_handler.ax_ky[idx_ky], color='black', linestyle='--', linewidth = 1.5)

        self.fig.tight_layout()
        
    def update_image(self):
        """Update the 2D image based on selected E and delay."""
        self.get_current_values()
        idx_kx, idx_ky, idx_E, idx_delay = self.data_handler.get_closest_indices(self.kx, self.ky, self.E, 100)
        frame_temp = np.transpose(self.data_handler.I[:, :, idx_E-2:idx_E+3,:].sum(axis = (2,3)))
        self.im_1.set_data(frame_temp/np.max(frame_temp))  # Update image for new E
        self.ax[0].set_title("E = " + str(self.data_handler.ax_E[idx_E]) + ' eV')
        self.fig.canvas.draw()

    def update_time_traces(self):
        """Update the time traces when the square is moved or resized."""
        self.get_current_values()
        idx_kx, idx_ky, idx_E, idx_delay = self.data_handler.get_closest_indices(self.kx, self.ky, self.E, 100)
        time_trace = self.data_handler.I[idx_kx-2:idx_kx+3, idx_ky-2:idx_ky+3, idx_E-2:idx_E+3, :].sum(axis=(0,1,2))
        time_trace = time_trace - np.mean(time_trace[6:15])
        self.fig.canvas.draw()

        # Update the time trace plots
        self.time_trace_1.set_ydata(time_trace/np.max(time_trace))        

    def plot_square(self, square):
        """Plot a square on the 2D image."""
        square_x, square_y = square.get_coordinates()
        self.ax[0].plot(square_x, square_y, color='cyan', lw=2)

    def update_square(self, square):
        """Update square position on plot."""
        self.ax[0].clear()  # Clear the plot for fresh update
        self.update_image(E=0)  # Re-draw the image
        self.plot_square(square)  # Plot the updated square
        self.update_time_traces(square)  # Update time traces
        self.fig.canvas.draw_idle()  # Force the figure to re-render

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
    def __init__(self, slider_manager, plot_manager):
        self.slider_manager = slider_manager
        self.plot_manager = plot_manager
        self.press_horizontal = False
        self.press_vertical = False
        
    def on_press(self, event):
        """Handle mouse press events and update square or lines."""
        if self.plot_manager.horizontal_line_0.contains(event)[0]:
            self.press_horizontal = True    
        if self.plot_manager.vertical_line_0.contains(event)[0]:
            self.press_vertical = True 
            #self.plot_manager.square.update_position(event.xdata, event.ydata)  # Update square position
            #self.plot_manager.update_square(self.plot_manager.square)  # Redraw square and update traces
    
    def on_motion(self, event):
        if self.press_horizontal:    
            new_ky = event.ydata
            E =  self.slider_manager.E_slider.val
            self.plot_manager.horizontal_line_0.set_ydata(y = new_ky)
            self.plot_manager.horizontal_line_2.set_ydata(y = E)
            self.plot_manager.vertical_line_2.set_xdata(x = new_ky)
            self.plot_manager.update_time_traces()
            self.plot_manager.fig.canvas.draw()

        if self.press_vertical:    
            new_kx = event.xdata
            E =  self.slider_manager.E_slider.val
            self.plot_manager.vertical_line_0.set_xdata(x = new_kx)
            self.plot_manager.vertical_line_1.set_xdata(x = new_kx)
            self.plot_manager.horizontal_line_1.set_ydata(y = E)
            self.plot_manager.update_time_traces()
            self.plot_manager.fig.canvas.draw()

    def on_release(self, event):
        """Handle mouse release events."""
        self.press_horizontal = False
        self.press_vertical = False

class SliderManager:
    def __init__(self, plot_manager):
        self.plot_manager = plot_manager
        self.E_slider = self.create_sliders()

    def create_sliders(self):
        """Create the sliders for energy and delay."""
        E_slider = Slider(plt.axes([0.045, 0.6, 0.03, 0.25]), 'E, eV', -4, 3.5, valinit=0, valstep = 0.05, color = 'black', orientation = 'vertical')
        #delay_slider = Slider(plt.axes([0.8, 0.475, 0.15, 0.03]), 'delay, fs', -200, 1000, valinit=0)
        return E_slider

    def on_slider_update(self, val):
        """Update plots based on slider values."""
        E = self.E_slider.val
        plot_manager.E = E
        self.plot_manager.update_image()
        self.plot_manager.horizontal_line_1.set_ydata(y = E)
        self.plot_manager.horizontal_line_2.set_ydata(y = E)
        self.plot_manager.update_time_traces()