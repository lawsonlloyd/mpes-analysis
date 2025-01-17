# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:36:40 2024

@author: lloyd
"""

#Manager.py

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, Button

class DataHandler:
    def __init__(self, value_manager, I, ax_kx, ax_ky, ax_E, ax_delay, *offsets):
        self.value_manager = value_manager
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
    
    def calculate_dk(self):
        dk = self.ax_kx[2] - self.ax_kx[1]
        
        return dk
    
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

    def get_momentum_map(self):
        k_int, kx, ky, E, delay = self.value_manager.get_values()
        idx_kx, idx_ky, idx_E, idx_delay = self.get_closest_indices(kx, ky, E, delay)
        if self.I.ndim > 3:
            mm = np.transpose(self.I[:, :, idx_E-2:idx_E+3, :].sum(axis=(2,3)))
        else:
            mm = np.transpose(self.I[:, :, idx_E-2:idx_E+3].sum(axis=(2)))
        return mm
    
    def get_kx_map(self):
        k_int, kx, ky, E, delay = self.value_manager.get_values()
        idx_kx, idx_ky, idx_E, idx_delay = self.get_closest_indices(kx, ky, E, delay)
        if self.I.ndim > 3:
            kx_map = np.transpose(self.I[:, idx_ky-2:idx_ky+3, :, :].sum(axis=(1,3)))
        else:
            kx_map = np.transpose(self.I[:, idx_ky-2:idx_ky+3, :].sum(axis=(1)))
        return kx_map
    
    def get_ky_map(self):
        k_int, kx, ky, E, delay = self.value_manager.get_values()
        idx_kx, idx_ky, idx_E, idx_delay = self.get_closest_indices(kx, ky, E, delay)
        if self.I.ndim > 3:
            ky_map = np.transpose(self.I[idx_kx-2:idx_kx+3, :, :, :].sum(axis=(0,3)))
        else:
            ky_map = np.transpose(self.I[idx_kx-2:idx_kx+3, :, :].sum(axis=(0)))
        return ky_map
    
    def get_edc(self):
        k_int, kx, ky, E, delay = self.value_manager.get_values()
        idx_kx, idx_ky, idx_E, idx_delay = self.get_closest_indices(kx, ky, E, delay)
        dk = self.get_dk()
        idx_k_int = np.round(dk/k_int)
        
        edc = self.I[idx_kx-idx_k_int:idx_kx+idx_k_int, idx_ky-idx_k_int:idx_ky+idx_k_int, :, delay-3:delay+3].sum(axis=(0,1,3))
        
        return edc
    
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

class FigureHandler:
    def __init__(self):
        test = 0
        self.fig, self.ax = self.create_fig_axes()
        
    def create_fig_axes(self):
        # Create the figure and axes for subplots
        fig, ax = plt.subplots(nrows=2, ncols=2, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1]})
        fig.set_size_inches(15, 10)
        
        return fig, ax.flatten()
        
class PlotHandler:
    def __init__(self, figure_handler, data_handler, value_manager, check_button_manager):
        self.figure_handler = figure_handler
        self.data_handler = data_handler
        self.value_manager = value_manager
        self.check_button_manager = check_button_manager
        self.fig, self.ax = self.figure_handler.fig, self.figure_handler.ax
        self.cmap = self.custom_colormap(mpl.cm.viridis, 0.25)
        self.im_1, self.im_2, self.im_3, self.im_4 = None, None, None, None
        self.time_trace_1 = None
        self.t0 = self.data_handler.get_t0()

        # Initial setup for plots
        self.initialize_plots()

    def initialize_plots(self,):
        
        # Define intial kx, ky, Energy, and Delay Points for Images
        k_int, kx, ky, E, delay = self.value_manager.get_values()
        idx_kx, idx_ky, idx_E, idx_delay = self.data_handler.get_closest_indices(kx, ky, E, delay)
    
        # Initial Momentum Map kx, ky Image  (top left)
        frame_temp = self.data_handler.get_momentum_map()
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
        frame_temp = self.data_handler.get_kx_map()
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
        self.ax[2].set_ylim(-4,3)

        # Initial ky vs E Image (bottom left)
        frame_temp = self.data_handler.get_ky_map()
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
        self.ax[3].set_ylim(-4,3)
        
        # Initial Dynamics Time Trace (top right)
        if self.data_handler.I.ndim > 3:
            self.plot_time_trace()  
        else:
            self.plot_edc()      

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
        
    def plot_edc(self):
        k_int, kx, ky, E, delay = self.value_manager.get_values()
        idx_kx, idx_ky = [0, 0], [0, 0]
        idx_kx[0], idx_ky[0], _, _ = self.data_handler.get_closest_indices(kx-k_int/2, ky-k_int/2, E, delay)
        idx_kx[1], idx_ky[1], _, _ = self.data_handler.get_closest_indices(kx+k_int/2, ky+k_int/2, E, delay)

        if self.data_handler.I.ndim > 3:
            edc = self.data_handler.I[idx_kx[0]:idx_kx[1], idx_ky[0]:idx_ky[1], :, :].sum(axis=(0,1,3))
        else:
            edc = self.data_handler.I[idx_kx[0]:idx_kx[1], idx_ky[0]:idx_ky[1], :].sum(axis=(0,1))

        edc = edc/np.max(edc)
        
        self.im_4, = self.ax[1].plot(self.data_handler.ax_E, edc, color = 'black')
       # self.ax[1].set_xticks(np.arange(-400,1250,200))
        #for label in self.ax[1].xaxis.get_ticklabels()[1::2]:
         #   label.set_visible(False)
        self.ax[1].set_ylim([-0.1, 1.1])
        self.ax[1].set_xlim([self.data_handler.ax_E[0], self.data_handler.ax_E[-1]])
        self.ax[1].set_title("EDC")
        self.ax[1].set_xlabel("Energy, eV")
        self.ax[1].set_ylabel("Intensity")
                
    def plot_time_trace(self):
        k_int, kx, ky, E, delay = self.value_manager.get_values()
        idx_kx, idx_ky = [0, 0], [0, 0]
        idx_kx[0], idx_ky[0], idx_E, _ = self.data_handler.get_closest_indices(kx-k_int/2, ky-k_int/2, E, delay)
        idx_kx[1], idx_ky[1], _, _ = self.data_handler.get_closest_indices(kx+k_int/2, ky+k_int/2, E, delay)

        if self.data_handler.I.ndim > 3:
            time_trace = self.data_handler.I[idx_kx[0]:idx_kx[1], idx_ky[0]:idx_ky[1], idx_E-2:idx_E+3, :].sum(axis=(0,1,2))
            time_trace = time_trace #- np.mean(time_trace[5:self.t0-6])
            
            self.im_4, = self.ax[1].plot(self.data_handler.ax_delay, time_trace/np.max(time_trace), color = 'black')
            self.ax[1].set_xticks(np.arange(-300,1250,100))
            for label in self.ax[1].xaxis.get_ticklabels()[1::2]:
                label.set_visible(False)
            self.ax[1].set_xlim([self.data_handler.ax_delay[1], self.data_handler.ax_delay[-1]])

        else:
            time_trace = np.zeros(1)
            
        self.ax[1].set_ylim([-0.2, 1.1])
        self.ax[1].set_title("Dynamics")
        self.ax[1].set_xlabel("Delay, fs")
        self.ax[1].set_ylabel("Intensity")
                   
    def update_kxky_image(self):
        k_int, kx, ky, E, delay = self.value_manager.get_values()
        idx_kx, idx_ky, idx_E, idx_delay = self.data_handler.get_closest_indices(kx, ky, E, delay)
        
        frame_temp = self.data_handler.get_momentum_map()
        self.im_1.set_data(frame_temp/np.max(frame_temp))  # Update image for new E
        self.ax[0].set_title("E = " + str(round(self.data_handler.ax_E[idx_E],1)) + ' eV')
        
    def update_kx_image(self):
        k_int, kx, ky, E, delay = self.value_manager.get_values()
        idx_kx, idx_ky, idx_E, idx_delay = self.data_handler.get_closest_indices(kx, ky, E, delay)
        frame_temp = self.data_handler.get_kx_map()
        frame_temp = frame_temp/np.max(frame_temp)
        if self.check_button_manager.enhance_button_status == True:    
            mask_start = (np.abs(self.data_handler.ax_E - 0.95)).argmin()
            frame_temp[mask_start:,:] *= 1/np.max(frame_temp[mask_start:,:])
        self.im_2.set_data(frame_temp/np.max(frame_temp))  # Update image for new E

    def update_ky_image(self):
        k_int, kx, ky, E, delay = self.value_manager.get_values()
        idx_kx, idx_ky, idx_E, idx_delay = self.data_handler.get_closest_indices(kx, ky, E, delay)
        frame_temp = self.data_handler.get_ky_map()
        frame_temp = frame_temp/np.max(frame_temp)
        if self.check_button_manager.enhance_button_status == True:    
            mask_start = (np.abs(self.data_handler.ax_E - 0.95)).argmin()
            frame_temp[mask_start:,:] *= 1/np.max(frame_temp[mask_start:,:])
        self.im_3.set_data(frame_temp/np.max(frame_temp))  # Update image for new E

    def update_time_trace(self):
        """Update the time traces when the square is moved or resized."""
        k_int, kx, ky, E, delay = self.value_manager.get_values()
        idx_kx, idx_ky = [0, 0], [0, 0]
        idx_kx[0], idx_ky[0], idx_E, _ = self.data_handler.get_closest_indices(kx-k_int/2, ky-k_int/2, E, delay)
        idx_kx[1], idx_ky[1], _, _ = self.data_handler.get_closest_indices(kx+k_int/2, ky+k_int/2, E, delay)
        
        # Update the time trace plots
        self.ax[1].set_title("Dynamics")
        self.ax[1].set_xlabel("Delay, fs")
        self.ax[1].set_ylabel("Intensity")
        
        if self.data_handler.I.ndim > 3:
            time_trace = self.data_handler.I[idx_kx[0]:idx_kx[1], idx_ky[0]:idx_ky[1], idx_E-2:idx_E+3, :].sum(axis=(0,1,2))
            time_trace = time_trace #- np.mean(time_trace[5:self.t0-5])
            #self.ax[1].set_xticks(np.arange(-200,1250,200))
            #for label in self.ax[1].xaxis.get_ticklabels()[1::2]:
             #   label.set_visible(False)
            #self.im_4.set_xdata(self.data_handler.ax_delay)
            self.im_4.set_ydata(time_trace/np.max(time_trace))
            #self.ax[1].set_xlim([self.data_handler.ax_delay[5], self.data_handler.ax_delay[-5]])
        else:
            time_trace = np.zeros(1)


    def update_edc(self):
        """Update the time traces when the square is moved or resized."""
        k_int, kx, ky, E, delay = self.value_manager.get_values()
        idx_kx, idx_ky = [0, 0], [0, 0]
        idx_kx[0], idx_ky[0], _, _ = self.data_handler.get_closest_indices(kx-k_int/2, ky-k_int/2, E, delay)
        idx_kx[1], idx_ky[1], _, _ = self.data_handler.get_closest_indices(kx+k_int/2, ky+k_int/2, E, delay)

        if self.data_handler.I.ndim > 3:
            edc = self.data_handler.I[idx_kx[0]:idx_kx[1], idx_ky[0]:idx_ky[1], :, :].sum(axis=(0,1,3))
        else:
            edc = self.data_handler.I[idx_kx[0]:idx_kx[1], idx_ky[0]:idx_ky[1], :].sum(axis=(0,1))
        
        edc = edc/np.max(edc)

        if self.check_button_manager.enhance_button_status == True:    
            mask_start = (np.abs(self.data_handler.ax_E - 0.95)).argmin()
            edc[mask_start:] *= 1/np.max(edc[mask_start:])
            
        # Update the edc plots
        self.im_4.set_xdata(self.data_handler.ax_E)
        self.im_4.set_ydata(edc/np.max(edc))
        self.ax[1].set_ylim([-0.1, 1.1])
        self.ax[1].set_xticks(np.arange(-6,4,0.5))
        for label in self.ax[1].xaxis.get_ticklabels()[1::2]:
            label.set_visible(False)
        self.ax[1].set_xlim([self.data_handler.ax_E[0], self.data_handler.ax_E[-1]])
        self.ax[1].set_title("EDC")
        self.ax[1].set_xlabel("Energy, eV")
        self.ax[1].set_ylabel("Intensity")
        #self.ax[1].axvline(E, color = 'black', linestyle = 'dashed')
        
    def update_lines(self):
        k_int, kx, ky, E, delay = self.value_manager.get_values()
        self.horizontal_line_1.set_ydata(y = E)
        self.horizontal_line_2.set_ydata(y = E)
        
    def update_square(self):
        """Update square position on plot."""
        k_int, kx, ky, E, delay = self.value_manager.get_values()
        
        square_x, square_y = self.make_square(kx, ky, k_int)
        self.square_0.set_data(square_x, square_y)
     
    #def save_trace_plot(self)
    
    def custom_colormap(self, CMAP, lower_portion_percentage):
        # create a colormap that consists of
        # - 1/5 : custom colormap, ranging from white to the first color of the colormap
        # - 4/5 : existing colormap
        
        # set upper part: 4 * 256/4 entries
        
        upper =  CMAP(np.arange(256))
        upper = upper[56:,:]
        
        # - initialize all entries to 1 to make sure that the alpha channel (4th column) is 1
        lower_portion = int(1/lower_portion_percentage) - 1
        
        lower = np.ones((int(200/lower_portion),4))
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
    def __init__(self, value_manager, slider_manager, plot_manager, check_button_manager):
        self.slider_manager = slider_manager
        self.plot_manager = plot_manager
        self.check_button_manager = check_button_manager
        self.value_manager = value_manager
        self.press_horizontal = False
        self.press_vertical = False
        
    def on_press(self, event):
        """Handle mouse press events and update square or lines."""
        if self.plot_manager.horizontal_line_0.contains(event)[0]:
            self.press_horizontal = True    
        if self.plot_manager.vertical_line_0.contains(event)[0]:
            self.press_vertical = True 
        
        if self.check_button_manager.trace_check_button.get_status()[0]: # if EDC button selected
            self.check_button_manager.trace_button_status = True
            self.plot_manager.update_edc()
            self.plot_manager.fig.canvas.draw()
        elif self.check_button_manager.trace_check_button.get_status()[0] is False and self.plot_manager.data_handler.I.ndim > 3:
            self.check_button_manager.trace_button_status = False            
            self.plot_manager.update_time_trace()
            self.plot_manager.fig.canvas.draw()

        if self.check_button_manager.enhance_check_button.get_status()[0] is False: # if Enhance CB feature visibility OFF
            self.check_button_manager.enhance_button_status = False
            self.plot_manager.update_kx_image()
            self.plot_manager.update_ky_image()
            self.plot_manager.fig.canvas.draw()
        elif self.check_button_manager.enhance_check_button.get_status()[0] is True: # if Enhance CB feature visibility ON
            self.check_button_manager.enhance_button_status = True
            self.plot_manager.update_kx_image()
            self.plot_manager.update_ky_image()
            if self.check_button_manager.trace_check_button.get_status()[0] is True:
                self.plot_manager.update_edc()
            self.plot_manager.fig.canvas.draw()

    def on_motion(self, event):
        if self.press_horizontal:    
            new_ky = event.ydata
            self.value_manager.update_ky_value(new_ky)
            self.plot_manager.horizontal_line_0.set_ydata(y = new_ky)
            self.plot_manager.vertical_line_2.set_xdata(x = new_ky)
            self.plot_manager.update_kx_image()            
            
            if self.check_button_manager.trace_button_status:
                self.plot_manager.update_edc()
            else:
                self.plot_manager.update_time_trace()
                
            self.plot_manager.update_square()  # Update square position
            self.plot_manager.fig.canvas.draw()

        if self.press_vertical:    
            new_kx = event.xdata
            self.value_manager.update_kx_value(new_kx)
            self.plot_manager.vertical_line_0.set_xdata(x = new_kx)
            self.plot_manager.vertical_line_1.set_xdata(x = new_kx)
            self.plot_manager.update_ky_image()
            
            if self.check_button_manager.trace_button_status:
                self.plot_manager.update_edc()
            else:
                self.plot_manager.update_time_trace()
                
            self.plot_manager.update_square()  # Update square position
            self.plot_manager.fig.canvas.draw()
        
    def on_release(self, event):
        """Handle mouse release events."""
        self.press_horizontal = False
        self.press_vertical = False

class CheckButtonManager:
    def __init__(self):
        self.trace_check_button = self.create_trace_check_button()
        self.enhance_check_button = self.create_enhance_check_button()
        
        self.trace_button_status = False # for EDC
        self.enhance_button_status = False #for enhance CB
    
    def create_trace_check_button(self):
        trace_check_button = CheckButtons(plt.axes([0.005, 0.5, 0.06, 0.05]), ['EDC'])
        
        return trace_check_button

    def create_enhance_check_button(self):
        enhance_check_button = CheckButtons(plt.axes([0.045, 0.5, 0.08, 0.05]), ['Enhance CB'])
        
        return enhance_check_button
    
class ClickButtonManager:
    def __init__(self, plot_manager, check_button_manager):
        self.plot_manager = plot_manager
        self.check_button_manager = check_button_manager
        self.save_button = self.create_save_button()
        self.clear_button = self.create_clear_button()

        # Connect the button actions
        self.save_button.on_clicked(self.save_trace)
        self.clear_button.on_clicked(self.clear_traces)

        # Store saved traces separately
        self.saved_lines = []
        
    def create_save_button(self):
        save_button = Button(plt.axes([0.02, 0.94, 0.075, 0.04]), 'Keep Trace')

        return save_button
    
    def create_clear_button(self):
        clear_button = Button(plt.axes([0.02, 0.89, 0.075, 0.04]), 'Clear Traces')

        return clear_button

    def save_trace(self, event):
        # Get all the current dynamic lines from the plot (assuming they haven't been saved yet)
        lines = self.plot_manager.ax[1].get_lines()
        
        if lines:
            # Find the last unsaved line (assuming the latest plotted line is dynamic)
            current_line = lines[0]  
            
            # Copy the current line's data to "freeze" it as a saved trace
            x_data, y_data = current_line.get_xdata(), current_line.get_ydata()
            
            # Plot the saved trace with a different style (e.g., dashed gray line)
            saved_line, = self.plot_manager.ax[1].plot(x_data, y_data, linestyle='--')

            # Add the saved line to the list of saved lines, so it won't be modified later
            self.saved_lines.append(saved_line)
            
            # Redraw the canvas to ensure the saved trace is displayed
            self.plot_manager.fig.canvas.draw()

    def clear_traces(self, event):
        # Clear all lines from ax[1]
        self.plot_manager.ax[1].cla()  # Clear the axis
        
        if self.check_button_manager.trace_button_status:
            self.plot_manager.plot_edc()
        else:
            self.plot_manager.plot_time_trace()
            
        self.saved_lines.clear()  # Clear the saved traces list
        self.plot_manager.fig.canvas.draw()  # Redraw the canvas
        
class SliderManager:
    def __init__(self, value_manager, plot_manager, check_button_manager):
        self.plot_manager = plot_manager
        self.value_manager = value_manager
        self.check_button_manager = check_button_manager
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
        
        self.plot_manager.update_lines()
        self.plot_manager.update_square()
        self.plot_manager.update_kxky_image()
        
        if self.check_button_manager.trace_button_status:
            self.plot_manager.update_edc()
        else:
            self.plot_manager.update_time_trace()

        self.plot_manager.fig.canvas.draw()
    