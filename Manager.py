# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:36:40 2024

@author: lloyd
"""

#Manager.py

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import tkinter as tk
from tkinter import Menu, filedialog, Tk
from functools import partial
from matplotlib.widgets import Slider, CheckButtons, Button
from scipy.ndimage import map_coordinates
import mpes
from mpes import cmap_LTL, cmap_LTL2

class DataHandler:
    def __init__(self, value_manager, I):
        self.value_manager = value_manager
        self.I = I

    def calculate_dt(self):
        if self.I.ndim > 3:
            dt = self.I.delay.values[1] - self.I.delay.values[0]
            return dt
        else:
            return 1
    
    def calculate_dk(self):
        dk = self.I.kx.values[2] - self.I.kx.values[1]
        
        return dk

class ValueHandler:
    def __init__(self):
        self.k_int, self.kx, self.ky, self.E, self.E_int, self.delay, self.delay_int = 0.4, 0, 0, 0, 0.100, 100, 1000

    def update_k_int_value(self, k_int):
        self.k_int = k_int
        
    def update_kx_value(self, kx):
        self.kx = kx        
    
    def update_ky_value(self, ky):
        self.ky = ky   
        
    def update_E_value(self, E):
        self.E = E

    def update_E_int_value(self, E_int):
        self.E_int = E_int
    
    def update_delay_value(self, delay):
        self.delay = delay 
        
    def update_delay_int_value(self, delay_int):
        self.delay_int = delay_int 
        
    def get_values(self):
        return self.k_int, self.kx, self.ky, self.E, self.E_int, self.delay, self.delay_int

class FigureHandler:
    def __init__(self):
        self.fig, self.ax = self.create_fig_axes()
        
    def create_fig_axes(self):
        # Create the figure and axes for subplots
        fig, ax = plt.subplots(nrows=2, ncols=2, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1]})
        fig.set_size_inches(15, 10)
        #fig.canvas.manager.set_window_title("MPES GUI")

        return fig, ax.flatten()
        
class PlotHandler:
    def __init__(self, figure_handler, data_handler, value_manager, check_button_manager):
        self.figure_handler = figure_handler
        self.data_handler = data_handler
        self.I = data_handler.I
        self.value_manager = value_manager
        self.check_button_manager = check_button_manager
        self.fig, self.ax = self.figure_handler.fig, self.figure_handler.ax
        self.cmap = self.custom_colormap(mpl.cm.viridis, 0.25)
        self.im_1, self.im_2, self.im_3, self.im_4 = None, None, None, None
        self.time_trace_1 = None
        self.Energy_limits = None
        self.E_enhance = 1
        
        # Initial setup for plots
        self.initialize_plots()

    def initialize_plots(self):
        
        # Define intial kx, ky, Energy, and Delay Points for Images
        k_int, kx, ky, E, E_int, delay, delay_int = self.value_manager.get_values()

        # Initial Momentum Map kx, ky Image (top left)
        frame_temp = mpes.get_momentum_map(self.I, E, E_int, delay, delay_int)
        frame_temp = frame_temp/np.max(frame_temp)
        
        self.im_1 = frame_temp.plot.imshow(ax = self.ax[0], clim = None, vmin = 0, vmax = 1, cmap = self.cmap, add_colorbar=False)
            
        self.ax[0].set_title("E = " + str(E) + ' eV')
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
        frame_temp = mpes.get_kx_E_frame(self.I, ky, k_int, delay, delay_int)
        frame_temp = (frame_temp)/np.max(frame_temp)
        
        self.Energy_limits = (frame_temp.E.values.min(), frame_temp.E.values.max())
        self.im_2 = frame_temp.T.plot.imshow(ax = self.ax[2], clim = None, vmin = 0, vmax = 1, cmap = self.cmap, add_colorbar=False)

        self.ax[2].set_title("Energy Cut")
        self.ax[2].set_xlabel('$k_x$', fontsize = 14)
        self.ax[2].set_ylabel("E, eV")
        self.ax[2].set_aspect('auto')
        self.ax[2].set_xticks(np.arange(-2,3.1,0.5))
        for label in self.ax[2].xaxis.get_ticklabels()[1::2]:
            label.set_visible(False)
        self.ax[2].set_yticks(np.arange(-10,5.1,0.5))
        for label in self.ax[2].yaxis.get_ticklabels()[1::2]:
            label.set_visible(False)   
        self.ax[2].set_xlim(-2.25,2.25)
        self.ax[2].set_ylim(self.Energy_limits[0],self.Energy_limits[1])

        # Initial ky vs E Image (bottom left)
        frame_temp = mpes.get_ky_E_frame(self.I, kx, k_int, delay, delay_int)
        frame_temp = (frame_temp)/np.max(frame_temp)
        
        self.im_3 = frame_temp.T.plot.imshow(ax = self.ax[3], clim = None, vmin = 0, vmax = 1, cmap = self.cmap, add_colorbar=False)

        self.ax[3].set_title("Energy Cut")
        self.ax[3].set_xlabel('$k_y$', fontsize = 14)
        self.ax[3].set_ylabel("E, eV")
        self.ax[3].set_aspect('auto')
        self.ax[3].set_xticks(np.arange(-3,3.1,0.5))
        for label in self.ax[3].xaxis.get_ticklabels()[1::2]:
            label.set_visible(False)
        self.ax[3].set_yticks(np.arange(-10,5.1,0.5))
        for label in self.ax[3].yaxis.get_ticklabels()[1::2]:
            label.set_visible(False)   
        self.ax[3].set_xlim(-2.25,2.25)
        self.ax[3].set_ylim(self.Energy_limits[0],self.Energy_limits[1])
        
        # Initial Dynamics Time Trace (top right)
        if self.I.ndim > 3:
            self.plot_time_trace()
            print('hi')  
        else:
            self.plot_edc()      

        # Add interactive horizontal and vertical lines (for cuts)
        self.horizontal_line_0 = self.ax[0].axhline(y=kx, color='black', linestyle='--', linewidth = 1.5)
        self.vertical_line_0 = self.ax[0].axvline(x=ky, color='black', linestyle='--', linewidth = 1.5)
    
        self.horizontal_line_1 = self.ax[2].axhline(y=E, color='black', linestyle='--', linewidth = 1.5)
        self.vertical_line_1 = self.ax[2].axvline(x=kx, color='black', linestyle='--', linewidth = 1.5)
                
        self.horizontal_line_2 = self.ax[3].axhline(y=E, color='black', linestyle='--', linewidth = 1.5)
        self.vertical_line_2 = self.ax[3].axvline(x=ky, color='black', linestyle='--', linewidth = 1.5)

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
        k_int, kx, ky, E, E_int, delay, delay_int = self.value_manager.get_values()

        edc = mpes.get_edc(self.I, kx, ky, (k_int, k_int), delay, delay_int)
        edc = edc/np.max(edc)
        
        self.im_4, = self.ax[1].plot(self.I.E.values, edc, color = 'black')
       # self.ax[1].set_xticks(np.arange(-400,1250,200))
        #for label in self.ax[1].xaxis.get_ticklabels()[1::2]:
         #   label.set_visible(False)
        self.ax[1].set_ylim([-0.1, 1.1])
        self.ax[1].set_xlim([self.I.E.values[0], self.I.E.values[-1]])
        self.ax[1].set_title("EDC")
        self.ax[1].set_xlabel("Energy, eV")
        self.ax[1].set_ylabel("Intensity")
                
    def plot_time_trace(self):
        k_int, kx, ky, E, E_int, delay, delay_int = self.value_manager.get_values()
        k = (kx, ky)
        norm_trace = True
        subtract_neg = False

        if self.I.ndim > 3:
            time_trace = mpes.get_time_trace(self.I, E, E_int, k, (k_int, k_int), norm_trace = True, subtract_neg=False, neg_delays = [-200,-50])
#            time_trace = self.data_handler.I[idx_kx[0]:idx_kx[1], idx_ky[0]:idx_ky[1], idx_E-2:idx_E+3, :].sum(axis=(0,1,2))
            
            self.im_4, = self.ax[1].plot(self.I.delay.values, time_trace/np.max(time_trace), color = 'black')
            self.ax[1].set_xticks(np.arange(-300,1250,100))
            for label in self.ax[1].xaxis.get_ticklabels()[1::2]:
                label.set_visible(False)
            self.ax[1].set_xlim([self.I.delay.values[1], self.I.delay.values[-1]])

        else:
            time_trace = np.zeros(1)
            
        self.ax[1].set_ylim([-0.2, 1.1])
        self.ax[1].set_title("Dynamics")
        self.ax[1].set_xlabel("Delay, fs")
        self.ax[1].set_ylabel("Intensity")

    def create_waterfall_plot(self):
        k_int, kx, ky, E, E_int, delay, delay_int = self.value_manager.get_values()
        
        self.ax[1].cla()
        if self.check_button_manager.difference_button_status is True:
            subtract_neg = True
        else:
            subtract_neg = False

        energy_limits=[0.75, 2.5]
        
        f, a, wf = mpes.plot_waterfall(
            self.I, kx, k_int, ky, k_int,
            fig = self.fig, ax = self.ax[1],
            subtract_neg=subtract_neg, energy_limits=energy_limits
            )
        
        #self.ax[1].set_ylim(energy_limits[0], energy_limits[1])
        #self.ax[1].set_xlim(self.I.delay.values[1], self.I.delay.values[-2])
            
    def update_kxky_image(self):
        k_int, kx, ky, E, E_int, delay, delay_int = self.value_manager.get_values()
        frame_temp = mpes.get_momentum_map(self.I, E, E_int, delay, delay_int)

        self.im_1.set_data(frame_temp/np.max(frame_temp))  # Update image for new E
        self.ax[0].set_title("E = " + str(round(E,2)) + ' eV')
        
    def update_kx_image(self):
        k_int, kx, ky, E, E_int, delay, delay_int = self.value_manager.get_values()

        frame_temp = mpes.get_kx_E_frame(self.I, ky, 0.1, delay, delay_int)
        frame_temp = frame_temp.T/np.max(frame_temp)
        #self.E_enhance = -0.9
        if self.check_button_manager.enhance_button_status == True:    
            mask_start = (np.abs(self.I.E.values - self.E_enhance)).argmin()
            frame_temp[mask_start:,:] *= 1/np.max(frame_temp[mask_start:,:])
        self.im_2.set_data(frame_temp)  # Update image for new E

    def update_ky_image(self):
        k_int, kx, ky, E, E_int, delay, delay_int = self.value_manager.get_values()

        frame_temp = mpes.get_ky_E_frame(self.I, kx, 0.1, delay, delay_int)
        frame_temp = frame_temp.T/np.max(frame_temp)
        #self.E_enhance = -0.9

        if self.check_button_manager.enhance_button_status == True:    
            mask_start = (np.abs(self.I.E.values - self.E_enhance)).argmin()
            frame_temp[mask_start:,:] *= 1/np.max(frame_temp[mask_start:,:])
            
        self.im_3.set_data(frame_temp)  # Update image for new E
    
    def plot_k_cut(self, k_frame):
        i = 1
        self.im_k_frame = k_frame.plot.imshow(ax=self.ax[i], cmap=self.cmap, add_colorbar=False, vmin=0, vmax=1) #kx, ky, t
        
        self.ax[i].set_xticks(np.arange(-2,3.5,.5))
        for label in self.ax[i].xaxis.get_ticklabels()[1::2]:
            label.set_visible(False)
        self.ax[i].set_yticks(np.arange(-4,4.1,0.5))
        for label in self.ax[i].yaxis.get_ticklabels()[1::2]:
            label.set_visible(False)
#        self.ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        self.ax[i].set_xlabel(r'$k_{//}$, $\AA^{{-1}}$', fontsize = 18)
        self.ax[i].set_ylabel(r'$E - E_{{VBM}}, eV$', fontsize = 18)
        self.ax[i].set_title("E vs k slice", color = 'black', fontsize = 18)
        self.ax[i].tick_params(axis='both', labelsize=16)
        self.ax[i].set_xlim(0,k_frame.k.values.max())
        self.ax[i].set_ylim(k_frame.E.values[0], k_frame.E.values[-1])
        
        print('plotting k cut')

    def update_k_cut(self, k_frame):
        self.im_k_frame.set_data(k_frame)
        self.ax[1].set_xlim(0,k_frame.k.values.max())
        self.ax[1].set_ylim(k_frame.E.values[0], k_frame.E.values[-1])
        print('updating k cut')
        
    def update_time_trace(self):
        """Update the time traces when the square is moved or resized."""
        k_int, kx, ky, E, E_int, delay, delay_int = self.value_manager.get_values()
        k = (kx, ky)
        # Update the time trace plots
        self.ax[1].set_title("Dynamics")
        self.ax[1].set_xlabel("Delay, fs")
        self.ax[1].set_ylabel("Intensity")
        
        if self.data_handler.I.ndim > 3:
            if self.check_button_manager.difference_button_status is True:
                time_trace = mpes.get_time_trace(self.I, E, E_int, k, (k_int, k_int), norm_Trace = True, subtract_neg = True, neg_delays =  [-280,-100])
                time_trace = time_trace/np.max(np.abs(time_trace))
                self.im_4.set_data(time_trace.delay.values, time_trace/np.max(time_trace))

                print(time_trace)
                self.ax[1].set_ylim([-0.2, 1.1])
                self.ax[1].set_xlim([self.I.delay.values[1], self.I.delay.values[-1]])
                print('difference negative!!!')

            else:
                print('not negative!!!')
                time_trace = mpes.get_time_trace(self.I, E, E_int, k, (k_int, k_int), norm_Trace = True, subtract_neg = False)
                time_trace = time_trace/np.max(np.abs(time_trace))
                self.im_4.set_data(time_trace.delay.values, time_trace/np.max(time_trace))
                self.ax[1].set_ylim(0, 1.1)

        else:
            time_trace = np.zeros(1)
            print('no time delay data..')

    def update_edc(self):
        """Update the time traces when the square is moved or resized."""
        k_int, kx, ky, E, E_int, delay, delay_int = self.value_manager.get_values()

        if self.data_handler.I.ndim > 3:
            edc = mpes.get_edc(self.I, kx, ky, (k_int, k_int), delay, delay_int)
        
        else:
            edc = mpes.get_edc(self.I, kx, ky, (k_int, k_int), delay, delay_int)
            
        edc = edc/np.max(edc)

        if self.check_button_manager.enhance_check_button.get_status()[0] is True:
            mask_start = (np.abs(self.I.E.values - 1.0)).argmin()
            edc[mask_start:] *= 1/np.max(edc[mask_start:])
            
        # Update the edc plots
        self.im_4.set_xdata(self.I.E.values)
        self.im_4.set_ydata(edc/np.max(edc))
        self.ax[1].set_ylim([-0.1, 1.1])
        self.ax[1].set_xticks(np.arange(-6,4,0.5))
        for label in self.ax[1].xaxis.get_ticklabels()[1::2]:
            label.set_visible(False)
        self.ax[1].set_xlim([self.I.E.values[0], self.I.E.values[-1]])
        self.ax[1].set_title("EDC")
        self.ax[1].set_xlabel("Energy, eV")
        self.ax[1].set_ylabel("Intensity")
        #self.ax[1].axvline(E, color = 'black', linestyle = 'dashed')
        
    def update_lines(self):
        k_int, kx, ky, E, E_int, delay, delay_int = self.value_manager.get_values()
        self.horizontal_line_1.set_ydata(y = E)
        self.horizontal_line_2.set_ydata(y = E)
        
    def update_square(self):
        """Update square position on plot."""
        k_int, kx, ky, E, E_int, delay, delay_int = self.value_manager.get_values()
        
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
    def __init__(self, value_manager, slider_manager, plot_manager, check_button_manager, arbitrary_cut_handler, waterfallHandler):
        self.slider_manager = slider_manager
        self.plot_manager = plot_manager
        self.check_button_manager = check_button_manager
        self.value_manager = value_manager
        self.press_horizontal = False
        self.press_vertical = False
        self.arbitrary_cut_handler = arbitrary_cut_handler
        self.waterfall_handler = waterfallHandler
        
        def enable_right_click_menu(fig, menu_items, target_ax=None):
                root = tk.Tk()
                root.withdraw()
            
                def on_right_click(event):
                    if event.button == 3 and (target_ax is None or event.inaxes == target_ax):
                        menu = Menu(root, tearoff=0)
                        for label, callback in menu_items:
                            menu.add_command(label=label, command=callback)

                        # Use tkinter to get pointer position (safe for all backends)
                        x, y = root.winfo_pointerx(), root.winfo_pointery()

                        try:
                            menu.tk_popup(x, y)
                        finally:
                            menu.grab_release()

                fig.canvas.mpl_connect("button_press_event", on_right_click)
        
        enable_right_click_menu(
            fig=self.plot_manager.fig,
            menu_items=[
                ("Waterfall Plot", partial(self.waterfall_plot)),
                ("Subtract Neg Delays",partial(self.show_difference_spectra)),
                ("Chose k-cut",partial(self.choose_k_cut)),
                ("Revert",partial(self.revert_to_original))
            ]
        )

    def waterfall_plot(self):
        ax = self.plot_manager.ax[1]
        print('waterfalling')
        self.check_button_manager.waterfall_button_status = True

        if self.check_button_manager.waterfall_button_status is True:
            ax.cla()  # Clear the subplot
            self.waterfall_handler.enable()
        elif self.check_button_manager.waterfall_button_status is False and self.check_button_manager.kcut_button_status is False:
            self.plot_manager.plot_time_trace()

        self.plot_manager.fig.canvas.draw()

    def show_difference_spectra(self):
        ax = self.plot_manager.ax[1]
        print('create difference')
        self.check_button_manager.difference_button_status = True
        
        if self.check_button_manager.waterfall_button_status is True:
            ax.cla()  # Clear the subplot
            self.waterfall_handler.enable()
        elif self.check_button_manager.waterfall_button_status is False and self.check_button_manager.kcut_button_status is False:
            self.plot_manager.update_time_trace()

        self.plot_manager.fig.canvas.draw()

    def choose_k_cut(self):
        ax = self.plot_manager.ax[1]
        #ax.cla()  # Clear the subplot
        print('choosing k cut')
        self.check_button_manager.difference_button_status = False
        self.check_button_manager.waterfall_button_status = False
        self.check_button_manager.kcut_button_status = True
        ax.cla()  # Clear the subplot

        if self.check_button_manager.waterfall_button_status is True:
            self.waterfall_handler.enable()
        elif self.check_button_manager.waterfall_button_status is False and self.check_button_manager.kcut_button_status is False:
            self.plot_manager.plot_time_trace()
        elif self.check_button_manager.kcut_button_status is True:
            self.arbitrary_cut_handler.enable()

        self.plot_manager.fig.canvas.draw()

    def revert_to_original(self):
        ax = self.plot_manager.ax[1]
        #ax.cla()  # Clear the subplot
        print('reverting')
        self.check_button_manager.difference_button_status = False

        ax.cla()  # Clear the subplot

        if self.check_button_manager.waterfall_button_status is True:
            self.waterfall_handler.disable()
            self.check_button_manager.waterfall_button_status = False
        elif self.check_button_manager.kcut_button_status is True:
            self.arbitrary_cut_handler.disable()
            self.check_button_manager.kcut_button_status = False
        else:
            self.plot_manager.plot_time_trace()

        self.plot_manager.fig.canvas.draw()

    def on_checkbox_change(self, label):
        # K-Cut
        if label == "k-Cut":
            if self.check_button_manager.kcut_check_button.get_status()[0]:
                self.check_button_manager.kcut_button_status = True
                self.arbitrary_cut_handler.enable()
            else:
                self.check_button_manager.kcut_button_status = False
                self.arbitrary_cut_handler.disable()
            print('k-cut button')

        # Waterfall
        elif label == "Waterfall":
            if self.check_button_manager.waterfall_button.get_status()[0]:
                self.check_button_manager.waterfall_button_status = True
                self.waterfall_handler.enable()
                print('waterfall button')
            else:
                self.check_button_manager.waterfall_button_status = False
                self.waterfall_handler.disable()

        # Redraw everything after handling any change
        self.plot_manager.fig.canvas.draw()

    def on_press(self, event):
        """Handle mouse press events and update square or lines."""
        if self.plot_manager.horizontal_line_0.contains(event)[0]:
            self.press_horizontal = True    
        if self.plot_manager.vertical_line_0.contains(event)[0]:
            self.press_vertical = True 
        
        # if self.check_button_manager.trace_check_button.get_status()[0]: # if EDC button selected
        #     self.check_button_manager.trace_button_status = True
        #     self.plot_manager.update_edc()
        #     self.plot_manager.fig.canvas.draw()
        # elif self.check_button_manager.trace_check_button.get_status()[0] is False and self.plot_manager.data_handler.I.ndim > 3:
        #     self.check_button_manager.trace_button_status = False            
        #     self.plot_manager.update_time_trace()
        #     self.plot_manager.fig.canvas.draw()

        # if self.check_button_manager.enhance_check_button.get_status()[0] is False: # if Enhance CB feature visibility OFF
        #     self.check_button_manager.enhance_button_status = False
        #     self.plot_manager.update_kx_image()
        #     self.plot_manager.update_ky_image()
        #     self.plot_manager.fig.canvas.draw()
        # elif self.check_button_manager.enhance_check_button.get_status()[0] is True: # if Enhance CB feature visibility ON
        #     self.check_button_manager.enhance_button_status = True
        #     self.plot_manager.update_kx_image()
        #     self.plot_manager.update_ky_image()
        #     if self.check_button_manager.trace_check_button.get_status()[0] is True:
        #         self.plot_manager.update_edc()
        #     self.plot_manager.fig.canvas.draw()

        #self.check_button_manager.kcut_check_button.on_clicked(self.on_checkbox_change)
        #self.check_button_manager.waterfall_button.on_clicked(self.on_checkbox_change)

        # # Arb. k-cut Button
        # if self.check_button_manager.kcut_check_button.get_status()[0]:
        #     print('Enabled!')
        #     self.check_button_manager.kcut_button_status = True
        #     self.arbitrary_cut_handler.enable()
        
        #     #self.plot_manager.fig.canvas.draw()

        # else:
        #     self.check_button_manager.kcut_button_status = False
        #     self.arbitrary_cut_handler.disable()
        
        #     self.plot_manager.fig.canvas.draw()

        # # Waterfall Spectra Button
        # if self.check_button_manager.waterfall_button.get_status()[0]:
        #     print('taking waterfall spectra')

        #     self.check_button_manager.waterfall_button_status = True
        #     self.waterfall_handler.enable()
        #     self.plot_manager.fig.canvas.draw()
        # else:
        #     self.check_button_manager.waterfall_button_status = False

        #     self.waterfall_handler.disable()
        #     self.plot_manager.fig.canvas.draw()

        # Difference Spectra Button
        #if self.check_button_manager.difference_button.get_status()[0]:
         #   self.check_button_manager.difference_button_status = True

    def on_motion(self, event):
        if self.press_horizontal:    
            new_ky = event.ydata
            self.value_manager.update_ky_value(new_ky)
            self.plot_manager.update_kx_image()            
            self.plot_manager.horizontal_line_0.set_ydata(y = new_ky)
            self.plot_manager.vertical_line_2.set_xdata(x = new_ky)
            self.plot_manager.update_square()  # Update square position
            
            if self.check_button_manager.trace_button_status and self.check_button_manager.kcut_button_status is False and self.check_button_manager.waterfall_button_status is False:
                self.plot_manager.update_edc()
            elif  self.check_button_manager.trace_button_status is False and self.check_button_manager.kcut_button_status is False and self.check_button_manager.waterfall_button_status is False:
                self.plot_manager.update_time_trace()
                
            self.plot_manager.fig.canvas.draw()

        if self.press_vertical:    
            new_kx = event.xdata
            self.value_manager.update_kx_value(new_kx)
            self.plot_manager.update_ky_image()

            self.plot_manager.vertical_line_0.set_xdata(x = new_kx)
            self.plot_manager.vertical_line_1.set_xdata(x = new_kx)
            self.plot_manager.update_square()  # Update square position

            if self.check_button_manager.trace_button_status and self.check_button_manager.kcut_button_status is False and self.check_button_manager.waterfall_button_status is False:
                self.plot_manager.update_edc()
            elif  self.check_button_manager.trace_button_status is False and self.check_button_manager.kcut_button_status is False and self.check_button_manager.waterfall_button_status is False:
                self.plot_manager.update_time_trace()
                
            self.plot_manager.fig.canvas.draw()
        
    def on_release(self, event):
        """Handle mouse release events."""
        self.press_horizontal = False
        self.press_vertical = False

class ArbitraryCutHandler:
    def __init__(self, plot_manager, data_handler, check_button_manager):
        self.plot_manager = plot_manager
        self.check_button_manager = check_button_manager
        self.data_handler = data_handler
        self.ax = self.plot_manager.ax[0]
        self.kx_vals = self.data_handler.I.kx.values
        self.ky_vals = self.data_handler.I.ky.values
        self.E_vals = self.data_handler.I.E.values
        self.I = self.data_handler.I  # [kx, ky, E] or [kx, ky, E, delay]
        
        # Initial line endpoints in kx-ky space
        self.x1, self.y1 = -1.0, -1.0
        self.x2, self.y2 = 1.0, 1.0

        # Plot line and endpoints
        self.line = Line2D([self.x1, self.x2], [self.y1, self.y2], color='purple', linewidth=2)
        self.p1 = Circle((self.x1, self.y1), 0.05, color='purple', picker=True)
        self.p2 = Circle((self.x2, self.y2), 0.05, color='purple', picker=True)
        self.ax.add_line(self.line)
        self.ax.add_patch(self.p1)
        self.ax.add_patch(self.p2)
        
        # Hide initially
        self.line.set_visible(False)
        self.p1.set_visible(False)
        self.p2.set_visible(False)
        self.active_point = None

        # Connect events
        fig = self.ax.figure
        fig.canvas.mpl_connect('button_press_event', self.on_press)
        fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        fig.canvas.mpl_connect('button_release_event', self.on_release)

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        if self.p1.contains(event)[0]:
            self.active_point = self.p1
        elif self.p2.contains(event)[0]:
            self.active_point = self.p2

    def on_motion(self, event):
        if self.active_point and event.inaxes == self.ax:
            self.active_point.center = (event.xdata, event.ydata)
            self.x1, self.y1 = self.p1.center
            self.x2, self.y2 = self.p2.center
            self.line.set_data([self.x1, self.x2], [self.y1, self.y2])
            
            k_frame = mpes.get_k_cut(self.I, (self.x1, self.y1), (self.x2, self.y2), 500, 2000)
            k_frame = k_frame/np.max(np.abs(k_frame))
            self.plot_k_cut()
        
        self.plot_manager.fig.canvas.draw()

    def on_release(self, event):
        self.active_point = None
        self.x1, self.y1 = self.p1.center
        self.x2, self.y2 = self.p2.center
        #self.plot_k_cut()

        #self.plot_manager.fig.canvas.draw()

    def plot_k_cut(self):
        i = 1
        k_frame = mpes.get_k_cut(self.I, (self.x1, self.y1), (self.x2, self.y2), 500, 2000)
        k_frame = k_frame/np.max(np.abs(k_frame))

        if self.check_button_manager.enhance_button_status is True:
            k_frame = mpes.enhance_features(k_frame, 0.8, 1, True)

        self.plot_manager.plot_k_cut(k_frame)

    def enable(self):
        print('On!')
        #self.plot_manager.ax[0].cla()
        self.plot_k_cut()

        self.line.set_visible(True)
        self.p1.set_visible(True)
        self.p2.set_visible(True)
        #self.plot_manager.ax[1].cla()  # Clear the axis

        self.plot_manager.fig.canvas.draw()

    def disable(self):
        self.line.set_visible(False)
        self.p1.set_visible(False)
        self.p2.set_visible(False)
        self.plot_manager.ax[1].cla()  # Clear the axis
        # Revert to EDC or TimeTrace
        # if self.data_handler.I.ndim > 3:
        #     self.plot_handler.plot_time_trace()
        if self.check_button_manager.trace_button_status is True:
            self.plot_manager.plot_edc()
        
        elif self.check_button_manager.trace_button_status is False and self.check_button_manager.waterfall_button_status is False:
            self.plot_manager.plot_time_trace()
        #elif self.check_button_manager.waterfall_button_status is True:
         #   self.waterfall_handler.enable()

        self.plot_manager.fig.canvas.draw()

class waterfallHandler:
    def __init__(self, plot_manager, data_handler, check_button_manager):
        self.plot_manager = plot_manager
        self.check_button_manager = check_button_manager
        self.data_handler = data_handler

    #def update_waterfall(self):
        #self.plot_manager.create_waterfall_plot()
    def create_waterfall(self):
        self.plot_manager.create_waterfall_plot()

    def enable(self):
        self.create_waterfall()
        self.plot_manager.fig.canvas.draw()

    def disable(self):
        self.plot_manager.ax[1].cla()  # Clear the axis
        # Revert to EDC or TimeTrace
        # if self.data_handler.I.ndim > 3:
        #     self.plot_handler.plot_time_trace()
        #self.wf.set_visible(False)
        #self.plot_manager.im_4.set_visible(True)

        if self.check_button_manager.trace_button_status is True:
            self.plot_manager.plot_edc()
        elif self.check_button_manager.trace_button_status is False and self.check_button_manager.kcut_button_status is False:
            self.plot_manager.plot_time_trace()

        self.plot_manager.fig.canvas.draw()

class CheckButtonManager:
    def __init__(self):
        self.trace_check_button = self.create_trace_check_button()
        self.enhance_check_button = self.create_enhance_check_button()
        self.kcut_check_button = self.create_kcut_check_button()  # for arb. k cut
        self.waterfall_button = self.create_waterfall_button()  # for arb. k cut
        self.difference_button = self.create_difference_button()  # for arb. k cut

        self.trace_button_status = False # for EDC
        self.enhance_button_status = False #for enhance CB
        self.kcut_button_status = False  # for arb. k cut
        self.waterfall_button_status = False  # for k-intergrated waterfall dynamics
        self.difference_button_status = False  # for k-intergrated waterfall dynamics

    def create_trace_check_button(self):
        ax = plt.axes([0.005, 0.5, 0.06, 0.05])
        trace_check_button = CheckButtons(ax, ['EDC'])
        
        # Remove background and borders
        ax.set_facecolor('none')            # transparent background
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

        return trace_check_button

    def create_enhance_check_button(self):
        ax = plt.axes([0.045, 0.5, 0.08, 0.05])
        enhance_check_button = CheckButtons(ax, ['Enhance CB'])

        # Remove background and borders
        ax.set_facecolor('none')            # transparent background
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)    
        
        return enhance_check_button
    
    def create_kcut_check_button(self):
        ax = plt.axes([0.41, 0.89, 0.15, 0.08])
        kcut_check_button = CheckButtons(ax, ['k-Cut'])  # layout can be tweaked
        
        # Remove background and borders
        ax.set_facecolor('none')            # transparent background
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        return kcut_check_button
    
    def create_waterfall_button(self):
        ax = plt.axes([0.41, 0.85, 0.15, 0.08])
        waterfall_button = CheckButtons(ax, ['Waterfall'])  # layout can be tweaked
        
        # Remove background and borders
        ax.set_facecolor('none')            # transparent background
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        return waterfall_button

    def create_difference_button(self):
        ax = plt.axes([0.41, 0.5, 0.12, 0.1])
        difference_button = CheckButtons(ax, ['Difference'])  # layout can be tweaked
        
        # Remove background and borders
        ax.set_facecolor('none')            # transparent background
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        return difference_button
    
class ClickButtonManager:
    def __init__(self, plot_manager, check_button_manager, fig):
        self.plot_manager = plot_manager
        self.check_button_manager = check_button_manager
        self.save_button = self.create_save_button()
        self.clear_button = self.create_clear_button()
        self.save_fig_button = self.create_save_fig_button()
        self.fig = fig

        # Store saved traces separately
        self.saved_lines = []
        
        # Connect the button actions
        self.save_button.on_clicked(self.save_trace)
        self.clear_button.on_clicked(self.clear_traces)

        self.save_fig_button.on_clicked(self.save_figure)

    def create_save_fig_button(self):
        save_fig_button = Button(plt.axes([0.43, 0.97, 0.05, 0.02]), 'Save Fig')
                    
        return save_fig_button

    def create_save_button(self):
        save_button = Button(plt.axes([0.02, 0.94, 0.075, 0.04]), 'Keep Trace')

        return save_button
    
    def create_clear_button(self):
        clear_button = Button(plt.axes([0.02, 0.89, 0.075, 0.04]), 'Clear Traces')

        return clear_button

    def save_figure(self, event):
            # Hide tkinter root window
            root = Tk()
            root.withdraw()

            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png"), ("PDF File", "*.pdf"), ("SVG Vector", "*.svg")],
                title="Save figure as..."
            )

            if file_path:
                self.plot_manager.fig.savefig(file_path, dpi=300)
                print(f"Figure saved to {file_path}")

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
        self.plot_manager.fig.canvas.draw()

class SliderManager:
    def __init__(self, value_manager, plot_manager, check_button_manager, waterfall_handler):
        self.plot_manager = plot_manager
        self.value_manager = value_manager
        self.check_button_manager = check_button_manager
        self.waterfall_handler = waterfall_handler
        self.E_slider, self.E_int_slider, self.k_int_slider, self.delay_slider, self.delay_int_slider = self.create_sliders()
        
    def create_sliders(self):
        """Create the sliders for energy and delay."""
        E_slider = Slider(plt.axes([0.015, 0.6, 0.03, 0.25]), 'E, eV', -10, 5, valinit=0, valstep = 0.05, color = 'black', orientation = 'vertical')
        E_int_slider = Slider(plt.axes([0.057, 0.6, 0.03, 0.25]), '$\Delta$E, eV', 0, 500, valinit=100, valstep = 50, color = 'grey', orientation = 'vertical')
        k_int_slider = Slider(plt.axes([0.42, 0.6, 0.03, 0.25]), '$\Delta k$, $A^{-1}$', 0, 4, valinit=.5, valstep = 0.1, color = 'red', orientation = 'vertical')
        delay_slider = Slider(plt.axes([0.055, 0.02, 0.25, 0.03]), 'Delay, fs', -200, 1000, valinit=100, valstep = 20, color = 'purple', orientation = 'horizontal')
        delay_int_slider = Slider(plt.axes([0.055, 0.001, 0.25, 0.03]), 'Delay, fs', 0, 1000, valinit=1000, valstep = 20, color = 'violet', orientation = 'horizontal')

        return E_slider, E_int_slider, k_int_slider, delay_slider, delay_int_slider
    
    def on_slider_update(self, val):
        """Update plots based on slider values."""
        E = self.E_slider.val
        E_int = self.E_int_slider.val/1000
        k_int = self.k_int_slider.val
        delay = self.delay_slider.val
        delay_int = self.delay_int_slider.val

        self.value_manager.update_E_value(E)

        self.value_manager.update_E_int_value(E_int)

        self.value_manager.update_k_int_value(k_int)
        self.value_manager.update_delay_value(delay)
        self.value_manager.update_delay_int_value(delay_int)

        self.plot_manager.update_lines()
        self.plot_manager.update_square()
        self.plot_manager.update_kxky_image()
        #self.plot_manager.update_kx_image()
        #self.plot_manager.update_ky_image()
        if self.check_button_manager.trace_button_status is True and self.check_button_manager.kcut_button_status is False and self.check_button_manager.waterfall_button_status is False:
            self.plot_manager.update_edc()

        elif self.check_button_manager.trace_button_status is False and self.check_button_manager.kcut_button_status is False and self.check_button_manager.waterfall_button_status is False:
            print('updating time trace after slider')
            self.plot_manager.update_time_trace()

        elif self.check_button_manager.waterfall_button_status is True:
            
            self.waterfall_handler.enable()
            
        self.plot_manager.fig.canvas.draw()
    