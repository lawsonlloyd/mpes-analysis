# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:36:40 2024

@author: lloyd
"""

#Manager.py

import numpy as np

class DataHandler:
    def __init__(self, I, ax_kx, ax_ky, ax_E_offset, ax_delay_offset):
        self.I = I
        self.ax_kx = ax_kx
        self.ax_ky = ax_ky
        self.ax_E_offset = ax_E_offset
        self.ax_delay_offset = ax_delay_offset

    def calculate_dt(self):
        if self.I.ndim > 3:
            dt = self.ax_delay_offset[1] - self.ax_delay_offset[0]
            return dt
        else:
            return 1
    
    def get_t0(self):
        if self.I.ndim > 3:
            t0 = (np.abs(self.ax_delay_offset - 0)).argmin()
            return t0
        else:
            return 1
        
    def get_closest_indices(self, kx, ky, E, delay):
        kx_idx = (np.abs(self.ax_kx - kx)).argmin()
        ky_idx = (np.abs(self.ax_ky - ky)).argmin()
        E_idx = (np.abs(self.ax_E_offset - E)).argmin()
        delay_idx = (np.abs(self.ax_delay_offset - delay)).argmin()
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

    
class PlotManager:
    def __init__(self, data_handler):
        self.data_handler = data_handler
        self.fig, self.ax = self.create_fig_axes()
        self.cmap = 'terrain_r'
        self.im_1, self.im_2 = None, None
        self.time_trace_1, self.time_trace_2 = None, None
        self.horizontal_line, self.vertical_line = None, None

        # Square instance to manage the interactive region
        self.square = Square(0, 0, 0.5)  # Default square with center at (0, 0)

        # Initial setup for plots
        self.initialize_plots()

    def create_fig_axes(self):
        # Create the figure and axes for subplots
        fig, ax = plt.subplots(nrows=2, ncols=2, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1]})
        fig.set_size_inches(15, 10)
        return fig, ax.flatten()

    def initialize_plots(self):
        # Initial image plot (left)
        self.im_1 = self.ax[0].imshow(self.data_handler.I[:, :, 0], cmap=self.cmap, aspect='auto', origin='lower')
        self.ax[0].set_title("2D Intensity Map")

        # Time trace 1 (right)
        self.time_trace_1, = self.ax[1].plot(self.data_handler.ax_delay_offset, np.zeros_like(self.data_handler.ax_delay_offset))
        self.ax[1].set_title("Time Trace 1")
        self.ax[1].set_xlabel("Time Delay (fs)")
        self.ax[1].set_ylabel("Intensity")

        # Another plot for energy cut (bottom left)
        self.im_2 = self.ax[2].imshow(self.data_handler.I[:, :, 0], cmap=self.cmap, aspect='auto', origin='lower')
        self.ax[2].set_title("Energy Cut")

        # Time trace 2 (bottom right)
        self.time_trace_2, = self.ax[3].plot(self.data_handler.ax_delay_offset, np.zeros_like(self.data_handler.ax_delay_offset))
        self.ax[3].set_title("Time Trace 2")
        self.ax[3].set_xlabel("Time Delay (fs)")
        self.ax[3].set_ylabel("Intensity")

        # Add interactive horizontal and vertical lines (for cuts)
        self.horizontal_line = self.ax[0].axhline(y=0, color='r', linestyle='--')
        self.vertical_line = self.ax[0].axvline(x=0, color='r', linestyle='--')

    def update_image(self, E, delay):
        """Update the 2D image based on selected E and delay."""
        idx_E, idx_delay = self.data_handler.get_closest_indices(E, 0, 0, delay)
        self.im_1.set_data(self.data_handler.I[:, :, idx_E])  # Update image for new E
        self.ax[0].set_title(f"2D Intensity Map at E = {E} eV")

    def update_time_traces(self, square):
        """Update the time traces when the square is moved or resized."""
        x_idx, y_idx, _, t_idx = self.data_handler.get_closest_indices(square.x_center, square.y_center, 0, 0)
        intensity_profile_1 = self.data_handler.I[x_idx, y_idx, :, t_idx]
        intensity_profile_2 = self.data_handler.I[x_idx, y_idx, t_idx, :]
        
        # Update the time trace plots
        self.time_trace_1.set_ydata(intensity_profile_1)
        self.ax[1].set_title(f"Time Trace at x={square.x_center}, y={square.y_center}")
        
        self.time_trace_2.set_ydata(intensity_profile_2)
        self.ax[3].set_title(f"Energy Cut Trace")

    def plot_square(self, square):
        """Plot a square on the 2D image."""
        square_x, square_y = square.get_coordinates()
        self.ax[0].plot(square_x, square_y, color='cyan', lw=2)

    def update_square(self, square):
        """Update square position on plot."""
        self.ax[0].clear()  # Clear the plot for fresh update
        self.update_image(E=0, delay=0)  # Re-draw the image
        self.plot_square(square)  # Plot the updated square
        self.update_time_traces(square)  # Update time traces
        self.fig.canvas.draw_idle()  # Force the figure to re-render

class EventHandler:
    def __init__(self, plot_manager):
        self.plot_manager = plot_manager
        self.press_horizontal = False

    def on_press(self, event):
        """Handle mouse press events and update square or lines."""
        if event.inaxes == self.plot_manager.ax[0]:  # If event is in 2D image
            self.plot_manager.square.update_position(event.xdata, event.ydata)  # Update square position
            self.plot_manager.update_square(self.plot_manager.square)  # Redraw square and update traces

    def on_release(self, event):
        """Handle mouse release events."""
        self.press_horizontal = False

class SliderManager:
    def __init__(self, plot_manager):
        self.plot_manager = plot_manager
        self.E_slider, self.delay_slider = self.create_sliders()

    def create_sliders(self):
        """Create the sliders for energy and delay."""
        E_slider = Slider(plt.axes([0.045, 0.6, 0.03, 0.25]), 'E, eV', -4, 3.5, valinit=0)
        delay_slider = Slider(plt.axes([0.8, 0.475, 0.15, 0.03]), 'delay, fs', -200, 1000, valinit=0)
        return E_slider, delay_slider

    def on_slider_update(self, val):
        """Update plots based on slider values."""
        E = self.E_slider.val
        delay = self.delay_slider.val
        self.plot_manager.update_image(E, delay)
        self.plot_manager.update_time_traces(self.plot_manager.square)
