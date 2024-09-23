#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 16:45:17 2023

@author: lawsonlloyd
"""

#%%

import numpy as np
import matplotlib.pyplot as plt

from Loader import DataLoader
from Manager import DataHandler, PlotHandler, ValueHandler, SliderManager, EventHandler, ButtonManager

#%% Load File in your path...

fn = 'your_data_file.h5'
fn = 'Scan162_binned_100x100x200x150_CrSBr_RT_750fs_New_2.h5'

E_offset = -0.3
delay_offset = 100

offsets = [E_offset, delay_offset]

#%%

# Load data
data_loader = DataLoader(fn)
I, ax_kx, ax_ky, ax_E, ax_delay = data_loader.load()

#%%
    
def main():
    
    data_handler = DataHandler(I, ax_kx, ax_ky, ax_E, ax_delay, *offsets)
    value_manager =  ValueHandler()

    # Initialize plot manager
    plot_manager = PlotHandler(data_handler, value_manager)

    # Initialize sliders and attach update event
    slider_manager = SliderManager(plot_manager, value_manager)
    slider_manager.E_slider.on_changed(slider_manager.on_slider_update)
    slider_manager.k_int_slider.on_changed(slider_manager.on_slider_update)

    button_manager = ButtonManager(plot_manager)
    #button_manager.
#    save_button.on_clicked(self.save_trace)

    # Initialize event handler for interactivity
    event_handler = EventHandler(slider_manager, plot_manager, value_manager)
    plot_manager.fig.canvas.mpl_connect('button_press_event', event_handler.on_press)
    plot_manager.fig.canvas.mpl_connect('motion_notify_event', event_handler.on_motion)
    plot_manager.fig.canvas.mpl_connect('button_release_event', event_handler.on_release)
    #plot_manager.fig.canvas.mpl_connect('button_press_event', event_handler.on_press)

    plt.show()

if __name__ == "__main__":
    main()
