#!/usr/bin/env python3
# -*- coding: utf-8 -*-\

#%% Import the Loader and Main GUI File

from Loader import DataLoader
from Main import main
from Manager import DataHandler, FigureHandler, PlotHandler, ValueHandler, SliderManager, EventHandler, CheckButtonManager, ClickButtonManager
import matplotlib.pyplot as plt

#%% Specifiy filename of h5 file in your path.
# Include manual energy and time delay offsets for the axes, if required.

data_path = 'path_to_your_data'
filename = 'your_file_name.h5'

data_path = 'R:\Lawson\Data\metis'
#data_path = '/Users/lawsonlloyd/Desktop/Data/'

#filename, offsets = 'Scan682_binned.h5', [0,0]
filename, offsets = 'Scan162_RT_120x120x115x50_binned.h5', [0.8467, -120]
#filename, offsets = 'Scan789_binned.h5', [0.2, 0]
#filename, offsets = 'Scan162_binned_100x100x200x150_CrSBr_RT_750fs_New_2.h5', [-0.2, 90] # Axis Offsets: [Energy (eV), delay (fs)]
#filename, offsets = 'Scan163_binned_100x100x200x150_CrSBr_120K_1000fs_rebinned_distCorrected_New_2.h5', [0, 100]
#filename, offsets = 'Scan188_binned_100x100x200x155_CrSBr_120K_1000fs_rebinned_ChargeingCorrected_DistCorrected.h5', [0.05, 65]
#filename, offsets = 'Scan803_binned.h5', [0.2, -102]

#filename, offsets = 'Scan186_binned_100x100x200_CrSBr_120K_Static.h5', [0,0]
#filename, offsets = 'Scan62_binned_200x200x300_CrSBr_RT_Static_rebinned.h5', [0,0]

filename, offsets = 'Scan1004_binned.h5', [0, 0]
#%% Load the data and axes information

data_loader = DataLoader(data_path + '//' + filename, offsets)

I = data_loader.load()
I_res = I/np.max(I)
#I_res = I_res.T
I_res  = I_res.loc[{"E":slice(-9.5,10)}]
#%% Run the Interactive GUI for Data Exploration and Plotting

%matplotlib auto

#main(I, ax_kx, ax_ky, ax_E, ax_delay, *[E_offset, delay_offset])

value_manager =  ValueHandler()
data_handler = DataHandler(value_manager, I_res)

# Initialize plot manager and check and click button managers
figure_handler = FigureHandler()
check_button_manager = CheckButtonManager()
plot_manager = PlotHandler(figure_handler, data_handler, value_manager, check_button_manager)
click_button_manager = ClickButtonManager(plot_manager, check_button_manager)

# Initialize sliders and attach update event
slider_manager = SliderManager(value_manager, plot_manager, check_button_manager)
slider_manager.E_slider.on_changed(slider_manager.on_slider_update)
slider_manager.E_int_slider.on_changed(slider_manager.on_slider_update)
slider_manager.k_int_slider.on_changed(slider_manager.on_slider_update)
slider_manager.delay_slider.on_changed(slider_manager.on_slider_update)
slider_manager.delay_int_slider.on_changed(slider_manager.on_slider_update)

# Initialize event handler for interactivity
event_handler = EventHandler(value_manager, slider_manager, plot_manager, check_button_manager)
plot_manager.fig.canvas.mpl_connect('button_press_event', event_handler.on_press)
plot_manager.fig.canvas.mpl_connect('motion_notify_event', event_handler.on_motion)
plot_manager.fig.canvas.mpl_connect('button_release_event', event_handler.on_release)

plt.show(block=False)