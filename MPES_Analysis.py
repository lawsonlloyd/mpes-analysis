#!/usr/bin/env python3
# -*- coding: utf-8 -*-\

#%% Import the Loader and Main GUI File

from Loader import DataLoader
from Main import main

#%% Specifiy filename of h5 file in your path.

data_path = 'path_to_your_data'
filename = 'your_data_file.h5'

data_path = 'R:\Lawson\Analysis\data'
filename = 'Scan162_binned_100x100x200x150_CrSBr_RT_750fs_New_2.h5'

# Include manual energy and time delay offsets for the axes, if required.
E_offset = -0.1
delay_offset = 100

#%% Load the data and axes information

data_loader = DataLoader(data_path + '\\' + filename)
I, ax_kx, ax_ky, ax_E, ax_delay = data_loader.load()

#%% Run the Interactive GUI for Data Exploration and Plotting

%matplotlib auto

main(I, ax_kx, ax_ky, ax_E, ax_delay, *[E_offset, delay_offset])
