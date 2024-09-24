#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 16:45:17 2023

@author: lawsonlloyd
"""

#%%

from Loader import DataLoader
from Main import main

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

main(I, ax_kx, ax_ky, ax_E, ax_delay, *offsets)
