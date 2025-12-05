#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 15:51:29 2025

@author: lawsonlloyd
"""

import numpy as np
import csv
from Loader import DataLoader
import xarray as xr

#%%

def get_scan_info(data_path, filename, scan_info):
    with open(data_path + '//' + filename) as f:
        
        reader = csv.DictReader(f
                                )
        for row in reader:
            key = row.pop('Scan')
            if key in scan_info:
                # implement your duplicate row handling here
                pass
            scan_info[key] = row
    
    return scan_info

def load_data(data_path, scan, force_offset=False, **kwargs):
    
    scan_info = kwargs.get("scan_info", None)

    energy_offset = kwargs.get("energy_offset", 0)
    delay_offset = kwargs.get("delay_offset", 0)
    delay_scan = kwargs.get("delay_scan", True)
    tilt_scan = kwargs.get("tilt_scan", False)    
    
    filename = f"Scan{scan}.h5"
    data_loader = DataLoader(data_path + '//' + filename, offsets = [energy_offset, delay_offset])
    
    if force_offset is True: # Force the loaded data to offset given values when calling the function
        energy_offset = energy_offset
        delay_offset = delay_offset 
    
    elif force_offset is False and scan_info is None: # Default case: No offsets
        energy_offset = energy_offset
        delay_offset = delay_offset 

    elif force_offset is False and scan_info is not None: # Take the offsets from provided scan info
        if scan_info[str(scan)]['t0_offset'] == '':
            delay_offset = delay_offset 
        else:
            delay_offset = float(scan_info[str(scan)]['t0_offset'])
            
        if scan_info[str(scan)]['E_offset'] == '':
            energy_offset = energy_offset
        else: 
            energy_offset = float(scan_info[str(scan)]['E_offset'])

    if tilt_scan is True:
        res = data_loader.load_phoibos(tilt_scan = True)
    else:
        res = data_loader.load_phoibos()

    res = res.assign_coords(E=(res.E-energy_offset))
    if "delay" in res.dims:
        res = res.assign_coords(delay=(res.delay-delay_offset))
    
    return res