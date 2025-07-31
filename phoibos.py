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

def load_data(data_path, scan, scan_info, energy_offset, delay_offset, force_offset):
    
    filename = f"Scan{scan}.h5"
    data_loader = DataLoader(data_path + '//' + filename, offsets = [energy_offset, delay_offset])
    
    if force_offset is True:
        energy_offset = energy_offset
        delay_offset = delay_offset 
    
    else:    
        if scan_info[str(scan)]['t0_offset'] == '':
            delay_offset = delay_offset 
        else:
            delay_offset = float(scan_info[str(scan)]['t0_offset'])
            
        if scan_info[str(scan)]['E_offset'] == '':
            energy_offset = energy_offset
        else: 
            energy_offset = float(scan_info[str(scan)]['E_offset'])

    res = data_loader.load_phoibos()
    
    res = res.assign_coords(E=(res.E-energy_offset))
    res = res.assign_coords(delay=(res.delay-delay_offset))
    
    return res

# Fucntion for Extracting time Traces
def get_time_trace(res, E, E_int, k , k_int, subtract_neg, norm_trace):
        
    trace = res.loc[{"Energy":slice(E-E_int/2, E+E_int/2), "Angle":slice(k-k_int/2, k+k_int/2)}].sum(dim=("Energy","Angle"))

    if subtract_neg is True : 
        trace = trace - np.mean(trace.loc[{"Delay":slice(-1000,-350)}])
    
    if norm_trace is True : 
        trace = trace/np.max(trace)
    
    return trace

def make_diff_ARPES(res, delays, E_inset):

    res_neg = res.loc[{'Delay':slice(-1000,-300)}]
    #res_pos = res.loc[{'Delay':slice(0,5000)}]
        
    res_diff_E_Ang = res.loc[{'Delay':slice(delays[0],delays[1])}].mean(axis=2) - res_neg.mean(axis=2)
    res_diff_E_Ang = res_diff_E_Ang/np.max(np.abs(res_diff_E_Ang))
    
    d1 = (res_diff_E_Ang.loc[{'Energy':slice(-1,E_inset)}])
    d2 = (res_diff_E_Ang.loc[{'Energy':slice(E_inset,3)}])
    d2 = d2/np.max(abs(d2))
    d3 = xr.concat([d1, d2], dim = "Energy")
    res_diff_sum_Angle_Normed = d3

    return res_diff_sum_Angle_Normed

def enhance_features(I_res, Ein, factor, norm):
    
    I1 = I_res.loc[{"Energy":slice(-3.5,Ein)}]
    I2 = I_res.loc[{"Energy":slice(Ein,3.5)}]

    if norm is True:
        I1 = I1/np.max(I1)
        I2 = I2/np.max(I2)
    else:
        I1 = I1/factor[0]
        I2 = I2/factor[1]
        
    I3 = xr.concat([I1, I2], dim = "Energy")
    
    return I3