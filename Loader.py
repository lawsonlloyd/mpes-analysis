# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:52:01 2024

@author: lloyd
"""
#Loader.py

import h5py
import numpy as np 
import xarray as xr

class DataLoader:
    def __init__(self, filename):
        self.filename = filename
        self.data = None
        self.ax_kx = None
        self.ax_ky = None
        self.ax_E = None
        self.ax_ADC = None
    
    def load(self):
        with h5py.File(self.filename, 'r') as f:
            # Print all root level object names (aka keys)
            #print("Keys: %s" % f.keys())

            # Get group keys
            group_keys = list(f.keys())
            if len(group_keys) < 2:
                raise ValueError("Not enough groups found in the file.")

            a_group_key = group_keys[0]
            #a_group_key2 = group_keys[1]

            # Load axes data
            self.ax_E = f['axes/ax2'][()].astype(np.float32)
            self.ax_kx = f['axes/ax0'][()].astype(np.float32)
            self.ax_ky = f['axes/ax1'][()].astype(np.float32)
            
            # Load binned data
            i_data = f['binned/BinnedData'][()]
            
            # Determine the ADC axis
            if 'delay' in list(f[a_group_key]):
                self.ax_ADC = f['axes/delay'][()].astype(np.float32)
            elif 'theta' in list(f[a_group_key]):
                self.ax_ADC = f['axes/theta'][()].astype(np.float32)
            elif 'ax3' in list(f[a_group_key]):
                self.ax_ADC = f['axes/ax3'][()].astype(np.float32)
            else:
                self.ax_ADC = np.zeros(1)

            # Initialize the data cube
            if self.ax_ADC.size > 1:
                self.data = np.zeros((len(self.ax_kx), len(self.ax_ky), len(self.ax_E), len(self.ax_ADC)), dtype='float32')
            else:
                self.data = np.zeros((len(self.ax_kx), len(self.ax_ky), len(self.ax_E)), dtype='float32')

            self.data = i_data

            print('The data shape is: ' + str(self.data.shape))
            print('"'+ self.filename + '"' + ' has been loaded! Happy Analysis...')
            
            if i_data.ndim > 3:
                res = xr.DataArray(i_data, dims = ("kx", "ky", "E", "delay"), coords = [self.ax_kx, self.ax_ky, self.ax_E, self.ax_ADC])
            elif i_data.ndim < 4:
                res = xr.DataArray(i_data, dims = ("kx", "ky", "E"), coords = [self.ax_kx, self.ax_ky, self.ax_E])
            
            return res
            #return self.data, self.ax_kx, self.ax_ky, self.ax_E, self.ax_ADC
        
    def load_phoibos(self):
        with h5py.File(self.filename, 'r') as f:
        
            group_keys = list(f.keys())
            a_group_key = group_keys[0]

            # Load axes and data
            data = f['binned/BinnedData'][()].astype(np.float32)
            ax_angle = f['axes/ax0'][()].astype(np.float32)
            ax_E = f['axes/ax1'][()].astype(np.float32)
            
            if 'ax2' in list(f[a_group_key]):
                ax_delay = f['axes/ax2'][()].astype(np.float32)
                return xr.DataArray(data, dims = ("Angle", "Energy", "Delay"), coords = [ax_angle, ax_E, ax_delay])
            else:
                return xr.DataArray(data, dims = ("Angle", "Energy"), coords = [ax_angle, ax_E])

            print('"'+ self.filename + '"' + ' has been loaded! Happy Analysis...')


