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
        self.dt = self.calculate_dt()

    def calculate_dt(self):
        if self.I.ndim > 3:
            return self.ax_delay_offset[1] - self.ax_delay_offset[0]
        else:
            return 1

    def get_closest_indices(self, x, y, E, t):
        x_idx = (np.abs(self.ax_kx - x)).argmin()
        y_idx = (np.abs(self.ax_ky - y)).argmin()
        E_idx = (np.abs(self.ax_E_offset - E)).argmin()
        t_idx = (np.abs(self.ax_delay_offset - t)).argmin()
        return x_idx, y_idx, E_idx, t_idx