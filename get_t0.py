#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 15:35:03 2025

@author: lawsonlloyd
"""


#%% Define t0 from Exciton Rise

from scipy.special import erf

scans = [9219, 9217, 9218, 9216, 9220, 9228]
offsets_t0 = [-162.1, -152.7, -183.2, -118.4, -113.2, -125.0]

scans =      [9227, 9219, 9217, 9218, 9216, 9220, 9228, 9231, 9525, 9517, 9526] # Scans to analyze and fit below: 910 nm + 400 nm
offsets_t0 = [-191.7, -157.4, -152.7, -183.1, -118.5, -113.7, -125.2, -147.6, -77, -151.1, -200.6]

#scans = [9227, 9219, 9217, 9218, 9216, 9220, 9228, 9231,    9525, 9517, 9526]
#offsets_t0 = [-191.7, -162.4, -152.7, -183.1, -118.5, -113.7, -125.2, -147.6, -77, -151.1, -200.6]

t0_offsets = []

fig, ax = plt.subplots(4, 3)
fig.set_size_inches(8, 12, forward=False)
ax = ax.flatten()

i = 0    
for scan_i in scans:
    
    res = load_data(scan_i, 19.72, offsets_t0[i], False)

    ### Plot EDCs at GAMMA vs time
    
    (kx, ky), k_int = (0 , 0), 24
    trace = res.loc[{'Angle':slice(-12,12), 'Energy':slice(E1-E_int/2, E1+E_int/2)}].sum(axis=(0,1))
    trace = trace - trace[3:8].mean()
    trace = trace/np.max(trace)
#    trace.plot(ax = ax[i], color = 'red')
    
    trace_ex = res.loc[{"Angle":slice(kx-k_int/2,kx+k_int/2), 'Energy':slice(E1-E_int/2, E1+E_int/2)}].sum(dim=("Angle","Energy"))
    trace_ex = trace_ex - trace_ex[2:6].mean()
    trace_ex = trace_ex/np.max(trace_ex.loc[{"Delay":slice(-200,250)}])
    
    t0 = 0
    tau = 55
    def rise_erf(t, t0, tau):
        r = 0.5 * (1 + erf((t - t0) / tau))
        return r
    
    rise = rise_erf(res.Delay, 30, 45)
    
    p0 = [-30, 45]
    popt, pcov = curve_fit(rise_erf, trace_ex.loc[{"Delay":slice(-180,200)}].Delay.values ,
                                    trace_ex.loc[{"Delay":slice(-180,200)}].values,
                                    p0, method="lm")
    
    rise_fit = rise_erf(np.linspace(-200,200,50), *popt)
    
    ax[i].plot(res.Delay, trace_ex, 'ko')
    ax[i].plot(np.linspace(-200,200,50), rise_fit, 'pink')
    #ax[1].plot(I_res.delay, rise, 'red')
    
    ax[i].axvline(0, color = 'grey', linestyle = 'dashed')
    
    ax[i].set_xlim([-250, 500]) 
    ax[i].set_ylim(-.1,1.25)
    ax[i].set_title(f"t0 offset = {round(popt[0],3)}")
    print(round(popt[0],3))
    t0_offsets.append(popt[0])
    i += 1

fig.tight_layout()