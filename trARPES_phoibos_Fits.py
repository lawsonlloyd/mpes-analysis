#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 14:52:43 2025

@author: lawsonlloyd
"""

#%% Define Exponential Fitting Functions

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve

%matplotlib inline

# Define the monoexponential decay function (Exciton)
def monoexp(t, A, tau):
    return A * np.exp(-t / tau) * (t >= 0)  # Ensure decay starts at t=0

# Define the biexponnential decay function (Exciton)    
def biexp(t, A, tau1, B, tau2):
    return ( A * np.exp(-t / tau1) + B * np.exp(-t / tau2))  * (t >= 0)  # Ensure decay starts at t=0

# Define the conduction band model: exponential rise + decay
def exp_rise_monoexp_decay(t, C, tau_rise, tau_decay1):
    return C * (1 - np.exp(-t / tau_rise)) * (np.exp(-t / tau_decay1)) * (t >= 0)

def exp_rise_biexp_decay(t, C, tau_rise, D, tau_decay1, tau_decay2):
    return C * (1 - np.exp(-t / tau_rise)) * (D * np.exp(-t / tau_decay1) + (1-D) * np.exp(-t / tau_decay2)) * (t >= 0)

# Define the Instrumental Response Function (IRF) as a Gaussian
def IRF(t, sigma_IRF):
    return np.exp(-t**2 / (2 * sigma_IRF**2)) / (sigma_IRF * np.sqrt(2 * np.pi))

# Convolution of the signal with the IRF
def convolved_signal(t, signal_function, sigma_IRF, *params):
    dt = np.mean(np.diff(t))  # Time step
    signal = signal_function(t, *params)  # Compute signal
    irf = IRF(t - t[len(t)//2], sigma_IRF)  # Shift IRF to center
    irf /= np.sum(irf) * dt  # Normalize IRF
    convolved = fftconvolve(signal, irf, mode='same') * dt  # Convolve with IRF
    return convolved

#%% Independent Fitting EX and CBM Signals to Single Exponentials

# Fitting functions (excluding IRF from fit)
def exciton_model(t, A, tau1):
    return convolved_signal(t, monoexp, sigma_IRF, A, tau1)  # IRF is fixed

def cbm_model(t, C, tau_rise, tau_decay1):
    return convolved_signal(t, exp_rise_monoexp_decay, sigma_IRF, C, tau_rise, tau_decay1)  # IRF is fixed

########

# Initialize figure
fig, ax = plt.subplots(6, 2, figsize=(12, 16))
ax = ax.flatten()

# Load Data
scans = [9219, 9217, 9218, 9216, 9220, 9228]
power = [8.3, 20.9, 41.7, 65.6, 83.2, 104.7]
E, E_int = [1.35, 2.1], 0.1 # center energies, half of full E integration range

#scans = [9241, 9237, 9240]
popt_ex, popt_cbm = np.zeros((len(scans),2)), np.zeros((len(scans),3))
perr_ex, perr_cbm = np.zeros((len(scans),2)), np.zeros((len(scans),3))

fwhm_IRF = 80
sigma_IRF = fwhm_IRF/2.355   # Fixed IRF width (fs)

for s in range(len(scans)):

    res = phoibos.load_data(data_path, scans[s], scan_info, 19.7, 0, False)
    delay_axis = res.Delay.values
    
    ### Extract time traces ###
    delay_limit = [-200, 3050]
    delay_axis = res.Delay.loc[{"Delay":slice(delay_limit[0], delay_limit[1])}].values
    
    # EXCITON
    trace_1 = res.loc[{'Angle':slice(-12,12), 'Energy':slice(E[0]-E_int/2, E[0]+E_int/2)}].sum(axis=(0,1))
    trace_1 = trace_1-trace_1.loc[{"Delay":slice(-600,-200)}].mean()
    trace_1 = trace_1.loc[{"Delay":slice(delay_limit[0],delay_limit[1])}]
    norm_factor = np.max(trace_1)
    
    # CBM
    trace_2 = res.loc[{'Angle':slice(-12,12), 'Energy':slice(E[1]-E_int/2, E[1]+E_int/2)}].sum(axis=(0,1))
    trace_2 = trace_2-trace_2.loc[{"Delay":slice(-600,-200)}].mean()
    trace_2 = trace_2.loc[{"Delay":slice(delay_limit[0],delay_limit[1])}]
    
    trace_1 = trace_1/norm_factor
    trace_2 = trace_2/norm_factor
    
    # Initial guesses for fitting
    init_guess_exciton = [0.8, 100]  # A, tau1
    init_guess_conduction = [2, 1500, 5000]  # C, tau_rise, tau_decay
    
    # Bounds for fitting
    bnds_exciton = [ [0,0], [2,15000] ]
    bnds_cbm =  [[0,0,0], [6, 2000, 15000] ]
    
    # Fit noisy exciton data
    popt_exciton, pcov = curve_fit(exciton_model, delay_axis, trace_1, p0=init_guess_exciton, bounds = bnds_exciton)
    perr_exciton = np.sqrt(np.diag(pcov))

    # Fit noisy conduction band data
    popt_conduction, pcov = curve_fit(cbm_model, delay_axis, trace_2, p0=init_guess_conduction, bounds = bnds_cbm)
    perr_conduction = np.sqrt(np.diag(pcov))

    # Save Parameters
    popt_ex[s,:] = popt_exciton
    popt_cbm[s,:] = popt_conduction
    
    perr_ex[s,:] = perr_exciton
    perr_cbm[s,:] = perr_conduction

    # Generate fitted curves\
    t = np.linspace(-500,5000,550)
    exciton_fitted = exciton_model(t, *popt_exciton)
    conduction_fitted = cbm_model(t, *popt_conduction)
    
    # Print best-fit parameters
    #    print("Exciton Fit Parameters:")
    #   print(f"A = {popt_exciton[0]:.3f}, tau1 = {popt_exciton[1]:.1f} fs, B = {popt_exciton[2]:.3f}, tau2 = {popt_exciton[3]:.1f} fs")
    
    #  print("\nConduction Band Fit Parameters:")
    # print(f"tau_rise = {popt_conduction[0]:.1f} fs, C = {popt_conduction[1]:.1f}, tau1 = {popt_conduction[2]:.1f} fs, D = {popt_conduction[3]:.1f}, tau2 = {popt_conduction[2]:.1f} fs")
    
    ### Plot results ###
    
    # Exciton decay plot
    ax[2*s].plot(delay_axis, trace_1, label='Data', color='black', alpha=1)
    #ax[0].plot(t, exciton_signal, label='True Exciton Model', color='red', linestyle='dashed')
    ax[2*s].plot(t, exciton_fitted, label=f"tau_1 = {popt_exciton[1]:.0f} fs", color='grey', linestyle='solid')
    ax[2*s].axvline(0, linestyle='--', color='gray', alpha=0.5)
    ax[2*s].set_xlabel('Time (fs)')
    ax[2*s].set_ylabel('Norm. Int., a.u.')
    ax[2*s].set_title('Exciton')
    ax[2*s].set_xlim(-250,3000)
    ax[2*s].legend(frameon=False)
    if s == 7:
        ax[2*s].plot(delay_axis, monoexp(delay_axis, 1.1, 500), color = 'blue')
    
    # Conduction band rise + decay plot
    ax[2*s+1].plot(delay_axis, trace_2, label='Data', color='maroon', alpha=1)
    #ax[1].plot(t, conduction_band_signal, label='True Conduction Model', color='red', linestyle='dashed')
    ax[2*s+1].plot(t, conduction_fitted, label= f"tau_r = {popt_conduction[1]:.0f} fs, tau_1 = {popt_conduction[2]:.0f} fs", color='red', linestyle='solid')
    ax[2*s+1].axvline(0, linestyle='--', color='gray', alpha=0.5)
    ax[2*s+1].set_xlabel('Time (fs)')
    ax[2*s+1].set_ylabel('Norm. Int., a.u.')
    ax[2*s+1].set_title('CBM')
    ax[2*s+1].set_xlim(-250,3000)
    ax[2*s+1].legend(frameon=False)
    if s == 7:
        ax[2*s+1].plot(delay_axis, monoexp(delay_axis, .5, 1250), color = 'blue')

fig.tight_layout()
plt.show()

#%% # PLOT DECAY CONSTANTS AND AMPLITUDES FROM THE Single Exp. Fit

fig, ax = plt.subplots(2, 2, figsize=(8,6), sharex=False)
ax = ax.flatten()

n = [p**1 for p in power]
#n = power

# A, tau1, B, tau2
# C, tau_rise, D, tau_decay1, tau_decay2

#ax[0].plot(n, popt_ex[:,3], 'ko', label = 'Slow')
#ax[0].plot(n, popt_cbm[:,2], 'ro', label = 'Dec')
# Slow Exp Decay Component
ax[0].errorbar(n, popt_ex[:,1], yerr = perr_ex[:,1], marker = 'o', color = 'k')
ax[0].errorbar(n, popt_cbm[:,2], yerr = perr_cbm[:,2], marker = 'o', color = 'r')
ax[0].set_title('Exp. Decay (slow)')
ax[0].set_ylabel('tau, fs')
ax[0].set_xlabel('n')
ax[0].set_ylim(0,12000)

# CBM Rise Time Component
#ax[1].plot(n, popt_cbm[:,1], color = 'red', marker = '*')
ax[1].errorbar(n, popt_cbm[:,1], yerr = perr_cbm[:,1], marker = 'o', color = 'purple')
ax[1].set_ylim(0, 300)
ax[1].set_title('CBM Rise Time')
ax[1].set_ylabel('tau, fs')
ax[1].set_xlabel('n')

# FAST Exp Decay Component
#ax[3].plot(n, popt_ex[:,1], 'k*', label = 'Fast')
#ax[2].errorbar(n, popt_ex[:,1], yerr = perr_ex[:,1], marker = '*', color = 'grey', label = 'Fast')
#ax[2].errorbar(n, popt_cbm[:,3], yerr = perr_cbm[:,3], marker = '*', color = 'pink', label = 'Fast')
ax[2].set_title('Exp. Decay (Fast)')
ax[2].set_ylabel('tau, fs')
ax[2].set_xlabel('n')
ax[2].set_ylim(-100,1500)

#ax[2].plot(n, popt_ex[:,0], 'ko', label = 'Fast')
#ax[2].plot(n, popt_ex[:,2], 'k*', label = 'Slow')
#ax[2].plot(n, popt_cbm[:,0], 'ro', label = 'CBM')
# Exp AMPLITUDES Components
ax[3].errorbar(n, popt_ex[:,0], yerr = perr_ex[:,0], marker = 'o', color = 'grey', label = 'Fast')
#ax[3].errorbar(n, popt_ex[:,2], yerr = perr_ex[:,2], marker = 's', color = 'k', label = 'slow')
ax[3].errorbar(n, popt_cbm[:,0], yerr = perr_cbm[:,0], marker = '*', color = 'pink', label = 'Fast')
#ax[3].errorbar(n, (1- popt_cbm[:,2]), yerr = perr_cbm[:,2], marker = '*', color = 'r', label = 'Slow')
#ax[3].errorbar(n, popt_cbm[:,0], yerr = perr_cbm[:,0], marker = '*', color = 'purple', label = 'RISE')

ax[3].set_title('Exp. Amplitudes')
ax[3].set_ylabel('Amp.')
ax[3].set_xlabel('n')
ax[3].set_ylim(-.2,1.5)
ax[3].legend(frameon=False, fontsize = 8)

fig.tight_layout()

#%% Independent Fitting EX and CBM Signals to Bi-Exponentials

def exciton_model(t, A, tau1, B, tau2):
    return convolved_signal(t, biexp, sigma_IRF, A, tau1, B, tau2)  # IRF is fixed

def cbm_model(t, C, tau_rise, D, tau_decay1, tau_decay2):
    return convolved_signal(t, exp_rise_biexp_decay, sigma_IRF, C, tau_rise, D, tau_decay1, tau_decay2)  # IRF is fixed

########

# Initialize figure
fig, ax = plt.subplots(6, 2, figsize=(12, 16))
ax = ax.flatten()

# Load Data
E, Eint = [1.35, 2.1], 0.1 # center energies, half of full E integration range
scans = [9219, 9217, 9218, 9216, 9220, 9228]
power = [8.3, 20.9, 41.7, 65.6, 83.2, 104.7]
# Load Data
#scans = [9227, 9219, 9217, 9218, 9216, 9220, 9228, 9231] # Scans to analyze and fit below: 910 nm + 400 nm
trans_percent = [float(scan_info[str(s)].get("Percent")) for s in scans] # Retrieve the percentage
power = [500*0.01*t for t in trans_percent]

#scans = [9241, 9237, 9240]
popt_ex, popt_cbm = np.zeros((len(scans),4)), np.zeros((len(scans),5))
perr_ex, perr_cbm = np.zeros((len(scans),4)), np.zeros((len(scans),5))

fwhm_IRF = 80
sigma_IRF = fwhm_IRF/2.355   # Fixed IRF width (fs)

for s in range(len(scans)):

    res = phoibos.load_data(data_path, scans[s], scan_info, energy_offset, delay_offset, False)
    delay_axis = res.Delay.values
    
    ### Extract time traces ###
    delay_limit = [-300, 3050]
    delay_axis = res.Delay.loc[{"Delay":slice(delay_limit[0], delay_limit[1])}].values
    
    # EXCITON
    trace_1 = res.loc[{'Angle':slice(-12,12), 'Energy':slice(E[0]-E_int/2, E[0]+E_int/2)}].sum(axis=(0,1))
    trace_1 = trace_1-trace_1.loc[{"Delay":slice(-600,-200)}].mean()
    trace_1 = trace_1.loc[{"Delay":slice(delay_limit[0],delay_limit[1])}]
    norm_factor = np.max(trace_1)
    
    # CBM
    trace_2 = res.loc[{'Angle':slice(-12,12), 'Energy':slice(E[1]-E_int/2, E[1]+E_int/2)}].sum(axis=(0,1))
    trace_2 = trace_2-trace_2.loc[{"Delay":slice(-600,-200)}].mean()
    trace_2 = trace_2.loc[{"Delay":slice(delay_limit[0],delay_limit[1])}]
    
    trace_1 = trace_1/norm_factor
    trace_2 = trace_2/norm_factor
    
    # Initial guesses for fitting
    init_guess_exciton = [0.8, 100, 0.2, 1000]  # A, tau1, B, tau2
    init_guess_conduction = [6, 200, .5, 300, 5000]  # C, tau_rise, D, tau_decay1, tau_decay2
    
    # Bounds for fitting
    bnds_exciton = [ [0,0,0,0], [2,1000,2,30000] ]
    bnds_cbm =  [[0,0,0,0,0], [6, 1000, 10, 2000, 30000] ]
    
    # Fit noisy exciton data
    popt_exciton, pcov = curve_fit(exciton_model, delay_axis, trace_1, p0=init_guess_exciton, bounds = bnds_exciton)
    perr_exciton = np.sqrt(np.diag(pcov))

    # Fit noisy conduction band data
    popt_conduction, pcov = curve_fit(cbm_model, delay_axis, trace_2, p0=init_guess_conduction, bounds = bnds_cbm)
    perr_conduction = np.sqrt(np.diag(pcov))

    # Save Parameters
    popt_ex[s,:] = popt_exciton
    popt_cbm[s,:] = popt_conduction
    
    perr_ex[s,:] = perr_exciton
    perr_cbm[s,:] = perr_conduction

    # Generate fitted curves
    exciton_fitted = exciton_model(t, *popt_exciton)
    conduction_fitted = cbm_model(t, *popt_conduction)
    
    # Print best-fit parameters
    print("Exciton Fit Parameters:")
    print(f"A = {popt_exciton[0]:.3f}, tau1 = {popt_exciton[1]:.1f} fs, B = {popt_exciton[2]:.3f}, tau2 = {popt_exciton[3]:.1f} fs")
    
    print("\nConduction Band Fit Parameters:")
    print(f"tau_rise = {popt_conduction[1]:.1f} fs, C = {popt_conduction[0]:.1f}, tau1 = {popt_conduction[3]:.1f} fs, D = {popt_conduction[2]:.1f}, tau2 = {popt_conduction[4]:.1f} fs")
    
    ### Plot results ###
    
    # Exciton decay plot
    ax[2*s].plot(delay_axis, trace_1, label='Data', color='black', alpha=1)
    #ax[0].plot(t, exciton_signal, label='True Exciton Model', color='red', linestyle='dashed')
    ax[2*s].plot(t, exciton_fitted, label=f"tau_1 = {popt_exciton[1]:.0f} fs, tau_2 = {popt_exciton[3]:.0f} fs", color='grey', linestyle='solid')
    ax[2*s].axvline(0, linestyle='--', color='gray', alpha=0.5)
    ax[2*s].set_xlabel('Time (fs)')
    ax[2*s].set_ylabel('Norm. Int., a.u.')
    ax[2*s].set_title('Exciton')
    ax[2*s].set_xlim(-250,3000)
    ax[2*s].legend(frameon=False)
    
    # Conduction band rise + decay plot
    ax[2*s+1].plot(delay_axis, trace_2, label='Data', color='maroon', alpha=1)
    #ax[1].plot(t, conduction_band_signal, label='True Conduction Model', color='red', linestyle='dashed')
    ax[2*s+1].plot(t, conduction_fitted, label= f"tau_r = {popt_conduction[1]:.0f} fs, tau_1 = {popt_conduction[3]:.0f} fs, tau_2 = {popt_conduction[4]:.0f} fs", color='red', linestyle='solid')
    ax[2*s+1].axvline(0, linestyle='--', color='gray', alpha=0.5)
    ax[2*s+1].set_xlabel('Time (fs)')
    ax[2*s+1].set_ylabel('Norm. Int., a.u.')
    ax[2*s+1].set_title('CBM')
    ax[2*s+1].set_xlim(-250,3000)
    ax[2*s+1].legend(frameon=False)

fig.tight_layout()
plt.show()

#%% # PLOT DECAY CONSTANTS AND AMPLITUDES FROM THE Biexp. FIT

fig, ax = plt.subplots(2, 2, figsize=(8,6), sharex=False)
ax = ax.flatten()

n = [p**1 for p in power]
#n = power

# A, tau1, B, tau2
# C, tau_rise, D, tau_decay1, tau_decay2

#ax[0].plot(n, popt_ex[:,3], 'ko', label = 'Slow')
#ax[0].plot(n, popt_cbm[:,2], 'ro', label = 'Dec')
# Slow Exp Decay Component
ax[0].errorbar(n, popt_ex[:,3], yerr = perr_ex[:,3], marker = 'o', color = 'k')
ax[0].errorbar(n, popt_cbm[:,4], yerr = perr_cbm[:,4], marker = 'o', color = 'r')
ax[0].set_title('Exp. Decay (slow)')
ax[0].set_ylabel('tau, fs')
ax[0].set_xlabel('n')
ax[0].set_ylim(0,12000)

# CBM Rise Time Component
#ax[1].plot(n, popt_cbm[:,1], color = 'red', marker = '*')
ax[1].errorbar(n, popt_cbm[:,1], yerr = perr_cbm[:,1], marker = 'o', color = 'purple')
ax[1].set_ylim(0, 500)
ax[1].set_title('CBM Rise Time')
ax[1].set_ylabel('tau, fs')
ax[1].set_xlabel('n')

# FAST Exp Decay Component
#ax[3].plot(n, popt_ex[:,1], 'k*', label = 'Fast')
ax[2].errorbar(n, popt_ex[:,1], yerr = perr_ex[:,1], marker = '*', color = 'grey', label = 'Fast')
ax[2].errorbar(n, popt_cbm[:,3], yerr = perr_cbm[:,3], marker = '*', color = 'pink', label = 'Fast')
ax[2].set_title('Exp. Decay (Fast)')
ax[2].set_ylabel('tau, fs')
ax[2].set_xlabel('n')
ax[2].set_ylim(-100,1500)

#ax[2].plot(n, popt_ex[:,0], 'ko', label = 'Fast')
#ax[2].plot(n, popt_ex[:,2], 'k*', label = 'Slow')
#ax[2].plot(n, popt_cbm[:,0], 'ro', label = 'CBM')
# Exp AMPLITUDES Components
ax[3].errorbar(n, popt_ex[:,0], yerr = perr_ex[:,0], marker = 'o', color = 'grey', label = 'Fast')
ax[3].errorbar(n, popt_ex[:,2], yerr = perr_ex[:,2], marker = 's', color = 'k', label = 'slow')
ax[3].errorbar(n, popt_cbm[:,2], yerr = perr_cbm[:,2], marker = '*', color = 'pink', label = 'Fast')
ax[3].errorbar(n, (1- popt_cbm[:,2]), yerr = perr_cbm[:,2], marker = '*', color = 'r', label = 'Slow')
ax[3].errorbar(n, popt_cbm[:,0], yerr = perr_cbm[:,0], marker = '*', color = 'purple', label = 'RISE')

ax[3].set_title('Exp. Amplitudes')
ax[3].set_ylabel('Amp.')
ax[3].set_xlabel('n')
ax[3].set_ylim(-.2,1.5)
ax[3].legend(frameon=False, fontsize = 8)

fig.tight_layout()

#%% Fit EXCITON AND CBM Traces Simultaneously (Shared Exp.)

# Define the monoexponential decay function (Exciton)
def monoexp(t, A, tau):
    return A * np.exp(-t / tau) * (t >= 0)  # Ensure decay starts at t=0

# Define the biexponnential decay function (Exciton)    
def biexp(t, A, tau1, B, tau2):
    return ( A * np.exp(-t / tau1) + B * np.exp(-t / tau2))  * (t >= 0)  # Ensure decay starts at t=0

# Define the conduction band model: exponential rise + decay
def exp_rise_decay(t, C, tau_rise, D, tau_decay1, tau_decay2):
    return C * (1 - np.exp(-t / tau_rise)) * (D * np.exp(-t / tau_decay1) + (1-D) * np.exp(-t / tau_decay2)) * (t >= 0)

# Define the Instrumental Response Function (IRF) as a Gaussian
def IRF(t, sigma_IRF):
    return np.exp(-t**2 / (2 * sigma_IRF**2)) / (sigma_IRF * np.sqrt(2 * np.pi))

# Convolution of the signal with the IRF
def convolved_signal(t, signal_function, sigma_IRF, *params):
    dt = np.mean(np.diff(t))  # Time step
    signal = signal_function(t, *params)  # Compute signal
    irf = IRF(t - t[len(t)//2], sigma_IRF)  # Shift IRF to center
    irf /= np.sum(irf) * dt  # Normalize IRF
    convolved = fftconvolve(signal, irf, mode='same') * dt  # Convolve with IRF
    return convolved

# Fitting functions (excluding IRF from fit)
def exciton_model(t, A, tau1, B, tau2):
    return convolved_signal(t, biexp, sigma_IRF, A, tau1, B, tau2)  # IRF is fixed

def cbm_model(t, C, tau_rise, D, tau_decay1, tau_decay2):
    return convolved_signal(t, exp_rise_decay, sigma_IRF, C, tau_rise, D, tau_decay1, tau_decay2)  # IRF is fixed

def combined_model(t, A, tau1, B, tau2, C, tau_rise, D):
    exciton = exciton_model(t, A, tau1, B, tau2)
    cb = cbm_model(t, C, tau_rise, D, tau1, tau2)
    return np.concatenate((exciton, cb))  # Stack both signals

sigma_IRF = 80   # Fixed IRF width (fs)

#####

# Initialize figure
fig, ax = plt.subplots(8, 2, figsize=(12, 20))
ax = ax.flatten()

# Load Data
scans = [9227, 9219, 9217, 9218, 9216, 9220, 9228, 9231] # Scans to analyze and fit below: 910 nm + 400 nm
offsets_t0 = [-191.7, -162.4, -152.7, -183.1, -118.5, -113.7, -125.2, -147.6]
trans_percent = [float(scan_info[str(s)].get("Percent")) for s in scans] # Retrieve the percentage
power = [500*0.01*t for t in trans_percent]

#scans = [9219, 9217, 9218, 9216, 9220, 9228]
#offsets_t0 = [-162.4, -152.7, -183.1, -118.5, -113.7, -125.2]
#power = [8.3, 20.9, 41.7, 65.6, 83.2, 104.7]

popt_combined, perr_combined =  np.zeros((len(scans),7)), np.zeros((len(scans),7))

for s in range(len(scans)):

    res = phoibos.load_data(data_path, scans[s], scan_info, energy_offset, delay_offset, force_offset)
    delay_axis = res.Delay.values
    
    ### Extract time traces ###
    E, Eint = [1.35, 2.1], 0.1 # center energies, half of full E integration range
    delay_limit = [-200, 3050]
    delay_axis = res.Delay.loc[{"Delay":slice(delay_limit[0], delay_limit[1])}].values
    
    # EXCITON
    trace_1 = res.loc[{'Angle':slice(-12,12), 'Energy':slice(E[0]-E_int/2, E[0]+E_int/2)}].sum(axis=(0,1))
    trace_1 = trace_1-trace_1.loc[{"Delay":slice(-600,-200)}].mean()
    trace_1 = trace_1.loc[{"Delay":slice(delay_limit[0],delay_limit[1])}]
    norm_factor = np.max(trace_1)
    
    # CBM
    trace_2 = res.loc[{'Angle':slice(-12,12), 'Energy':slice(E[1]-E_int/2, E[1]+E_int/2)}].sum(axis=(0,1))
    trace_2 = trace_2-trace_2.loc[{"Delay":slice(-600,-200)}].mean()
    trace_2 = trace_2.loc[{"Delay":slice(delay_limit[0],delay_limit[1])}]
    
    trace_1 = trace_1/norm_factor
    trace_2 = trace_2/norm_factor
    
    traces_combined = np.concatenate((trace_1, trace_2))

    # Fit Combined Data Together
    initial_guess_combined = [1, 1000, 1, 15000, 1, 150, .5] # A, tau1, B, tau2, C, tau_rise, D
    bnds_combined = [[0, 100, 0, 2000, 0, 0, 0], [2, 1200, 2, 25000, 2, 1000, 1]]
    popt_comb, pcov = curve_fit(combined_model, delay_axis, traces_combined, p0=initial_guess_combined, bounds = bnds_combined)
    perr_comb = np.sqrt(np.diag(pcov))
    
    # Save Parameters
    popt_combined[s,:] = popt_comb
    perr_combined[s,:] = perr_comb

    # Generate fitted curves
    A, tau1, B, tau2, C, tau_rise, D = popt_comb
    exciton_fitted, conduction_fitted = exciton_model(t, A, tau1, B, tau2), cbm_model(t, C, tau_rise, D, tau1, tau2)
    
    # Print best-fit parameters
    print("Exciton Fit Parameters:")
    print(f"A = {A:.3f}, tau1 = {tau1:.1f} fs, B = {B:.3f}, tau2 = {tau2:.1f} fs")
    
    print("\nConduction Band Fit Parameters:")
    print(f"tau_rise = {tau_rise:.1f} fs, C = {C:.1f}, D = {D:.1f}")
    
    ### Plot results ###
    
    # Exciton decay plot
    ax[2*s].plot(delay_axis, trace_1, label='Data', color='black', alpha=1)
    #ax[0].plot(t, exciton_signal, label='True Exciton Model', color='red', linestyle='dashed')
    ax[2*s].plot(t, exciton_fitted, label=f"tau_1 = {tau1:.0f} fs, tau_2 = {tau2:.0f} fs", color='grey', linestyle='solid')
    ax[2*s].axvline(0, linestyle='--', color='gray', alpha=0.5)
    ax[2*s].set_xlabel('Time (fs)')
    ax[2*s].set_ylabel('Norm. Int., a.u.')
    ax[2*s].set_title('Exciton')
    ax[2*s].set_xlim(-250,3000)
    ax[2*s].legend(frameon=False)
    
    # Conduction band rise + decay plot
    ax[2*s+1].plot(delay_axis, trace_2, label='Data', color='maroon', alpha=1)
    #ax[1].plot(t, conduction_band_signal, label='True Conduction Model', color='red', linestyle='dashed')
    ax[2*s+1].plot(t, conduction_fitted, label= f"tau_r = {tau_rise:.0f} fs, tau_1 = {tau1:.0f} fs, tau_2 = {tau2:.0f} fs", color='red', linestyle='solid')
    ax[2*s+1].axvline(0, linestyle='--', color='gray', alpha=0.5)
    ax[2*s+1].set_xlabel('Time (fs)')
    ax[2*s+1].set_ylabel('Norm. Int., a.u.')
    ax[2*s+1].set_title('CBM')
    ax[2*s+1].set_xlim(-250,3000)
    ax[2*s+1].legend(frameon=False)

fig.tight_layout()
plt.show()

#%% Plot Fit Results: Combined

# Plot results
fig, ax = plt.subplots(2, 2, figsize=(8,6), sharex=False)
ax = ax.flatten()

n = [p**1 for p in power]
#n = power

# A, tau1, B, tau2, C, tau_rise, D

# Slow Exp Decay Component
ax[0].errorbar(n, popt_combined[:,3], yerr = perr_combined[:,3], marker = 'o', color = 'g')
#ax[0].errorbar(n, popt_cbm[:,4], yerr = perr_cbm[:,4], marker = 'o', color = 'r')
ax[0].set_title('Exp. Decay (slow)')
ax[0].set_ylabel('tau, fs')
ax[0].set_xlabel('n')
ax[0].set_ylim(0,12000)

# CBM Rise Time Component
#ax[1].plot(n, popt_cbm[:,1], color = 'red', marker = '*')
ax[1].errorbar(n, popt_combined[:,5], yerr = perr_combined[:,5], marker = 'o', color = 'purple')
ax[1].set_ylim(0, 500)
ax[1].set_title('CBM Rise Time')
ax[1].set_ylabel('tau, fs')
ax[1].set_xlabel('n')

# FAST Exp Decay Component
#ax[3].plot(n, popt_ex[:,1], 'k*', label = 'Fast')
ax[2].errorbar(n, popt_combined[:,1], yerr = perr_combined[:,1], marker = '*', color = 'g', label = 'Fast')
#ax[2].errorbar(n, popt_cbm[:,3], yerr = perr_cbm[:,3], marker = '*', color = 'pink', label = 'Fast')
ax[2].set_title('Exp. Decay (Fast)')
ax[2].set_ylabel('tau, fs')
ax[2].set_xlabel('n')
ax[2].set_ylim(-100,1500)

# Exp AMPLITUDES Components
ax[3].errorbar(n, popt_combined[:,0], yerr = perr_combined[:,0], marker = 'o', color = 'grey', label = 'Fast EX')
ax[3].errorbar(n, popt_combined[:,2], yerr = perr_combined[:,2], marker = 's', color = 'k', label = 'slow EX')
ax[3].errorbar(n, popt_combined[:,4], yerr = perr_combined[:,4], marker = '*', color = 'purple', label = 'RISE')
ax[3].errorbar(n, popt_combined[:,6], yerr = perr_combined[:,6], marker = '*', color = 'pink', label = 'Fast CBM')
ax[3].errorbar(n, (1-popt_combined[:,6]), yerr = perr_combined[:,6], marker = '*', color = 'red', label = 'Slow CBM')

ax[3].set_title('Exp. Amplitudes')
ax[3].set_ylabel('Amp.')
ax[3].set_xlabel('n')
ax[3].set_ylim(-.2,4)
ax[3].legend(frameon=False, fontsize = 8)

fig.tight_layout()

#%%
# Initialize figure
fig, ax = plt.subplots(1, 2, figsize=(10, 6))
plt.gcf().set_dpi(300)
ax = ax.flatten()

# Load Data
scans = [9526]

popt_ex, popt_cbm = np.zeros((len(scans),4)), np.zeros((len(scans),5))
perr_ex, perr_cbm = np.zeros((len(scans),4)), np.zeros((len(scans),5))

sigma_IRF = 100   # Fixed IRF width (fs)

for s in range(len(scans)):

    res = phoibos.load_data(data_path, scans[s], scan_info, energy_offset, delay_offset, False)
    delay_axis = res.Delay.values
    
    ### Extract time traces ###
    E, E_int = [1.37, 2.125], 0.1
    E, E_int = [2.125, 1.37], 0.1

    delay_limit = [-300, 3050]
    delay_axis = res.Delay.loc[{"Delay":slice(delay_limit[0], delay_limit[1])}].values
    
    # EXCITON
    trace_1 = res.loc[{'Angle':slice(-12,12), 'Energy':slice(E[0]-E_int/2, E[0]+E_int/2)}].sum(axis=(0,1))
    trace_1 = trace_1-trace_1.loc[{"Delay":slice(-600,-200)}].mean()
    trace_1 = trace_1.loc[{"Delay":slice(delay_limit[0],delay_limit[1])}]
    norm_factor = np.max(trace_1)
    
    # CBM
    trace_2 = res.loc[{'Angle':slice(-12,12), 'Energy':slice(E[1]-E_int/2, E[1]+E_int/2)}].sum(axis=(0,1))
    trace_2 = trace_2-trace_2.loc[{"Delay":slice(-600,-200)}].mean()
    trace_2 = trace_2.loc[{"Delay":slice(delay_limit[0],delay_limit[1])}]
    
    trace_1 = trace_1/norm_factor
    trace_2 = trace_2/norm_factor
    
    # Initial guesses for fitting
    init_guess_exciton = [0.8, 100, 0.2, 1000]  # A, tau1, B, tau2
    init_guess_conduction = [6, 200, .5, 300, 5000]  # C, tau_rise, D, tau_decay1, tau_decay2
    
    # Bounds for fitting
    bnds_exciton = [ [0,0,0,0], [2,1000,2,30000] ]
    bnds_cbm =  [[0,0,0,0,0], [10, 3500, 10, 3000, 30000] ]
    
    # Fit noisy exciton data
    popt_exciton, pcov = curve_fit(exciton_model, delay_axis, trace_1, p0=init_guess_exciton, bounds = bnds_exciton)
    perr_exciton = np.sqrt(np.diag(pcov))

    # Fit noisy conduction band data
    popt_conduction, pcov = curve_fit(cbm_model, delay_axis, trace_2, p0=init_guess_conduction, bounds = bnds_cbm)
    perr_conduction = np.sqrt(np.diag(pcov))

    # Save Parameters
    popt_ex[s,:] = popt_exciton
    popt_cbm[s,:] = popt_conduction
    
    perr_ex[s,:] = perr_exciton
    perr_cbm[s,:] = perr_conduction

    # Generate fitted curves
    exciton_fitted = exciton_model(t, *popt_exciton)
    conduction_fitted = cbm_model(t, *popt_conduction)
    
 #   conduction_fitted = biexp(t-150, 3.5, 400, 1, 10000)
    
    # Print best-fit parameters
    print("Exciton Fit Parameters:")
    print(f"A = {popt_exciton[0]:.3f}, tau1 = {popt_exciton[1]:.1f} fs, B = {popt_exciton[2]:.3f}, tau2 = {popt_exciton[3]:.1f} fs")
    
    print("\nConduction Band Fit Parameters:")
    print(f"tau_rise = {popt_conduction[1]:.1f} fs, C = {popt_conduction[0]:.1f}, D = {popt_conduction[2]:.1f}, tau1 = {popt_conduction[3]:.1f} fs, tau2 = {popt_conduction[4]:.1f} fs")
    
    ### Plot results ###
    
    # Exciton decay plot
    ax[2*s].plot(delay_axis, trace_1, label='Data', color='black', alpha=1)
    #ax[0].plot(t, exciton_signal, label='True Exciton Model', color='red', linestyle='dashed')
    ax[2*s].plot(t, exciton_fitted, label=f"tau_1 = {popt_exciton[1]:.0f} fs, tau_2 = {popt_exciton[3]:.0f} fs", color='grey', linestyle='solid')
    ax[2*s].axvline(0, linestyle='--', color='gray', alpha=0.5)
    ax[2*s].set_xlabel('Time (fs)')
    ax[2*s].set_ylabel('Norm. Int., a.u.')
    ax[2*s].set_title('Exciton')
    ax[2*s].set_xlim(-250,3000)
    ax[2*s].legend(frameon=False)
    
    # Conduction band rise + decay plot
    ax[2*s+1].plot(delay_axis, trace_2, label='Data', color='maroon', alpha=1)
    #ax[1].plot(t, conduction_band_signal, label='True Conduction Model', color='red', linestyle='dashed')
    ax[2*s+1].plot(t, conduction_fitted, label= f"tau_r = {popt_conduction[1]:.0f} fs, tau_1 = {popt_conduction[3]:.0f} fs, tau_2 = {popt_conduction[4]:.0f} fs", color='red', linestyle='solid')
    ax[2*s+1].axvline(0, linestyle='--', color='gray', alpha=0.5)
    ax[2*s+1].set_xlabel('Time (fs)')
    ax[2*s+1].set_ylabel('Norm. Int., a.u.')
    ax[2*s+1].set_title('CBM')
    ax[2*s+1].set_xlim(-250,3000)
 #   ax[2*s+1].legend(frameon=False)

fig.tight_layout()
plt.show()
#%% Fit 400 nm hc Relaxtion

scan = 9525
res = phoibos.load_data(data_path, scan, scan_info, energy_offset, delay_offset, force_offset)

res_neg = res.loc[{'Delay':slice(-1000,-300)}]

res_diff = res - res_neg.mean(axis=2)
res_diff_sum_Angle = res_diff.loc[{'Angle':slice(-A-A_int/2,A+A_int/2)}].sum(axis=0)
res_diff_sum_Angle = res_diff_sum_Angle/np.max(res_diff_sum_Angle)
res_diff_sum_Angle = phoibos.enhance_features(res_diff_sum_Angle, .8, _ , True)

E_int = 0.05
d1, d2 = -100, 500
e1, e2 = 1.8, 3.075
trace_max = np.zeros((res.Energy.loc[{"Energy":slice(e1,e2)}].values.shape))
ei = 0
for e in res.Energy.loc[{"Energy":slice(e1,e2)}].values:
    
    trace = res_diff_sum_Angle.loc[{"Energy":slice(e-E_int/2, e+E_int/2)}].mean(dim="Energy")
    trace = trace / np.max(trace)
    
    p0 = [.9, 10, 100, 0] # Fitting params initial guess [amp, center, width, offset]
    bnds = ((0.5, -100, 0.0, 0), (1.5, 600, 300, .5))

    popt, pcov = curve_fit(gaussian, trace.Delay.loc[{"Delay":slice(d1,d2)}].values, trace.loc[{"Delay":slice(d1,d2)}].values, p0, method=None, bounds = bnds)

    tmax_i = (trace-np.max(trace)).argmax()   
    trace_max[ei] = res.Delay.values[tmax_i]
    
    trace_max[ei] = popt[1]

    ei += 1
    
plt.figure()
plt.plot(trace_max, res.Energy.loc[{"Energy":slice(e1,e2)}].values)
    
# peak_centers = np.zeros((res.Delay.loc[{"Delay":slice(-10,500)}].values.shape))
# di = 0    
# for d in res.Delay.loc[{"Delay":slice(-10,500)}].values:

#     edc = res_diff_sum_Angle.loc[{"Delay":slice(d-50/2, d+50/2)}].mean(dim="Delay")
#     edc = edc/np.max(edc.loc[{"Energy":slice(1,3)}])
#     e1 = 1.8
#     e2 = 3
    
#     p0 = [.9, 2, .12, 0] # Fitting params initial guess [amp, center, width, offset]
#     bnds = ((0.5, 1.9, 0.0, 0), (1.5, 3, 1, .5))
#     popt, pcov = curve_fit(gaussian, edc.Energy.loc[{"Energy":slice(e1,e2)}].values, edc.Energy.loc[{"Energy":slice(e1,e2)}].values, p0, method=None, bounds = bnds)

#     peak_centers[di] = popt[1]
    
#     di += 1
    
# plt.figure()
# plt.plot(res.Delay.loc[{"Delay":slice(-10,500)}].values, peak_centers)    
