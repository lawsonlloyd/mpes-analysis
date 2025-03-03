# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:46:45 2025

@author: lloyd
"""


#%%

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
from skimage.draw import disk
from scipy.optimize import curve_fit
import csv
from Loader import DataLoader
import xarray as xr

#%%

filename = '2024 Bulk CrSBr Phoibos.csv'

scan_info = {}
data_path = 'R:\Lawson\Data'
data_path = '/Users/lawsonlloyd/Desktop/phoibos'

with open(data_path + '//' + filename) as f:
    
    reader = csv.DictReader(f
                            )
    for row in reader:
        key = row.pop('Scan')
        if key in scan_info:
            # implement your duplicate row handling here
            pass
        scan_info[key] = row
        
            
def load_data(scan, energy_offset, delay_offset):
    filename = f"Scan{scan}.h5"
    
    data_loader = DataLoader(data_path + '//' + filename)
    res = data_loader.load_phoibos()
    
    res = res.assign_coords(Energy=(res.Energy-energy_offset))
    res = res.assign_coords(Delay=(res.Delay-delay_offset))
    
    return res

#%%


def objective(params, x, data):
    
    g1, g2, offset = two_gaussians(x, **params)
    fit = g1+g2+offset
    resid = np.abs(data-fit)**2
    
    return resid

#%%

phoibos = True

#data_path = 'path_to_your_data'
#filename = 'your_file_name.h5'

#data_path = 'R:\Lawson\Data\phoibos'
#data_path = '/Users/lawsonlloyd/Desktop/Data/'

scan = 9228
energy_offset = + 19.72
delay_offset = -80

res = load_data(scan, energy_offset, delay_offset)

#%%

scans = [9219, 9217, 9218, 9216, 9220, 9228]
offsets_t0 = [-162.4, -152.7, -183.1, -118.5, -113.7, -125.2]

scans = [9227, 9219, 9217, 9218, 9216, 9220, 9228, 9231, 9525, 9517, 9526] # Scans to analyze and fit below: 910 nm + 400 nm
offsets_t0 = [-191.7, -162.4, -152.7, -183.1, -118.5, -113.7, -125.2, -147.6, -77, -151.1, -200.6]

#res = load_data(scan_i, 19.72,  offsets_t0[i])

trans_percent = [float(scan_info[str(s)].get("Percent")) for s in scans] # Retrieve the percentage
power = [4.7, 8.3, 20.9, 41.7, 65.6, 83.2, 104.7, 151, 20, 36.3, 45]

#%%

# Give the data points for Exciton and CB signal...for all Fluences...

E = [1.35, 2.05]

data_910nm = np.zeros((8,2,69))
data_400nm = np.zeros((3,2,49))
data_ALL_WL = np.zeros((11,2,69))
counts = [] 

delay_axes = []
data_traces = []

s = 0
for scan in scans:

#    E, Eint = [21.75, 21], 0.1 # center energies, half of full E integration range
    E, Eint = [1.35, 2.1], 0.1 # center energies, half of full E integration range
    a, aint, a_full = [-12, 15], 20, [-11, 15]

#    res_to_plot = vars()[str('res_'+str(scans[s]))]
    res = load_data(scan, 19.72,  offsets_t0[s])

    delay_limit = [np.min(res.Delay.values), np.max(res.Delay.values)]
    delay_limit[0] = -450
    delay_limit[1] = 3030

    ### Extract time data ###
    
    #EX
    trace_1 = res.loc[{'Angle':slice(-12,12), 'Energy':slice(E[0]-E_int/2, E[0]+E_int/2)}].sum(axis=(0,1))
    trace_1 = trace_1-trace_1.loc[{"Delay":slice(-600,-200)}].mean()
    trace_1 = trace_1.loc[{"Delay":slice(delay_limit[0],delay_limit[1])}]
    norm_factor = np.max(trace_1)
    
    #CB
    trace_2 = res.loc[{'Angle':slice(-12,12), 'Energy':slice(E[1]-E_int/2, E[1]+E_int/2)}].sum(axis=(0,1))
    trace_2 = trace_2-trace_2.loc[{"Delay":slice(-600,-200)}].mean()
    trace_2 = trace_2.loc[{"Delay":slice(delay_limit[0],delay_limit[1])}]
    
    R = np.max(trace_2)/np.max(trace_1)
    counts.append(np.max(trace_2.values))

    trace_1 = trace_1/norm_factor
    trace_2 = trace_2/norm_factor

    # Collect all data together
    delay_axes.append(trace_2.Delay.values)
    data_traces.append(trace_1.values)   
    data_traces.append(trace_2.values)   

    #data_ALL[s,:,:] = np.asarray([power[s]*trace_2/np.max(trace_2), power[s]*trace_1/np.max(trace_2)])
    # if s < 7:
    #     #data_910nm[s,:,:] = np.asarray([1*trace_2/np.max(trace_2),1*trace_1/np.max(trace_2)])
    #     delay_times_910nm = trace_2.Delay.values
    #     data_ALL_WL[s,:,:] = np.asarray([trace_2, trace_1])
        
    # if s == 7:
    #     delay_times_9231 = trace_2.Delay.values
    #     trace_2 = np.interp(delay_times_910nm, delay_times_9231, trace_2)
    #     trace_1 = np.interp(delay_times_910nm, delay_times_9231, trace_1)
    #     #data_400nm[0,:,:] = np.asarray([1*trace_2/np.max(trace_2), 1*trace_1/np.max(trace_2)]) #By default normalize to relative ratio   
    #     data_ALL_WL[s,:,:] = np.asarray([trace_2, trace_1])
    # if s > 7:
    #     delay_times_400nm = trace_2.Delay.values
    #     trace_2 = np.interp(delay_times_910nm, delay_times_400nm, trace_2)
    #     trace_1 = np.interp(delay_times_910nm, delay_times_400nm, trace_1)
    #     R = np.max(trace_2)/np.max(trace_1)
    #     trace_2 = trace_2/np.max(trace_2)
    #     trace_1 = trace_1/np.max(trace_1)
    #     trace_2 = R*trace_2

    #     #data_400nm[0,:,:] = np.asarray([1*trace_2/np.max(trace_2), 1*trace_1/np.max(trace_2)]) #By default normalize to relative ratio   
    #     data_ALL_WL[s,:,:] = np.asarray([trace_2, trace_1])
    
    # t_values = delay_times_910nm
    
    s += 1
    
#%%

from scipy.integrate import solve_ivp

tau_ex_r = 20000
tau_EEA = 2500
tau_ex_f = 100
tau_hc = 100
t_0 = 100
fwhm = 80
F = 1
H = 0
fi = 100

def global_fit_Excitons_and_CB_ALL_WL(t, N, fi, F, H, t_0, fwhm, tau_ex_r, tau_EEA, tau_ex_f, tau_hc):
    
    #G = np.exp(-(t-t_0)**2/(np.sqrt(2)*fwhm/2.3548200)**2)/(fwhm/2.3548200*np.sqrt(2*np.pi))
    G = np.exp(-(t-0)**2/(np.sqrt(2)*fwhm/2.3548200)**2)/(fwhm/2.3548200*np.sqrt(2*np.pi))

    f0 = 8.3
    h0 = 20
    
    #EEA Auger Model
    Nex_prime = F*(fi/f0)*G - N[0]/tau_ex_r - (N[0]**2)/(tau_EEA) + (N[1]**2)/tau_ex_f

    Ncb_prime = -1*(N[1]**2)/tau_ex_f + N[2]/(tau_hc) + 0.5*(N[0]**2)/(tau_EEA)
    #Ncb_prime = -N[1]**2/tau_ex_f + N[2]/(tau_hc) + N[0]**2/tau_EEA

    #Ncb_prime = H*G - N[1]**2/tau_ex_f + 0.5*N[0]**2/tau_EEA # + N[2]/tau_hc
    Nhc_prime = H*(fi/h0)*G - N[2]/tau_hc #+ 0.5*N[0]**2/tau_EEA)

    return [Nex_prime, Ncb_prime, Nhc_prime]

params = (fi, F, H, t_0, fwhm, tau_ex_r, tau_EEA, tau_ex_f, tau_hc)
N0 = [0, 0, 0]
#res = solve_ivp(fit_Excitons_and_CB_400nm, t_span=(-300, delay_limit[1]), t_eval=np.linspace(-300,  delay_limit[1], int((delay_limit[1]+300)/50)) , y0=N0, args = params)
res_ALL_2 = solve_ivp(global_fit_Excitons_and_CB_ALL_WL, t_span=(-200, 3000), t_eval=np.linspace(-200, 3000, 100), y0=N0, args = params)

#%%

%matplotlib inline

plt.plot(res_ALL_2.t, res_ALL_2.y.T/np.max(res_ALL_2.y[0,:]), label = ('X', 'CB', 'hc'))
plt.legend(frameon=False)
plt.axvline(0, linestyle ='dashed', color = 'grey')
plt.show()

#%%

def ode_resolution_GLOBAL(t, fi, i, a0, a1,a2,a3,a4,a5,a6,a7,a8,a9,a10, b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10, F, H, fwhm, tau_ex_r, tau_EEA, tau_ex_f, tau_hc):
    N0 = [0, 0, 0]
    A = [a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10]
    B = [b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10]
    t_0 = 0
    
    t = t - t_0
    if i > 7:
        F = 0
    else:
        H = 0
        
    res_model = solve_ivp(global_fit_Excitons_and_CB_ALL_WL,  t_span = (t[0], t[-1]), t_eval=t, y0=N0, args = (fi, F, H, t_0, fwhm, tau_ex_r, tau_EEA, tau_ex_f, tau_hc))
    #ret = np.asarray([A[i]*res.y[0]/ res.y[0].max(), B[i]*res.y[1]/ res.y[1].max()])
    
    if i > 7: #FOR HC EXCITATION
        ret = np.asarray([A[i]*res_model.y[0]/ res_model.y[1].max(), B[i]*res_model.y[1]/ res_model.y[1].max()])
    else: #FOR EXCITON EXCITATION
        ret = np.asarray([A[i]*res_model.y[0]/ res_model.y[0].max(), B[i]*res_model.y[1]/ res_model.y[0].max()])
    
    return ret

#%%

def objective_GLOBAL(params, ts, data):
    """Calculate total residual for fits"""
    
#    resid = 0.0*data[:]
    resid = []
    for i in range(len(ts)):
        fi = power[i]  
        t = ts[i]
        
        res_GLOBAL_model = ode_resolution_GLOBAL(t, fi, i, **params)
        #res_GLOBAL = ode_resolution_GLOBAL(x, **params)
    
        # make residual per data set
        resid.extend((data[2*i:2*i+2] - res_GLOBAL_model)**2) #/sigma
    
    # now flatten this to a 1D array, as minimize() needs
    return np.concatenate(resid)

#%% Testing with Hand-Picked Paramters

%matplotlib inline

t_values = np.arange(-500,3100,50)

tau_ex_r = 15000
tau_EEA = 4500
tau_ex_f = 300 #40
tau_hc = 0 #258
fwhm = 80
F = 1
H = 1

t_0 = 0
a = [.9, .92, .94, .99, .93, .90, 0.88, .95, 0.4, .3, .32]
b = [2.4, 2.5, 2.83, 3.91, 4.57, 5.22, 5.4, 5.17, 4, 3.2, 2.75]

#t_0 = np.zeros(11)
#a = [1.0*x for x in a]
#b = [4.2*x for x in b]
#b = np.ones((8))
#b = [4, 4, 4, 4, 4 ,4]
#params_GLOBAL = {"F": F, "t_0": t_0, "fwhm": fwhm, "tau_ex_r":tau_ex_r, "tau_EEA":tau_EEA, "tau_ex_f":tau_ex_f, "tau_hc":tau_hc}

params_GLOBAL = {}
for i in range(10+1):
    params_GLOBAL.update({str("a"+str(i)): a[i]})
for i in range(10+1):
    params_GLOBAL.update({str("b"+str(i)): b[i]})

#for i in range(10+1):
#    params_GLOBAL.update({str("t0"+str(i)): t_0[i]})
    
params_GLOBAL.update({"F": F, "H": H, "fwhm": fwhm, "tau_ex_r":tau_ex_r, "tau_EEA":tau_EEA, "tau_ex_f":tau_ex_f, "tau_hc":tau_hc})
    
res_ALL_TEST = np.zeros((11,2,len(t_values)))

for i in range(10+1):
    fi = power[i]
    res_ALL_TEST[i,:,:] = ode_resolution_GLOBAL(t_values, fi, i, **params_GLOBAL)

fig, ax = plt.subplots(3,4)
fig.set_size_inches(12, 8, forward=False)
ax = ax.flatten()

for s in range(11):
    
    d1, d2 = data_traces[2*s], data_traces[2*s+1]
    #d1, d2 = d1/np.max(d1), d2/np.max(d2)
    
    r1, r2 = res_ALL_TEST[s,0,:], res_ALL_TEST[s,1,:]
    #r1, r2 = r1/np.max(r1), 0.8*r2/np.max(r2)
    gaussian_pulse = gaussian(t_values, 1, 0, fwhm/2.355, 0)

    ax[s].set_title(str(round(power[s],1)) + ' mW')
    ax[s].plot(delay_axes[s], d1, 'ko')
    ax[s].plot(delay_axes[s], d2, 'ro')
    ax[s].plot(t_values, r1, 'blue')
    ax[s].plot(t_values, r2, 'green')
#    ax[s].plot(t_values, gaussian_pulse, linestyle = 'dashed', color='green')
    #ax[s].axvline(0, linestyle = 'dashed', color='green')
    ax[s].set_xlabel('Delay, fs')
    ax[s].set_ylabel('Norm. Int., a.u.')
    ax[s].set_xlim([-700,3000])

#test = np.exp(-t/800)+ np.exp(-t/15000)

#ax[s].plot(t, test/np.max(test), 'b*')    
fig.tight_layout()

#%%

# Define Global Fit Paramters

from lmfit import Parameters, minimize, report_fit

tau_ex_r = 15000
tau_EEA = 4500
tau_ex_f = 200 #40
tau_hc = 50 #258
fwhm = 80
F = 1
H = 1

t_0 = 0
a = [.9, .92, .94, .99, .93, .90, 0.88, .95, 0.4, .3, .32]
b = [2.4, 2.5, 2.83, 3.91, 4.57, 5.22, 5.4, 5.17, 4, 3.2, 2.75]

#a = 0.98*np.ones((11))
#b = 4.4*np.ones((11))

fit_params = Parameters()
for i in range(10+1):
    fit_params.add(str("a"+str(i)), value=a[i], min=0.2, max=5, vary=True)
    fit_params.add(str("b"+str(i)), value=b[i], min=0.2, max=10, vary=True)

fit_params.add("F", value=F, min=0.01, max=10, vary=False)
fit_params.add("H", value=H, min=0.01, max=10, vary=False)

fit_params.add("fwhm", value=fwhm, min=10, max=120, vary=False)

fit_params.add("tau_ex_r", value=tau_ex_r, min=10, max=30000, vary=False)
fit_params.add("tau_EEA", value=tau_EEA, min=5000, max=10000, vary=True)
fit_params.add("tau_ex_f", value=tau_ex_f, min=50, max=1200, vary=True)
fit_params.add("tau_hc", value=tau_hc, min=20, max=250, vary=True)

fit_params

#%% Do the Global fitting!

data = data_traces

# for s in range(10+1):
#     for f in range(2):
#         data[s,f,:] = data[s,f,:] #/np.max(data[s,f,:]) 

out = minimize(objective_GLOBAL, fit_params, args=(delay_axes, data_traces))
report_fit(out)

#%% GLOBAL: Plot the Results!
import time
ym = time.strftime("%Y%m%d")

%matplotlib inline

res_ALL = np.zeros((11,2,len(t_values)))

for i in range(0,11):

    fi = power[i]
    res_ALL[i,:,:] = ode_resolution_GLOBAL(t_values, fi, i, **out.params)
    #res_ALL[i,:,:] = ode_resolution_GLOBAL(t, **out.params)

fig, ax = plt.subplots(3,4)
fig.set_size_inches(12, 8, forward=False)
ax = ax.flatten()

for s in range(10+1):
    
    d1, d2 = data_traces[2*s], data_traces[2*s+1]
    #d1, d2 = d1/np.max(d1), d2/np.max(d2)
    
    r1, r2 = res_ALL[s,0,:], res_ALL[s,1,:]
    
    ax[s].set_title(str(round(power[s],1)) + ' mW')
    ax[s].plot(delay_axes[s], d1, 'ko')
    ax[s].plot(delay_axes[s], d2, 'ro')
    ax[s].plot(t_values, r1, 'blue')
    ax[s].plot(t_values, r2, 'green')
    #ax[s].axvline(0, linestyle = 'dashed', color='green')
    ax[s].set_xlabel('Delay, fs')
    ax[s].set_ylabel('Norm. Int., a.u.')
    ax[s].set_xlim([-500,3000])
    ax[s].set_ylim(-.1,1.2)

    if s > 7:
        ax[s].set_ylim(-.1, 3.5)
    #ax[s].plot(t_values_offset-t_0[s], gaussian_pulse, linestyle = 'dashed', color='green')
    
fig.tight_layout()
plt.show()


#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Example data (replace with actual data)
fluence = np.array([1, 2, 3, 5, 7, 10])  # Excitation densities
decay_times = np.array([10, 5.5, 3.8, 2.4, 1.8, 1.2])  # Extracted tau values from fits

# Convert to decay rates
decay_rates = 1 / decay_times

# Define power-law function
def power_law(n, gamma0, beta, alpha):
    return gamma0 + beta * n**alpha

# Fit to the data
popt, pcov = curve_fit(power_law, fluence, decay_rates)
gamma0_fit, beta_fit, alpha_fit = popt

# Plot results
plt.figure(figsize=(6,4))
plt.scatter(fluence, decay_rates, label="Data", color="red")
plt.plot(fluence, power_law(fluence, *popt), label=f"Fit: α = {alpha_fit:.2f}", linestyle="--", color="blue")
plt.xlabel("Excitation Density (n)")
plt.ylabel("Decay Rate (Γ = 1/τ)")
plt.legend()
plt.grid()
plt.show()

print(f"Fitted parameters: Gamma0 = {gamma0_fit:.3f},

#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve

# Define the monoexponential decay function (Exciton)
def monoexp(t, A, tau):
    return A * np.exp(-t / tau) * (t >= 0)  # Ensure decay starts at t=0

# Define the biexponnential decay function (Exciton)    
def biexp(t, A, tau1, B, tau2):
    return ( A * np.exp(-t / tau1) + B * np.exp(-t / tau2))  * (t >= 0)  # Ensure decay starts at t=0

# Define the conduction band model: exponential rise + decay
def exp_rise_decay(t, B, tau_rise, tau_decay):
    return B * (1 - np.exp(-t / tau_rise)) * np.exp(-t / tau_decay) * (t >= 0)

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

# Generate time axis in femtoseconds
t = np.arange(-500, 3500, 20)  # fs

# Set true parameters
A = 1.0      # Exciton amplitude
tau_exc = 300  # Exciton decay time (fs)
B = 0.5      # Conduction band amplitude
tau_rise = 200   # Conduction band rise time (fs)
tau_decay = 500  # Conduction band decay time (fs)
sigma_IRF = 29   # Fixed IRF width (fs)

# Generate convolved signals (true model)
exciton_signal = convolved_signal(t, monoexp, sigma_IRF, A, tau_exc)
conduction_band_signal = convolved_signal(t, exp_rise_decay, sigma_IRF, B, tau_rise, tau_decay)

# Generate noisy data (simulating experimental ARPES)
noise_level = 0.02
exciton_noisy = exciton_signal + noise_level * np.random.normal(size=t.shape)
conduction_noisy = conduction_band_signal + noise_level * np.random.normal(size=t.shape)

# Fitting functions (excluding IRF from fit)
def fit_exciton(t, A, tau1, B, tau2):
    return convolved_signal(t, biexp, sigma_IRF, A, tau1, B, tau2)  # IRF is fixed

def fit_conduction_band(t, C, tau_rise, tau_decay):
    return convolved_signal(t, exp_rise_decay, sigma_IRF, C, tau_rise, tau_decay)  # IRF is fixed

#####
#####
#####

fig, ax = plt.subplots(6, 2, figsize=(12, 16))
ax = ax.flatten()

popt_ex = np.zeros((6,4))
popt_cbm = np.zeros((6,5))

# Load Data
scans = [9219, 9217, 9218, 9216, 9220, 9228]
offsets_t0 = [-162.4, -152.7, -183.1, -118.5, -113.7, -125.2]
power = [8.3, 20.9, 41.7, 65.6, 83.2, 104.7]

for s in range(len(scans)):
    print(s)
    res = load_data(scans[s], 19.72,  offsets_t0[s])
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
    
    # Initial guesses for fitting
    init_guess_exciton = [0.8, 100, 0.2, 1000]  # A, tau1, B, tau2
    init_guess_conduction = [0.4, 150, 400]  # C, tau_rise, tau_decay
    
    # Bounds for fitting
    bnds_exciton = [ [0,0,0,0], [3,1000,3,25000] ]
    bnds_cbm =  [[0,0,0], [2,2000,25000] ]
    # Fit noisy exciton data
    popt_exciton, _ = curve_fit(fit_exciton, delay_axis, trace_1, p0=init_guess_exciton, bounds = bnds_exciton)

    # Fit noisy conduction band data
    popt_conduction, _ = curve_fit(fit_conduction_band, delay_axis, trace_2, p0=init_guess_conduction, bounds = bnds_cbm)
    
    # Save Parameters
    popt_ex[s,:] = popt_exciton
    popt_cbm[s,0:3] = popt_conduction
    
    # Generate fitted curves
    exciton_fitted = fit_exciton(t, *popt_exciton)
    conduction_fitted = fit_conduction_band(t, *popt_conduction)
    
    # Print best-fit parameters
    print("Exciton Fit Parameters:")
    print(f"A = {popt_exciton[0]:.3f}, tau = {popt_exciton[1]:.1f} fs")
    
    print("\nConduction Band Fit Parameters:")
    print(f"B = {popt_conduction[0]:.3f}, tau_rise = {popt_conduction[1]:.1f} fs, tau_decay = {popt_conduction[2]:.1f} fs")
    
    # Plot results
    
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
    ax[2*s+1].plot(t, conduction_fitted, label= f"tau_r = {popt_conduction[1]:.0f} fs, tau_1 = {popt_conduction[2]:.0f} fs", color='red', linestyle='solid')
    ax[2*s+1].axvline(0, linestyle='--', color='gray', alpha=0.5)
    ax[2*s+1].set_xlabel('Time (fs)')
    ax[2*s+1].set_ylabel('Norm. Int., a.u.')
    ax[2*s+1].set_title('CBM')
    ax[2*s+1].set_xlim(-250,3000)
    ax[2*s+1].legend(frameon=False)

fig.tight_layout()
plt.show()

#%%

# Plot results
fig, ax = plt.subplots(2, 2, figsize=(8,6), sharex=False)
ax = ax.flatten()

n = [p**1 for p in power]
#n = power
ax[0].plot(n, popt_ex[:,3], 'ko', label = 'Slow')
ax[0].plot(n, popt_cbm[:,2], 'ro', label = 'Dec')
ax[0].set_title('Exp. Decay')
ax[0].set_ylabel('tau, fs')
ax[0].set_xlabel('n')

ax[1].plot(n, tau_fits_cbm[:,1], color = 'red', marker = '*')
ax[1].set_ylim(0, 250)
ax[1].set_title('CBM Rise Time')
ax[1].set_ylabel('tau, fs')
ax[1].set_xlabel('n')

ax[2].plot(n, popt_ex[:,0], 'ko', label = 'Fast')
ax[2].plot(n, popt_ex[:,2], 'k*', label = 'Slow')
ax[2].plot(n, popt_cbm[:,0], 'ro', label = 'CBM')
ax[2].set_title('Exp. Amplitude')
ax[2].set_ylabel('Amp.')
ax[2].set_xlabel('n')
ax[2].legend(frameon=False, fontsize = 8)

ax[3].plot(n, popt_ex[:,1], 'k*', label = 'Fast')
ax[3].set_title('Exp. Decay (Fast)')
ax[3].set_ylabel('tau, fs')
ax[3].set_xlabel('n')

fig.tight_layout()

#%%

from scipy.optimize import curve_fit

# Define the function for fitting: 1/tau(n) = 1/tau_0 + gamma * n^m
def rate_model(n, tau_0, gamma, m):
    return 1/tau_0 + gamma * n**m

# Prepare data for fitting
inverse_tau_noisy = 1 / tau_n2_noisy  # Convert lifetime to rate

# Perform the fit
popt, pcov = curve_fit(rate_model, n_values, inverse_tau_noisy, p0=[300, 1e-9, 2])

# Extract fitted parameters
tau_0_fit, gamma_fit, m_fit = popt
print(f"Fitted parameters: tau_0 = {tau_0_fit:.2f} fs, gamma = {gamma_fit:.2e}, m = {m_fit:.2f}")


# Plot results
fig, ax = plt.subplots(2, 2, figsize=(8,6), sharex=False)
ax = ax.flatten()

n = [p**1 for p in power]
#n = power
ax[0].plot(n, 1/popt_ex[:,3], 'ko', label = 'Slow')
ax[0].plot(n, 1/popt_cbm[:,2], 'ro', label = 'Dec')
ax[0].set_title('Exp. Decay')
ax[0].set_ylabel('tau, fs')
ax[0].set_xlabel('n')

ax[1].plot(n, 1/tau_fits_cbm[:,1], color = 'red', marker = '*')
#ax[1].set_ylim(0, 250)
ax[1].set_title('CBM Rise Time')
ax[1].set_ylabel('tau, fs')
ax[1].set_xlabel('n')

ax[2].plot(n, popt_ex[:,0], 'ko', label = 'Fast')
ax[2].plot(n, popt_ex[:,2], 'k*', label = 'Slow')
ax[2].plot(n, popt_cbm[:,0], 'ro', label = 'CBM')
ax[2].set_title('Exp. Amplitude')
ax[2].set_ylabel('Amp.')
ax[2].set_xlabel('n')
ax[2].legend(frameon=False, fontsize = 8)

ax[3].plot(n, 1/popt_ex[:,1], 'k*', label = 'Fast')
ax[3].set_title('Exp. Decay (Fast)')
ax[3].set_ylabel('tau, fs')
ax[3].set_xlabel('n')

fig.tight_layout()
