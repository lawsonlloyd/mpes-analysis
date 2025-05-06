# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:46:45 2025

@author: lloyd
"""

#%% Load Packages

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
from skimage.draw import disk
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
import csv
from Loader import DataLoader
import xarray as xr
import phoibos

#%% Load Data Scan Info

filename = '2024 Bulk CrSBr Phoibos.csv'

scan_info = {}
data_path_info = 'R:\Lawson\mpes-analysis'
data_path = 'R:\Lawson\Data\phoibos'

data_path = '/Users/lawsonlloyd/Desktop/Data/phoibos'
data_path_info = '/Users/lawsonlloyd//GitHub/mpes-analysis'

energy_offset, delay_offset, force_offset = 19.62,  0, False

scan_info = phoibos.get_scan_info(data_path_info, filename, {})

#%% Specify Scan Data for the Model

scans = [9219, 9217, 9218, 9216, 9220, 9228]
scans = [9227, 9219, 9217, 9218, 9216, 9220, 9228, 9231, 9525, 9517, 9526] # Scans to analyze and fit below: 910 nm + 400 nm

trans_percent = [float(scan_info[str(s)].get("Percent")) for s in scans] # Retrieve the percentage
power = [4.7, 8.3, 20.9, 41.7, 65.6, 83.2, 104.7, 151, 20, 36.3, 45]
fluences = [0.11, .2, .35, .8, 1.74, 2.4, 2.9, 4.2, 3, 4.5, 6]

#%% Load the Exciton and CBM Traces

# Give the data points for Exciton and CB signal...for all Fluences...

E = [1.35, 2.05]
energy_offset, delay_offset, force_offset = 19.72, 0, True

data_910nm = np.zeros((8,2,69))
data_400nm = np.zeros((3,2,49))
data_ALL_WL = np.zeros((11,2,69))
counts = [] 

delay_axes = []
data_traces = []

s = 0
for scan_i in scans:

#    E, Eint = [21.75, 21], 0.1 # center energies, half of full E integration range
    E, Eint = [1.325, 2.075], 0.1 # center energies, half of full E integration range
    
    if s > 7:
        E, E_int = [1.33, 2], 0.1
    else:
        E, E_int = [1.33, 2.1], 0.1 # center energies, half of full E integration range

    a, aint, a_full = [-12, 15], 20, [-11, 15]

#    res_to_plot = vars()[str('res_'+str(scans[s]))]
    #res = phoibos.load_data(data_path, scan_i, scan_info, 19.72 , 0 , force_offset)
    res = phoibos.load_data(data_path, scan_i, scan_info, energy_offset, delay_offset, force_offset)

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
    
    norm_factor = np.max([np.max(trace_1), np.max(trace_2)])

    trace_1 = trace_1/norm_factor
    trace_2 = trace_2/norm_factor
    
#    trace_1 = trace_1/np.max(trace_1)
 #   trace_2 = trace_2/np.max(trace_2)
    
    
    #trace_1 = trace_1 * power[s]
    #trace_2 = trace_2 * power[s]

    # Collect all data together
    delay_axes.append(trace_2.Delay.values)
    data_traces.append(trace_1.values)   
    data_traces.append(trace_2.values)   
    
    s += 1
    
#%% Define Rate-Equation Model

def global_fit_Excitons_and_CB_ALL_WL(t, N, fi, F, H, t_0, fwhm, tau_ex_r, tau_EEA, tau_ex_f, tau_hc):
    
    #G = np.exp(-(t-t_0)**2/(np.sqrt(2)*fwhm/2.3548200)**2)/(fwhm/2.3548200*np.sqrt(2*np.pi))
    sigma = fwhm / 2.355
    G = np.exp(-0.5*(t-0)**2/(sigma**2)) / (sigma * np.sqrt(2*np.pi))
    #f0 = 4.7
    #h0 = 20
    
    f0 = 1
    h0 = 6
    
    #CBM and Exciton Formation/Relaxation
    #Nex_prime = F*(fi/f0)*G - N[0]/tau_ex_r - 0*(N[0]**2)/(tau_EEA) + (N[1])/tau_ex_f + (N[2])/tau_ex_f

    #EEA Auger Model
    Nex_prime = F*(fi/f0)*G - N[0]/tau_ex_r - (N[0]**2)/(tau_EEA) + (N[1])/tau_ex_f
    
    Ncb_prime = N[2]/(tau_hc) + 0.5*(N[0]**2)/(tau_EEA) - (N[1])/tau_ex_f

    Nhc_prime = H*(fi/h0)*G - N[2]/tau_hc # - (N[2])/tau_ex_f 

    return [Nex_prime, Ncb_prime, Nhc_prime]

#%% Test: Solve IVP

%matplotlib inline

tau_ex_r = 20000
tau_EEA = 1121 * 1e13
tau_ex_f = 50
tau_hc = 154
t_0 = 0
fwhm = 80
F = 0
H = 1
fi =  1e13
t_axis = np.linspace(-200, 3000, 250)

params = (fi, F, H, t_0, fwhm, tau_ex_r, tau_EEA, tau_ex_f, tau_hc)
N0 = [0, 0, 0]
#res = solve_ivp(fit_Excitons_and_CB_400nm, t_span=(-300, delay_limit[1]), t_eval=np.linspace(-300,  delay_limit[1], int((delay_limit[1]+300)/50)) , y0=N0, args = params)
res_ALL_2 = solve_ivp(global_fit_Excitons_and_CB_ALL_WL, t_span=(t_axis[0], t_axis[-1]), t_eval=t_axis, y0=N0, args = params)

#plt.plot(res_ALL_2.t, res_ALL_2.y.T/np.max(res_ALL_2.y[0,:]), label = ('X', 'CB', 'hc'))
plt.plot(res_ALL_2.t, res_ALL_2.y.T, label = ('X', 'CB', 'hc'))
#plt.plot(res_ALL_2.t, (res_ALL_2.y[0,:].T + res_ALL_2.y[1,:].T), label = ('sum'))

# fwhm_IRF = 80
# sigma_IRF = fwhm_IRF/2.355   # Fixed IRF width (fs)
# def exciton_model(t, N_0, gamma):
#     return convolved_signal(t, eea_exp, sigma_IRF, N_0, gamma)  # IRF is fixed
# test = exp_rise_monoexp_decay(res_ALL_2.t, 1*np.max(res_ALL_2.y[0,:]), 400, 3500)
# test = exciton_model(res_ALL_2.t, fi, 0.055e-15)

#plt.plot(res_ALL_2.t, test, color = 'grey')
#plt.axhline(0.5e13, linestyle = 'dashed', color = 'grey')
plt.axhline(0.5, linestyle = 'dashed', color = 'grey')
#plt.plot(res_ALL_2.t, res_ALL_2.y.T, label = ('X', 'CB', 'hc'))

plt.legend(frameon=False)
plt.axvline(0, linestyle ='dashed', color = 'grey')
plt.show()

#%% Define Objective function

def ode_resolution_GLOBAL(t, fi, i, a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10, b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10, F, H, fwhm, tau_ex_r, tau_EEA, tau_ex_f, tau_hc):
    N0 = [0, 0, 0]
    A = [a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10]
    B = [b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10]
    t_0 = 0
    
    t = t - t_0
    if i > 7:
        F = 0
        H = 1
    else:
        H = 0
        F = 1
        
    res_model = solve_ivp(global_fit_Excitons_and_CB_ALL_WL,  t_span = (t[0], t[-1]), t_eval=t, y0=N0, \
                          args = (fi, F, H, t_0, fwhm, tau_ex_r, tau_EEA, tau_ex_f, tau_hc))
    #ret = np.asarray([A[i]*res.y[0]/ res.y[0].max(), B[i]*res.y[1]/ res.y[1].max()])
    
    if i > 7: #FOR HC EXCITATION
        ret = np.asarray([A[i]*res_model.y[0]/ res_model.y[1].max(), B[i]*res_model.y[1]/ res_model.y[1].max()])
        #ret = np.asarray([A[i]*res_model.y[0]/ res_model.y[0].max(), B[i]*res_model.y[1]/ res_model.y[1].max()])

    else: #FOR EXCITON EXCITATION
        ret = np.asarray([A[i]*res_model.y[0]/ res_model.y[0].max(), B[i]*res_model.y[1]/ res_model.y[0].max()])
        #ret = np.asarray([A[i]*res_model.y[0]/ res_model.y[0].max(), B[i]*res_model.y[1]/ res_model.y[1].max()])

    return ret

def objective_GLOBAL(params, delay_axes, data):
    """Calculate total residual for fits"""
    
    resid = []
    for i in range(len(delay_axes)):        
    #for i in [9, 10]:        

        fi = power[i] 
        fi =  fluences[i] * 1e13
        t = delay_axes[i]
        res_GLOBAL_model = ode_resolution_GLOBAL(t, fi, i, **params)
        
        # make residual per data set
        
        #Normal
        resid.extend((data[2*i:2*i+2] - res_GLOBAL_model)**2) #/sigma

        #Exciton Only
        #resid.extend((data[2*i:2*i+1] - res_GLOBAL_model[0,:])**2) #/sigma
        
        #CBM Only
        #resid.extend((data[2*i+1:2*i+2] - res_GLOBAL_model[1,:])**2) #/sigma


    # now flatten this to a 1D array, as minimize() needs
    return np.concatenate(resid)

#%% Testing with Hand-Picked Paramters

%matplotlib inline

t_values = np.arange(-500,3100,10)

tau_ex_r = 20000
tau_EEA = 1121 * 1e13
tau_ex_f = 285
tau_hc = 154 #258
fwhm = 80
F = 1
H = 1

a = [.9, .92, .94, .99, .93, .90, 0.88, .95, \
     .05, .12, .2]
b = [2.4, 2.5, 2.83, 3.91, 3, 2.5, 2.5, 2.5, \
     1, 1, 1]
    
#b = np.ones(11)
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
    
params_GLOBAL.update({"F": F, "H": H, "fwhm": fwhm, "tau_ex_r":tau_ex_r, "tau_EEA":tau_EEA, "tau_ex_f":tau_ex_f, "tau_hc":tau_hc})
    
res_ALL_TEST = np.zeros((11,2,len(t_values)))

for i in range(10+1):
    fi = power[i]
    fi = fluences[i]*1e13
    res_ALL_TEST[i,:,:] = ode_resolution_GLOBAL(t_values, fi, i, **params_GLOBAL)

fig, ax = plt.subplots(3,4)
fig.set_size_inches(12, 8, forward=False)
ax = ax.flatten()

for s in range(11):
    
    d1, d2 = data_traces[2*s], data_traces[2*s+1]
    #d1, d2 = d1/np.max(d1), d2/np.max(d2)
    
    r1, r2 = res_ALL_TEST[s,0,:], res_ALL_TEST[s,1,:]
    #r1 = r1 / np.max(r1)
    #r1 = r1*0.4
    #r1, r2 = r1/np.max(r1), 0.8*r2/np.max(r2)
    #    gaussian_pulse = gaussian(t_values, 1, 0, fwhm/2.355, 0)

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

fig.delaxes(ax[11])
fig.text(0.77, 0.3, "Testing", fontsize=24)
fig.text(0.77, 0.25, fr"$\tau_{{hc}} = {tau_hc:.0f} \:fs$", fontsize=20)
fig.text(0.77, 0.2, fr"$\tau_{{f}} = {tau_ex_f:.0f} \: fs$", fontsize=20)
fig.text(0.77, 0.15, fr"$\tau_{{EEA}} = {tau_EEA/1e13:.0f} \: fs$", fontsize=20)
fig.text(0.77, 0.1, fr"$\tau_{{r}} = {tau_ex_r:.0f} \: fs$", fontsize=20)

#test = np.exp(-t/800)+ np.exp(-t/15000)

#ax[s].plot(t, test/np.max(test), 'b*')    
fig.tight_layout()

#%% Define Global Fit Paramters

from lmfit import Parameters, minimize, report_fit

tau_ex_r = 20000
tau_EEA = 800 * 1e13
tau_ex_f = 100
tau_hc = 254 #258
fwhm = 80
F = 1
H = 1
t_0 = 0
a = [.9, .92, .94, .99, .93, .90, 0.88, .95, \
     .05, .12, .2]
b = [2.4, 2.5, 2.83, 3.91, 3, 2.5, 2.5, 2.5, \
     1, 1, 1]

#no_fit = [8, 9, 10]
fit_params = Parameters()
for i in range(10+1):
    fit_params.add(str("a"+str(i)), value=a[i], min=0.05, max=30, vary=True)
    fit_params.add(str("b"+str(i)), value=b[i], min=0.1, max=50, vary=True)
    # if i in [9, 10]:
    #     fit_params.add(str("a"+str(i)), value=a[i], min=0.05, max=12, vary=True)
    #     fit_params.add(str("b"+str(i)), value=b[i], min=0.1, max=50, vary=True)
#for i in no_fit:
#    fit_params.add(str("a"+str(i)), value=a[i], min=0.2, max=5, vary=False)
#    fit_params.add(str("b"+str(i)), value=b[i], min=0.2, max=10, vary=False)
    
fit_params.add("F", value=F, min=0.01, max=10, vary=False)
fit_params.add("H", value=H, min=0.01, max=10, vary=False)
fit_params.add("fwhm", value=fwhm, min=10, max=120, vary=False)

fit_params.add("tau_ex_r", value=tau_ex_r, min=500, max=35000, vary=False)

fit_params.add("tau_EEA", value=tau_EEA, min=200*1e13, max=15000*1e13, vary=True)
fit_params.add("tau_ex_f", value=tau_ex_f, min=50, max=2000, vary=True)
#fit_params.add("tau_ex_f", value=tau_ex_f, min=5*1e13, max=2000*1e13, vary=True)
fit_params.add("tau_hc", value=tau_hc, min=50, max=1000, vary=True)

fit_params

#%% Do the Global fitting!

data = data_traces

# for s in range(10+1):
#     for f in range(2):
#         data[s,f,:] = data[s,f,:] #/np.max(data[s,f,:]) 

out = minimize(objective_GLOBAL, fit_params, args=(delay_axes, data_traces))
report_fit(out)

#%% Global Fitting: Plot the Results!

import time
ym = time.strftime("%Y%m%d")

%matplotlib inline

save_figure = True
figure_file_name = 'MODEL_Fig4'
image_format = 'svg'

res_ALL = np.zeros((11,2,len(t_values)))

for i in range(0,11):

    fi = power[i]
    fi = fluences[i]*1e13
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
    ax[s].plot(delay_axes[s], d1, 'ko', alpha = 0.4)
    ax[s].plot(delay_axes[s], d2, 'ro', alpha = 0.4)
    ax[s].plot(t_values, r1, 'orange', linewidth = 2.5)
    ax[s].plot(t_values, r2, 'purple', linewidth = 2.5)
    #ax[s].axvline(0, linestyle = 'dashed', color='green')
    ax[s].set_xlabel('Delay, ps')
    ax[s].set_ylabel('Norm. Int., a.u.')
    ax[s].set_xticks(np.arange(-1000,3100,500))
    xticks = np.arange(-1000, 3500, 500)
    xtick_labels = [str(tick/1000) if i % 2 == 0 else "" for i, tick in enumerate(xticks)]
    ax[s].set_xticklabels(xtick_labels)
    for label in ax[s].xaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
    ax[s].set_xlim([-500,3000])
    ax[s].set_ylim(-.1,1.1)

    #if s > 7:
     #   ax[s].set_ylim(-.1, 3.5)
    #ax[s].plot(t_values_offset-t_0[s], gaussian_pulse, linestyle = 'dashed', color='green')

fig.delaxes(ax[11])
fig.text(0.77, 0.25, fr"$\tau_{{hc}} = {out.params['tau_hc'].value:.0f} \:fs$", fontsize=20)
fig.text(0.77, 0.2, fr"$\tau_{{f}} = {out.params['tau_ex_f'].value:.0f} \: fs$", fontsize=20)
fig.text(0.77, 0.15, fr"$\tau_{{EEA}} = {out.params['tau_EEA'].value/1e13:.0f} \: fs$", fontsize=20)
fig.text(0.77, 0.1, fr"$\tau_{{EEA}} = {1/(out.params['tau_EEA'].value/1e13):.2f} \: fs$", fontsize=18)
fig.text(0.77, 0.05, fr"$\tau_{{r}} = {out.params['tau_ex_r'].value:.0f} \: fs$", fontsize=20)

plt.rcParams['svg.fonttype'] = 'none'

fig.tight_layout()
plt.show()

if save_figure is True:
    fig.savefig(figure_file_name + '.'+ image_format, bbox_inches='tight', format=image_format) 
    
#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Example data (replace with actual data)
fluence = n  # Excitation densities

# Convert to decay rates 
decay_rates = 1 / popt_combined[:,1] # A, tau1, B, tau2, C, tau_rise, D

# Define power-law function
def power_law(n, gamma0, beta, alpha):
    return gamma0 + beta * n**alpha

# Fit to the data
init_guess = [popt_combined[0,1], 1, .5] 
popt, pcov = curve_fit(power_law, fluence, decay_rates, p0=init_guess)
gamma0_fit, beta_fit, alpha_fit = popt

# Plot results
plt.figure(figsize=(6,4))
plt.scatter(fluence, decay_rates, label="Data", color="red")
#plt.errorbar(fluence, decay_rates, yerr=1/perr_combined[:,5], color = 'red')
plt.plot(fluence, power_law(fluence, *popt), label=f"Fit: α = {alpha_fit:.2f}", linestyle="--", color="blue")
plt.xlabel("Excitation Density (n)")
plt.ylabel("Decay Rate (Γ = 1/τ)")
plt.legend()
plt.grid()
#plt.ylim(0,0.02)
plt.show()

print(f"Fitted parameters: Gamma0 = {gamma0_fit:.3f}")
