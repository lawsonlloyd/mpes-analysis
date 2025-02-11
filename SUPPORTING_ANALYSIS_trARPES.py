# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 10:43:43 2025

@author: lloyd
"""

import scipy.constants as scpc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
from skimage.draw import disk
from scipy.optimize import curve_fit

#%%

### ADDITIONAL SUPPORTING ANALYSIS ###

%matplotlib inline

save_figure = True
figure_file_name = 'pump_spectra'

#data = np.loadtxt("920nm_51fsFROG_.txt", skiprows = 1)

#ds910 = "910nm_opa_pumpspectrum_18042024_beforechamber.txt"
ds915 = "OPA_pump_spectrum_910nm_DATA_2.txt"
ds800 = "800nm_opa_pumpspectrum_070624.txt"
ds700 = "700nm_opa_pumpspectrum_24042024.txt"
ds680 = "680nm_opa_pumpspectrum_24042024.txt"
ds640 = "640nm_opa_pumpspectrum_24042024.txt"
ds400 = "400nm_opa_pumpspectrum_2904_directonSpectrometer_2.txt"

data_string = [ds915, ds800, ds700, ds640, ds400]
pump_WLs = [915, 800, 700, 640, 400]

pump_colors = ['black', 'brown', 'red', 'orange', 'blue']
#data = np.loadtxt("760nm.txt", skiprows = 1)
fig = plt.figure()
plt.gcf().set_dpi(300)

for p in np.arange(0,len(pump_WLs)):
    data = np.loadtxt(data_string[p], skiprows = 1)

    wl = data[:,0]
    lam = np.abs(wl - pump_WLs[p]).argmin()
    
    amp = data[:,1] - np.mean(data[-300:-1,1])
    amp = amp/np.max(amp)
    
    if p == 0:
        amp = amp[lam-300:lam+350]
        wl = wl[lam-300:lam+350]
    else:
        amp = amp[lam-250:lam+350]
        wl = wl[lam-250:lam+350]

    #amp = amp - np.min(amp)
    
    plt.plot(wl, amp, color = pump_colors[p])
    plt.title('Pump Spectra', fontsize = 24)
    #plt.xlim(870, 970)
    #plt.xlim(710, 810)
    plt.xlabel('$\lambda$, nm', fontsize = 20)
    plt.ylabel('Int., arb. units.', fontsize=22)
    plt.tick_params(axis='both', labelsize=18)
    #plt.axvline(915, linestyle = 'dashed')
    
plt.xticks(np.arange(300,1000,50))
for label in plt.gca().xaxis.get_ticklabels()[1::2]:
        label.set_visible(False)  
plt.xlim([350,1000])

if save_figure is True:
    fig.savefig((figure_file_name +'.svg'), format='svg')
    
#%% Power Calilbration

percentage = np.asarray([28.84, 19.95, 15.85, 12.58, 10, 7.94, 3.898, 1.99, 1.58, 0.9])
power = 1.05*np.asarray([153, 111, 91, 66, 47, 32, 15, 10, 8, 5])

fig = plt.figure()
plt.plot(percentage, power, 'o')

#%%

%matplotlib inline

# CALCULATING ABSORBED EXCITATION FLUENCE.

thickness =  1 * 0.57e-9 #2e-9 # m, thickness of first layer (?)
AOI = 40

### Load Dielectric Constant

real_epsilon =  np.loadtxt('Real_epsilon.txt')
imag_epsilon = np.loadtxt('Imaginary_epsilon.txt')

real_epsilon = np.transpose(real_epsilon)
imag_epsilon = np.transpose(imag_epsilon)
real_ = real_epsilon[1,:]
imag_ = imag_epsilon[1,:]

energy = np.asarray(imag_epsilon[0,:])

theta = np.pi*AOI/180 #deg
c = 299792458 # m/s
hbar = 6.58e-16  # eV.s

### Calculate the Complex Refractive Index
n = np.sqrt((np.sqrt((real_)**2 + (imag_)**2) + real_) / 2)
k = np.sqrt( (np.sqrt((real_)**2 + (imag_)**2) - real_) / 2)

### Calculate the Penetration Depth and Absorption
pen_depth = np.sqrt(1-((np.sin(theta))**2/n**2)) * (c*hbar/(2*k*energy))
ABS = 1 - np.exp(-thickness/pen_depth)

A_ = 1 - np.exp(-thickness * energy*imag_/(hbar*c))

#absorbance_2 = np.asarray([(1 - np.exp(-z*aa)) for aa in A_s_2])
alpha = (2*energy)/(c*hbar) * np.sqrt((np.sqrt((real_)**2 + (imag_)**2) - real_) / 2) #1/m
A_s = 4*n*np.cos(theta)/(n**2 + k**2+2*n*np.cos(theta)+(np.cos(theta))**2)

### Make a Few Plots
fig, ax1 = plt.subplots()

plt.plot(energy, imag_, color = 'red', label = 'Imaginary')
plt.plot(energy, real_, color = 'blue', label = 'Real')
plt.ylabel('Int')
plt.xlabel('Energy (eV)')
plt.xlim([1.25,2.25])
plt.ylim([-30,90])
plt.title('Calculating Absorption')
plt.legend(frameon=False)

ax2 = ax1.twinx()
ax2.plot(energy, 100*ABS, color = 'pink', linestyle = 'dashed', label = 'Calc. Absorbance')
plt.ylabel('% Abs')
#plt.plot(energy, Abs_Huber,  color = 'purple', linestyle = 'dashed', label = 'Abs. Huber')
plt.ylim([0,10])

plt.legend(frameon=False)

#%%

### Define experimental pump parameters
lam = 915
average_power = 116.6 #mW
fwhm = 0.110 #mm #110
pump_pol = 's'
rep_rate = 500000 # 475000

### Load Pump Spectrum
if lam == 800:
    laserspectrum = np.loadtxt('800nm_opa_pumpspectrum_070624.txt',skiprows=1)
elif lam == 915:
    laserspectrum = np.loadtxt('OPA_pump_spectrum_910nm_DATA_2.txt',skiprows=1)
elif lam == 700:
    laserspectrum = np.loadtxt('700nm_opa_pumpspectrum_24042024.txt',skiprows=1)

### Account for s- and p-pol Reflection
e_i = np.abs(energy - 4.1357e-15*c*1e9/lam).argmin()
n1 = 1
n2 = n[e_i]
Rs = ( (n1* np.cos(theta) - n2 * np.sqrt(1 - ((n1/n2)*np.sin(theta))**2 )) / (n1* np.cos(theta) + n2 * np.sqrt(1 - ((n1/n2)*np.sin(theta))**2 )) )**2
Rp = ( (n1*np.sqrt(1- ((n1/n2)*np.sin(theta))**2 ) - n2*np.cos(theta)) / (n1*np.sqrt(1- ((n1/n2)*np.sin(theta))**2 ) + n2*np.cos(theta)) )**2
Ts = 1 - Rs
Tp = 1 - Rp

if pump_pol == 's':
    average_power = average_power*Ts
elif pump_pol == 'p':
    average_power = average_power*Tp    
    
### Provdie the (Calculated) Absorption Spectrum    
absorbance_to_use = ABS

wl = laserspectrum[:,0]    
spectrum = laserspectrum[:,1]
spectrum = spectrum - np.min(spectrum)
spectrum = spectrum/np.max(spectrum)
wl_i = np.abs(wl - lam).argmin()

#wl_i = np.abs(wl-915).argmin()
#abs_spec = np.zeros((laserspectrum.shape[0]))

lam_from_energy = np.asarray([1239.8/e for e in energy])
lam_from_energy = lam_from_energy[::-1]

absorbance_test = absorbance_to_use[::-1]
abs_spec = np.interp(wl , lam_from_energy, absorbance_test)

### Calculate the beam spot size and the excitation fluence
beam_rad = 0.1*fwhm*1.699*0.5 #Get 1/e^2 beam radius, cm (from FWHM)
beam_rad_2 = 0.1*fwhm*1.699*0.5*(1/np.cos(theta)) #Get 1/e^2 beam radius, cm (from FWHM)
spot_size = np.pi*beam_rad*beam_rad_2/2 #cm^2
pulse_energy = (average_power/1000)/rep_rate #Joules
energy_density = 1000*pulse_energy/(spot_size) #mJ/cm^2

h, c = 6.6261E-34, 299792458 # constants

ph_E = h*c/(lam*1E-9)
num_ph = pulse_energy/ph_E

spec_integration = np.sum(spectrum)
spectrum_ph = num_ph*spectrum/spec_integration

neh_abs = abs_spec*spectrum_ph
exc_density = (sum(neh_abs)/spot_size)/1e13

##############
### Make Plots
fig, ax1 = plt.subplots()
plt.plot(wl, abs_spec, label = 'Calc. Absorbance, %', color = 'black')
plt.ylim([0, 0.12])
plt.legend(frameon=False)
plt.ylabel('Abs.')
plt.xlabel('Wavelength, nm')

ax2 = ax1.twinx()
ax2.plot(wl[wl_i-400:wl_i+400],spectrum_ph[wl_i-400:wl_i+400]/np.max(spectrum_ph), color = 'red',label = 'Norm. Laser Spectrum')
#plt.plot(wl,abs_spec*spectrum_ph/spectrum_ph.max(), color = 'pink', linestyle = 'dashed', label = 'Effective Absorbed')
plt.xlim([600, 1000])
plt.ylim([0,1.05])
plt.title('Estimated Carrier Density')
plt.axvline(lam, linestyle = 'dashed', color = 'pink')
plt.legend(frameon=False)
print('Est. Carrier Density for ' + str(round(average_power,1)) +  ' mW, ' + str(round(energy_density, 2)) + ' mJ/cm2 (' + str(lam) + 'nm) ' ': ' + str(round(exc_density, 2)) + ' E13/cm2')
plt.show()

aB = 1 #bohr radius, nm
rx = 1e7*(1/np.sqrt(np.pi*aB**2*0.3*exc_density*1e6*1e6)) ; print(rx)

#%%

%matplotlib inline

# CALCULATING ABSORBED EXCITATION FLUENCE: ALTERNATIVE METHOD

thickness = 0.518e-9 #2e-9 # m, thickness of first layer (?)
AOI = 40

### Load Dielectric Constant

real_epsilon =  np.loadtxt('Real_epsilon.txt')
imag_epsilon = np.loadtxt('Imaginary_epsilon.txt')
Abs_Huber_load = np.loadtxt('Abs_Huber2.txt')

real_epsilon = np.transpose(real_epsilon)
imag_epsilon = np.transpose(imag_epsilon)
real_ = real_epsilon[1,:]
imag_ = imag_epsilon[1,:]

energy = np.asarray(imag_epsilon[0,:])

Abs_Huber_load = np.transpose(Abs_Huber_load)
Abs_Huber = np.interp(energy, Abs_Huber_load[0,:], Abs_Huber_load[1,:])

theta = np.pi*AOI/180 #deg
c = 299792458 # m/s
hbar = 6.582e-16  # eV.s
h = 4.1357e-15 # eV.s

### Calculate Penetration Depth and Refractive Index
n = np.sqrt( (np.sqrt((real_)**2 + (imag_)**2) + real_) / 2)
k =  np.sqrt((np.sqrt((real_)**2 + (imag_)**2) - real_) / 2)

#nu_pump = energy/(4.1357e-15)
alpha_0 = 1/( np.sqrt(1-((np.sin(theta))**2/n**2)) * (c*hbar/(2*k*energy)))
alpha_1 = (2*energy)/(c*hbar) * k

alpha_2 = 1/ (np.sqrt(real_ + imag_) * c * (1 / (2*np.pi*imag_*energy/h)))


#A_s = 4*n*np.cos(theta)/(n**2 + k**2+2*n*np.cos(theta)+(np.cos(theta))**2)
z = .518e-9
A_s_2 = 1/pen_depth

absorbance = alpha * z
absorbance_2 = np.asarray([(1 - np.exp(-z*aa)) for aa in A_s_2])



penetration_depth = np.asarray(1/alpha)
#A_s_2 = 1/pen_depth
#pen_depth_2 = 1/A_s

z = 15e-9
absorbance = alpha * z
absorbance = 1 - np.exp(-z*alpha)

z = np.linspace(220,0,176)
nn =  np.exp(-alpha*z) * np.exp(-z/15)
fig, ax1 = plt.subplots()

plt.plot(energy, imag_, color = 'red', label = 'Imaginary')
plt.plot(energy, real_, color = 'blue', label = 'Real')
plt.ylabel('Int', color = 'purple')
plt.xlabel('Energy (eV)')
plt.xlim([1,2.25])
plt.ylim([-20,90])
plt.title('Calculating Absorption')
plt.legend(frameon=False)

ax2 = ax1.twinx()
ax2.plot(energy, 100*absorbance, color = 'pink', linestyle = 'dashed', label = 'Calc. Absorbance')
plt.ylabel('% Abs', color = 'pink')
plt.legend(frameon=False)

#plt.plot(energy, 100*absorbance_2, color = 'black', linestyle = 'dashed', label = 'Calc. Absorbance')
#plt.plot(energy, 100*absorbance, color = 'grey', linestyle = 'dashed', label = 'Absorbance')
#plt.plot(energy, 100*A_s, color = 'pink', linestyle = 'dashed', label = 'A_s')
#plt.plot(energy, 100*A_s_2, color = 'purple', linestyle = 'dashed', label = 'A_s')
#plt.plot(energy, Abs_Huber,  color = 'green', linestyle = 'dashed', label = 'Abs. Huber')

#%%

