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

### ADDITIONAL SUPPORTING ANALYSIS

#%%

# CALCULATING ABSORBED EXCITATION FLUENCE.

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
hbar = 6.58e-16  # eV.s

### Calculate Penetration Depth and Refractive Index
k =  np.sqrt((np.sqrt((real_)**2 + (imag_)**2) - real_) / 2)
n = np.sqrt((np.sqrt((real_)**2 + (imag_)**2) + real_) / 2)
pen_depth = np.sqrt(1-((np.sin(theta))**2/n**2))*(c*hbar/(2*k*energy))
ABS = 1 - np.exp(-thickness/pen_depth)

absorbance_2 = np.asarray([(1 - np.exp(-z*aa)) for aa in A_s_2])
alpha = (2*energy)/(c*hbar) * np.sqrt((np.sqrt((real_)**2 + (imag_)**2) - real_) / 2) #1/m
A_s = 4*n*np.cos(theta)/(n**2 + k**2+2*n*np.cos(theta)+(np.cos(theta))**2)

penetration_depth = np.asarray(1/alpha)
A_s_2 = 1/pen_depth
pen_depth_2 = 1/A_s

z = 2e-9
absorbance = alpha * z
absorbance_2 = np.asarray([(1 - np.exp(-z*aa)) for aa in A_s_2])

fig, ax1 = plt.subplots()

plt.plot(energy, imag_, color = 'red', label = 'Imaginary')
plt.plot(energy, real_, color = 'blue', label = 'Real')
plt.ylabel('Int', color = 'purple')
plt.xlabel('Energy (eV)')
plt.xlim([1.3,2.4])
plt.ylim([-20,90])
plt.title('Calculating Absorption')
plt.legend(frameon=False)

ax2 = ax1.twinx()
ax2.plot(energy, 100*ABS, color = 'pink', linestyle = 'dashed', label = 'Calc. Absorbance')
plt.ylabel('% Abs', color = 'pink')
plt.legend(frameon=False)

#plt.plot(energy, 100*absorbance_2, color = 'black', linestyle = 'dashed', label = 'Calc. Absorbance')
#plt.plot(energy, 100*absorbance, color = 'grey', linestyle = 'dashed', label = 'Absorbance')
#plt.plot(energy, 100*A_s, color = 'pink', linestyle = 'dashed', label = 'A_s')
#plt.plot(energy, 100*A_s_2, color = 'purple', linestyle = 'dashed', label = 'A_s')
#plt.plot(energy, Abs_Huber,  color = 'green', linestyle = 'dashed', label = 'Abs. Huber')

#%%

lam = 800
average_power = 120 #mW
fwhm = 0.110 #mm #110
rep_rate = 475000 # 475000

absorbance_to_use = ABS

laserspectrum = np.loadtxt('910nm_opa_pumpspectrum_18042024_beforechamber.txt',skiprows=1)
wl = laserspectrum[:,0]
spectrum = laserspectrum[:,1]
spectrum = spectrum - np.min(spectrum)
spectrum = spectrum/np.max(spectrum)

#wl_i = np.abs(wl-915).argmin()
#abs_spec = np.zeros((laserspectrum.shape[0]))

lam_from_energy = np.asarray([1239.8/e for e in energy])
lam_from_energy = lam_from_energy[::-1]

#absorbance_to_use = absorbance_2
#absorbance_to_use = 0.5*A_s
#absorbance_to_use = Abs_Huber/100

absorbance_test = absorbance_to_use[::-1]
abs_spec = np.interp(wl , lam_from_energy, absorbance_test)

#abs_spec[wl_i-200:wl_i+200] = 0.25

h = 6.6261E-34
c = 299792458

rep_rate = 475000 # 475000
beam_rad = 0.1*fwhm*1.699*0.5 #Get 1/e^2 beam radius, cm (from FWHM)
beam_rad_2 = 0.1*fwhm*1.699*0.5*(1/np.cos(theta)) #Get 1/e^2 beam radius, cm (from FWHM)
spot_size = np.pi*beam_rad*beam_rad_2/2 #cm^2

#print(beam_rad, beam_rad_2)

pulse_energy = (average_power/1000)/rep_rate #Joules
energy_density = 1000*pulse_energy/(spot_size) #mJ/cm^2

ph_E = h*c/(lam*1E-9)

num_ph = pulse_energy/ph_E

spec_integration = np.sum(spectrum)
spectrum_ph = num_ph*spectrum/spec_integration

neh_abs = abs_spec*spectrum_ph

exc_density = (sum(neh_abs)/spot_size)/1e13

###
fig, ax1 = plt.subplots()

plt.plot(wl,spectrum_ph/np.max(spectrum_ph), color = 'red',label = 'Norm. Laser Spectrum')
plt.legend(frameon=False)
plt.ylabel('Int., arb. un.')

ax2 = ax1.twinx()
ax2.plot(wl,abs_spec, label = 'Calc. Absorbance, %', color = 'black')
#plt.plot(wl,abs_spec*spectrum_ph/spectrum_ph.max(), color = 'pink', linestyle = 'dashed', label = 'Effective Absorbed')
plt.xlim([640,1000])
plt.xlabel('Wavelength, nm')
plt.ylabel('Int., arb. un.')
plt.title('Estimated Carrier Density')
plt.axvline(915, linestyle = 'dashed')
plt.legend(frameon=False)
print('Estimated Carrier Density for ' + str(average_power) + ' mW, ' + str(round(energy_density, 2)) + ' mJ/cm2 : ' + str(round(exc_density, 2)) + ' E13/cm2')
plt.show()