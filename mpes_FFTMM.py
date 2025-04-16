#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 08:22:54 2025

@author: lawsonlloyd
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as col
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
from skimage.draw import disk
from scipy.optimize import curve_fit
from scipy.optimize import curve_fit
from scipy import signal
from scipy.fft import fft, fftshift
#from lmfit import Parameters, minimize, report_fit
from obspy.imaging.cm import viridis_white
import xarray as xr
from math import nan

from Loader import DataLoader
from Main import main
from Manager import DataHandler, FigureHandler, PlotHandler, ValueHandler, SliderManager, EventHandler, CheckButtonManager, ClickButtonManager

import mpes
from mpes import cmap_LTL

#%% Specifiy filename of h5 file in your path.
# Include manual energy and time delay offsets for the axes, if required.

data_path = 'path_to_your_data'
filename = 'your_file_name.h5'

#data_path = 'R:\Lawson\Data\metis'
data_path = '/Users/lawsonlloyd/Desktop/Data/metis/'

filename, offsets = 'Scan162_RT_120x120x115x50_binned.h5', [0.8467, -120]
filename, offsets = 'Scan163_120K_120x120x115x75_binned.h5',  [0.6369, -132]
#filename, offsets = 'Scan188_120K_120x120x115x77_binned.h5', [0.5660, -110]

#%% Load the data and axes information

data_loader = DataLoader(data_path + '//' + filename)
value_manager =  ValueHandler()

#I, ax_kx, ax_ky, ax_E, ax_delay = data_loader.load()
#data_handler = DataHandler(value_manager, I, ax_kx, ax_ky, ax_E, ax_delay, *offsets)

I = data_loader.load()
#data_handler = DataHandler(value_manager, I, ax_kx, ax_ky, ax_E, ax_delay, *offsets)

I = I.assign_coords(E=(I.E-offsets[0]))
I = I.assign_coords(delay=(I.delay-offsets[1]))
if filename == 'Scan163_120K_120x120x115x75_binned.h5':    
    I = I.assign_coords(ky=(I.ky-0.075))

I_res = I/np.max(I)

#%% Functions for Fourier Transform Analysis

def window_MM(kspace_frame, kx, ky, kx_int, ky_int, win_type, alpha):    
    
    ### Deconvolve k-space momentum broadening, Gaussian with FWHM 0.063A-1
    fwhm = 0.063
    fwhm_pixel = fwhm/dkx
    sigma = fwhm_pixel/2.355
    # gaussian_kx = signal.gaussian(len(ax_kx), std = sigma)
    # gaussian_kx = gaussian_kx/np.max(gaussian_kx)
    # gaussian_ky = signal.gaussian(len(ax_ky), std = sigma)
    # gaussian_ky = gaussian_ky/np.max(gaussian_ky)
    
    # gaussian_kxky = np.outer(gaussian_kx, gaussian_ky)
    # gaussian_kxky = gaussian_kxky/np.max(gaussian_kxky)
    # gaussian_kxky = np.outer(gaussian_kx, gaussian_ky)
    #kx_cut_deconv = signal.deconvolve(kx_cut, gaussian_kx)
    
    ### Symmetrize Data
    #kspace_frame_sym = xr.DataArray(np.zeros(kspace_frame.shape), coords = {"ky": ax_kx, "kx": ax_ky})
    kspace_frame_ = kspace_frame.values
    kspace_frame_rev = kspace_frame_[:,::-1]
    
    kspace_frame_sym = kspace_frame_ + kspace_frame_rev
    kspace_frame_sym =  kspace_frame_sym/2
    kspace_frame_sym = xr.DataArray(kspace_frame_sym, coords = {"ky": ax_ky, "kx": ax_kx})

    ### Generate the Windows to Apodize the signal
    k_x_i = np.abs(ax_kx.values-(kx-kx_int/2)).argmin()
    k_x_f = np.abs(ax_kx.values-(kx+kx_int/2)).argmin()
    k_y_i = np.abs(ax_kx.values-(ky-ky_int/2)).argmin()
    k_y_f = np.abs(ax_kx.values-(ky+ky_int/2)).argmin()
    #I_res.indexes["kx"].get_indexer([kx-kx_int/2], method = 'nearest')[0]
    
    # kx Axis
    win_1_tuk = np.zeros(kspace_frame.shape[0])
    win_1_box = np.zeros(kspace_frame.shape[0])

    tuk_1 = signal.windows.tukey(k_x_f-k_x_i, alpha = 0.1)
    box_1 = signal.windows.boxcar(k_x_f-k_x_i)
    win_1_tuk[k_x_i:k_x_f] = tuk_1
    win_1_box[k_x_i:k_x_f] = box_1

    # ky Axis
    win_2_tuk = np.zeros(kspace_frame.shape[1])
    win_2_box = np.zeros(kspace_frame.shape[1])

    tuk_2 = signal.windows.tukey(k_y_f-k_y_i, alpha = alpha)
    box_2 = signal.windows.boxcar(k_y_f-k_y_i)
    win_2_tuk[k_y_i:k_y_f] = tuk_2
    win_2_box[k_y_i:k_y_f] = box_2

    # Combine to (kx, ky) Window
    window_2D_tukey = np.outer(win_2_tuk, win_1_tuk) # 2D tukey
    window_2D_box = np.outer(win_2_box, win_1_box) # 2D Square Window
    window_tukey_box = np.outer(win_2_tuk, win_1_box) # Tukey + Box

    if win_type == 'gaussian':
        win_1_gauss = np.zeros(kspace_frame.shape[0])
        gaus_1 = signal.windows.gaussian(k_x_f-k_x_i, alpha)
        win_1_gauss[k_x_i:k_x_f] = gaus_1
        window_2D_gaussian = np.outer(win_1_gauss, win_2_box)
        kspace_window = window_2D_gaussian
        
    # 2D Tukey    
    if win_type == 1:
        kspace_window = xr.DataArray(window_2D_tukey, coords = {"ky": ax_ky, "kx": ax_kx})

    if win_type == 'square':
        kspace_window = window_2D_box

    if win_type == 'tukey, square':
        kspace_window = xr.DataArray(window_tukey_box, coords = {"ky": ax_ky, "kx": ax_kx})

    kspace_frame_sym_win = kspace_frame_sym*kspace_window
    kspace_frame_win = kspace_frame*(kspace_window)
        
    return kspace_frame_sym, kspace_frame_win, kspace_frame_sym_win, kspace_window/np.max(kspace_window)

def FFT_MM(MM_frame, zeropad):
    
    I_MM = MM_frame

    ##############################
    
    # Define real-space axis
    k_step = dkx
    #k_step_y = k-step
    #k_length_y = len(ax_ky)
    zplength = zeropad #5*k_length+1
    max_r = 1/(2*k_step)

    #r_axis = np.linspace(-max_r, max_r, num = k_length)
    #r_axis = r_axis/(10)

    # Shuo Method ?
    N = 1 #(zplength)Fs
    Fs = 1/((2*np.max(ax_kx.values))/len(ax_kx.values))
    r_axis = np.arange(0,zplength)*Fs/1
    r_axis = r_axis - (np.max(r_axis)/2)
    r_axis = r_axis/(1*zplength)
   
    # Use np to define
    r_axis = np.linspace(-max_r, max_r, num = zplength)
    r_axis = 2*np.pi * np.fft.fftshift(np.fft.fftfreq(zplength, d=dkx)) #Include 2pi factor
    r_axis = 0.1 * r_axis # Covnert to nm from Angstrom

    ### Do the FFT operations to get --> |Psi(x,y)|^2 ###
    I_MM = np.abs(I_MM)/np.max(I_MM)
    root_I_MM = np.sqrt(I_MM)
    fft_frame = np.fft.fft2(root_I_MM, [zplength, zplength])
    fft_frame = np.fft.fftshift(fft_frame, axes = (0,1))

    fft_frame = np.abs(fft_frame) 
    I_xy = np.square(np.abs(fft_frame)) #frame squared
    I_xy = I_xy/np.max(I_xy)
    
    ### Take x and y cuts and extract bohr radius
    ky_cut = I_MM[:,int(len(I_MM[0])/2)-1-4:int(len(I_MM[0])/2)-1+4].sum(axis=1)
    ky_cut = ky_cut/np.max(ky_cut)
    kx_cut = I_MM[int(len(I_MM[0])/2)-1-4:int(len(I_MM[0])/2)-1+4,:].sum(axis=0)
    kx_cut = kx_cut/np.max(kx_cut)

    ### real space Psi*^2 cut
    y_cut = I_xy[:,int(zplength/2)-1]
    x_cut = I_xy[int(zplength/2)-1,:]
    x_cut = x_cut/np.max(x_cut)
    y_cut = y_cut/np.max(y_cut)

    x_brad = (np.abs(x_cut[int(zplength/2)-10:int(zplength/2)+200] - 0.5)).argmin()
    y_brad = (np.abs(y_cut[int(zplength/2)-10:] - 0.5)).argmin()
    x_brad = int(zplength/2)-10 + x_brad
    y_brad = int(zplength/2)-10 + y_brad
    x_brad = r_axis[x_brad]
    y_brad = r_axis[y_brad]
    
    ### real space Psi cut : |r*Psi(r)|^2
    r2_cut_y = fft_frame[:,int(zplength/2)-1] 
    r2_cut_y = np.square(np.abs(r2_cut_y*r_axis)) 
    r2_cut_y = r2_cut_y/np.max(r2_cut_y)
    
    r2_cut_x = fft_frame[int(zplength/2)-1,:]
    r2_cut_x = np.square(np.abs(r2_cut_x[0:1090]*r_axis[0:1090]))
    r2_cut_x = r2_cut_x/np.max(r2_cut_x)

    rdist_brad_x = np.argmax(r2_cut_x[int(zplength/2)-10:int(zplength/2)+90])
    rdist_brad_y = np.argmax(r2_cut_y[int(zplength/2)-10:int(zplength/2)+150])

    rdist_brad_x = r_axis[int(zplength/2)-10 + rdist_brad_x]
    rdist_brad_y = r_axis[int(zplength/2)-10 + rdist_brad_y]

    return r_axis, I_xy, x_cut, y_cut, rdist_brad_x, rdist_brad_y, x_brad, y_brad
    
def lorentzian(x, amp_1, mean_1, stddev_1, offset):
    
    b = (x - mean_1)/(stddev_1/2)
    l1 = amp_1/(1+b**2) + offset
    
    return l1
    
def gaussian(x, amp_1, mean_1, stddev_1, offset):
    
    g1 = amp_1 * np.exp(-0.5*((x - mean_1) / stddev_1)**2)+offset
    
    return g1

#%% Do the 2D FFT of MM to Extract Real-Space Information

%matplotlib inline

a, b = 3.508, 4.763 # CrSBr values
X, Y = np.pi/a, np.pi/b
x, y = -2*X, 0*Y

E, E_int  = 1.35, 0.150 #Energy and total width in eV
#E, E_int  = 1.35, 0.100 #Energy and total width in eV

kx, kx_int = (1*X+dkx), 2.1*X # 1.25*X
kx, kx_int = (1.5*X+dkx), 1.1*X # 1.25*X
kx, kx_int = (0.5*X+dkx), 1.1*X # 1.25*X

ky, ky_int = 0, 0.65
delays, delay_int = 600, 800 

win_type = 1 #0, 1 = 2D Tukey, 2, 3
alpha = 0.5
zeropad = 2048

frame_pos = mpes.get_momentum_map(I_res, E, E_int, delays, delay_int)  # Get Positive Delay MM frame (takes mean over ranges)
frame_neg = mpes.get_momentum_map(I_res, E, E_int, -130, 50) # Get Negative Delay MM frame (takes mean over ranges)
frame_diff = frame_pos - frame_neg

testing = 0
if testing == 1:
    ax_kx, ax_ky = np.linspace(-2,2,120), np.linspace(-2,2,120)
    ax_kx = xr.DataArray(ax_kx, coords = {"kx": ax_kx})
    ax_ky = xr.DataArray(ax_ky, coords = {"ky": ax_ky})

    dkx = (ax_kx.values[1] - ax_kx.values[0])
    g_test = gaussian(ax_kx, *[1, 0, 0.15, 0])
    kspace_frame_test = np.zeros((g_test.shape[0], g_test.shape[0]))
    i, f = round(0.45*len(g_test)), round(0.85*len(g_test))
    kspace_frame_test = np.ones((g_test.shape[0], g_test.shape[0]))
    
    kspace_frame_test[:,i:f] = np.tile(g_test, (f-i,1)).T
    kspace_frame = xr.DataArray(kspace_frame_test, coords = {"ky": ax_ky, "kx": ax_kx})

elif testing == 0:
    kspace_frame = frame_pos/np.max(frame_pos) #Define MM of itnerested for FFT
    ax_kx, ax_ky = I_res.kx, I_res.ky
    dkx = (ax_kx.values[1] - ax_kx.values[0])

background = kspace_frame.loc[{"kx":slice(0.2,1.8), "ky":slice(0.5,0.6)}].mean(dim=("kx","ky"))
background = 0.05
#background = 0.1

#kspace_frame = kspace_frame - background
kspace_frame = kspace_frame/np.max(kspace_frame)

kspace_frame_sym, kspace_frame_win, kspace_frame_sym_win, kspace_window = window_MM(kspace_frame, kx, ky, kx_int, ky_int, win_type, alpha) # Window the MM

MM_frame = kspace_frame_win # Choose which kspace frame to FFT
r_axis, rspace_frame, x_cut, y_cut, rdist_brad_x, rdist_brad_y, x_brad, y_brad = FFT_MM(MM_frame, zeropad) # Do the 2D FFT and extract real-space map and cuts

#%% Setup k-space Deconvolution

# ------------------ Peak Shapes ------------------ #

def lorentzian_(x, x0, gamma, amp, offset):
    return amp / (1 + ((x - x0) / gamma)**2) + offset

def gaussian_(x, x0, sigma, amp, offset):
    return amp * np.exp(-((x - x0)**2) / (2 * sigma**2)) + offset

def gaussian_irf(x, sigma):
    return np.exp(-(x**2) / (2 * sigma**2))

# ------------------ Model and Fit ------------------ #

def convolved_model(x, x0, width, amp, offset, sigma_irf, shape):
    if shape == 'lorentzian':
        peak = lorentzian_(x, x0, width, amp, offset)
    elif shape == 'gaussian':
        peak = gaussian_(x, x0, width, amp, offset)
    else:
        raise ValueError("Shape must be 'lorentzian' or 'gaussian'")
    
    irf = gaussian_irf(x - x0, sigma_irf)
    irf /= np.trapz(irf, x)
    return fftconvolve(peak, irf, mode='same')

def fit_convolved_model(x, y, sigma_irf, p0, bnds, shape):
    def fit_func(x, x0, width, amp, offset):
        return convolved_model(x, x0, width, amp, offset, sigma_irf, shape)
    popt, pcov = curve_fit(fit_func, x, y, p0=p0, bounds = bnds)
    return popt, pcov

#%% Fit to Deconvolve k-space: Choose Peak Shape

# Define MM Map to Fit...
test_frame = kspace_frame_sym #kspace_frame_sym
test_frame = test_frame/np.max(test_frame)

# Choose intrinsic function
shape = 'lorentzian'
#shape = 'gaussian'
sigma_irf = 0.02675 

intrinsic_fit = np.zeros(test_frame.shape)
fitted_model = np.zeros(test_frame.shape)

popts = np.zeros((test_frame.shape[1],4))
fit_errors = np.zeros((test_frame.shape[1],4))

i = 0
for kx_i in ax_kx.loc[{"kx":slice(-2*X-.1, 2*X+.1)}]:
    
#    kx_i = ax_kx.values[i]
    i = np.abs(ax_kx.values - kx_i.values).argmin()
    data_cut = test_frame.loc[{"kx":slice(kx_i-.05,kx_i+.05)}].mean(dim="kx")
    ylim = [-.35, .35]
    initial_guess = (0.0, .1, 0.1, 0.00) #x0, gamma, amp, offset
    
    if kx_i > -0.25 and kx_i < 0.2:
        ylim = [-.3, .3]
        initial_guess = (0.0, .07, 0.1, 0.005) #x0, gamma, amp, offset
        initial_guess = (0.0, .15, 0.1, 0.005) #x0, gamma, amp, offset

    bnds = [ [-.1, 0, 0, 0], [0.2, 1, .1, .01] ] #x0, gamma, amp, offset
    popt, pcov = fit_convolved_model(ax_ky.loc[{"ky":slice(ylim[0],ylim[1])}].values, data_cut.loc[{"ky":slice(ylim[0],ylim[1])}], sigma_irf, initial_guess, bnds, shape)
    
    #perr = np.sqrt(np.diag(pcov))
    #fit_errors[i,:] = perr
    popts[i,:] = popt
    popt[0] = popt[0] - 0.05
    #popt[3] = 0
    
    # Extract fitted intrinsic and convolved model
    if shape == 'gaussian':
        intrinsic_fit[i,:] = gaussian_(ax_ky.values, *popt)
    else:
        intrinsic_fit[i,:] = lorentzian_(ax_ky.values, *popt)

    fitted_model[i,:] = convolved_model(ax_ky.values, *popt, sigma_irf, shape=shape)

intrinsic_fit = intrinsic_fit.T / np.max(intrinsic_fit) 
fitted_model = fitted_model.T / np.max(fitted_model)

intrinsic_fit = xr.DataArray(intrinsic_fit, coords = {"ky": ax_ky, "kx": ax_kx})
fitted_model = xr.DataArray(fitted_model, coords = {"ky": ax_ky, "kx": ax_kx})

#%% Re-process Using Deonvolved Data

kspace_frame_fit = xr.DataArray(intrinsic_fit, coords = {"ky": ax_ky, "kx": ax_kx})
kspace_frame_sym_fit, kspace_frame_win_fit, kspace_frame_sym_win_fit, kspace_window = window_MM(kspace_frame_fit, kx, ky, kx_int, ky_int, win_type, alpha) # Window the MM

MM_frame_fit = kspace_frame_win_fit # Choose which kspace frame to FFT
r_axis, rspace_frame, x_cut, y_cut, rdist_brad_x, rdist_brad_y, x_brad, y_brad = FFT_MM(MM_frame_fit, zeropad) # Do the 2D FFT and extract real-space map and cuts

#%% Plot Results of Deconvolution
hfont = {'fontname':'Helvetica'}

save_figure = True
figure_file_name = 'deconvolve_lor' 
image_format = 'svg'

fit_difference = test_frame/np.max(test_frame) - fitted_model/np.max(fitted_model)
fit_difference = fit_difference.loc[{"ky":slice(-.75,.75)}]
difference_scale = np.max(np.abs(fit_difference))
#fit_difference = fit_difference/np.max(np.abs(fit_difference.loc[{"kx":slice(-1.8,2*X)}]))

fig, ax = plt.subplots(4,2, sharey = False)
fig.set_size_inches(8,11)
plt.gcf().set_dpi(300)
ax = ax.flatten()

extent = [ax_kx[0],ax_kx[-1],ax_ky[0],ax_ky[-1]]

im0 = ax[0].imshow(kspace_frame, cmap = cmap_LTL, origin = 'lower', extent = extent)
im1 = ax[1].imshow(test_frame, cmap = cmap_LTL, origin = 'lower', extent = extent)
#ax[0].imshow(fit_difference, vmin = -1, vmax = 1, cmap = 'seismic_r', origin = 'lower', extent = [-2,2,-2,2])
im2 = ax[2].imshow(fitted_model, cmap = cmap_LTL, vmin=0, vmax = 1, origin = 'lower', extent = extent)
#ax[2].imshow(intrinsic_fit.T, cmap = cmap_LTL, origin = 'lower', extent = [-2,2,-2,2])
im3 = ax[3].imshow(fit_difference, vmin = -1*difference_scale, vmax = 1*difference_scale, cmap = 'RdBu_r', origin = 'lower', extent = [ax_kx[0],ax_kx[-1],-0.75,0.75])
im4 = ax[4].imshow(intrinsic_fit, cmap = cmap_LTL, origin = 'lower', extent = extent)
im5 = ax[5].imshow(kspace_frame_win_fit, cmap = cmap_LTL, origin = 'lower', extent = extent)

ax[0].set_title('Data', fontsize = 18)
ax[1].set_title('Symm. Data', fontsize = 18)
ax[2].set_title('Convolved Fit', fontsize = 18)
ax[3].set_title('Difference', fontsize = 18)
ax[4].set_title('Retrieved Peak', fontsize = 18)
ax[5].set_title('Windowed Peak', fontsize = 18)

ax[0].set_ylabel('$k_y$, $\AA^{-1}$', fontsize = 18)
ax[2].set_ylabel('$k_y$, $\AA^{-1}$', fontsize = 18)
ax[3].set_ylabel('$k_y$, $\AA^{-1}$', fontsize = 18)

for a in [0, 1, 2, 3, 4, 5]:
    
    ax[a].set_xlabel('$k_x$, $\AA^{-1}$', fontsize = 18)
    ax[a].set_ylim(-0.8,0.8)
    ax[a].set_aspect(1)
    ax[a].set_xticks(np.arange(-2,2.2,.5))
    for label in ax[a].xaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
   # label.set_xticklabels(tick_labels.astype(int))

ax[5].axvline(0, linestyle='dashed',color='black', linewidth = 1.5)
ax[5].axvline(X, linestyle='dashed',color='black', linewidth = 1.5)
ax[5].axvline(2*X, linestyle='dashed',color='black', linewidth = 1.5)
ax[5].axvline(2*X, linestyle='dashed',color='black', linewidth = 1.5)

#ax[1].set_ylabel('$k_y$, $\AA^{-1}$', fontsize = 14)
#ax[2].set_ylabel('$k_y$, $\AA^{-1}$', fontsize = 14)

# add space for colour bar
fig.subplots_adjust(right=0.5)
cbar_ax = fig.add_axes([.98, 0.77, 0.02, 0.1])
fig.colorbar(im2, cax=cbar_ax)

fig.subplots_adjust(right=0.5)
cbar_ax = fig.add_axes([.98, 0.46, 0.02, 0.1])
fig.colorbar(im3, cax=cbar_ax)

fig.text(.04, 0.9, "(a)", fontsize = 18, fontweight = 'regular')
fig.text(.52, 0.9, "(b)", fontsize = 18, fontweight = 'regular')
fig.text(.04, 0.58, "(c)", fontsize = 18, fontweight = 'regular')
fig.text(.52, 0.58, "(d)", fontsize = 18, fontweight = 'regular')
fig.text(.04, 0.35, "(e)", fontsize = 18, fontweight = 'regular')
fig.text(.52, 0.35, "(f)", fontsize = 18, fontweight = 'regular')

im0 = ax[6].imshow(rspace_frame/np.max(rspace_frame), clim = None, origin='lower', vmax = 1, cmap=cmap_LTL, interpolation='none', extent = [r_axis[0], r_axis[-1], r_axis[0], r_axis[-1]]) #kx, ky, t
#single_k_circle = plt.Circle((single_ky, single_kx), single_rad, color='red', linestyle = 'dashed', linewidth = 1.5, clip_on=False, fill=False)
#ax[1].add_patch(single_k_circle)
ax[6].set_aspect(1)
ax[7].set_aspect(1)

ax[6].set_xticks(np.arange(-2,2.2,.5))
for label in ax[6].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
   # label.set_xticklabels(tick_labels.astype(int))
    
ax[6].set_yticks(np.arange(-2,2.2,.5))
for label in ax[6].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)

ax[7].set_xticks(np.arange(-2,2.2,.5))
for label in ax[7].xaxis.get_ticklabels()[1::2]:
    label.set_visible(True)
   # label.set_xticklabels(tick_labels.astype(int))
    
ax[7].set_yticks(np.arange(-2,2.1,.5))
for label in ax[7].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)

ax[6].set_xlim(-2,2)
ax[6].set_ylim(-2,2)
#ax[0].set_box_aspect(1)
ax[6].set_xlabel('$r_x$, nm', fontsize = 18)
ax[6].set_ylabel('$r_y$, nm', fontsize = 18)
ax[6].tick_params(axis='both', labelsize=18)
ax[6].set_title('2D FFT', fontsize = 18)

#ax[2].plot(r_axis, x_cut/np.max(1), color = 'black', label = '$r_b$')
ax[7].plot(r_axis, x_cut/np.max(x_cut), color = 'royalblue', label = '$r_x$')
#ax[3].plot(r_axis, r2_cut_x, color = 'black', linestyle = 'dashed')
ax[7].plot(r_axis, y_cut/np.max(y_cut), color = 'crimson', label = '$r_y$')
#ax[3].plot(r_axis, r2_cut_y, color = 'red', linestyle = 'dashed')

ax[7].axvline(x_brad, linestyle = 'dashed', color = 'navy', linewidth = 1.5)
ax[7].axvline(y_brad, linestyle = 'dashed', color = 'red', linewidth = 1.5)
ax[7].axvline(rdist_brad_x, linestyle = 'dashed', color = 'black', linewidth = .5)
ax[7].axvline(rdist_brad_y, linestyle = 'dashed', color = 'red', linewidth = .5)

ax[7].set_xlabel('$r$, nm', fontsize = 18)
ax[7].set_ylabel('Norm. Int.', fontsize = 18)
ax[7].set_title(f"$r^*_{{x,y}}$ = ({round(x_brad,2)}, {round(y_brad,2)}) nm", fontsize = 18)
ax[7].tick_params(axis='both', labelsize=18)
ax[7].set_yticks(np.arange(-0,1.5,0.5))
ax[7].set_xlim([0, 2])
ax[7].set_ylim([-0.025, 1.025])
ax[7].set_aspect(2/(1.025+0.025))
ax[7].set_xlabel('$r$, nm', fontsize = 18)
ax[7].legend(frameon=False, fontsize = 16)
ax[7].text(1.05, 0.55,  f"({np.round(rdist_brad_x,2)}, {np.round(rdist_brad_y,2)})", size=14)

new_rc_params = {'text.usetex': False, "svg.fonttype": 'none', "font.family":'helvetica'}
params = {'lines.linewidth' : 2.5, 'axes.linewidth' : 2, 'axes.labelsize' : 20, 
              'xtick.labelsize' : 16, 'ytick.labelsize' : 16, 'axes.titlesize' : 20, 'legend.fontsize' : 16}
plt.rcParams.update(params)
plt.rcParams.update(new_rc_params)

fig.tight_layout()

if save_figure is True:
    fig.savefig(figure_file_name + '.'+ image_format, bbox_inches='tight', format=image_format)

#%% Plot Raw, Symm., and Windowed MMs

save_figure = False
figure_file_name = '2DFFT_Windowing' 

fig, ax = plt.subplots(1, 3, sharey=True)
plt.gcf().set_dpi(300)
ax = ax.flatten()

im0 = ax[0].imshow(kspace_frame/kspace_frame.max(), origin='lower', vmax = 1, cmap=cmap_LTL, extent = [ax_kx[0],ax_kx[-1],ax_ky[0],ax_ky[-1]])
im1 = ax[1].imshow(kspace_frame_sym/kspace_frame_sym.max(), origin='lower', vmax = 1, cmap=cmap_LTL, extent = [ax_kx[0],ax_kx[-1],ax_ky[0],ax_ky[-1]])
im2 = ax[2].imshow(kspace_frame_sym_win/kspace_frame_sym_win.max(), origin='lower', vmax = 1, cmap=cmap_LTL, extent = [ax_kx[0],ax_kx[-1],ax_ky[0],ax_ky[-1]])

for i in np.arange(3):
    #ax[i].axhline(0, color='black', linewidth = 1, linestyle = 'dashed')
    ax[i].axvline(0, color='black', linewidth = 1, linestyle = 'dashed')
    ax[i].axvline(-X, color='black', linewidth = 1, linestyle = 'dashed')
    ax[i].axvline(X, color='black', linewidth = 1, linestyle = 'dashed')
    ax[i].axvline(2*X, color='black', linewidth = 1, linestyle = 'dashed')
    ax[i].axvline(-2*X, color='black', linewidth = 1, linestyle = 'dashed')
    
    ax[i].set_aspect(1)
    #ax[0].axhline(y,color='black')
    #ax[0].axvline(x,color='black')
    
    ax[i].set_xticks(np.arange(-2,2.2,1))
    for label in ax[i].xaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
        
    ax[i].set_yticks(np.arange(-2,2.1,1))
    for label in ax[i].yaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
    
    ax[i].set_xlim(-2,2)
    ax[i].set_ylim(-2,2)
    #ax[0].set_box_aspect(1)

    ax[i].set_xlabel('$k_x$, $\AA^{-1}$', fontsize = 14)
    ax[i].set_ylabel('$k_y$, $\AA^{-1}$', fontsize = 14)
    ax[i].tick_params(axis='both', labelsize=12)
    ax[i].set_title('$E$ = ' + str(E) + ' eV', fontsize = 16)

fig.text(.03, 0.75, "(a)", fontsize = 16, fontweight = 'regular')
fig.text(.36, 0.75, "(b)", fontsize = 16, fontweight = 'regular')
fig.text(.69, 0.75, "(c)", fontsize = 16, fontweight = 'regular')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([1, 0.35, 0.025, 0.3])
fig.colorbar(im2, cax=cbar_ax, ticks = [0,1])
cbar_ax.set_yticklabels(['min', 'max'], fontsize=16)  # vertically oriented colorbar

#fig.colorbar(im, fraction=0.046, pad=0.04)
fig.tight_layout()
plt.show()

if save_figure is True:
    fig.savefig((figure_file_name +'.svg'), format='svg')
    
#%% Plot 2D FFT and Line Cuts

### PLOT ###

save_figure = True
figure_file_name = 'MM_FFT_120k_deconv' 
image_format = 'svg'

fig, ax = plt.subplots(1, 2)
fig.set_size_inches(6,4)
plt.gcf().set_dpi(300)
ax = ax.flatten()

im0 = ax[0].imshow(rspace_frame/np.max(rspace_frame), clim = None, origin='lower', vmax = 1, cmap=cmap_LTL, interpolation='none', extent = [r_axis[0], r_axis[-1], r_axis[0], r_axis[-1]]) #kx, ky, t
#single_k_circle = plt.Circle((single_ky, single_kx), single_rad, color='red', linestyle = 'dashed', linewidth = 1.5, clip_on=False, fill=False)
#ax[1].add_patch(single_k_circle)
ax[0].set_aspect(1)
ax[1].set_aspect(1)

#ax[0].axhline(y,color='black')
#ax[0].axvline(x,color='bl ack')

ax[0].set_xticks(np.arange(-2,2.2,.5))
for label in ax[0].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
   # label.set_xticklabels(tick_labels.astype(int))
    
ax[0].set_yticks(np.arange(-2,2.2,.5))
for label in ax[0].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)

ax[1].set_xticks(np.arange(-2,2.2,.5))
for label in ax[1].xaxis.get_ticklabels()[1::2]:
    label.set_visible(True)
   # label.set_xticklabels(tick_labels.astype(int))
    
ax[1].set_yticks(np.arange(-2,2.1,.5))
for label in ax[1].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)

ax[0].set_xlim(-2,2)
ax[0].set_ylim(-2,2)
#ax[0].set_box_aspect(1)
ax[0].set_xlabel('$r_x$, nm', fontsize = 14)
ax[0].set_ylabel('$r_y$, nm', fontsize = 14)
ax[0].tick_params(axis='both', labelsize=10)
ax[0].set_title('2D FFT', fontsize = 12)

#ax[2].plot(r_axis, x_cut/np.max(1), color = 'black', label = '$r_b$')
ax[1].plot(r_axis, x_cut/np.max(x_cut), color = 'royalblue', label = '$r_x$')
#ax[3].plot(r_axis, r2_cut_x, color = 'black', linestyle = 'dashed')
ax[1].plot(r_axis, y_cut/np.max(y_cut), color = 'crimson', label = '$r_y$')
#ax[3].plot(r_axis, r2_cut_y, color = 'red', linestyle = 'dashed')

ax[1].axvline(x_brad, linestyle = 'dashed', color = 'navy', linewidth = 1.5)
ax[1].axvline(y_brad, linestyle = 'dashed', color = 'red', linewidth = 1.5)
ax[1].axvline(rdist_brad_x, linestyle = 'dashed', color = 'black', linewidth = .5)
ax[1].axvline(rdist_brad_y, linestyle = 'dashed', color = 'red', linewidth = .5)

ax[1].set_ylim([-0.025, 1.025])
ax[1].set_xlabel('$r$, nm', fontsize = 14)
ax[1].set_ylabel('Norm. Int.', fontsize = 14)
ax[1].set_title(f"$r^*_{{x,y}}$ = ({round(x_brad,2)}, {round(y_brad,2)}) nm", fontsize = 12)
ax[1].tick_params(axis='both', labelsize=10)
ax[1].set_yticks(np.arange(-0,1.5,0.5))
ax[1].set_xlim([0, 2])
ax[1].set_aspect(2/(1.025+0.025))
ax[1].set_xlabel('$r$, nm')
ax[1].legend(frameon=False, fontsize = 12)
ax[1].text(1.05, 0.55,  f"({np.round(rdist_brad_x,2)}, {np.round(rdist_brad_y,2)})", size=10)

#fig.subplots_adjust(right=0.58, top = 1.1)
fig.tight_layout()
new_rc_params = {'text.usetex': False, "svg.fonttype": 'none'}
plt.rcParams.update(new_rc_params)

fig.text(.03, 0.8, "(e)", fontsize = 14, fontweight = 'regular')
fig.text(0.5, 0.8, "(f)", fontsize = 14, fontweight = 'regular')

if save_figure is True:
    fig.savefig(figure_file_name + '.'+ image_format, bbox_inches='tight', format=image_format)
    
#print("x: " + str(round(rdist_brad_x,3)))
#print("y: " + str(round(rdist_brad_y,3)))
    
#%% # Plot MM, Windowed Map, I_xy, and r-space cuts

# %matplotlib inline

# ### PLOT ###

# save_figure = False
# figure_file_name = 'MM_FFT' 

# fig, ax = plt.subplots(2, 2)
# fig.set_size_inches(6,8)
# plt.gcf().set_dpi(300)
# ax = ax.flatten()

# im0 = ax[0].imshow(kspace_frame/np.max(kspace_frame), clim = None, origin = 'lower', vmax = 1, cmap=cmap_LTL, interpolation = 'none', extent = [ax_kx[0], ax_kx[-1], ax_ky[0], ax_ky[-1]])
# im1 = ax[1].imshow(MM_frame/np.max(MM_frame), clim = None, origin = 'lower', vmax = 1, cmap=cmap_LTL, interpolation = 'none', extent = [ax_kx[0], ax_kx[-1], ax_ky[0], ax_ky[-1]])
# im2 = ax[2].imshow(rspace_frame/np.max(rspace_frame), clim = None, origin='lower', vmax = 1, cmap=cmap_LTL, interpolation='none', extent = [r_axis[0], r_axis[-1], r_axis[0], r_axis[-1]]) #kx, ky, t
# #single_k_circle = plt.Circle((single_ky, single_kx), single_rad, color='red', linestyle = 'dashed', linewidth = 1.5, clip_on=False, fill=False)
# #ax[1].add_patch(single_k_circle)
# ax[0].set_aspect(1)
# ax[1].set_aspect(1)
# ax[2].set_aspect(1)

# #ax[0].axhline(y,color='black')
# #ax[0].axvline(x,color='bl ack')

# ax[0].set_xticks(np.arange(-2,2.2,1))
# for label in ax[0].xaxis.get_ticklabels()[1::2]:
#     label.set_visible(False)
#    # label.set_xticklabels(tick_labels.astype(int))
    
# ax[0].set_yticks(np.arange(-2,2.2,1))
# for label in ax[0].yaxis.get_ticklabels()[1::2]:
#     label.set_visible(False)

# ax[1].set_xticks(np.arange(-2,2.2,1))
# for label in ax[1].xaxis.get_ticklabels()[1::2]:
#     label.set_visible(False)
#    # label.set_xticklabels(tick_labels.astype(int))
    
# ax[1].set_yticks(np.arange(-2,2.1,1))
# for label in ax[1].yaxis.get_ticklabels()[1::2]:
#     label.set_visible(False)

# ax[2].set_xticks(np.arange(-8,8,.5))
# for label in ax[2].xaxis.get_ticklabels()[1::2]:
#     label.set_visible(False)
    
# ax[2].set_yticks(np.arange(-8,8.1,.5))
# for label in ax[2].yaxis.get_ticklabels()[1::2]:
#     label.set_visible(False)
    
# ax[3].set_xticks(np.arange(0,5.2,.5))
# #for label in ax[3].xaxis.get_ticklabels()[1::2]:
#     #label.set_visible(False)    

# ax[0].axvline(0, color='black', linewidth = 1, linestyle = 'dashed')
# #ax[0].axhline(0, color='black', linewidth = 1, linestyle = 'dashed')
# ax[0].axvline(-X, color='black', linewidth = 1, linestyle = 'dashed')
# ax[0].axvline(X, color='black', linewidth = 1, linestyle = 'dashed')
# ax[0].axvline(2*X, color='black', linewidth = 1, linestyle = 'dashed')
# ax[0].axvline(-2*X, color='black', linewidth = 1, linestyle = 'dashed')

# ax[1].axvline(X, color='black', linewidth = 1, linestyle = 'dashed')
# ax[1].axvline(2*X, color='black', linewidth = 1, linestyle = 'dashed')

# ax[0].set_xlim(-2,2)
# ax[0].set_ylim(-2,2)
# #ax[0].set_box_aspect(1)
# ax[0].set_xlabel('$k_x$, $\AA^{-1}$', fontsize = 16)
# ax[0].set_ylabel('$k_y$,  $\AA^{-1}$', fontsize = 16)
# ax[0].tick_params(axis='both', labelsize=10)
# ax[0].set_title('$E$ = ' + str(E) + ' eV, ' + '$\Delta$E = ' + str(E_int) + ' eV', fontsize = 14)
# ax[0].set_title('$E$ = ' + str(E) + ' eV ', fontsize = 14)
# #fig.suptitle('E = ' + str(E) + ' eV, $\Delta$E = ' + str(1000*E_int) + ' meV,  $\Delta$$k_{rad}$ = ' + str(window_k_width) + ' $\AA^{-1}$', fontsize = 18)
# ax[0].text(-1.9, 1.5,  f"$\Delta$t = {round(delays-delay_int/2)} to {round(delay_int+delay_int/2)} fs", size=12)

# ax[1].set_xlim(-2,2)
# ax[1].set_ylim(-2,2)
# #ax[0].set_box_aspect(1)
# ax[1].set_xlabel('$k_x$, $\AA^{-1}$', fontsize = 16)
# ax[1].set_ylabel('$k_y$,  $\AA^{-1}$', fontsize = 16)
# ax[1].tick_params(axis='both', labelsize=10)
# ax[1].set_title(f'$\Delta$k = ({kx_int:.2f}, {ky_int})', fontsize = 15)
# # ax[1].axvline(0, color='black', linewidth = 1, linestyle = 'dashed')
# # ax[1].axhline(0, color='black', linewidth = 1, linestyle = 'dashed')
# # ax[1].axvline(-X, color='black', linewidth = 1, linestyle = 'dashed')
# # ax[1].axvline(X, color='black', linewidth = 1, linestyle = 'dashed')

# ax[2].set_xlim(-1,1)
# ax[2].set_ylim(-1,1)
# #ax[0].set_box_aspect(1)
# ax[2].set_xlabel('$r_x$, nm', fontsize = 16)
# ax[2].set_ylabel('$r_y$, nm', fontsize = 16)
# ax[2].tick_params(axis='both', labelsize=10)
# ax[2].set_title('2D FFT', fontsize = 15)

# #ax[2].plot(r_axis, x_cut/np.max(1), color = 'black', label = '$r_b$')
# ax[3].plot(r_axis, x_cut/np.max(x_cut), color = 'black', label = '$r_x$')
# #ax[3].plot(r_axis, r2_cut_x, color = 'black', linestyle = 'dashed')
# ax[3].plot(r_axis, y_cut/np.max(y_cut), color = 'red', label = '$r_y$')
# #ax[3].plot(r_axis, r2_cut_y, color = 'red', linestyle = 'dashed')

# ax[3].axvline(x_brad, linestyle = 'dashed', color = 'black', linewidth = 1.5)
# ax[3].axvline(y_brad, linestyle = 'dashed', color = 'red', linewidth = 1.5)
# ax[3].axvline(rdist_brad_x, linestyle = 'dashed', color = 'black', linewidth = .5)
# ax[3].axvline(rdist_brad_y, linestyle = 'dashed', color = 'red', linewidth = .5)

# ax[3].set_ylim([-0.025, 1.025])
# ax[3].set_xlabel('$r$, nm', fontsize = 16)
# ax[3].set_ylabel('Norm. Int.', fontsize = 16)
# ax[3].set_title(f"$r^*_{{x,y}}$ = ({round(x_brad,2)}, {round(y_brad,2)}) nm", fontsize = 14)
# ax[3].tick_params(axis='both', labelsize=10)
# ax[3].set_yticks(np.arange(-0,1.5,0.5))
# ax[3].set_xlim([0, 1])
# ax[3].set_aspect(1)
# ax[3].set_xlabel('$r$, nm')
# ax[3].legend(frameon=False, fontsize = 12)
# ax[3].text(1.05, 0.55,  f"({np.round(rdist_brad_x,2)}, {np.round(rdist_brad_y,2)})", size=10)

# fig.subplots_adjust(right=0.58, top = 1.1)
# fig.tight_layout()
# new_rc_params = {'text.usetex': False, "svg.fonttype": 'none'}
# plt.rcParams.update(new_rc_params)

# fig.text(.03, 0.45, "(a)", fontsize = 16, fontweight = 'regular')
# fig.text(.42, 0.45, "(b)", fontsize = 16, fontweight = 'regular')

# if save_figure is True:
#     fig.savefig((figure_file_name +'.svg'), format='svg')
    
# #print("x: " + str(round(rdist_brad_x,3)))
# #print("y: " + str(round(rdist_brad_y,3)))


#%% Do line fits analysis to cross check values

test_frame = kspace_frame_sym
test_frame_win = kspace_frame_sym_win

kx = X/2
kx_cut = test_frame.loc[{"ky":slice(-0.05,0.05)}].mean(dim="ky")
ky_cut = test_frame.loc[{"kx":slice(kx-.05,kx+.05)}].mean(dim="kx")
kx_win_cut = test_frame_win.loc[{"ky":slice(-0.05,0.05)}].mean(dim="ky")
ky_win_cut = test_frame_win.loc[{"kx":slice(kx-.05,kx+.05)}].mean(dim="kx")
window_kx_cut = kspace_window.loc[{"ky":slice(-0.05,0.05)}].mean(dim="ky")
window_ky_cut = kspace_window.loc[{"kx":slice(kx-.05,kx+.05)}].mean(dim="kx")
    
#kx_win_cut = MM_frame.loc[{"ky":slice(-.25,.25)}].sum(dim="ky")
#ky_win_cut = MM_frame.loc[{"kx":slice(-.05,1.1)}].sum(dim="kx")

#ky_cut = MM_frame[:,int(len(MM_frame[0])/2)-1-4:int(len(MM_frame[0])/2)-1+4].sum(axis=1)
#kx_cut = MM_frame[int(len(MM_frame[0])/2)-1-4:int(len(MM_frame[0])/2)-1+4,:].sum(axis=0)

ky_cut = ky_cut/np.max(ky_cut)
kx_cut = kx_cut/np.max(kx_cut)

kx_win_cut = kx_win_cut/np.max(kx_cut.loc[{"kx":slice(0,1.5)}])
ky_win_cut = ky_win_cut/np.max(ky_win_cut)

# Fit kx Cut
xlim = [-0.1, 1]
p0 = [0.5, 0.9, 0.5, 0.0]
bnds = ((0.1, -0.5, .2, 0), (1.5, 1.5, 1.2, 0.8))
popt_kx, pcov = curve_fit(gaussian, ax_kx.loc[{"kx":slice(xlim[0],xlim[1])}], kx_cut.loc[{"kx":slice(xlim[0],xlim[1])}], p0, method=None, bounds = bnds)
g_fit_kx = gaussian(ax_kx, *popt_kx)
k_sig_fit_x = popt_kx[2]
#plt.plot(ax_kx, kx_cut) ; plt.plot(ax_kx, g_fit_kx)

# Fit ky Cut
ylim = [-.2, .2]
p0 = [1, 0.0, .105, 0.0]
bnds = ((0.5, -0.2, 0, 0), (1.5, 0.2, .2, .5)) #Amp, mean, std. offset
popt_ky, pcov = curve_fit(gaussian, ax_ky.loc[{"ky":slice(ylim[0],ylim[1])}], ky_cut.loc[{"ky":slice(ylim[0],ylim[1])}], p0, method=None, bounds = bnds)
g_fit_ky = gaussian(ax_ky, *popt_ky)

#initial_guess = (.1, .05, 1, 0)
#popt_ky = fit_convolved_model(ax_ky.loc[{"ky":slice(ylim[0],ylim[1])}], ky_cut.loc[{"ky":slice(ylim[0],ylim[1])}], sigma_irf, initial_guess)
# Retrieve fit components
#g_fit_ky = lorentzian_(ax_kx.values, *popt_ky)
#fitted_model = convolved_model(ax_kx.values, *popt, sigma_irf)
#popt_ky, pcov = curve_fit(lorentzian, ax_ky.loc[{"ky":slice(ylim[0],ylim[1])}], ky_cut.loc[{"ky":slice(ylim[0],ylim[1])}], p0, method=None, bounds = bnds)
#popt_ky = [0.88, 0.0, .3, 0.12]
#g_fit_ky = lorentzian(ax_ky, *popt_ky)

k_sig_fit_y = popt_ky[2]
#plt.plot(ax_ky, ky_cut) ; plt.plot(ax_ky, g_fit_ky)

#Fit r-x Cut
p0 = [1, 0, 0.2, 0]
bnds = ((0.5, -1, .05, 0), (1.2, 2, 5, 0.4))
popt_rx, pcov_r = curve_fit(gaussian, r_axis, x_cut/np.max(x_cut), p0, method=None, bounds = bnds)
g_fit_rx = gaussian(r_axis, *popt_rx)
r_sig_fit_x = popt_rx[2]

#Fit r-y Cut
p0 = [1, 0, 0.2, 0]
bnds = ((0.5, -1, .05, 0), (1.2, 2, 5, 0.4))
popt_ry, pcov_r = curve_fit(gaussian, r_axis, y_cut/np.max(y_cut), p0, method=None, bounds = bnds)
g_fit_ry = gaussian(r_axis, *popt_ry)
r_sig_fit_y = popt_ry[2]

# Do the FFT of the fit in the k-space to get verify r-space
g_fit_fft = np.abs(np.fft.fftshift(np.fft.fft((g_fit_kx-popt_kx[3])**0.5, zeropad)))  # Compute FFT
g_fit_fft_x_root = g_fit_fft / np.max(g_fit_fft)
g_fit_fft_x = g_fit_fft_x_root**2

g_fit_fft = np.abs(np.fft.fftshift(np.fft.fft((g_fit_ky-popt_ky[3])**0.5, zeropad)))  # Compute FFT
g_fit_fft_y_root = g_fit_fft / np.max(np.abs(g_fit_fft))
g_fit_fft_y = g_fit_fft_y_root**2
g_fit_fft_y_r2 = (g_fit_fft_y_root*r_axis)**2
r2_brad_y = r_axis[np.argmax(g_fit_fft_y_r2[1024:])+1024]

#Fit r-y Cut from FFT of k-fit
p0 = [1, 0, 0.2, 0]
bnds = ((0.5, -1, .05, 0), (1.2, 2, 5, 0.4))
popt_ry_kfit, pcov_r = curve_fit(gaussian, r_axis, g_fit_fft_y/np.max(g_fit_fft_y), p0, method=None, bounds = bnds)
g_fit_ry_kfit = gaussian(r_axis, *popt_ry_kfit)
r_sig_fit_y_kfit = popt_ry_kfit[2]

### Fourier Transform: k-space to r-space
#Prediced from FT relations
r_sig_x = 0.1*1/(2*k_sig_fit_x) #Ang to nm
r_sig_rad_x = np.sqrt(2)*r_sig_x # Rad from Gaussian Relation Considering fit to k-data before fft

r_sig_y = 0.1*1/(2*k_sig_fit_y) #Ang to nm
r_sig_rad_y = np.sqrt(2)*r_sig_y # Rad from Gaussian Relation Considering fit to k-data before fft

# Extracted from Fit in Real Space
r_sig_rad_fit_x = np.sqrt(2)*r_sig_fit_x #Rad from fit to r-data from fft
r_sig_rad_fit_y = np.sqrt(2)*r_sig_fit_y #Rad from fit to r-data from fft
r_sig_rad_fit_y_fitk = np.sqrt(2)*r_sig_fit_y_kfit #Rad from fit to r-data from fft of k-fit (y)

#print("predicted x: " + str(round(x_pr,3)))

#####################################################
save_figure = False
figure_file_name = '2DFFT_Windowing' 

fig, ax = plt.subplots(2, 2, sharey=False, gridspec_kw={'width_ratios': [1, 1], 'height_ratios':[1, 1]})
fig.set_size_inches(10, 6, forward=False)
plt.gcf().set_dpi(300)
ax = ax.flatten()

#test_frame.plot.imshow(ax = ax[0], cmap = cmap_LTL, origin = 'lower')

ax[0].plot(ax_ky, g_fit_kx, linewidth = 4, color = 'pink', label = 'Fit to k-Data')
ax[0].plot(ax_kx, kx_cut, color =  'purple', linewidth = 2, label = 'Data')
ax[0].plot(ax_kx, kx_win_cut, color =  'black', linestyle = 'solid', linewidth = 1.5, label = 'Win. Data.')
ax[0].plot(ax_kx, window_kx_cut, color = 'grey', linestyle = 'solid', linewidth = 1.5, label =  'WINDOW')
ax[0].axvline(0, color='black',linestyle = 'dashed',linewidth = 1.5)
ax[0].axvline(2*X, color='black',linestyle = 'dashed',linewidth = 1.5)
ax[0].axvline(1*X, color='black',linestyle = 'dashed',linewidth = 1.5)

ax[1].plot(ax_ky, g_fit_ky, linewidth = 3, color = 'pink', label = 'Fit to k-Data')
#ax[2].plot(ax_kx, ky_win, color =  'grey', linestyle = 'solid', linewidth = 1.5)
ax[1].plot(ax_ky, ky_cut, color =  'purple', linewidth = 2, label = 'Data')
ax[1].plot(ax_ky, ky_win_cut, color =  'black', linestyle = 'solid', linewidth = 2, label = 'Win. Data.')
ax[1].plot(ax_ky, window_ky_cut, color = 'grey', linestyle = 'solid', linewidth = 1.5, label =  'WINDOW')

for i in [0,1,2,3]:
    ax[i].set_xticks(np.arange(-2,2.2,1))
    for label in ax[i].xaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
        
    ax[i].set_yticks(np.arange(0,1.1,.5))
    for label in ax[i].yaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
        
    ax[i].set_ylabel('Norm. Int.', fontsize = 14)
    ax[i].set_ylim(0,1.1)
    ax[i].set_xlim(-2,2)

#ax[0].axhline(xi, color='black', linewidth = 1, linestyle = 'dashed')
#ax[0].axvline(yi, color='black', linewidth = 1, linestyle = 'dashed')
ax[0].set_title(f'$k_y$ = {ky:.2f} $\AA^{{-1}}$', fontsize = 18)
ax[1].set_title(f'$k_x$ = {kx:.2f} $\AA^{{-1}}$', fontsize = 18)
ax[0].set_xlabel('$k_x$, $\AA^{-1}$', fontsize = 16)
ax[1].set_xlabel('$k_y$, $\AA^{-1}$', fontsize = 16)
ax[2].set_xlabel('$r_x$, nm', fontsize = 16)
ax[3].set_xlabel('$r_y$, nm', fontsize = 16)
ax[2].set_title('FFT', fontsize = 18)
ax[3].set_title('FFT', fontsize = 18)

ax[2].plot(r_axis, g_fit_fft_x, linewidth = 3, color = 'pink', label = 'FFT of k-fit (w/o offset)')
#ax[4].plot(r_axis, g_fit_rx/np.max(g_fit_rx), linewidth = 3, color = 'green', label = 'fit to FFT of Win. k-data')
#ax[5].plot(r_axis, y_cut, color =  'grey', linestyle = 'solid', linewidth = 1.5)
ax[2].plot(r_axis, g_fit_rx/np.max(g_fit_rx), linewidth = 3, color = 'green', label = 'fit to FFT of Win. k-data')
ax[2].plot(r_axis, x_cut, color =  'black', linewidth = 2, label = 'FFT of Win k-data')
#ax[5].plot(r_axis, kx_win_cut, color =  'black', linestyle = 'dashed', linewidth = 1.5, label = 'FFT Data')
ax[2].set_xlim(-2,2)

ax[3].plot(r_axis, g_fit_fft_y, linewidth = 4, color = 'pink', label = 'FFT of k-fit (w/o offset)')
ax[3].plot(r_axis, g_fit_ry_kfit, linewidth = 4, color = 'purple', linestyle = 'dashed', label = 'fit to FFT of k-fit (w/o offset)')

ax[3].plot(r_axis, g_fit_ry/np.max(g_fit_ry), linewidth = 3, color = 'green', label = 'fit to FFT of Win. k-data')
#ax[5].plot(r_axis, y_cut, color =  'grey', linestyle = 'solid', linewidth = 1.5)
ax[3].plot(r_axis, y_cut, color =  'black', linewidth = 2, label = 'FFT of Win k-data')
#ax[5].plot(r_axis, kx_win_cut, color =  'black', linestyle = 'dashed', linewidth = 1.5, label = 'FFT Data')
ax[3].set_xlim(-2,2)

print(f"Pred. Rx (rad) from Fit of k Peak ({round(k_sig_fit_x,4)} A^-1): {round(r_sig_rad_x,3)} nm")
#print(f"Rx (radius) from Fit of Real-space After FFT: {round(r_sig_rad_fit_x,3)} nm")
print(f"Rx (rad) from Fit of Real-space After FFT: {round(r_sig_rad_fit_x,3)} nm")
print()
print(f"Pred. Ry (rad) from Fit of k Peak ({round(k_sig_fit_y,4)} A^-1): {round(r_sig_rad_y,3)} nm")
print(f"Ry (rad) from Fit of Real-space After FFT: {round(r_sig_rad_fit_y,3)} nm")
print(f"Ry (rad) from Fit of Real-space After FFT of k-fit: {round(r_sig_rad_fit_y_fitk,3)} nm")
print(f"Ry (r^2 Bohr Rad) from Fit of Real-space After FFT of k-fit: {round(r2_brad_y,3)} nm")

ax[1].legend(frameon=False, fontsize = 12)
ax[3].legend(frameon=False, fontsize = 12)
#fig.subplots_adjust(right=0.8)

fig.text(.03, 0.975, "(a)", fontsize = 16, fontweight = 'regular')
fig.text(.51, 0.975, "(b)", fontsize = 16, fontweight = 'regular')
fig.text(.03, 0.48, "(c)", fontsize = 16, fontweight = 'regular')
fig.text(.51, 0.48, "(d)", fontsize = 16, fontweight = 'regular')

fig.tight_layout()
plt.show()

if save_figure is True:
    fig.savefig(figure_file_name + '.'+ image_format, bbox_inches='tight', format=image_format)

#%%