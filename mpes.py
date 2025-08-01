# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 11:15:40 2025

@author: lloyd
""" 
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as col
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.patches import Polygon
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import LogNorm
import xarray as xr
from scipy.special import erf
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve
from scipy.optimize import curve_fit
import numpy as np
from scipy.interpolate import RegularGridInterpolator

#%% Useful Functions and Definitions for Manipulating Data

# Partition Data into + and - Delays
def get_data_chunks(I, neg_times, t0, ax_delay_offset):
    if I.ndim > 3:
        tnf1 = (np.abs(ax_delay_offset - neg_times[0])).argmin()
        tnf2 = (np.abs(ax_delay_offset - neg_times[1])).argmin()

        I_neg = I[:,:,:,tnf1:tnf2+1] #Sum over delay/polarization/theta...
        neg_length = I_neg.shape[3]
        I_neg = I_neg
        I_neg_sum = I_neg.sum(axis=(3))/neg_length
    
        I_pos = I[:,:,:,t0+1:]
        pos_length = I_pos.shape[3]
        I_pos = I_pos #Sum over delay/polarization/theta...
        I_pos_sum = I_pos.sum(axis=(3))/pos_length
    
        I_sum = I[:,:,:,:].sum(axis=(3))
        
        return I_sum, I_pos, I_pos_sum, I_neg, I_neg_sum

    else:
        I_neg = I[:,:,:] #Sum over delay/polarization/theta...
        I_pos = I[:,:,:]
        I_sum = I

# Function for Creating MM Constant Energy kx, ky slice 
def get_momentum_map(I_res, E, E_int, delay=None, delay_int=None):
    
    # Integrate over energy window
    I_E = I_res.loc[{"E":slice(E - E_int / 2, E + E_int / 2)}].mean(dim="E")

    if "delay" in I_res.dims:
        if delay is not None and delay_int is not None:
            # Integrate over a delay window
            frame = I_E.loc[{"delay":slice(delay - delay_int / 2, delay + delay_int / 2)}].mean(dim="delay").T
        else:
            # No delay specified: average over entire delay axis
            frame = I_E.mean(dim="delay").T
    else:
        frame = I_E.T

    return frame

def get_kx_E_frame(I_res, ky, ky_int, delay, delay_int):
    
    if I_res.ndim > 3:    
        frame = I_res.loc[{"ky":slice(ky-ky_int/2, ky+ky_int/2), "delay":slice(delay-delay_int/2, delay+delay_int/2)}].mean(dim="ky").mean(dim="delay")
    
    else:
        frame = I_res.loc[{"ky":slice(ky-ky_int/2, ky+ky_int/2)}].mean(dim="ky")

    return frame

def get_ky_E_frame(I_res, kx, kx_int, delay, delay_int):

    if I_res.ndim > 3:    
        frame = I_res.loc[{"kx":slice(kx-kx_int/2, kx+kx_int/2), "delay":slice(delay-delay_int/2, delay+delay_int/2)}].mean(dim="kx").mean(dim="delay")
    else:
        frame = I_res.loc[{"kx":slice(kx-kx_int/2, kx+kx_int/2)}].mean(dim="kx")

    return frame

def get_waterfall(I_res, kx, kx_int, ky, ky_int):
    
    frame = I_res.loc[{"kx":slice(kx-kx_int/2, kx+kx_int/2), "ky":slice(ky-ky_int/2, ky+ky_int/2)}].mean(dim=("kx","ky"))

    return frame

def get_k_cut(I, k_start, k_end, num_k=200):
    """
    Extract an E vs k slice along an arbitrary line in kx-ky space.

    Parameters:
    - I: xarray.DataArray with dims ('kx', 'ky', 'energy')
    - k_start: (kx0, ky0) tuple — start point in k-space
    - k_end: (kx1, ky1) tuple — end point in k-space
    - num_k: number of points along the k-space cut

    Returns:
    - I_cut: 2D array of shape (len(E), num_k)
    - k_vals: 1D array of k-distance along the cut
    - E_vals: 1D array of energies
    """
    assert all(dim in I.dims for dim in ['kx', 'ky', 'E']), "Data must have kx, ky, energy dims"

    # Coordinate arrays
    kx_vals = I.kx.values
    ky_vals = I.ky.values
    E_vals = I.E.values

    # Create k-space cut line
    k_start = np.array(k_start)
    k_end = np.array(k_end)
    k_line = np.linspace(k_start, k_end, num_k)
    kx_line = k_line[:, 0]
    ky_line = k_line[:, 1]

    # Interpolator
    interp = RegularGridInterpolator(
        (kx_vals, ky_vals, E_vals),
        I.transpose('kx', 'ky', 'E').values,
        bounds_error=False,
        fill_value=np.nan
    )

    # For each point along the k-line, extract the I(E) spectrum
    I_cut = []
    for kx_i, ky_i in zip(kx_line, ky_line):
        pts = np.column_stack([np.full_like(E_vals, kx_i),
                               np.full_like(E_vals, ky_i),
                               E_vals])
        spectrum = interp(pts)
        I_cut.append(spectrum)

    I_cut = np.array(I_cut).T  # shape: (energy, k_index)

    # Compute 1D distance along cut
    dk = np.linalg.norm(k_end - k_start)
    k_vals = np.linspace(0, dk, num_k)

    return I_cut, k_vals, E_vals

# Fucntion for Extracting time Traces
def get_time_trace(I_res, E, E_int, k, k_int, norm_trace = False, **kwargs):
    
    # At the top of your function
    if isinstance(k, (int, float)):
        k = (k,)
    if isinstance(k_int, (int, float)):
        k_int = (k_int,)

    subtract_neg = kwargs.get("subtract_neg", False)
    neg_delays = kwargs.get("neg_delays", [-200, -100])

    d1, d2 = neg_delays[0], neg_delays[1]
    
    if "angle" in I_res.dims:
        (angle,) = k
        (angle_int,) = k_int
        trace = I_res.loc[{"E":slice(E-E_int/2, E+E_int/2), "angle":slice(angle-angle_int/2, angle+angle_int/2)}].mean(dim=("angle", "E"))
    
    elif "kx" in I_res.dims and "ky" in I_res.dims:
        (kx, ky) = k
        (kx_int, ky_int) = k_int
        trace = I_res.loc[{"E":slice(E-E_int/2, E+E_int/2), "kx":slice(kx-kx_int/2, kx+kx_int/2), "ky":slice(ky-ky_int/2, ky+ky_int/2)}].mean(dim=("kx", "ky", "E"))
    
    else:
        raise ValueError("Data must contain either ('angle') or ('kx', 'ky') dimensions.")

    if subtract_neg is True : 
        trace = trace - np.mean(trace.loc[{"delay":slice(d1,d2)}])
    
    if norm_trace is True : 
        trace = trace/np.max(trace)
    
    return trace

def get_edc(I_res, kx, ky, k_int, delay, delay_int):
    
    (kx_int, ky_int) = k_int
    
    if I_res.ndim > 3:    

        edc = I_res.loc[{"kx":slice(kx-kx_int/2, kx+kx_int/2), "ky":slice(ky-ky_int/2, ky+ky_int/2), "delay":slice(delay-delay_int/2,delay+delay_int/2)}].mean(dim=("kx", "ky", "delay"))
    else:
        edc = I_res.loc[{"kx":slice(kx-kx_int/2, kx+kx_int/2), "ky":slice(ky-ky_int/2, ky+ky_int/2)}].mean(dim=("kx", "ky"))

    return edc

def enhance_features(I_res, Ein, factor, norm):
    
    I1 = I_res.loc[{"E":slice(-3.5,Ein)}]
    I2 = I_res.loc[{"E":slice(Ein,3.5)}]

    if norm is True:
        I1 = I1/np.max(I1)
        I2 = I2/np.max(I2)
    else:
        I1 = I1/factor[0]
        I2 = I2/factor[1]
        
    I3 = xr.concat([I1, I2], dim = "E")
    
    return I3

def find_E0(edc, energy_window, p0, fig, ax):
    
    def gaussian(x, amp_1, mean_1, stddev_1, offset):
        
        g1 = amp_1 * np.exp(-0.5*((x - mean_1) / stddev_1)**2)+offset
        
        return g1
    
    #plt.legend(frameon = False)
    ax[1].set_xlim([-1, 1]) 
    #ax[1].set_ylim([0, 1.1])
    ax[1].set_xlabel('E - E$_{VBM}$, eV')
    ax[1].set_ylabel('Norm. Int.')
    #ax[1].axvline(0, color = 'black', linestyle = 'dashed', linewidth = 0.5)
    #ax[1].set_yscale('log')
    #plt.ax[1].gca().set_aspect(2)
    
    ##### VBM #########
    #e1 = -.15
    #e2 = 0.6
    #    p0 = [1, .02, 0.17, 0] # Fitting params initial guess [amp, center, width, offset]
    bnds = ((0.5, -.155, 0.0, 0), (1.5, 0.75, 1.5, .5))

    e1, e2 = energy_window[0], energy_window[1]    
    try:
        popt, pcov = curve_fit(gaussian, edc.loc[{"E":slice(e1,e2)}].E.values, edc.loc[{"E":slice(e1,e2)}].values, p0, method=None, bounds = bnds)
    except ValueError:
        popt = [0, 0, 0, 0]
        pcov = [0, 0, 0, 0]
        print('oops!')
        
    perr = np.sqrt(np.diag(pcov))
            
    vb_fit = gaussian(edc.E, *popt)
    ax[1].plot(edc.E, edc, color = 'black', label = 'Data')
    ax[1].plot(edc.E, vb_fit, linestyle = 'dashed', color = 'red', label = 'Fit')
    ax[1].legend(frameon=False, loc = 'upper left', fontsize = 11)
    print(f'E_VBM = {popt[1]:.3f} +- {perr[1]:.3f} eV')
    
def find_t0(trace_ex, delay_limits, fig, ax):

    def rise_erf(t, t0, tau):
        r = 0.5 * (1 + erf((t - t0) / (tau)))
        return r
            
    p0 = [-30, 45]
    #delay_limits = [-200,60]
    popt, pcov = curve_fit(rise_erf, trace_ex.loc[{"delay":slice(delay_limits[0],delay_limits[1])}].delay.values ,
                           trace_ex.loc[{"delay":slice(delay_limits[0],delay_limits[1])}].values,
                           p0, method="lm")
    
    perr = np.sqrt(np.diag(pcov))
    
    rise_fit = rise_erf(np.linspace(delay_limits[0],delay_limits[1], 50), *popt)
    
    ax[1].plot(trace_ex.delay, trace_ex, 'ko',label='Data')
    ax[1].plot(np.linspace(delay_limits[0],delay_limits[1],50), rise_fit, 'red',label='Fit')
    #ax[1].plot(I_res.delay, rise, 'red')
    ax[1].set_xlabel('Delay, fs')
    ax[1].set_ylabel('Norm. Int.')
    ax[1].axvline(0, color = 'grey', linestyle = 'dashed')
    
    ax[1].set_xlim([-150, 150]) 
    ax[1].set_ylim(-.1,1.05)
    #ax[1].axvline(30)
    ax[1].legend(frameon=False)
    print(fr't0 = {popt[0]:.1f} +/- {perr[0]:.1f} fs')
    print(fr'width = {popt[1]:.1f} +/- {perr[1]:.1f} fs')
    
    return rise_fit

#%% Useful Functions and Definitions for Plotting Data

def save_figure(fig, name, image_format):
    
    fig.savefig(name + '.'+ image_format, bbox_inches='tight', format=image_format)

def plot_momentum_maps(I, E, E_int, delays=None, delay_int=None, fig=None, ax=None, **kwargs):
    """
    Plot momentum maps at specified energies and delays with optional layout and styling.
    
    Parameters:
    - I: xarray dataset (e.g., I_diff or I_res).
    - E: list of energies.
    - E_int: total energy integration width (float).
    - delays: list of delays (same length as E). Ignored if no 'delay' in data.
    - delay_int: integration window for delays (float). Ignored if no 'delay' in data.
    
    Optional kwargs:
    - fig: matplotlib figure object (optional).
    - ax: array of axes (optional). If not provided, subplots are created.
    - cmap: colormap (default 'viridis').
    - scale: list [vmin, vmax] (default [0, 1]).
    - panel_labels: list of text labels (e.g., ['(a)', '(b)', ...]) (optional).
    - label_positions: tuple (x, y) in axes fraction coords for labels (default: (0.03, 0.9)).
    - fontsize: int for all text (default: 14).
    - figsize: tuple for fig size (only used if fig is created here).
    - nrows, ncols: layout for auto subplot creation (optional).
    - colorbar
    
    Returns:
    - fig, ax, im (image handle for colorbar)
    """
    E = np.atleast_1d(E)

    has_delay = "delay" in I.dims

    delays = np.atleast_1d(delays)
    
    if has_delay:
        delays = np.atleast_1d(delays)
        if len(delays) != len(E):
            delays = np.resize(delays, len(E))
    else:
        # Static data – ignore delays entirely
        delays = [None] * len(E)

    cmap = kwargs.get("cmap", "viridis")
    scale = kwargs.get("scale", [0, 1])
    panel_labels = kwargs.get("panel_labels", False)
    label_positions = kwargs.get("label_positions", (0.0, 1.1))
    fontsize = kwargs.get("fontsize", 14)
    figsize = kwargs.get("figsize", (8, 5))
    nrows = kwargs.get("nrows", 1)
    ncols = kwargs.get("ncols", int(np.ceil(len(E) / nrows)))
    colorbar = kwargs.get("colorbar", False)

    if ax is None or fig is None:
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        ax = np.ravel(ax)
    else:
        ax = [ax]
        
    for i in range(len(E)):
        frame = get_momentum_map(I, E[i], E_int, delays[i], delay_int)
        frame = frame / frame.max()

        im = frame.plot.imshow(
            ax=ax[i],
            vmin=scale[0],
            vmax=scale[1],
            cmap=cmap,
            add_colorbar=False
        )

        # Consistent formatting for all axes
        ax[i].set_aspect(1)
        ax[i].set_xlim(-2, 2)
        ax[i].set_ylim(-2, 2)
        ax[i].set_xticks(np.arange(-2, 2.2, 1))
        for label in ax[i].xaxis.get_ticklabels()[1::2]:
            label.set_visible(False)

        ax[i].set_yticks(np.arange(-2, 2.1, 1))
        for label in ax[i].yaxis.get_ticklabels()[1::2]:
            label.set_visible(False)

        ax[i].set_xlabel('$k_x$, $\AA^{-1}$', fontsize=fontsize)
        ax[i].set_ylabel('$k_y$, $\AA^{-1}$', fontsize=fontsize)
        ax[i].set_title(f"$E$ = {E[i]:.2f} eV", fontsize=fontsize)

        # Optional panel label
        if panel_labels is True:
            labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
            ax[i].text(
                label_positions[0], label_positions[1],
                labels[i],
                transform=ax[i].transAxes,
                fontsize=fontsize,
                fontweight='regular'
            )
    
    if colorbar == True:
        # Add colorbar
        cbar_ax = fig.add_axes([1.02, 0.36, 0.025, 0.35])
        cbar = fig.colorbar(im, cax=cbar_ax, ticks=[0, 1])
        cbar.ax.set_yticklabels(['min', 'max'])

    fig.tight_layout()

    return fig, ax, im

def plot_kx_frame(I_res, ky, ky_int, delays, delay_int, fig=None, ax=None, **kwargs):
    """
    Plot time traces of momentum frame at specified kx for multiple energies.
    
    Parameters:
    - I_res: xarray dataset (e.g., I_diff or I_res).
    - E_list: list of energies for which to plot time traces.
    - E_int: energy integration width (float).
    - kx: the specific kx value to plot.
    - kx_int: kx integration window (float).
    - delays: list of delay values.
    - delay_int: delay integration window (float).
    
    Optional kwargs:
    - fig: matplotlib figure object (optional).
    - ax: axis (optional). If not provided, a new axis is created.
    - cmap: colormap (default 'viridis').
    - norm_trace: whether to normalize the trace (default: True).
    - subtract_neg: whether to subtract the negative delays (default: True).
    - neg_delays: the range of negative delays to subtract (default: (-3, 0)).
    - fontsize: int for all text (default: 14).
    - nrows, ncols: layout for auto subplot creation (optional).
    - colorbar
    """
    delays = np.atleast_1d(delays)

    nrows = kwargs.get("nrows", 1)
    ncols = kwargs.get("ncols", int(np.ceil(len(delays) / nrows)))
    figsize = kwargs.get("figsize", (8, 5))
    fontsize = kwargs.get("fontsize", 14)
    cmap = kwargs.get("cmap", "viridis")
    scale = kwargs.get("scale", [0, 1])
    energy_limits=kwargs.get("energy_limits", (1,3))
    E_enhance = kwargs.get("E_enhance", None)

    if ax is None or fig is None:
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        ax = np.ravel(ax)
    else:
        ax = [ax]
    
    # Loop over the energy list to plot time traces at each energy
    for i, delay in enumerate(delays):
        # Get the frame for the given energy, kx, and delay
        kx_frame = get_kx_E_frame(I_res, ky, ky_int, delay, delay_int)
        if E_enhance is not None:    
            kx_frame = enhance_features(kx_frame, E_enhance, factor = 0, norm = True)
            ax[i].axhline(E_enhance, linestyle = 'dashed', color = 'black', linewidth = 1)

        im = kx_frame.T.plot.imshow(ax=ax[i], cmap=cmap, add_colorbar=False, vmin=scale[0], vmax=scale[1]) #kx, ky, t
        
        #ax[2].set_aspect(1)
        ax[i].set_xticks(np.arange(-2,2.2,1))
        for label in ax[i].xaxis.get_ticklabels()[1::2]:
            label.set_visible(False)
        ax[i].set_yticks(np.arange(-2,4.1,.25))
        for label in ax[i].yaxis.get_ticklabels()[1::2]:
            label.set_visible(False)
        ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax[i].set_xlabel('$k_x$, $\AA^{-1}$', fontsize = 18)
        ax[i].set_ylabel('$E - E_{VBM}, eV$', fontsize = 18)
        ax[i].set_title(f'$k_y$ = {ky} $\pm$ {ky_int/2} $\AA^{{-1}}$', fontsize = 18)
        ax[i].tick_params(axis='both', labelsize=16)
        ax[i].set_xlim(-2,2)
        ax[i].set_ylim(energy_limits[0], energy_limits[1])
        #ax[i].text(-1.9, 2.7,  f"$\Delta$t = {delay} $\pm$ {delay_int/2:.0f} fs", size=14)
    
    # Adjust layout
    fig.tight_layout()
    
    return fig, ax, im

def plot_ky_frame(I_res, kx, kx_int, delays, delay_int, fig=None, ax=None, **kwargs):
    """
    Plot time traces of momentum frame at specified kx for multiple energies.
    
    Parameters:
    - I_res: xarray dataset (e.g., I_diff or I_res).
    - E_list: list of energies for which to plot time traces.
    - E_int: energy integration width (float).
    - ky: the specific kx value to plot.
    - ky_int: kx integration window (float).
    - delays: list of delay values.
    - delay_int: delay integration window (float).
    
    Optional kwargs:
    - fig: matplotlib figure object (optional).
    - ax: axis (optional). If not provided, a new axis is created.
    - cmap: colormap (default 'viridis').
    - norm_trace: whether to normalize the trace (default: True).
    - subtract_neg: whether to subtract the negative delays (default: True).
    - neg_delays: the range of negative delays to subtract (default: (-3, 0)).
    - fontsize: int for all text (default: 14).
    - nrows, ncols: layout for auto subplot creation (optional).
    - colorbar
    """
    nrows = kwargs.get("nrows", 1)
    ncols = kwargs.get("ncols", int(np.ceil(len(delays) / nrows)))
    figsize = kwargs.get("figsize", (8, 5))
    fontsize = kwargs.get("fontsize", 14)
    cmap = kwargs.get("cmap", "viridis")
    scale = kwargs.get("scale", [0, 1])
    energy_limits=kwargs.get("energy_limits", (1,3))

    delays = np.atleast_1d(delays)
    
    if ax is None or fig is None:
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        ax = np.ravel(ax)
    else:
        ax = [ax]
    
    # Loop over the energy list to plot time traces at each energy
    for i, delay in enumerate(delays):
        # Get the frame for the given energy, kx, and delay
        ky_frame = get_ky_E_frame(I_res, kx, kx_int, delay, delay_int)
        ky_frame = enhance_features(ky_frame, 0.9, factor = 0, norm = True)
        ky_frame.T.plot.imshow(ax=ax[i], cmap=cmap, add_colorbar=False, vmin=scale[0], vmax=scale[1]) #kx, ky, t
        
        #ax[2].set_aspect(1)
        ax[i].set_xticks(np.arange(-2,2.2,1))
        for label in ax[i].xaxis.get_ticklabels()[1::2]:
            label.set_visible(False)
        ax[i].set_yticks(np.arange(-2,4.1,.25))
        for label in ax[i].yaxis.get_ticklabels()[1::2]:
            label.set_visible(False)
        ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax[i].set_xlabel('$k_y$, $\AA^{-1}$', fontsize = 18)
        ax[i].set_ylabel('$E - E_{VBM}, eV$', fontsize = 18)
        ax[i].set_title(f'$k_x$ = {kx} $\pm$ {kx_int/2} $\AA^{{-1}}$', fontsize = 18)
        ax[i].tick_params(axis='both', labelsize=16)
        ax[i].set_xlim(-2,2)
        ax[i].set_ylim(energy_limits[0], energy_limits[1])
        ax[i].text(-1.9, 2.7,  f"$\Delta$t = {delay} $\pm$ {delay_int/2:.0f} fs", size=14)
        ax[i].axhline(0.9, linestyle = 'dashed', color = 'black', linewidth = 1)
    
    # Adjust layout
    fig.tight_layout()
    
    return fig, ax

def plot_time_traces(I_res, E, E_int, k, k_int, norm_trace=True, subtract_neg=True, neg_delays=(-500, -150), fig=None, ax=None, **kwargs):
    """
    Plot time traces at a specific energy and momentum coordinates with optional styling.
    
    Parameters:
    - I_res: xarray dataset (e.g., I_diff or I_res).
    - E: list of energies for the time trace plot.
    - kx, ky: momentum coordinates at which to extract the time trace.
    - kx_int, ky_int: momentum integration widths for the time trace.
    - E_int: energy integration width.
    - norm_trace: whether to normalize the trace (default: True).
    - subtract_neg: whether to subtract the mean of the negative delays (default: True).
    - neg_delays: range for background subtraction (default: (-200, -50)).
    - fig: matplotlib figure object (optional).
    - ax: axes object (optional).
    - panel_labels: list of panel labels (e.g., ['(a)', '(b)', ...]).
    - label_positions: position for panel labels (default: (0.03, 0.9)).
    - fontsize: font size for all text (default: 14).
    
    Returns:
    - fig, ax (figure and axis objects).
    """
    fontsize = kwargs.get("fontsize", 14)
    colors = kwargs.get("colors", ['Black', 'Maroon', 'Blue', 'Purple', 'Green', 'Grey'])
    legend = kwargs.get("legend", True)

    (kx, ky), (kx_int, ky_int) = k, k_int

    E = np.atleast_1d(E)
    kx = np.atleast_1d(kx)

    if len(E) > len(kx):
        kx = np.resize(kx, len(E))
    
    if len(E) < len(kx):
        E = np.resize(E, len(kx))        
    
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    for i, (E, kx) in enumerate(zip(E, kx)):
        trace = get_time_trace(I_res, E, E_int, (kx, ky), (kx_int, ky_int), norm_trace=norm_trace, subtract_neg=subtract_neg, neg_delays=neg_delays)
        
        ax.plot(trace.coords['delay'].values, trace.values, label=f'E = {E:.2f} eV', color = colors[i], linewidth=2)
    
    # Formatting
    ax.set_xlabel('Delay, fs', fontsize=fontsize)
    ax.set_ylabel('Intensity' , fontsize=fontsize)
    ax.set_xlim(I_res.delay[1], I_res.delay[-1])

    if legend is True:
        ax.legend(fontsize=fontsize, frameon=False)
    
    fig.tight_layout()

    return fig, ax

def plot_waterfall(I_res, kx, kx_int, ky, ky_int, fig=None, ax=None, **kwargs):
    """
    Plot the waterfall of intensity across both kx and ky slices.

    Parameters:
    - I_res: xarray dataset with intensity data.
    - kx: kx value around which to extract the data (in 1/Å).
    - kx_int: integration window for kx (in 1/Å).
    - ky: ky value around which to extract the data (in 1/Å).
    - ky_int: integration window for ky (in 1/Å).

    Optional kwargs:
    - cmap: colormap for the waterfall plot (default 'viridis').
    - scale: [vmin, vmax] for normalization (default [0, 1]).
    - xlabel: label for the x-axis (default 'Delay, ps').
    - ylabel: label for the y-axis (default 'Intensity').
    - fontsize: font size for the labels (default: 14).
    - figsize: figure size (default (10, 6)).

    Returns:
    - fig, ax: figure and axis handles for the plot.
    """

    cmap = kwargs.get("cmap", "viridis")
    scale = kwargs.get("scale", [0, 1])
    xlabel = kwargs.get("xlabel", 'Delay, ps')
    ylabel = kwargs.get("ylabel", 'Intensity')
    fontsize = kwargs.get("fontsize", 14)
    figsize = kwargs.get("figsize", (10, 6))
    energy_limits=kwargs.get("energy_limits", (1,3))
    subtract_neg = kwargs.get("subtract_neg", False)
    neg_delays = kwargs.get("neg_delays", [-250, -120])
    
    d1, d2 = neg_delays[0], neg_delays[1]
    
    if ax is None or fig is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    waterfall = get_waterfall(I_res, kx, kx_int, ky, ky_int)

    if subtract_neg is True : 
        waterfall = waterfall - waterfall.loc[{"delay":slice(d1,d2)}].mean(dim='delay')

    waterfall = enhance_features(waterfall, energy_limits[0], factor = 0, norm = True)
    
    waterfall.plot.imshow(ax = ax, vmin = scale[0], vmax = scale[1], cmap = cmap, add_colorbar=False)
    #waterfall.plot.imshow(ax = ax, cmap = cmap, add_colorbar=False)
   
    ax.set_xlabel('Delay, fs', fontsize = 18)
    ax.set_ylabel('E - E$_{VBM}$, eV', fontsize = 18)
    ax.set_yticks(np.arange(-1,3.5,0.25))
    ax.set_xlim(I_res.delay[1], I_res.delay[-1])
    ax.set_ylim(energy_limits[0], energy_limits[1])
    ax.set_title('$k$-Integrated')
    ax.axhline(energy_limits[0], linestyle = 'dashed', color = 'black', linewidth = 1)
    
    for label in ax.yaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
    hor = I_res.delay[-1] - I_res.delay[1]
    ver =  energy_limits[1] - energy_limits[0]
    aspra = hor/ver 
    #ax[1].set_aspect(aspra)
    ax.set_aspect("auto")

    # Adjust layout to avoid overlap
    fig.tight_layout()

    return fig, ax

def add_rect(dim1, dim1_int, dim2, dim2_int, ax, **kwargs):
        edgecolor = kwargs.get("edgecolor", None)
        facecolor = kwargs.get("facecolor", 'grey')
        alpha = kwargs.get("alpha", 0.5)

        rect = (Rectangle((dim1-dim1_int/2, dim2-dim2_int/2), dim1_int, dim2_int , linewidth=.5,\
                             edgecolor=edgecolor, facecolor=facecolor, alpha = alpha))
        ax.add_patch(rect) #Add rectangle to plot
        
        return rect

def overlay_bz(shape_type, a, b, ax, color, **kwargs):

    """
    Overlays a custom Brillouin zone polygon on an imshow plot.

    Parameters:
    - shape: 
    - a, b: lattice constants in x and y direction (used to scale Γ, X, Y point labels).
    - ax: matplotlib axes object to draw on.
    - color: color for the polygon edge.
    """

    repeat=kwargs.get("repeat", 0)
    rotation_deg=kwargs.get("rotation_deg", 0)

    def make_rect_bz(a, b):
        X = np.pi / a
        Y = np.pi / b
        return [(-X, -Y), (X, -Y), (X, Y), (-X, Y)]

    def make_hex_bz(a):
        radius = 4 * np.pi / (3 * a)
        angles = np.linspace(0, 2*np.pi, 7)[:-1] + np.pi/6
        return [(radius * np.cos(a), radius * np.sin(a)) for a in angles]

    def rotate_shape(coords, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        return [(x*cos_a - y*sin_a, x*sin_a + y*cos_a) for x, y in coords]
        
    # Choose the shape
    if shape_type == 'rectangular':
        base_shape = make_rect_bz(a, b)
        dx, dy = 2 * np.pi / a, 2 * np.pi / b
    elif shape_type == 'hexagonal':
        base_shape = make_hex_bz(a)
        # approximate hexagon repetition spacing
        dx, dy = 4 * np.pi / (3 * a), 4 * np.pi / (3 * a)
    else:
        raise ValueError(f"Unsupported shape type: {shape_type}")
    
    # Rotate shape
    shape = rotate_shape(base_shape, rotation_deg)

    # Translate shape to center
    center = (0,0)
    cx, cy = center

    if shape_type == 'hexagonal':
        # Use reciprocal lattice vectors for proper hex tiling
        #b1 = np.array([4 * np.pi / (3 * a), 0])
        #b2 = np.array([-2 * np.pi / (3 * a), 2 * np.pi / (np.sqrt(3) * a)])
        b1 = (2 * np.pi / a) * np.array([1, -1 / np.sqrt(3)])
        b2 = (2 * np.pi / a) * np.array([0, 2 / np.sqrt(3)])
        for i in range(-repeat, repeat + 1):
            for j in range(-repeat, repeat + 1):
                offset = cx + i * b1[0] + j * b2[0], cy + i * b1[1] + j * b2[1]
                translated_shape = [(x + offset[0], y + offset[1]) for x, y in shape]
                patch = Polygon(translated_shape, closed=True, edgecolor=color,
                                facecolor='none', linewidth=2, alpha=0.75)
                ax.add_patch(patch)
                if i == 0 and j == 0 and np.allclose(center, (0, 0)):
                    ax.plot(0, 0, 'ko', markersize=4, alpha=0.75)
                    ax.text(0.1, 0.1, r'$\Gamma$', size=12)

    else:
        # Rectangular grid tiling
        for i in range(-repeat, repeat + 1):
            for j in range(-repeat, repeat + 1):
                offset_x = cx + i * dx
                offset_y = cy + j * dy
                translated_shape = [(x + offset_x, y + offset_y) for x, y in shape]
                patch = Polygon(translated_shape, closed=True, edgecolor=color,
                                facecolor='none', linewidth=2, alpha=0.75)
                ax.add_patch(patch)
                if i == 0 and j == 0 and np.allclose(center, (0, 0)):
                    ax.plot(0, 0, 'ko', markersize=4, alpha=0.75)
                    ax.text(0.1, 0.1, r'$\Gamma$', size=12)

    #bz = Rectangle((0-X, 0-Y), 2*X, 2*Y , linewidth=2, edgecolor=color, facecolor='none', alpha = 0.75)

    #ax.add_patch(bz) #Add bz to plot
    #ax.plot(0,0, 'ko', markersize = 4, alpha = 0.75)
    #ax.plot([0, 0], [Y-0.1, Y+0.1], color = 'black', alpha = 0.75)
    #ax.plot([-X-0.1, -X+0.1], [0, 0], color = 'black', alpha = 0.75)
    #ax.text(-X-0.45, 0, 'X', size=12)
    #ax.text(0, Y+0.15, 'Y', size=12)
    #ax.text(0.1, 0.1, fr'$\Gamma$', size=12)

#I_sum, I_pos, I_pos_sum, I_neg, I_neg_sum = get_data_chunks([-180,-100], t0, ax_delay_offset) #Get the Neg and Pos delay time arrays
def custom_colormap(CMAP, lower_portion_percentage):
    # create a colormap that consists of
    # - 1/5 : custom colormap, ranging from white to the first color of the colormap
    # - 4/5 : existing colormap
    
    # set upper part: 4 * 256/4 entries
    CMAP = plt.get_cmap(CMAP)
    upper =  CMAP(np.arange(256))
    upper = upper[56:,:]
    #upper = upper[0:,:]

    # - initialize all entries to 1 to make sure that the alpha channel (4th column) is 1
    lower_portion = int(1/lower_portion_percentage) - 1
    
    lower = np.ones((int(200/lower_portion),4))
    # - modify the first three columns (RGB):
    #   range linearly between white (1,1,1) and the first color of the upper colormap
    for i in range(3):
      lower[:,i] = np.linspace(1, upper[0,i], lower.shape[0])
    
    # combine parts of colormap
    cmap = np.vstack(( lower, upper ))
    
    # convert to matplotlib colormap
    custom_cmap = mpl.colors.ListedColormap(cmap, name='custom', N=cmap.shape[0])
    
    return custom_cmap

def create_custom_diverging_colormap(map1, map2):
    # Create the negative part from seismic (from 0.5 to 1 -> blue to white)
    seismic = plt.get_cmap(map1)
    seismic_colors = seismic(np.linspace(0, 1, 128))  # -1 to 0

    # Create the positive part from viridis (from 0 to 1)
    viridis = plt.get_cmap(map1)
    viridis = cmap_LTL
    viridis_colors = viridis(np.linspace(0, 1, 128))  # 0 to 1

    # Combine both
    combined_colors = np.vstack((seismic_colors[::-1], viridis_colors))

    # Create a new colormap
    custom_colormap = LinearSegmentedColormap.from_list('seismic_viridis', combined_colors)

    return custom_colormap

#%% Functions For Fitting Data: Time Traces

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
def convolved_signal_1(t, signal_function, sigma_IRF, *params):
    dt = np.mean(np.diff(t))  # Time step
    signal = signal_function(t, *params)  # Compute signal
    irf = IRF(t - t[len(t)//2], sigma_IRF)  # Shift IRF to center
    irf /= np.sum(irf) * dt  # Normalize IRF
    convolved = fftconvolve(signal, irf, mode='same') * dt  # Convolve with IRF
    return convolved

def convolved_signal(t, signal_function, sigma_IRF, *params):
    dt = np.mean(np.diff(t))

    # Extend the time axis on both sides to avoid edge effects
    pad_width = int(5 * sigma_IRF / dt)  # enough padding for Gaussian tail
    t_pad = np.linspace(t[0] - pad_width * dt, t[-1] + pad_width * dt, len(t) + 2 * pad_width)

    # Evaluate signal on the extended time axis
    signal_ext = signal_function(t_pad, *params)

    # Create centered Gaussian IRF
    irf = np.exp(-((t_pad - np.median(t_pad)) ** 2) / (2 * sigma_IRF ** 2))
    irf /= np.sum(irf) * dt  # Normalize area under the IRF to 1

    # Convolve using FFT
    conv_ext = fftconvolve(signal_ext, irf, mode='same') * dt

    # Trim back to original t range
    convolved = conv_ext[pad_width : -pad_width]
    
    return convolved

def make_convolved_model(base_model, t, sigma_IRF):
    """
    Returns a callable f(t, *params) which evaluates the convolved model at time t.
    """
    dt = np.mean(np.diff(t))
    pad_width = int(5 * sigma_IRF / dt)
    t_pad = np.linspace(t[0] - pad_width * dt, t[-1] + pad_width * dt, len(t) + 2 * pad_width)

    # Centered Gaussian IRF
    irf = np.exp(-((t_pad - np.median(t_pad)) ** 2) / (2 * sigma_IRF ** 2))
    irf /= np.sum(irf) * dt

    def model(t_fit, *params):
        signal_ext = base_model(t_pad, *params)
        conv_ext = fftconvolve(signal_ext, irf, mode='same') * dt
        return conv_ext[pad_width : -pad_width]

    return model

model_dict = {
        'monoexp': monoexp,
        'exp_rise_monoexp_decay': exp_rise_monoexp_decay,
        'biexp': biexp,
        'exp_rise_biexp_decay': exp_rise_biexp_decay
}


def fit_time_trace(fit_model, delay_axis, time_trace, p0, bounds, convolve=False, sigma_IRF=None):
    """
    Fit a time trace using a specified model, optionally convolved with an IRF.

    Parameters:
    - fit_model (str): Name of the model ('monoexp', 'exp_rise_monoexp_decay')
    - delay_axis (array): Time delay values
    - time_trace (array): Measured time trace
    - p0 (tuple/list): Initial guess for fit parameters
    - bounds (2-tuple): Bounds for fit parameters ((lower_bounds), (upper_bounds))
    - convolve (bool): Whether to convolve the model with an IRF
    - sigma_IRF (float): Width of the Gaussian IRF (if convolve=True)

    Returns:
    - popt: Optimal parameters from curve_fit
    - pcov: Covariance of the parameters
    - fit_curve: Evaluated fit curve
    """

    if fit_model not in model_dict:
        raise ValueError(f"Unsupported model: {fit_model}")

    base_model = model_dict[fit_model]

    if convolve:
        if sigma_IRF is None:
            raise ValueError("sigma_IRF must be provided if convolve=True")
        def model_func(t, *params):
            return convolved_signal(t, base_model, sigma_IRF, *params)
    else:
        model_func = base_model

    if convolve:
        model_func = make_convolved_model(base_model, delay_axis, sigma_IRF)
    else:
        model_func = base_model

    popt, pcov = curve_fit(model_func, delay_axis, time_trace, p0=p0, bounds=bounds)
    fit_curve = model_func(delay_axis, *popt)

    return popt, pcov, fit_curve

import numpy as np

def print_fit_results(model_name, popt, pcov):
    """
    Print fit parameters and uncertainties based on model name.
    """

    def build_param_list(popt, perr, param_names):
        
        return {
            name: val for name, val in zip(param_names, popt)
        } | {
            "errors": {name: err for name, err in zip(param_names, perr)}
        }

    model_param_names = {
        'monoexp': ['A', 'tau'],
        'biexp': ['A', 'tau1', 'B', 'tau2'],
        'exp_rise_monoexp_decay': ['A', 'tau_rise', 'tau_decay'],
        'exp_rise_biexp_decay': ['C', 'tau_rise', 'A', 'tau_decay1', 'tau_decay2']
        # Add more models here as needed
    }

    if model_name not in model_param_names:
        raise ValueError(f"Unsupported model: {model_name}")

    param_names = model_param_names[model_name]
    errors = np.sqrt(np.diag(pcov))

    print(f"\nFit Results for model: {model_name}")
    print("-" * 40)
    for name, val, err in zip(param_names, popt, errors):
        print(f"{name:10s} = {val:10.4f} ± {err:7.4f}")
    print("-" * 40)
    
    return build_param_list(popt, errors, param_names)

cmap_LTL = custom_colormap('viridis', 0.2)
cmap_LTL2 = create_custom_diverging_colormap('Blues', 'viridis')
