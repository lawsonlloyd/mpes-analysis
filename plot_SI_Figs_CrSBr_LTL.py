#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 07:55:10 2025

@author: lawsonlloyd
"""
###################################################
### Extra helpful figures, + SI figures for LTL ###
###################################################

#%% Plot Dynamics: Extract Traces At Different Energies and Momenta: Distinct k Points

save_figure = False
figure_file_name = 'distinct_K_Points'
image_format = 'pdf'

subtract_neg, neg_delays = True, [-110,-70] #If you want to subtract negative time delay baseline
norm_trace = False

E_ex, E_cbm, E_int = 1.25, 2.05, 0.2

(kx, ky), (kx_int, ky_int) = ((-2*X, -1.5*X, -X, -X/2, 0.0, X/2, X, 1.5*X, 2*X), 0), (.25, .25) # Central (kx, ky) point and k-integration
colors = ['crimson', 'purple', 'blue', 'darkblue', 'dodgerblue', 'yellow', 'darkorange', 'lightcoral', 'black'] #colors for plotting the traces

#######################
### Do the Plotting ###
#######################

fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 1.5], 'height_ratios':[1, 1]})
fig.set_size_inches(8, 6, forward=False)
ax = ax.flatten()

# Momenutm Map
mpes.plot_momentum_maps(
    I_res, 1.3, E_int=0.2, delays=500, delay_int=1000,
    fig = fig, ax = ax[0],
    cmap='BuPu',
    panel_labels=False, fontsize=16,
    nrows=2, figsize=(8, 6)
)

# Plot kx frame
mpes.plot_kx_frame(
    I_res, 0, 0.5, delays=[500], delay_int=1000,
    subtract_neg=subtract_neg, neg_delays=neg_delays,
    fig = fig, ax = ax[2],
    cmap = 'BuPu'
)

### for the Exciton

# Plot time traces
mpes.plot_time_traces(
    I_res, E_ex, E_int, (kx, ky), (kx_int, ky_int),
    norm_trace=True, subtract_neg=True, neg_delays=[-110,-70],
    fig = fig, ax = ax[1],
    colors = colors, legend=False, fontsize=16
)

### for the CBM

# Plot time traces
mpes.plot_time_traces(
    I, E_cbm, E_int, (kx, ky), (kx_int, ky_int),
    norm_trace=True, subtract_neg=True, neg_delays=[-110,-70],
    fig = fig, ax = ax[3],
    colors = colors, legend=False, fontsize=16
)

ax[1].set_title(f"Exciton")
ax[3].set_title(f"CBM")

for i in np.arange(len(kx)):

    mpes.add_rect(kx[i], kx_int, E_ex, E_int, colors[i], ax[2])
    mpes.add_rect(kx[i], kx_int, E_cbm, E_int, colors[i], ax[2])
    mpes.add_rect(kx[i], kx_int, ky, ky_int, colors[i], ax[0])


# ax[2].text(0+0.04, 2.75, f"$\Gamma$", size=12)
# ax[2].text(-X+0.04, 2.75, f"$X$", size=12)
# ax[2].text(X+0.04, 2.75, f"$X$", size=12)
# ax[2].text(-2*X+0.04, 2.75, f"$\Gamma$", size=14)
# ax[2].text(2*X-0.35, 2.75, f"$\Gamma$", size=14)
# ax[2].text(-1.82, 2.475,  f"$\Delta$t = {delay} fs", size=13)

k_point_label = ['$\Gamma_{-1,0}$', '$-X$', '$\Gamma_{0}$', '$+X$', '$\Gamma_{1,0}$']
k_point_label = ['$\Gamma_{-1,0}$', '$-3X/2$', '$-X$','$-X/2$', '$\Gamma_{0}$', '$+X/2$', '$+X$', '$+3X/2$', '$\Gamma_{1,0}$']

fig.text(.03, 0.975, "(a)", fontsize = 18, fontweight = 'regular')
fig.text(.03, 0.5, "(c)", fontsize = 18, fontweight = 'regular')
fig.text(.425, 0.975, "(b)", fontsize = 18, fontweight = 'regular')
fig.text(.425, 0.5, "(d)", fontsize = 18, fontweight = 'regular')

params = {'lines.linewidth' : 2.5, 'axes.linewidth' : 2, 'axes.labelsize' : 20, 
              'xtick.labelsize' : 16, 'ytick.labelsize' : 16, 'axes.titlesize' : 20, 'legend.fontsize' : 16}
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update(params)

fig.tight_layout()

#ax[1].set_tick_params(axis='both', labelsize=16)
#plt.gca().set_aspect(200)

if save_figure is True:
    fig.savefig(figure_file_name + '.'+ image_format, bbox_inches='tight', format=image_format) 

#%% Plot Waterfall With Different Traces

E_trace, E_int = [2.05, 2.3, 2.5, 2.7, 2.9, 3.1], .1 # Energies for Plotting Time Traces ; 1st Energy for MM
(kx, ky), (kx_int, ky_int) = (0, 0), (4, 4) # Central (kx, ky) point and k-integration

colors = ['crimson', 'violet', 'purple', 'midnightblue', 'orange', 'grey'] #colors for plotting the traces
colors = colors[::-1]
E_trace = E_trace[::-1]

fig, axs = plt.subplots(2, 1)
fig.set_size_inches(8, 8, forward=False)
axs = axs.flatten()

# Plot waterfall
mpes.plot_waterfall(
    I_res, kx, kx_int, ky, ky_int,
    fig = fig, ax = axs[0],
    cmap= cmocean.cm.balance, scale=[0,1]
)

# Plot time traces
mpes.plot_time_traces(
    I, E_trace, E_int, (kx, ky), (kx_int, ky_int),
    norm_trace=False, subtract_neg=True, neg_delays=[-110,-70],
    fig = fig, ax = axs[1],
    colors = colors,
    fontsize=16
)

#%% Extract k-dispersion and Eb momentum-depenence

save_figure = False
figure_file_name = 'eb-dispersion_120k'
image_format = 'pdf'

E_trace, E_int = [1.35, 2.05], .1 # Energies for Plotting Time Traces ; 1st Energy for MM
(kx, ky), (kx_int, ky_int) = (0.0, 0.0), (0.2, 0.2) # Central (kx, ky) point and k-integration
delay, delay_int = 500, 1000
subtract_neg  = True

kx_frame = mpes.get_kx_E_frame(I_res, ky, ky_int, delay, delay_int)
    
e1, e2 = 1.1, 3
k1, k2 = -1.83, 1.8
ax_kx = I.loc[{"kx":slice(k1,k2)}].kx.values
kx_fits = np.zeros((len(kx_frame.loc[{"kx":slice(k1,k2)}].kx.values),2))
kx_fits_error = np.zeros(kx_fits.shape)
eb_kx = np.zeros(kx_fits.shape[0])
eb_kx_error = np.zeros(kx_fits.shape[0])

i = 0
for k in ax_kx:
    kx_int = 0.2
    
    kx_edc = kx_frame.loc[{"kx":slice(k-kx_int/2,k+kx_int/2)}].mean(dim="kx")
    kx_edc = kx_edc/np.max(kx_edc.loc[{"E":slice(0.8,3)}])
    
    ##### X and CBM ####
    p0 = [1, 0.3,  1.35, 2.1,  0.2, 0.2, 0] # Fitting params initial guess [amp, center, width, offset]
    bnds = ((0.5, 0.1, 1.1, 1.9, 0.1, 0.1, 0), (1.5, 0.7, 1.5, 2.3, 0.9, 0.9, .3))
    
    popt, pcov = curve_fit(two_gaussians, kx_edc.loc[{"E":slice(e1,e2)}].E.values, kx_edc.loc[{"E":slice(e1,e2)}].values, p0, method=None, bounds = bnds)
    perr = np.sqrt(np.diag(pcov))
    g, g1, g2, offset = two_gaussians_report(kx_edc.loc[{"E":slice(0,3)}].E.values, *popt)
    kx_fits[i,0] = popt[2]
    kx_fits[i,1] = popt[3]
    kx_fits_error[i,0] = perr[2]
    kx_fits_error[i,1] = perr[3]
    eb_kx[i] = kx_fits[i,1] - kx_fits[i,0]
    eb_kx_error[i] = np.sqrt(perr[3]**2+perr[2]**2)
    
    if k < 0.02 and k > -0.7:
        kx_fits[i,:] = nan
        eb_kx[i] = nan
        eb_kx_error[i] = nan
        kx_fits_error[i,:] = nan
    i += 1

## Plot ##

fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1], 'height_ratios':[1]})
fig.set_size_inches(9, 3.5, forward=False)
ax = ax.flatten()

##### X and CBM ####
#Fit to Time an kx integrated
kx, kx_int = 0, 3.8
kx_edc = kx_frame.loc[{"kx":slice(kx-kx_int/2,kx+kx_int/2)}].mean(dim="kx")
kx_edc = kx_edc/np.max(kx_edc.loc[{"E":slice(0.8,3)}])
p0 = [1, 0.3,  1.35, 2.1,  0.2, 0.2, 0] # Fitting params initial guess [amp, center, width, offset]
bnds = ((0.5, 0.1, 1.1, 1.9, 0.1, 0.1, 0), (1.5, 0.7, 1.5, 2.3, 0.9, 0.9, .3))

popt, pcov = curve_fit(two_gaussians, kx_edc.loc[{"E":slice(e1,e2)}].E.values, kx_edc.loc[{"E":slice(e1,e2)}].values, p0, method=None, bounds = bnds)
perr = np.sqrt(np.diag(pcov))
g, g1, g2, offset = two_gaussians_report(kx_edc.loc[{"E":slice(0,3)}].E.values, *popt)
Eb = round(popt[3] - popt[2],3)
Eb_err = np.sqrt(perr[3]**2+perr[2]**2) 

mpes.plot_kx_frame(
    I_res, ky, ky_int, delay, delay_int,
    subtract_neg=subtract_neg, neg_delays=neg_delays,
    fig = fig, ax = ax[0], E_enhance = 1,
    cmap = cmap, scale=[0,1], energy_limits=[1,3]
)
#ax[0].set_aspect(1)
#ax[0].text(-1.9, 2.7,  f"$\Delta$t = {delay} $\pm$ {delay_int/2:.0f} fs", size=16)

ax[0].text(0+0.05, 2.75, f"$\Gamma$", size=18)
ax[0].text(-X+0.05, 2.75, f"$X$", size=18)
ax[0].text(X+0.05, 2.75, f"$X$", size=18)
ax[0].text(-2*X+0.05, 2.75, f"$\Gamma$", size=18)
ax[0].text(2*X+0.05, 2.75, f"$\Gamma$", size=18)

#ax[0].axhline(Ein, linestyle = 'dashed', color = 'black', linewidth = 1.5)
ax[0].axvline(0, linestyle = 'dashed', color = 'black', linewidth = 1.5)
ax[0].axvline(X, linestyle = 'dashed', color = 'black', linewidth = 1.5)
ax[0].axvline(-X, linestyle = 'dashed', color = 'black', linewidth = 1.5)
ax[0].axvline(2*X, linestyle = 'dashed', color = 'black', linewidth = 1.5)
ax[0].axvline(-2*X, linestyle = 'dashed', color = 'black', linewidth = 1.5)
#ax[0].axhline(popt[2], linestyle = 'dashed', color = 'k', linewidth = 1.5)
#ax[0].axhline(popt[3], linestyle = 'dashed', color = 'r', linewidth = 1.5)
ax[0].plot(ax_kx, kx_fits[:,0], 'o', color = 'black')
ax[0].plot(ax_kx, kx_fits[:,1], 'o', color = 'crimson')
#ax[0].fill_between(kx_frame.loc[{"kx":slice(k1,k2)}].kx.values, kx_fits[:,0] - kx_fits_error[:,0], kx_fits[:,0] + kx_fits_error[:,0], color = 'grey', alpha = 0.5)
#ax[0].fill_between(kx_frame.loc[{"kx":slice(k1,k2)}].kx.values, kx_fits[:,1] - kx_fits_error[:,1], kx_fits[:,1] + kx_fits_error[:,1], color = 'crimson', alpha = 0.5)

#ax[0].fill_between(I.E.values, I.kx.values - kx_int/2, I.kx.values + kx_int/2, color = 'pink', alpha = 0.5)
#mpes.add_rect()

rect = (Rectangle((kx-kx_int/2, .5), kx_int, 3, linewidth=2.5,\
                         edgecolor='purple', facecolor='purple', alpha = 0.3))
#if kx_int < 4:
    #ax[0].add_patch(rect) #Add rectangle to plot

# Plot the Eb
ax[1].fill_between(ax_kx, 1000*eb_kx - 1000*eb_kx_error, 1000*eb_kx + 1000*eb_kx_error, color = 'violet', alpha = 0.5)
ax[1].plot(ax_kx, 1000*eb_kx, color = 'darkviolet')
ax[1].set_xlim(-2,2)
ax[1].set_ylim(700,900)
ax[1].set_ylabel('$E_{b}, meV$', fontsize = 18)
ax[1].set_xlabel('$k_x$, $\AA^{-1}$', fontsize = 18)
ax[1].set_title(f"$E_b = {1000*np.nanmean(eb_kx):.0f} \pm {np.nanmean(1000*eb_kx_error):.0f}$ meV")
ax[1].set_title(f"Extracted $E_b$")
ax[1].set_aspect('auto')

fig.text(.02, 0.975, "(a)", fontsize = 20, fontweight = 'regular')
fig.text(.51, 0.975, "(b)", fontsize = 20, fontweight = 'regular')

fig.tight_layout()

if save_figure is True:
    mpes.save_figure(fig, name = f'{figure_file_name}', image_format = 'pdf')

print(f'The kx-avg Eb is {np.nanmean(1000*eb_kx)} +- {1000*np.nanstd(eb_kx)}')
print(f"{popt[2]:.3f} +- {perr[2]:.3f}")
print(f"{popt[3]:.3f} +- {perr[3]:.3f}")
print(f"{1000*Eb:.3f} +- {1000*Eb_err:.3f}")

#%% TEST: Excited State EDC Fitting to Extract DYNAMIC Exciton Binding Energy and Peak Positions

save_figure = False
figure_name = 'kx_excited'

E_trace, E_int = [1.35, 2.05], .1 # Energies for Plotting Time Traces ; 1st Energy for MM
(kx, ky), (kx_int, ky_int) = (0, 0), (3.8, 0.2) # Central (kx, ky) point and k-integration
delay, delay_int = 500, 100

kx_frame = mpes.get_kx_E_frame(I_res, ky, ky_int, delay, delay_int)
kx_frame = kx_frame/np.max(kx_frame.loc[{"E":slice(0.8,3)}])

kx_edc = kx_frame.loc[{"kx":slice(kx-kx_int/2,kx+kx_int/2)}].mean(dim="kx")
kx_edc = kx_edc/np.max(kx_edc.loc[{"E":slice(0.8,3)}])

##### X and CBM ####
p0 = [1, 0.3,  1.35, 2.1,  0.2, 0.2, 0] # Fitting params initial guess [amp, center, width, offset]
bnds = ((0.5, 0.1, 1.0, 1.5, 0.1, 0.1, 0), (1.5, 0.7, 1.5, 2.3, 0.9, 0.9, .3))

popt, pcov = curve_fit(two_gaussians, kx_edc.loc[{"E":slice(e1,e2)}].E.values, kx_edc.loc[{"E":slice(e1,e2)}].values, p0, method=None, bounds = bnds)
perr = np.sqrt(np.diag(pcov))
g, g1, g2, offset = two_gaussians_report(kx_edc.loc[{"E":slice(0,3)}].E.values, *popt)
Eb = round(popt[3] - popt[2],3)

fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1.5, 1], 'height_ratios':[1]})
fig.set_size_inches(10, 4, forward=False)
ax = ax.flatten()

mpes.plot_kx_frame(
    I_res, ky, ky_int, delay, delay_int,
    subtract_neg=subtract_neg, neg_delays=neg_delays,
    fig = fig, ax = ax[0], E_enhance = 1,
    cmap = cmap, scale=[0,1], energy_limits=[1,3]
)

ax[0].set_ylim(0.8,3)
ax[0].text(-1.9, 2.7,  f"$\Delta$t = {delay} fs", size=16)
#ax[0].axhline(Ein, linestyle = 'dashed', color = 'grey', linewidth = 1.5)
ax[0].axvline(0, linestyle = 'dashed', color = 'grey', linewidth = 1.5)
ax[0].axvline(X, linestyle = 'dashed', color = 'grey', linewidth = 1.5)
ax[0].axvline(-X, linestyle = 'dashed', color = 'grey', linewidth = 1.5)
ax[0].axvline(2*X, linestyle = 'dashed', color = 'grey', linewidth = 1.5)
ax[0].axvline(-2*X, linestyle = 'dashed', color = 'grey', linewidth = 1.5)
ax[0].axhline(popt[2], linestyle = 'dashed', color = 'k', linewidth = 1.5)
ax[0].axhline(popt[3], linestyle = 'dashed', color = 'r', linewidth = 1.5)
#ax[0].fill_between(I.E.values, I.kx.values - kx_int/2, I.kx.values + kx_int/2, color = 'pink', alpha = 0.5)

if kx_int < 4:
    mpes.add_rect(kx, kx_int, 2, 4, ax[0], alpha=0.3, facecolor='purple', edgecolor='purple')

#kx_edc.plot(ax=ax[1], color = 'purple', alpha = 0.8)
ax[1].plot(kx_edc.loc[{"E":slice(0,3)}].E.values, g, color='grey',linestyle = 'solid',linewidth = 3)
ax[1].plot(kx_edc.loc[{"E":slice(0,3)}].E.values, g1, color='black',linestyle = 'dashed')
ax[1].plot(kx_edc.loc[{"E":slice(0,3)}].E.values, g2, color='red',linestyle = 'dashed')
kx_edc.plot(ax=ax[1], color = 'purple', alpha = 0.6)

ax[1].set_title(f"$E_b = {1000*Eb}$ meV")
ax[1].text(1.9, .8,  f"$\Delta$t = {delay} fs", size=16)
ax[1].text(1.8, .95,  f'$k_x$ = {kx:.1f} $\AA^{{-1}}$', size=16)

ax[1].set_xlim(0.5,3)
ax[1].set_ylim(0, 1.1)
ax[1].set_xlabel('$E - E_{VBM}, eV$', fontsize = 18)
ax[1].set_ylabel('Norm. Int.', fontsize = 18)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update(params)

fig.text(.02, 0.975, "(a)", fontsize = 20, fontweight = 'regular')
fig.text(.59, 0.975, "(b)", fontsize = 20, fontweight = 'regular')

fig.tight_layout()

if save_figure is True:
    fig.savefig(figure_file_name + '.'+ image_format, bbox_inches='tight', format=image_format) 
    
print(f"{popt[2]:.3f} +- {perr[2]:.3f}")
print(f"{popt[3]:.3f} +- {perr[3]:.3f}")
print(f"{1000*Eb:.3f} +- {1000*np.sqrt(perr[3]**2+perr[2]**2):.3f}")

#%% Do the Excited State Fits for all Delay Times

# Momenta and Time Integration
(kx, ky), (kx_int, ky_int) = (0, 0), (4, 0.25) # Central (kx, ky) point and k-integration
delay_int = 40

# Fitting Paramaters
e1, e2 = 1.1, 3
p0 = [1, 0.3,  1.3, 2.1,  0.2, 0.2, 0] # Fitting params initial guess [amp, center, width, offset]
bnds = ((0.5, 0.1, 1.0, 1.5, 0.1, 0.1, 0), (1.5, 0.7, 1.5, 2.3, 0.9, 0.9, .3))

centers_CBM = np.zeros(len(I.delay))
centers_EX = np.zeros(len(I.delay))
Ebs = np.zeros(len(I.delay))

p_fits_excited = np.zeros((len(I.delay),7))
p_err_excited = np.zeros((len(I.delay),7))
p_err_eb = np.zeros((len(I.delay)))

n = len(I.delay.values)
for t in range(n):

    kx_frame = mpes.get_kx_E_frame(I_res, ky, ky_int, I.delay.values[t], delay_int)

    kx_edc_i = kx_frame.loc[{"kx":slice(kx-kx_int/2,kx+kx_int/2)}].sum(dim="kx")
    kx_edc_i = kx_edc_i/np.max(kx_edc_i.loc[{"E":slice(0.8,3)}])
    
    try:
        popt, pcov = curve_fit(two_gaussians, kx_edc_i.loc[{"E":slice(e1,e2)}].E.values, kx_edc_i.loc[{"E":slice(e1,e2)}].values, p0, method=None, bounds = bnds)
    except ValueError:
        popt = [0,0,0,0]
   
    centers_EX[t] = popt[2]
    centers_CBM[t] = popt[3]
    Eb = round(popt[3] - popt[2],3)
    Ebs[t] = Eb
    perr = np.sqrt(np.diag(pcov))
    p_fits_excited[t,:] = popt
    
    p_err_excited[t,:] = perr 
    p_err_eb[t] = np.sqrt(perr[3]**2+perr[2]**2)

#%% Plot Excited State EDC Fits and Binding Energy

figure_file_name = 'Eb_delays_allkx'
save_figure = False
image_format = 'pdf'

fig, ax = plt.subplots(1, 3,  gridspec_kw={'width_ratios': [1.25,1.5,1.5], 'height_ratios':[1]})
fig.set_size_inches(14, 4, forward=False)
ax = ax.flatten()

kx_frame = mpes.get_kx_E_frame(I_res, ky, ky_int, 500, delay_int)
kx_frame = kx_frame/np.max(kx_frame.loc[{"E":slice(0.8,3)}])

kx_edc = kx_frame.loc[{"kx":slice(kx-kx_int/2,kx+kx_int/2)}].mean(dim="kx")
kx_edc = kx_edc/np.max(kx_edc.loc[{"E":slice(0.8,3)}])

kx_edc.plot(ax=ax[0], color = 'darkgreen', label = 'Data')
ax[0].plot(kx_edc.loc[{"E":slice(0,3)}].E.values, g, color='grey',linestyle = 'solid', label = 'Fit', linewidth = 5)
ax[0].plot(kx_edc.loc[{"E":slice(0,3)}].E.values, g1, color='black',linestyle = 'dashed')
ax[0].plot(kx_edc.loc[{"E":slice(0,3)}].E.values, g2, color='red',linestyle = 'dashed')
#ax[0].set_title(f"$E_b = {1000*Eb}$ meV")
ax[0].set_title(f"$\Delta$t = {delay} fs")
ax[0].set_xticks(np.arange(0,3.2,.5))
for label in ax[0].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
ax[0].set_yticks(np.arange(0,1.5,0.25))
for label in ax[0].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
ax[0].set_xlim(0.5,3)
ax[0].set_ylim(0, 1.1)
ax[0].set_ylabel('Norm. Int.')
ax[0].set_xlabel('$E - E_{VBM}$, eV', color = 'black')
#ax[0].text(1.7, .8,  f"$\Delta$t = {delay} fs", size=16)
#ax[0].text(1.6, .95,  f'$k_x$ = {kx:.1f} $\AA^{{-1}}$', size=16)
#ax[0].text(1.6, .825,  f'$k_y$ = {kx:.1f} $\AA^{{-1}}$', size=16)
ax[0].set_aspect('auto')
ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax[0].legend(frameon=False)
# PLOT CBM and EX SHIFT DYNAMICS
#fig = plt.figure()
t = np.abs(I.delay.values-0).argmin() # Show only after 50 (?) fs
tt = np.abs(I.delay.values-100).argmin()
y_ex, y_ex_err = 1*(centers_EX[t:] - 0*centers_EX[-tt].mean()), 1*p_err_excited[t:,2]
y_cb, y_cb_err = 1*(centers_CBM[tt:]- 0*centers_CBM[-tt].mean()),  1*p_err_excited[tt:,3]

ax[1].plot(I.delay.values[t:], y_ex, color = 'black', label = 'Exciton')
ax[1].fill_between(I.delay.values[t:], y_ex - y_ex_err, y_ex + y_ex_err, color = 'grey', alpha = 0.5)
ax[1].set_xlim([0, I.delay.values[-1]])
#ax[1].set_ylim([1.1,2.3])
ax[1].set_xlabel('Delay, fs')
ax[1].set_xticks(np.arange(-400,1200,100))
for label in ax[1].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
ax[1].set_yticks(np.arange(0.5,2,0.05))
for label in ax[1].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
ax[1].set_xlim([0, I.delay.values[-1]])
ax[1].set_ylim([1.15,1.375])

ax2 = ax[1].twinx()
ax2.plot(I.delay.values[tt:], y_cb, color = 'red', label = 'CBM')
ax2.fill_between(I.delay.values[tt:], y_cb - y_cb_err, y_cb + y_cb_err, color = 'pink', alpha = 0.5)
ax2.set_ylim([1.7,2.25])
#ax[1].errorbar(I.delay.values[t:], 1*(centers_EX[t:]), yerr = p_err_excited[t:,2], marker = 'o', color = 'black', label = 'EX')
#ax[1].errorbar(I.delay.values[t:], 1*(centers_CBM[t:]), yerr = p_err_excited[t:,3], marker = 'o', color = 'red', label = 'CBM')
#ax[1].set_ylim([1.1,2.3])
ax[1].set_ylabel('$E_{Exciton}$, eV', color = 'black')
ax2.set_ylabel('$E_{CBM}$, eV', color = 'red')
#ax[1].set_title(f"From {round(I.delay.values[t])} fs")
ax[1].legend(frameon=False, loc = 'lower left')
ax2.legend(frameon=False, loc = 'lower right')
ax[1].arrow(250, 1.355, -80, 0, head_width = .025, width = 0.005, head_length = 40, fc='black', ec='black')
ax[1].arrow(700, 1.32, 80, 0, head_width = .025, width = 0.005, head_length = 40, fc='red', ec='red')
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax[2].plot(I.delay.values[tt:], 1000*Ebs[tt:], color = 'purple', label = '$E_{b}$')
ax[2].fill_between(I.delay.values[tt:], 1000*Ebs[tt:] - 1000*p_err_eb[tt:], 1000*Ebs[tt:] + 1000*p_err_eb[tt:], color = 'violet', alpha = 0.5)
ax[2].set_xlim([0, I.delay.values[-1]])
ax[2].set_ylim([700,900])
ax[2].set_xlabel('Delay, fs')
ax[2].set_ylabel('$E_{b}$, meV', color = 'black')
ax[2].legend(frameon=False, loc = 'lower right')
ax[2].set_xticks(np.arange(-400,1200,100))
for label in ax[2].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
ax[2].set_yticks(np.arange(600,1000,25))
for label in ax[2].yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
ax[2].set_ylim([700, 900])    
ax[2].set_xlim([0, I.delay.values[-1]])

fig.text(.02, 0.975, "(a)", fontsize = 20, fontweight = 'regular')
fig.text(.32, 0.975, "(b)", fontsize = 20, fontweight = 'regular')
fig.text(0.69, 0.975, "(c)", fontsize = 20, fontweight = 'regular')

# # PLOT VBM PEAK WIDTH DYNAMICS
# ax2 = ax[1].twinx()
# ax2.plot(edc_gamma.delay.values, 1000*(p_fits_VBM[:,2] - 0*p_fits_VBM[0:10,2].mean()), color = 'maroon')
# #ax2.set_ylim([-75,50])
# ax2.set_ylabel('${VBM}$ Peak Width, meV', color = 'maroon')

fig.tight_layout()

if save_figure is True:
    fig.savefig(figure_file_name + '.'+ image_format, bbox_inches='tight', format=image_format) 

#%% Plot and Fit EDCs of the VBM

%matplotlib inline

save_figure = True
figure_file_name = 'EDC_dyn_vb_metis'
image_format = 'pdf'

#I_res = I.groupby_bins('delay', 50)
#I_res = I_res.rename({"delay_bins":"delay"})
#I_res = I_res/np.max(I_res)
I_res = I/np.max(I)

fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 1], 'height_ratios':[1, 1]})
fig.set_size_inches(10, 8, forward=False)
ax = ax.flatten()
### Plot EDCs at GAMMA vs time

(kx, ky), k_int = (-2*X, 0), 0.12
edc_gamma = I_res.loc[{"kx":slice(kx-k_int/2,kx+k_int/2), "ky":slice(ky-k_int/2,ky+k_int/2)}].sum(dim=("kx","ky"))
edc_gamma = edc_gamma/np.max(edc_gamma)

im = edc_gamma.plot.imshow(ax = ax[0], cmap = cmap_LTL, add_colorbar = False)
# cbar_ax = fig.add_axes([.51, 0.275, 0.025, 0.5])
cbar = fig.colorbar(im, ax = ax[0], ticks = [0,1])
cbar.ax.set_yticklabels(['min', 'max'])  # vertically oriented colorbar

ax[0].set_ylim([-1,1])
ax[0].set_xlim([edc_gamma.delay[0],edc_gamma.delay[-1]])
#plt.axhline(0, linestyle = 'dashed', color = 'black')
#plt.axvline(0, linestyle = 'dashed', color = 'black')
ax[0].set_xlabel('Delay, fs')
ax[0].set_ylabel('E - E$_{VBM}$, eV')
ax[0].set_title('EDCs at $\Gamma$')

delays, delay_int  = [-120, 0, 50, 100, 500], 30
colors = ['black', 'red', 'orange', 'purple', 'blue', 'green', 'grey']
n = len(delays)
colors = mpl.cm.inferno(np.linspace(0,.8,n))
                    
for i in range(n):
    edc = I_res.loc[{"kx":slice(kx-k_int/2,kx+k_int/2), "ky":slice(ky-k_int/2,ky+k_int/2), "delay":slice(delays[i]-delay_int/2,delays[i]+delay_int/2)}].sum(dim=("kx","ky","delay"))
    edc = edc/np.max(edc)
    
    e = edc.plot(ax = ax[1], color = colors[i], label = f"{delays[i]} fs")

#plt.legend(frameon = False)
ax[1].set_xlim([-1.5, 1]) 
#ax[1].set_ylim([0, 1.1])
ax[1].set_xlabel('E - E$_{VBM}$, eV')
ax[1].set_ylabel('Norm. Int.')
#ax[1].set_title('EDCs at $\Gamma$')
#ax[1].axvline(0, color = 'black', linestyle = 'dashed', linewidth = 0.5)
ax[1].legend(frameon=False, loc = 'upper left', fontsize = 11)
#ax[1].set_yscale('log')
#plt.ax[1].gca().set_aspect(2)

###################
##### Fit EDCs ####
###################

##### VBM #########
e1 = -.2
e2 = 0.6
p0 = [1, 0, .2, 0] # Fitting params initial guess [amp, center, width, offset]
bnds = ((0.5, -1, 0.0, 0), (1.5, 0.5, 1, .5))

centers_VBM = np.zeros(len(I.delay))
p_fits_VBM = np.zeros((len(I.delay),4))
p_err_VBM = np.zeros((len(I.delay),2))

n = len(I.delay)
for t in np.arange(n):
    edc_i = edc_gamma.loc[{"E":slice(e1,e2)}][:,t].values
    edc_i = edc_i/np.max(edc_i)
    
    try:
        popt, pcov = curve_fit(gaussian, edc_gamma.loc[{"E":slice(e1,e2)}].E.values, edc_i, p0, method=None, bounds = bnds)
    except ValueError:
        popt = [0,0,0,0]
        
    centers_VBM[t] = popt[1]
    p_fits_VBM[t,:] = popt
    perr = np.sqrt(np.diag(pcov))
    p_err_VBM[t,:] = perr[1:2+1]

# VBM FIT TESTS FOR ONE POINT
t = 9
edc_neg = edc_gamma.loc[{"delay":slice(-120-10,-120+10)}].mean(dim='delay')
edc_neg = edc_neg/np.max(edc_neg)

gauss_test = gaussian(edc_gamma.E.values, *p_fits_VBM[t,:])
ax[2].plot(edc_gamma.E.values, edc_neg, color = 'black', label = 'Data')
ax[2].plot(edc_gamma.E.values, gauss_test, linestyle = 'dashed', color = 'grey', label = 'Fit')
#plt.axvline(trunc_e, linestyle = 'dashed', color = 'black')
ax[2].set_xlim([-2,1.5])
ax[2].set_xlabel('E - E$_{VBM}$, eV')
ax[2].set_ylabel('Norm. Int.')
ax[2].set_title(f'$\Delta$t = {-120} fs')
ax[2].legend(frameon=False, loc = 'upper left')
#ax[0].axvline(0, linestyle = 'dashed', color = 'grey')
#ax[0].axvline(e2, linestyle = 'dashed', color = 'black')

# PLOT VBM SHIFT DYNAMICS

t = 39 # Show only after 50 (?) fs
y_vb, y_vb_err = 1000*(p_fits_VBM[:,1] - p_fits_VBM[0:10,1].mean()), 1000*p_err_VBM[:,0]
y_vb_w, y_vb_w_err = 1000*(p_fits_VBM[:,2]),  1000*p_err_VBM[:,1]

ax[3].plot(I.delay.values, y_vb, color = 'navy', label = '$\Delta E_{VBM}$')
ax[3].fill_between(I.delay.values, y_vb - y_vb_err, y_vb + y_vb_err, color = 'navy', alpha = 0.5)

ax[3].set_xlim([edc_gamma.delay.values[1], edc_gamma.delay.values[-1]])
ax[3].set_ylim([-30,15])
ax[3].set_xlabel('Delay, fs')
ax[3].set_ylabel('$\Delta E_{VBM}$, meV', color = 'navy')
ax[3].set_title('Peak Dynamics')
ax[3].legend(frameon=False, loc = 'upper left')
#ax[3].arrow(250, 1.3, -75, 0, head_width = .025, width = 0.005, head_length = 40, fc='black', ec='navy')
#ax[3].arrow(650, 1.12, 75, 0, head_width = .025, width = 0.005, head_length = 40, fc='red', ec='maroon')

# PLOT VBM PEAK WIDTH DYNAMICS
ax2 = ax[3].twinx()
ax2.plot(I.delay.values, y_vb_w, color = 'maroon', label = '$\sigma_{VBM}$')
ax2.fill_between(I.delay.values, y_vb_w - y_vb_w_err, y_vb_w + y_vb_w_err, color = 'maroon', alpha = 0.5)
ax2.set_ylim([125,275])
ax2.legend(frameon=False, loc = 'upper right')
ax2.set_ylabel('${VBM}$ Peak Width, meV', color = 'maroon')

fig.text(.02, 1, "(a)", fontsize = 20, fontweight = 'regular')
fig.text(.5, 1, "(b)", fontsize = 20, fontweight = 'regular')
fig.text(.02, 0.5, "(c)", fontsize = 20, fontweight = 'regular')
fig.text(.5, 0.5, "(d)", fontsize = 20, fontweight = 'regular')
fig.tight_layout()

if save_figure is True:
    fig.savefig(figure_file_name + '.'+ image_format, bbox_inches='tight', format=image_format) 

#%% Fourier Transform Analaysis

figure_file_name = 'FFT_PEAKS'
save_figure = False
image_format = 'pdf'

def monoexp(t, A, tau):
   return A * np.exp(-t/tau) * (t >= 0)
    
i_start = np.abs(I.delay.values-100).argmin()
waitingfreq = (1/2.99793E10)*np.fft.fftshift(np.fft.fftfreq(len(I.delay.values[i_start:]), d=20E-15));
delay_trunc = I.delay.values[i_start:]

pk, color = Ebs, 'purple'
#pk, color = centers_EX, 'black'
#pk, color = centers_CBM, 'crimson'
pk = pk[i_start:] - np.mean(pk[i_start:])

omega = 2*np.pi/136
#pk = np.sin(delay_trunc*omega)
trace = np.abs(np.fft.fftshift(np.fft.fft(pk)))

fig, ax2 = plt.subplots(2,1)
fig.set_size_inches(8, 8, forward=False)
ax2 = ax2.flatten()

fig, ax = plt.subplots(1, 3)
fig.set_size_inches(12, 4, forward=False)
ax = ax.flatten()

# .plot(delay_trunc, centers_EX[i_start:]-np.mean(centers_EX[i_start:]), 'black')
# plt.plot(delay_trunc, centers_CBM[i_start:]-np.mean(centers_CBM[i_start:]), 'crimson')
# plt.plot(delay_trunc, Ebs[i_start:]-np.mean(Ebs[i_start:]), 'purple')
# plt.plot(delay_trunc, np.sin(delay_trunc*omega)*0.015, 'blue')

peaks = [centers_EX, centers_CBM, Ebs]
peak_labels = ['Exciton', 'CBM', '$E_{b}$']
colors = ['black', 'crimson', 'purple']
phonons = [110, 240, 350]

for i in [0,1,2]:
    
    pk = peaks[i]
    
    p0 = [1, 200]
    #popt, pcov = curve_fit(monoexp, I.delay.values[i_start], pk[i_start], p0, method=None)
    popt, pcov = curve_fit(monoexp, I.delay.values[i_start:], pk[i_start:], p0, method=None)

    pk_fit = monoexp(I.delay.values, *popt)
    trace = pk - pk_fit#- np.mean(pk[i_start:])
    trace = trace[i_start:]
    fft_trace = np.abs(np.fft.fftshift(np.fft.fft(trace)))

    ax2[0].plot(I_res.delay.values, pk, color = colors[i]) 
    ax2[0].plot(I_res.delay.values[i_start:], pk_fit[i_start:], color = colors[i], linestyle = 'dashed') 
    ax2[0].axvline(0, linestyle = 'solid', color = 'grey')
    
    residuals = pk-pk_fit
    ax2[1].plot(I_res.delay.values[i_start:], residuals[i_start:], color = colors[i]) 
    ax2[1].axvline(0, linestyle = 'solid', color = 'grey')
    
    
    ax[i].plot(waitingfreq, fft_trace, color = colors[i]) 
    ax[i].axvline(phonons[0], color = 'grey', linestyle = 'dashed') 
    ax[i].axvline(phonons[1], color = 'grey', linestyle = 'dashed') 
    ax[i].axvline(phonons[2], color = 'grey', linestyle = 'dashed')
    #plt.axvline(45, color = 'grey', linestyle = 'dashed')
    
    ax[i].set_xticks(np.arange(-1000,1000,100))
    for label in ax[i].xaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
    ax[i].set_yticks(np.arange(0,0.3,0.02))
    for label in ax[i].yaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
        
    ax[i].set_xlim(0,800)
    ax[i].set_ylim(0,0.2)
    ax[i].set_xlabel('Wavenumber, $cm^{-1}$')
    ax[i].set_ylabel('Amplitude')
    ax[i].set_title(f'{peak_labels[i]}')

fig.text(.02, .98, "(a)", fontsize = 20, fontweight = 'regular')
fig.text(.34, .98, "(b)", fontsize = 20, fontweight = 'regular')
fig.text(.67, .98, "(c)", fontsize = 20, fontweight = 'regular')

fig.tight_layout()

if save_figure is True:
    fig.savefig(figure_file_name + '.'+ image_format, bbox_inches='tight', format=image_format) 


#%% Plot Neg, Pos, and Difference Angle-Energy Panels

%matplotlib inline

save_figure = False
figure_file_name = 'ARPRES_Panels_diff'
image_format = 'pdf'

E, E_int = [1.375, 2.125], 0.1
scan = 9526
res = phoibos.load_data(data_path, scan, scan_info, energy_offset, delay_offset, force_offset)

colormap = cmap_LTL # 'bone_r'# 'Purples'

fig, axx = plt.subplots(1, 3)
fig.set_size_inches(12, 4, forward=True)
plt.gcf().set_dpi(300)
axx = axx.flatten()

E_inset = .8

res_neg = res.loc[{'Delay':slice(-1000,-300)}]
res_pos = res.loc[{'Delay':slice(0,5000)}]
res_neg_mean = res_neg.mean(axis=2)
res_pos_mean = res_pos.mean(axis=2)

res_diff_E_Ang = res_pos_mean - res_neg_mean

neg_enh = phoibos.enhance_features(res_neg_mean, E_inset, 1, True)
pos_enh = phoibos.enhance_features(res_pos_mean, E_inset, 1, True)
diff_enh = phoibos.enhance_features(res_diff_E_Ang, E_inset, 1, True) 
diff_enh = diff_enh / np.max(np.abs(diff_enh))

im1 = neg_enh.T.plot.imshow(ax = axx[0], cmap = colormap)
im2 = pos_enh.T.plot.imshow(ax = axx[1], cmap = colormap)
im3 = diff_enh.T.plot.imshow(ax = axx[2], cmap = 'RdBu_r')
axx[0].set_ylim(-1,2.65)
axx[1].set_ylim(-1,2.65)
axx[2].set_ylim(-1,2.65)

axx[2].axhline(E_inset, color = 'black', linewidth = 1)
axx[2].axhline(E_inset, color = 'black', linewidth = 1)
axx[2].axhline(E_inset, color = 'black', linewidth = 1)

axx[0].set_title('$\Delta$t < -300 fs')
axx[1].set_title('$\Delta$t > 0 fs ')
axx[2].set_title(f"Scan{scan}")
axx[2].set_title('Difference')

axx[0].set_ylabel('$E - E_{VBM}, eV$')
axx[1].set_ylabel('$E - E_{VBM}, eV$')
axx[2].set_ylabel('$E - E_{VBM}, eV$')

fig.text(.01, .95, "(a)", fontsize = 18, fontweight = 'regular')
fig.text(.33, .95, "(b)", fontsize = 18, fontweight = 'regular')
fig.text(.66, .95, "(c)", fontsize = 18, fontweight = 'regular')

params = {'lines.linewidth' : 3.5, 'axes.linewidth' : 2, 'axes.labelsize' : 20, 
              'xtick.labelsize' : 16, 'ytick.labelsize' : 16, 'axes.titlesize' : 20, 'legend.fontsize' : 16}
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update(params)

fig.tight_layout()
plt.show()

if save_figure is True:
    fig.savefig(figure_file_name + '.'+ image_format, bbox_inches='tight', format=image_format) 

#%% Waterfall difference Panel

save_figure = True
figure_file_name = 'WaterFallDifference_phoibos400nm'
image_format = 'pdf'

#E, E_int = [1.3, 2.0], 0.1
scan = 9526
res = phoibos.load_data(data_path, scan, scan_info, energy_offset, delay_offset, force_offset)

A, A_int = 0, 20
E_inset = 0.9

colormap, scale = cmap_LTL2, [-1,1] #'bone_r'
subtract_neg = True
norm_trace = False

###
WL = scan_info[str(scan)].get("Wavelength")
per = (scan_info[str(scan)].get("Percent"))
Temp = float(scan_info[str(scan)].get("Temperature"))

res_neg = res.loc[{'Delay':slice(-1000,-300)}]
res_pos = res.loc[{'Delay':slice(0,5000)}]

res_neg_mean = res_neg.mean(axis=2)
res_pos_mean = res_pos.mean(axis=2)

#res_diff_E_Ang = res_pos_mean - res_neg_mean

res_diff_E_Ang = res_pos_mean - res_neg_mean
#res_diff_E_Ang = res.loc[{'Delay':slice(-100,0)}].mean(axis=2) - res_neg_mean
#res_diff_E_Ang = res.loc[{'Delay':slice(250,350)}].mean(axis=2) - res_neg_mean

res_diff_E_Ang = res_diff_E_Ang/np.max(np.abs(res_diff_E_Ang))

res_sum_Angle = res.loc[{'Angle':slice(-A-A_int/2,A+A_int/2)}].sum(axis=0)
res_sum_Angle = res_sum_Angle/np.max(res_sum_Angle)

res_diff = res - res_neg.mean(axis=2)
res_diff_sum_Angle = res_diff.loc[{'Angle':slice(-A-A_int/2,A+A_int/2)}].sum(axis=0)
res_diff_sum_Angle = res_diff_sum_Angle/np.max(res_diff_sum_Angle)

res_sum_Angle = phoibos.enhance_features(res_sum_Angle, E_inset, _ , True)

res_diff_E_Ang = phoibos.enhance_features(res_diff_E_Ang, E_inset, _ , True)
res_diff_sum_Angle = phoibos.enhance_features(res_diff_sum_Angle, E_inset, _ , True)

trace_1 = phoibos.get_time_trace(res, E[0], E_int, A, A_int, subtract_neg, norm_trace)
trace_2 = phoibos.get_time_trace(res, E[1], E_int, A, A_int, subtract_neg, norm_trace)

trace_2 = trace_2/trace_1.max()
trace_1 = trace_1/trace_1.max()

############
### PLOT ###
############

fig, axx = plt.subplots(1, 2)
fig.set_size_inches(12, 4, forward=False)
plt.gcf().set_dpi(300)
axx = axx.flatten()

im1 = res_sum_Angle.plot.imshow(ax = axx[0], cmap = cmap_LTL, vmin = 0, vmax = scale[1])
#plt.colorbar(im1, ax=axx[0], extend='neither')

im2 = res_diff_sum_Angle.plot.imshow(ax = axx[1], cmap = 'RdBu_r', vmin = -1, vmax = 1)
#plt.colorbar(im2, ax=axx[1], extend='neither')

#fig.colorbar(im2, ax=axx[1])
#im_dyn = axx[2].plot(trace_1.Delay.loc[{"Delay":slice(0,50000)}].values, \
 #                  0.6*np.exp(-trace_1.Delay.loc[{"Delay":slice(0,50000)}].values/18000) +

  #                 0.3*np.exp(-trace_1.Delay.loc[{"Delay":slice(0,50000)}].values/2000))
#axx[0].axhline(E[0],  color = 'black')
#axx[0].axhline(E[1],  color = 'red')
axx[0].axhline(E_inset,  color = 'grey', linestyle = 'dashed')
axx[0].set_title('Integrated')
axx[1].set_title('Difference')
axx[0].set_xlim(-750,3000)
axx[0].set_ylim(E_inset-0.25,res.Energy.values.max())
axx[1].set_xlim(-750,3000)
axx[1].set_ylim(E_inset-0.25,res.Energy.values.max())
axx[1].axhline(E_inset,  color = 'grey', linestyle = 'dashed')

axx[0].set_xlabel('Delay, fs')
axx[1].set_xlabel('Delay, fs')
axx[0].set_ylabel('$E - E_{VBM}, eV$')
axx[1].set_ylabel('$E - E_{VBM}, eV$')

fig.text(.01, .95, "(d)", fontsize = 18, fontweight = 'regular')
fig.text(.5, .95, "(e)", fontsize = 18, fontweight = 'regular')

fig.tight_layout()

if save_figure is True:
    fig.savefig(figure_file_name + '.'+ image_format, bbox_inches='tight', format=image_format) 
    
#%% Total Excited State Population 

save_figure = False
figure_file_name = 'Combined'
image_format = 'pdf'

# Standard 915 nm Excitation
scans = [9219, 9217, 9218, 9216, 9220, 9228]
#offsets_t0 = [-162.4, -152.7, -183.1, -118.5, -113.7, -125.2]
power = 1.05*np.asarray([153, 111, 91, 66, 47, 32, 15, 10, 8, 5])
fluence = [.2, .35, .8, 1.74, 2.4, 2.9]

# Expanded 915 nm Excitation
#scans = [9227, 9219, 9217, 9218, 9216, 9220, 9228, 9231] # Scans to analyze and fit below: 910 nm + 400 nm
#offsets_t0 = [-191.7, -162.4, -152.7, -183.1, -118.5, -113.7, -125.2, -147.6]
#fluence = [4.7, 8.3, 20.9, 41.7, 65.6, 83.2, 104.7, 151]

####
k, k_int = (0), 24
E[0], E[1], E_int = 1.35, 2.1, 0.1
subtract_neg = True
norm_trace = False

cn = 100
p_min = .1
p_max = 3.5

fig, axx = plt.subplots()
fig.set_size_inches(5, 4, forward=False)
plt.gcf().set_dpi(300)

fluence = np.array(fluence)
boundaries = np.concatenate([[fluence[0] - (fluence[1] - fluence[0]) / 2],  # Leftmost edge
                             (fluence[:-1] + fluence[1:]) / 2,  # Midpoints
                             [fluence[-1] + (fluence[-1] - fluence[-2]) / 2]])  # Rightmost edge
midpoints = (boundaries[:-1] + boundaries[1:]) / 2


cmap = cm.get_cmap('inferno_r', len(scans))    # 11 discrete colors
norm = col.BoundaryNorm(boundaries, cmap.N)  # Normalize fluence values to colors

custom_colors = ['lightsteelblue', 'royalblue', 'mediumblue', 'salmon', 'indianred', 'firebrick'] #colors for plotting the traces
cmap = mcolors.ListedColormap(custom_colors)  # Create discrete colormap
norm = mcolors.BoundaryNorm(boundaries, cmap.N)  # Normalize boundaries

intensity = np.zeros(len(scans))
i = 0
for scan_i in scans:
    
    res = phoibos.load_data(data_path, scan_i, scan_info, energy_offset, delay_offset, force_offset)
    trace_1 = phoibos.get_time_trace(res, E[0], E_int, k, k_int, subtract_neg, norm_trace)
    trace_2 = phoibos.get_time_trace(res, E[1], E_int, k, k_int, subtract_neg, norm_trace)
    trace_2 = trace_2/np.max(trace_1)
    trace_1 = trace_1/np.max(trace_1)

    combined = trace_1 + trace_2
    combined = phoibos.get_time_trace(res, 1.7, 1, k, k_int, subtract_neg, norm_trace)
    
    intensity[i] = np.max(combined)
    combined = combined/np.max(combined)

    t3 = combined.plot(ax = axx, color = cmap(i), linewidth = 3)
    
    i += 1

axx.set_xticks(np.arange(-1000,3500,500))
for label in axx.xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)

axx.set_yticks(np.arange(-0.5,1.25,0.25))
for label in axx.yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
    

axx.set_xlim([-500,3000])
axx.set_ylim([-0.1,1.1])

axx.set_xlabel('Delay, fs')
axx.set_ylabel('Norm. Int.')

axx.set_title('Excited State Population')

fig.text(.01, 0.975, "(a)", fontsize = 20, fontweight = 'regular')

# Add colorbar for the discrete color mapping
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Required for colorbar to work
cbar_ax = fig.add_axes([1, 0.17, 0.02, 0.75])
cbar = fig.colorbar(sm, cax=cbar_ax, ticks=midpoints)
cbar.set_label("$n_{eh}$ ($x$10$^{13}$ cm$^{-2})$")
cbar.ax.set_yticklabels([f"{f:.2f}" for f in fluence])  # Format tick labels

params = {'lines.linewidth' : 3.5, 'axes.linewidth' : 2, 'axes.labelsize' : 20, 
              'xtick.labelsize' : 16, 'ytick.labelsize' : 16, 'axes.titlesize' : 20, 'legend.fontsize' : 16}
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update(params)

fig.tight_layout()

if save_figure is True:
    fig.savefig(figure_file_name + '.'+ image_format, bbox_inches='tight', format=image_format) 
    
#%% Plot Cold Data Scans

%matplotlib inline

save_figure = False
figure_file_name = 'Cold_Data_phoibos'
image_format = 'pdf'

# Scans to plot
# Standard 915 nm Excitation
scans = [9219, 9217, 9218, 9216, 9220, 9228]

# Combined
scans = [9241, 9240, 9137] #915 nm (top 3) ; 700 nm, 640 nm, 400 nm

#power = 1.05*np.asarray([153, 111, 91, 66, 47, 32, 15, 10, 8, 5])
fluence = [.2, .35, .8, 1.74, 2.4, 2.9]
fluence = [.2, .8, 2.9, 4.5, 3.6, 0]
ev = [1.355, 1.355, 3.10]
# Specify energy and Angle ranges
E, E_int = [1.325, 2.075], 0.1
E, E_int = [1.325, 2.025], 0.1

k, k_int = 0, 20
subtract_neg = True
norm_trace = False

# Plot
fig, ax = plt.subplots(1, 3)
fig.set_size_inches(11, 4, forward=False)
plt.gcf().set_dpi(300)
ax = ax.flatten()

for i in np.arange(len(scans)):
    scan_i = scans[i]
    res = phoibos.load_data(data_path, scan_i, scan_info, energy_offset, delay_offset, force_offset)
    WL = scan_info[str(scan_i)].get("Wavelength")
    temp = scan_info[str(scan_i)].get("Temperature")

    if i == 3:
        E, E_int = [1.3, 1.92], 0.1

#    res = phoibos.load_data(scan_i, energy_offset, offsets_t0[i])
    trace_1 = phoibos.get_time_trace(res, E[0], E_int, k, k_int, subtract_neg, norm_trace)
    trace_2 = phoibos.get_time_trace(res, E[1], E_int, k, k_int, subtract_neg, norm_trace)
    norm = np.max([trace_1,trace_2])
    #trace_2 = trace_2/np.max(trace_1)
    #trace_1 = trace_1/np.max(trace_1)
    trace_2 = trace_2/norm
    trace_1 = trace_1/norm
    
    t1 = trace_1.plot(ax = ax[i], color = 'black', linewidth = 3)
    t2 = trace_2.plot(ax = ax[i], color = 'crimson', linewidth = 3)
    #ax[i].text(1000, .9, f"$n_{{eh}} = {fluence[i]:.2f} x 10^{{13}} cm^{{-2}}$", fontsize = 14, fontweight = 'regular')

    # Set major ticks at every 500
    xticks = np.arange(-1000, 3500, 500)
    ax[i].set_xticks(xticks)
    
    # Hide every other label by replacing with an empty string
    xtick_labels = [str(tick/1000) if i % 2 == 0 else "" for i, tick in enumerate(xticks)]
    ax[i].set_xticklabels(xtick_labels)
    ax[i].set_yticks(np.arange(-0.5,1.25,0.25))
    for label in ax[i].yaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
    ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax[i].set_xlim([-500,3000])
    ax[i].set_ylim([-0.1,1.1])
    ax[i].set_xlabel('Delay, ps')
    ax[i].set_ylabel('Nom. Int.')
    ax[i].set_title(f'$hv_{{ex}} $ = {ev[i]} eV', fontsize = 22)
    ax[i].text(1000, .95, f'T = {temp} K', fontsize = 18)
    #ax[0].set_title('Exciton')
    #ax[1].set_title('CBM')

fig.text(.01, 1, "(a)", fontsize = 20, fontweight = 'regular')
fig.text(.33, 1, "(b)", fontsize = 20, fontweight = 'regular')
fig.text(.66, 1, "(c)", fontsize = 20, fontweight = 'regular')

# Add colorbar for the discrete color mapping
# sm = cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])  # Required for colorbar to work
# cbar_ax = fig.add_axes([1, 0.17, 0.02, 0.75])
# cbar = fig.colorbar(sm, cax=cbar_ax, ticks=midpoints)
# cbar.set_label("$n_{eh}$ ($x$10$^{13}$ cm$^{-2})$")
# cbar.ax.set_yticklabels([f"{f:.2f}" for f in fluence])  # Format tick labels

params = {'lines.linewidth' : 3.5, 'axes.linewidth' : 2, 'axes.labelsize' : 20, 
              'xtick.labelsize' : 16, 'ytick.labelsize' : 16, 'axes.titlesize' : 20, 'legend.fontsize' : 16}
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update(params)

fig.tight_layout()

if save_figure is True:
    fig.savefig(figure_file_name + '.'+ image_format, bbox_inches='tight', format=image_format)    
    