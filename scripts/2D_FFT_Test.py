import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import tukey

# Define parameters
t = np.linspace(-2, 2, 1000)  # Time array
sigma = 0.072  # Width of the Gaussian signal
offset = 0

# Define the Gaussian signal
def gaussian(x, amp_1, mean_1, stddev_1, offset):
    
    g1 = amp_1 * np.exp(-0.5*((x - mean_1) / stddev_1)**2)+offset
    
    return g1

signal_input = gaussian(t, 1, 0, sigma, offset)

d = 3
# Define a function to compute the FFT of the windowed signal
def compute_fft(signal, window_width, t, alpha):
    # Create a Tukey window using scipy
    window = np.pad(tukey(int(len(t)/d), alpha=alpha), ((int(len(t)/d)+1), int(len(t)/d)) )
    #window = (tukey(int(len(t)/1), alpha=alpha))

    windowed_signal = signal * window
    fft = np.fft.fftshift(np.fft.fft(windowed_signal))  # Compute FFT
    signal_fft = np.fft.fftshift(np.fft.fft(signal))  # Compute FFT
    window_fft = np.fft.fftshift(np.fft.fft(window))  # Compute FFT

    freq = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(len(t), d=t[1] - t[0]))  # Frequency array
    return windowed_signal, freq, np.abs(fft)/np.max(np.abs(fft)), window, np.abs(signal_fft)/np.max(np.abs(signal_fft)), np.abs(window_fft)/np.max(np.abs(window_fft))

# Define different window widths and Tukey alpha values
window_widths = [0.5, 1, 2, 4]
alphas = [0.1, 0.25, 0.5, 0.9]  # Different tapering parameters for the Tukey window
colors = ['blue', 'green', 'orange', 'red']

# Create a figure with subplots
fig, axes = plt.subplots(len(window_widths), 2, figsize=(14, 12))

# Loop through window widths
for i, (w, alpha, c) in enumerate(zip(window_widths, alphas, colors)):
    # Compute windowed signal and FFT
    windowed_signal, freq, fft, window, signal_fft, window_fft = compute_fft(signal_input**(0.5), w, t, alpha=alpha)
    
    signal_fft = signal_fft**2
    
    if i == 0:
        p0 = [1, 0, 1, 0]
        bnds = ( (0, -.5, 0, 0), (2, 0.5, 10, 1) )
        popt, pcov = curve_fit(gaussian, freq, signal_fft, p0, method=None, bounds = bnds)
        fft_sigma = popt[2]
        print(round(fft_sigma,3), round(1/(2*sigma),3), round(fft_sigma*2.355/2,3),  round(0.5*2.355/(2*sigma),3), round(np.sqrt(2)/(2*sigma),3))
        fit_test = gaussian(freq, *popt)
        axes[0, 1].plot(freq, fit_test, color = 'pink', label = 'FIT', linewidth = 10)
        
    # Plot original and windowed signal
    axes[i, 0].plot(t, signal_input, label="Orig. Sig.", color='black', linestyle='--')
    axes[i, 0].plot(t, window, label=f"Win. (a={alpha})", color=c, alpha=0.7)
    axes[i, 0].plot(t, windowed_signal, label="Windwd. Sig.", color=c)
    axes[i, 0].set_title(f"Gaussian & Tukey Win. Sig. (Width={w}, Alpha={alpha})")
    axes[i, 0].set_xlabel("Native")
    axes[i, 0].set_ylabel("Amplitude")
    axes[i, 0].legend()
    axes[i, 0].grid(True)
    
    # Plot FFT
    axes[i, 1].plot(freq, fft, label="FFT", color=c)
    axes[i, 1].plot(freq, signal_fft, label="FFT Original", color='black', linestyle = '--')
    axes[i, 1].plot(freq, window_fft, label="FFT Window", color=c, linestyle = '--')

    axes[i, 1].set_title(f"FFT of Win. Sig. (Width={w}, Alpha={alpha})")
    axes[i, 1].set_xlabel("Reciprocal")
    axes[i, 1].set_ylabel("Amplitude")
    axes[i, 1].legend()
    axes[i, 1].grid(True)
    axes[i, 1].set_xlim(-10,10)

# Adjust layout
plt.tight_layout()
plt.show()
