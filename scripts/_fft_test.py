import numpy as np
import matplotlib.pyplot as plt

# Define k-space parameters
N = 100  # Number of points
dk = 0.01  # Momentum space resolution in Å⁻¹
k = np.linspace(-2,2,N) # Generate k-space grid in Å⁻¹
k = np.arange(-2,2,dk)    
N = len(k)

# Construct the corresponding real-space grid in nm
dr = 1 / (N * dk)  # Compute real-space resolution in nm
r = np.fft.fftfreq(N, d=dk) 
r = np.fft.fftshift(r)  # Shift zero frequency to center
r = r*0.1

# Define a Gaussian in k-space
sigma_k = 0.12  # Width in Å⁻¹
gaussian_k = np.exp(- (k**2) / (2 * sigma_k**2))  # Smooth Gaussian function
#gaussian_k = np.sqrt(gaussian_k)

# Perform inverse FFT to obtain real-space function
gaussian_r = np.fft.ifft(np.fft.ifftshift(gaussian_k))  # IFFT to real space
gaussian_r = np.fft.fftshift(np.abs(gaussian_r))  # Shift zero frequency to center
#gaussian_r = gaussian_r**2
gaussian_r = gaussian_r/np.max(gaussian_r)

# Compute expected sigma_r from Fourier relation
sigma_r_expected = 1 / (2 * np.pi * sigma_k)  * 0.1  # Convert to nm
sigma_r_expected_2 = 1 / (sigma_k)  * 0.1  # Convert to nm

# Fit the FFT results to obtain actual real-space width
p0 = [.8, 0, .1, 0.2]
bnds = ((0.1, -0.2, 0, 0), (1, 0.2, .8, .5))
popt_r, pcov = curve_fit(gaussian, r, gaussian_r, p0, method=None, bounds = bnds)
fit_r = gaussian(r, *popt_r)

# Plot results
plt.figure(figsize=(10, 4))

# Plot k-space Gaussian
plt.subplot(1, 2, 1)
plt.plot(k, gaussian_k, label=f'$\sigma_k$ = {sigma_k:.3f}')
plt.xlabel(r'$k$ ($\mathrm{\AA}^{-1}$)')
plt.ylabel('Amplitude')
plt.title('Gaussian in Momentum Space')
plt.legend(frameon=False, loc = 'upper right')
plt.grid()
plt.xlim(-1,1)

# Plot real-space Gaussian
plt.subplot(1, 2, 2)
plt.plot(r, gaussian_r, linewidth = 4, label = rf'FFT of k-space')
plt.plot(r, fit_r, linestyle = 'dashed', label = 'Fit of FFT', linewidth = 4)
plt.plot(r, gaussian(r, 1, 0, sigma_r_expected, 0), linestyle = 'dashed', label = rf'Pred1. $\approx$ ${sigma_r_expected:.2f}$ nm')
plt.plot(r, gaussian(r, 1, 0, sigma_r_expected_2, 0), linestyle = ':', label = rf'Pred2 $\approx$ ${sigma_r_expected_2:.2f}$ nm')

plt.xlabel(r'$r$ (nm)')
plt.ylabel('Amplitude')
plt.title('Gaussian in Real Space')
plt.legend(frameon=False)
plt.grid()
plt.xlim(-2,2)

plt.tight_layout()
plt.show()

print(f"Expected real-space width (σ_r): {sigma_r_expected:.3f} nm")
print(f"Fit real-space width (σ_r): {popt_r[2]:.3f} nm")

#print(f"Real-space resolution (dr): {dr:.4f} nm")


#%%

import numpy as np
import matplotlib.pyplot as plt

# Parameters
dk = 0.04 # in inverse Angstroms
N = 2048  # Total points
k_min = -2  # k-space minimum value (in inverse Angstroms)
k_max = 2  # k-space maximum value (in inverse Angstroms)

# Define k-space axis (range from -2 to 2 inverse Angstroms)
k_space_axis = np.linspace(k_min, k_max, N)

# Create a Gaussian function in k-space (with a width of 0.12 inverse Angstroms)
sigma_k = 0.12  # in inverse Angstroms
k_space_data = np.exp(-0.5 * (k_space_axis / sigma_k)**2)
k_space_data = k_space_data/np.max(k_space_data)

# Perform the Inverse FFT to get the real-space data
real_space_data = np.fft.ifftshift(np.fft.ifft(k_space_data))

# Normalize the real-space data (apply scaling factor of N)
real_space_data = real_space_data/np.max(real_space_data) # * N

# Define the real-space axis (based on the k-space axis)
# The total length of the real space is related to the k-space size
# The spacing in real space is the inverse of the spacing in k-space (since dk = 1/delta_x)
delta_x = 1 / (dk * N)  # real space spacing (in Angstroms)
real_space_axis = np.fft.fftfreq(N, d=dk)  # real space axis in inverse Angstroms
real_space_axis = 2*3.14*np.fft.fftshift(real_space_axis)  # Shift to center the axis

# Convert the real-space axis to nanometers (1 Angstrom = 0.1 nm)
real_space_axis_nm = real_space_axis * 0.1  # now in nm

# Plotting
plt.figure(figsize=(10, 6))

# Plot the Gaussian in k-space
plt.subplot(1, 2, 1)
plt.plot(k_space_axis, np.abs(k_space_data))
plt.title("Gaussian in k-space")
plt.xlabel("Wavevector (inverse Angstroms)")
plt.ylabel("Amplitude")
plt.xlim(-.25,.25)

# Plot the real-space data
plt.subplot(1, 2, 2)
plt.plot(real_space_axis_nm, np.abs(real_space_data))
plt.plot(real_space_axis_nm, gaussian(real_space_axis_nm, 1, 0, .0105, 0))
plt.plot(real_space_axis_nm, gaussian(real_space_axis_nm, 1, 0, .07, 0))

plt.title("Gaussian in real-space (after FFT)")
plt.xlabel("Position (nm)")
plt.ylabel("Amplitude")
plt.xlim(-.2,.2)

plt.tight_layout()
plt.show()

# Output the expected real-space width (in nm)
sigma_r = 0.1 * 1 / sigma_k  # in nm (after converting from inverse Angstroms)
print(f"From k-space {sigma_k:.3f} A^-1, Expected real-space width: {sigma_r:.3f} nm")


#%%

import numpy as np
import matplotlib.pyplot as plt

# Parameters
dk = 0.02  # in inverse Angstroms (spacing in k-space)
N = 2048  # Total number of points
k_min = -2  # k-space minimum value (in inverse Angstroms)
k_max = 2  # k-space maximum value (in inverse Angstroms)

# Define k-space axis (range from -2 to 2 inverse Angstroms)
k_space_axis = np.linspace(k_min, k_max, N)

# Create a Gaussian function in k-space (with a width of 0.12 inverse Angstroms)
sigma_k = 0.12  # in inverse Angstroms
k_space_data = np.exp(-0.5 * (k_space_axis / sigma_k)**2)

# Perform the Inverse FFT to get the real-space data
real_space_data = np.fft.fftshift(np.fft.ifft(k_space_data))

# Normalize the real-space data (apply scaling factor of N)
real_space_data = real_space_data * N

# Define the real-space axis (this is where we correct the definition)
# The length of real space (L) is the total size of the real-space grid
L_real = 1 / (2 * dk)  # Length of real space in Angstroms
real_space_axis = np.linspace(-L_real / 2, L_real / 2, N)

# Convert the real-space axis to nanometers (1 Angstrom = 0.1 nm)
real_space_axis_nm = real_space_axis * 0.1  # now in nm

# Plotting
plt.figure(figsize=(10, 6))

# Plot the Gaussian in k-space
plt.subplot(1, 2, 1)
plt.plot(k_space_axis, np.abs(k_space_data))
plt.title("Gaussian in k-space")
plt.xlabel("Wavevector (inverse Angstroms)")
plt.ylabel("Amplitude")

# Plot the real-space data
plt.subplot(1, 2, 2)
plt.plot(real_space_axis_nm, np.abs(real_space_data))
plt.title("Gaussian in real-space (after FFT)")
plt.xlabel("Position (nm)")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()

# Output the expected real-space width (in nm)
sigma_r = 1 / sigma_k  # in nm (after converting from inverse Angstroms)
print(f"From k-space {sigma_k:.3f} A^-1, Expected real-space width: {sigma_r:.3f} nm")
