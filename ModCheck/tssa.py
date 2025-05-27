from ROOT import TLorentzVector
import numpy as np
import math
import ROOT
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import random

phi_up = []
phi_down = []
muon_mass = 0.1056
A_injected = 0.1

file_= ROOT.TFile("data_in/phi_angles_with_asym.root");
tree = file_.Get("phi_tree")

count_up = 0
count_down = 0


for i, event in enumerate(tree):
    if event.spin == 1:
        phi_up.append(event.phi)
    elif event.spin == -1:
        phi_down.append(event.phi)

def sin_fit(x, A):
    return A * np.sin(x)

def cos_fit(x, A):
    return A * np.cos(x)


bins = np.linspace(-np.pi, np.pi, 13)
centers = 0.5 * (bins[1:] + bins[:-1])

N_up, _ = np.histogram(phi_up, bins=bins)
N_down, _ = np.histogram(phi_down, bins=bins)

with np.errstate(divide='ignore', invalid='ignore'):
    A_N = (N_up - N_down) / (N_up + N_down)
    A_N_err = np.sqrt(1. / (N_up + N_down))
    A_N[np.isnan(A_N)] = 0
    A_N_err[np.isnan(A_N_err)] = 0

# Fit the data
popt, pcov = curve_fit(cos_fit, centers, A_N, sigma=A_N_err, absolute_sigma=True)
A_fit = popt[0]
A_fit_err = np.sqrt(np.diag(pcov))[0]

# Calculate chi-squared
residuals = A_N - cos_fit(centers, *popt)
chi2 = np.sum((residuals / A_N_err) ** 2)
ndf = len(centers) - len(popt)
chi2_ndf = chi2 / ndf

# Plot
plt.errorbar(centers, A_N, yerr=A_N_err, fmt='o', label='Asymmetry Data')
x_fit = np.linspace(-np.pi, np.pi, 1000)
plt.plot(x_fit, cos_fit(x_fit, A_fit), 'r-',
         label=f'Fit: A = {A_fit:.3f} ± {A_fit_err:.3f}, χ²/ndf = {chi2_ndf:.2f}')
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('φ_CS [rad]')
plt.ylabel('A_N(φ_CS)')
plt.title('TSSA from φ')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("tssa_phi_cs.png", dpi=300)  # Save before showing
plt.show()


# Experimental parameters
polarization = 0.8
packing_fraction = 4.67
dilution_factor = 0.18

effective_factor = polarization * packing_fraction * dilution_factor
A_N_final = A_fit / effective_factor
A_N_final_err = A_fit_err / effective_factor

#print(f'TSSA from φ_CS\nA_N_final = {A_N_final:.3f} ± {A_N_final_err:.3f}')
#plt.plot(x_fit, cos_fit(x_fit, A_fit), 'r-', label=f'Fit: A_meas = {A_fit:.3f} ± {A_fit_err:.3f}')
#plt.title(f'TSSA from φ_CS\nA_N_final = {A_N_final:.3f} ± {A_N_final_err:.3f}')



err_up = np.sqrt(N_up)
err_down = np.sqrt(N_down)

# Plot spin-up and spin-down distributions with Poisson error bars
plt.errorbar(centers, N_up, yerr=err_up, fmt='o', label='Spin Up', color='blue', capsize=3)
plt.errorbar(centers, N_down, yerr=err_down, fmt='o', label='Spin Down', color='orange', capsize=3)

# Plot formatting
plt.xlabel(r'$\phi$ [rad]')
plt.ylabel('Counts')
#plt.title(r'Distribution of $\phi_{\mathrm{CS}}$ for Spin Up/Down')
plt.axhline(0, color='gray', linestyle='--')
plt.legend()
#plt.grid(True)
plt.tight_layout()
plt.savefig("phi_distribution.png", dpi=300)
plt.show()
