import numpy as np

import math
# Input counts for different spin setups (grouped by detector)
N_L_up = 300 
N_L_down = 263
N_R_up = 290 
N_R_down = 276

P_up = 0.86
P_down = 0.83
dilution_factor = 0.18
packing_fraction = 0.60

# Mean polarization
P_mean = 0.5 * (P_up + P_down)
factor = dilution_factor * packing_fraction * P_mean
print("factor: ", factor)

# Raw asymmetries for Left and Right detectors
A_L = (N_L_up - N_L_down) / (N_L_up + N_L_down)
A_R = (N_R_up - N_R_down) / (N_R_up + N_R_down)

# Final Analyzing Power
A_N_final = (1.0 / factor) * ((A_L - A_R) / 2.0)

# Asymmetry error formula for spin-up and spin-down counts
#def asymmetry_error(N1, N2):
#    return 2 * np.sqrt(N1 * N2) / (N1 + N2)**2

def asymmetry_error(N_up, N_down):
    numerator = 4 * N_up * N_down
    denominator = (N_up + N_down) ** 3
    return math.sqrt(numerator / denominator)

A_L_err = asymmetry_error(N_L_up, N_L_down)
A_R_err = asymmetry_error(N_R_up, N_R_down)

# Final error propagation
A_N_final_err = 0.5 * np.sqrt(A_L_err**2 + A_R_err**2) / factor

# Output results
print(f"A_L          = {A_L:.5f} ± {A_L_err:.5f}")
print(f"A_R          = {A_R:.5f} ± {A_R_err:.5f}")
print(f"A_N_final    = {A_N_final:.5f} ± {A_N_final_err:.5f}")
