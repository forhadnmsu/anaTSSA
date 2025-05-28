import numpy as np

# Input counts in different spin setup
N_L_up = 13277 
N_L_down = 11645

N_R_up = 11713
N_R_down = 13365

#P_up = 0.86
#P_down = 0.83
#dilution_factor = 0.18
#packing_fraction = 0.60
#relative_luminosity = 1.0 has been used

# Mean polarization
#P_mean = 0.5 * (P_up + P_down)

#factor = dilution_factor*packing_fraction*P_mean
factor =1
print("factor: ", factor)

# Raw asymmetries
A_raw_up = (N_L_up - N_R_up) / (N_L_up + N_R_up)
A_raw_down = (N_L_down - N_R_down) / (N_L_down + N_R_down)

A_N_final = (1.0 /factor)* ((A_raw_up - A_raw_down) / 2.0)

def asymmetry_error(NL, NR):
    return np.sqrt(4 * NL * NR / ((NL + NR) ** 3))


A_raw_up_err = asymmetry_error(N_L_up, N_R_up) 
A_raw_down_err = asymmetry_error(N_L_down, N_R_down)

numerator_err = 0.5 * np.sqrt(A_raw_up_err**2 + A_raw_down_err**2)

print(f"A_raw_up     = {A_raw_up:.5f} ± {A_raw_up_err:.5f}")
print(f"A_raw_down   = {A_raw_down:.5f} ± {A_raw_down_err:.5f}")

# Constant
pi_over_2 = np.pi / 2

# Calculate A_N^up and A_N^down
A_N_up = A_raw_up * pi_over_2
A_N_down = -A_raw_down * pi_over_2  # Note the minus sign
A_N_up_err = A_raw_up_err * pi_over_2
A_N_down_err = A_raw_down_err * pi_over_2

A_N_avg = (A_N_up + A_N_down) / 2
A_N_avg_err = 0.5 * np.sqrt(A_N_up_err**2 + A_N_down_err**2)

print(f"A_N^up     = {A_N_up:.5f} ± {A_N_up_err:.5f}")
print(f"A_N^down   = {A_N_down:.5f} ± {A_N_down_err:.5f}")
print(f"A_N^avg    = {A_N_avg:.5f} ± {A_N_avg_err:.5f}")
