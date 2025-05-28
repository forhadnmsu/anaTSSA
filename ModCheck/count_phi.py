import uproot
import numpy as np

file = uproot.open("data_in/phi_angles_with_asym.root")
tree = file["phi_tree"]

phi = tree["phi"].array(library="np")
spin = tree["spin"].array(library="np")

left_side = ((phi >= -np.pi/2) & (phi <= np.pi/2))
right_side = ((phi > np.pi/2) & (phi <= np.pi)) | ((phi >= -np.pi) & (phi < -np.pi/2))

spin_up = (spin == 1)
spin_down = (spin == -1)

# Calculate counts for each condition
left_spin_up = np.sum(left_side & spin_up)
left_spin_down = np.sum(left_side & spin_down)
right_spin_up = np.sum(right_side & spin_up)
right_spin_down = np.sum(right_side & spin_down)

print(f"Left side (phi: [-π/2, π/2]):")
print(f"  Spin Up (spin = 1): {left_spin_up} events")
print(f"  Spin Down (spin = -1): {left_spin_down} events")
print(f"Right side (phi: (π/2, π] or [-π, -π/2)):")
print(f"  Spin Up (spin = 1): {right_spin_up} events")
print(f"  Spin Down (spin = -1): {right_spin_down} events")

