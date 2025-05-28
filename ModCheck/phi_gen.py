import ROOT
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from array import array

# Constants
muon_mass = 0.1056
A_injected = 0.1
max_prob = 1 + A_injected

# Step 1: Simulate and save events with spin conditions
output_file = ROOT.TFile("data_in/phi_angles_with_asym.root", "RECREATE")
phi = array('f', [0.0])
spin = array('i', [0])
tree_out = ROOT.TTree("phi_tree", "phi and spin")
tree_out.Branch("phi", phi, "phi/F")
tree_out.Branch("spin", spin, "spin/I")

phi_up = []
phi_down = []

count_up = 0
count_down = 0
total_generated = 0

while count_up + count_down < 50000:
    # Generate a random phi uniformly in [-pi, pi]
    phi_cs = random.uniform(-math.pi, math.pi)
    cos_phi = math.cos(phi_cs)
    # Alternate spin state
    spin_state = 1 if total_generated % 2 == 0 else -1
    r = random.uniform(0, 1)

    if spin_state == 1:
        prob = (1 + A_injected * cos_phi) / max_prob
        if r < prob:
            phi[0] = phi_cs
            phi_up.append(phi_cs)
            spin[0] = 1
            tree_out.Fill()
            count_up += 1
    else:
        prob = (1 - A_injected * cos_phi) / max_prob
        if r < prob:
            phi[0] = phi_cs
            phi_down.append(phi_cs)
            spin[0] = -1
            tree_out.Fill()
            count_down += 1

    total_generated += 1
output_file.Write()
output_file.Close()
