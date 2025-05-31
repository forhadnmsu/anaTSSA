import ROOT
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm

# Initialize random seed
#seed = int(time.time())
#print(f"Using random seed: {seed}")
#ROOT.gRandom.SetSeed(seed)
#np.random.seed(seed)

TARGET_COUNTS = {
    "N_L_up":  300,
    "N_R_up":  290,
    "N_L_down":263,
    "N_R_down":276
}


for key, value in TARGET_COUNTS.items():
    print(f"{key}: {value}")


def compute_asymmetry(n_left: float, n_right: float) -> float:
    total = n_left + n_right
    return (n_left - n_right) / total if total != 0 else 0

def compute_a(asymmetry: float) -> float:
    return asymmetry * math.pi / 2
#Based on the LR asymmetry, now we generate the distributions in lab phi

def define_distributions(a_up: float, a_down: float) -> tuple:
    spin_up_left = ROOT.TF1("spin_up_left", f"1 + {a_up}*cos(x)", -math.pi/2, math.pi/2)
    spin_down_left = ROOT.TF1("spin_down_left", f"1 - {a_down}*cos(x)", -math.pi/2, math.pi/2)
    spin_up_right = ROOT.TF1("spin_up_right", "((x < -TMath::Pi()/2) || (x > TMath::Pi()/2)) * (1 + [0]*cos(x))", -math.pi, math.pi)
    spin_up_right.SetParameter(0, a_up)
    spin_down_right = ROOT.TF1("spin_down_right", "((x < -TMath::Pi()/2) || (x > TMath::Pi()/2)) * (1 - [0]*cos(x))", -math.pi, math.pi)
    spin_down_right.SetParameter(0, a_down)
    return spin_up_left, spin_down_left, spin_up_right, spin_down_right


def generate_events(distribution: ROOT.TF1, num_events: int, nbins: int = 1000) -> list:
    x_vals = np.linspace(distribution.GetXmin(), distribution.GetXmax(), nbins)
    y_vals = np.array([distribution.Eval(x) for x in x_vals])
    y_sum = np.sum(y_vals)
    probs = y_vals / y_sum
    # Introduce Poisson fluctuations
    counts = np.random.poisson(probs * num_events)
    bin_width = x_vals[1] - x_vals[0]
    event_list = []
    for x, count in zip(x_vals, counts):
        event_list.extend([x + np.random.uniform(-bin_width/2, bin_width/2) for _ in range(count)])
    return event_list


def cos_fit(x: np.ndarray, a: float) -> np.ndarray:
    return a * np.cos(x)

def main():
    # Compute initial parameters
    asymmetry_up = compute_asymmetry(TARGET_COUNTS["N_L_up"], TARGET_COUNTS["N_R_up"])
    asymmetry_down = compute_asymmetry(TARGET_COUNTS["N_L_down"], TARGET_COUNTS["N_R_down"])
    a_up = compute_a(asymmetry_up)  # A_up = \pi /2 \times A_LR^up
    a_down = -compute_a(asymmetry_down) # A_down = - \pi /2 \times A_LR^down

    print(f"Initial Calculations:\nA_up: {a_up:.5f}\nA_down: {a_down:.5f}\n"
          f"asymmetry_LR_up: {asymmetry_up:.5f}\nasymmetry_LR_down: {asymmetry_down:.5f}\n")

    spin_up_left, spin_down_left, spin_up_right, spin_down_right = define_distributions(a_up, a_down)

    # Simulation parameters
    NUM_TRIALS = 1000
    NBINS = 16
    PLOT_TRIAL = 16
    a_n_results = []
    a_n_errors = []
    chi2_ndf_list = []

    for trial in range(NUM_TRIALS):
        # Generate events
        phi_up_left = generate_events(spin_up_left, TARGET_COUNTS["N_L_up"])
        phi_up_right = generate_events(spin_up_right, TARGET_COUNTS["N_R_up"])
        phi_down_left = generate_events(spin_down_left, TARGET_COUNTS["N_L_down"])
        phi_down_right = generate_events(spin_down_right, TARGET_COUNTS["N_R_down"])
        phi_up = phi_up_left + phi_up_right
        phi_down = phi_down_left + phi_down_right

        # Create histograms
        hist_up_np, bins = np.histogram(phi_up, bins=NBINS, range=(-math.pi, math.pi))
        hist_down_np, _ = np.histogram(phi_down, bins=NBINS, range=(-math.pi, math.pi))
        centers = (bins[:-1] + bins[1:]) / 2

        # Compute asymmetry and errors
        a_n = (hist_up_np - hist_down_np) / (hist_up_np + hist_down_np)
        n_total = hist_up_np + hist_down_np
        a_n_err = np.sqrt((4 * hist_up_np * hist_down_np) / (n_total ** 3))
        #a_n_err = np.where(np.isnan(a_n_err) | (a_n_err == 0), 1e-10, a_n_err)

        # Plot for specific trial
        if trial == PLOT_TRIAL:
            plt.figure(figsize=(8, 6))
            plt.errorbar(centers, a_n, yerr=a_n_err, fmt='o', color='blue', label='Asymmetry data')
            plt.xlabel(r'$\phi$ (radians)')
            plt.ylabel(r'$A_N$')
            plt.title(f'Asymmetry for Trial {PLOT_TRIAL}')
            plt.grid(True)

        # Fit asymmetry
        try:
            popt, pcov = curve_fit(cos_fit, centers, a_n, sigma=a_n_err, absolute_sigma=True,
                                   p0=[np.mean(a_n)], bounds=(-1, 1), method='trf')
            a_fit, a_fit_err = popt[0], np.sqrt(pcov[0, 0])
            a_n_results.append(a_fit)
            a_n_errors.append(a_fit_err)

            chi2 = np.sum(((a_n - cos_fit(centers, a_fit)) / a_n_err) ** 2)
            ndf = len(centers) - 1
            chi2_ndf_list.append(chi2 / ndf)

            if trial < 16:
                plt.figure(figsize=(8, 6))
                plt.rcParams.update({'font.size': 16})
                plt.errorbar(centers, a_n, yerr=a_n_err, fmt='o', color='blue', label='Asymmetry data')
                x_fit = np.linspace(-math.pi, math.pi, 100)
                plt.plot(x_fit, cos_fit(x_fit, a_fit), 'r-', label=f'Fit: A={a_fit:.3f} ± {a_fit_err:.3f}, $\chi^2$/ndf={chi2/ndf:.2f}')
                plt.xlabel(r'$\phi$ (radians)')
                plt.ylabel(r'$A_N$')
                plt.title(f'Asymmetry for Trial {trial + 1}')
                plt.grid(True)
                plt.legend()
                plt.savefig(f'plot/asym_fit/asymmetry_fit_trial_{trial}.png', dpi=300)
                plt.close()

        except RuntimeError:
            print(f"Fit failed on trial {trial}")
            a_n_results.append(0)
            a_n_errors.append(0)

    # Analyze results
    valid_a_n_results = [x for x in a_n_results if x != 0]
    valid_a_n_errors = [e for e in a_n_errors if e != 0]
    mean, std = norm.fit(valid_a_n_results)
    mean_a_n_err, std_a_n_err = norm.fit(valid_a_n_errors)
    fit_failures = sum(1 for x in a_n_results if x == 0)

    print(f"Results from {NUM_TRIALS} trials:\n"
          f"Mean A_fit: {mean:.5f}\nStandard deviation of A_fit: {std:.5f}\n"
          f"Mean fit error: {mean_a_n_err:.5f}\nStandard deviation of fit errors: {std_a_n_err:.5f}\n"
          f"Number of fit failures: {fit_failures}/{NUM_TRIALS}")
    if abs(std - mean_a_n_err) / mean_a_n_err > 0.1:
        print(f"Warning: Standard deviation of A_fit ({std:.5f}) differs from "
              f"mean fit error ({mean_a_n_err:.5f}) by more than 10%")

    # Plot A_fit distribution
    counts, bin_edges = np.histogram(valid_a_n_results, bins=20, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    errors = np.sqrt(counts)
    bin_width = bin_edges[1] - bin_edges[0]
    x = np.linspace(bin_edges[0], bin_edges[-1], 1000)
    pdf = norm.pdf(x, mean, std) * len(valid_a_n_results) * bin_width

    plt.figure(figsize=(8, 6))
    plt.errorbar(bin_centers, counts, yerr=errors, fmt='o', color='skyblue',
                 ecolor='gray', capsize=3, label='Histogram counts with Poisson errors')
    plt.plot(x, pdf, 'r-', label=f'Gaussian fit: μ={mean:.3f}, σ={std:.3f}')
    plt.axvline(mean, color='blue', linestyle='--', label=f'Mean: {mean:.3f}')
    plt.xlabel(r'$A_{\text{fit}}$')
    plt.ylabel('Counts')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('AN_distribution_gaussfit_poisson.png', dpi=300)
    plt.show()

    # Plot fit error distribution
    counts_err, bin_edges_err = np.histogram(valid_a_n_errors, bins=20, density=False)
    bin_centers_err = (bin_edges_err[:-1] + bin_edges_err[1:]) / 2
    errors_err = np.sqrt(counts_err)
    bin_width_err = bin_edges_err[1] - bin_edges_err[0]
    x_err = np.linspace(bin_edges_err[0], bin_edges_err[-1], 1000)
    pdf_err = norm.pdf(x_err, mean_a_n_err, std_a_n_err) * len(valid_a_n_errors) * bin_width_err

    plt.figure(figsize=(8, 6))
    plt.errorbar(bin_centers_err, counts_err, yerr=errors_err, fmt='o', color='lightgreen',
                 ecolor='gray', capsize=3)
    plt.plot(x_err, pdf_err, 'r-', label=f'Gaussian fit: μ={mean_a_n_err:.3f}, σ={std_a_n_err:.3f}')
    plt.axvline(mean_a_n_err, color='blue', linestyle='--')
    plt.xlabel(r'$A_{\text{fit_err}}$')
    plt.ylabel('Counts')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('A_fit_error_distribution_gaussfit.png', dpi=300)
    plt.show()

    # Plot chi-square distribution
    plt.figure(figsize=(8, 6))
    plt.hist(chi2_ndf_list, bins=20, color='skyblue', label='Chi2/ndf')
    plt.xlabel('Chi2/ndf')
    plt.ylabel('Counts')
    plt.title('Chi-Square/NDF')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('chi2_ndf_distribution.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
