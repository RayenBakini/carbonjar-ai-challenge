# ============================================
# Monte Carlo Simulation for Decarbonization
# Author: Rayen Bakini
# ============================================

import numpy as np
import matplotlib.pyplot as plt

def simulate_scenario(investment):
    """
    Run a single decarbonization scenario with uncertainty.
    - cost_reduction_factor (0.10, 0.02)
    - emission_cut_factor (0.05, 0.01)
    """
    # draw random factors
    cost_reduction_factor = np.random.normal(loc=0.10, scale=0.02)
    emission_cut_factor  = np.random.normal(loc=0.05, scale=0.01)
    emission_value       = 50  # value per ton cut

    # compute components
    cost_reduction = investment * cost_reduction_factor
    emission_cut   = investment * emission_cut_factor

    # net benefit = revenue from emission cuts minus net investment cost
    net_benefit = (emission_cut * emission_value) - (investment - cost_reduction)
    return net_benefit

def run_monte_carlo(investment, runs=1000):
    """
    Execute the simulation multiple times and return an array of net benefits.
    """
    results = [simulate_scenario(investment) for _ in range(runs)]
    return np.array(results)

if __name__ == "__main__":
    # parameters
    investment = 100_000   # example budget in currency units
    n_runs     = 1000      # number of Monte Carlo runs

    # run simulation
    benefits = run_monte_carlo(investment, n_runs)
    print(f" Monte Carlo completed ({n_runs} runs)")
    print(f"Mean net benefit : {benefits.mean():.2f}")
    print(f"Std dev net benefit: {benefits.std():.2f}")

    # plot histogram
    plt.hist(benefits, bins=30)
    plt.title("Distribution of Net Benefit (Monte Carlo Simulation)")
    plt.xlabel("Net Benefit")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
