# Question 15 – Scenario Simulation for Decarbonization 🎲

## Problem
Clients need to explore “what-if” decarbonization scenarios under uncertainty in cost savings and emission reductions. 
A single deterministic run is not enough to understand risk.

---

## What I Did

1. **Added randomness** to the original `simulate_scenario`:
   - `cost_reduction_factor` drawn from a Normal(0.10, 0.02).
   - `emission_cut_factor` drawn from a Normal(0.05, 0.01).
2. **Wrapped it** in `run_monte_carlo(...)` to execute 1 000 runs.
3. **Collected** all net benefits in an array.
4. **Printed** summary statistics (mean, standard deviation).
5. **Plotted** a histogram of the net benefit distribution.

---

