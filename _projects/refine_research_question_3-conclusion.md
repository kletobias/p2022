
## Research Question 3 - Conclusion

> Show that the distribution of the Average Dust Value (ADV) converges towards
> a normal distribution \( N(\mu_{N}, \sigma_{N}) \) where \( \mu_{N} =
> \hat{\mu} \) and \( \sigma_{N} = \hat{\sigma} \), with \( \hat{\mu} \) and \(
> \hat{\sigma} \) representing the empirical mean and standard deviation from
> the simulation. This observation aligns with the predictions of the central
> limit theorem.

The histogram represents the relative frequency of dust means per trial across
the simulation series. It reveals that smaller sample sizes and fewer packs per
trial yield distributions less representative of a normal distribution.
Conversely, the simulation with 40 packs per trial and 100,000 trials, which is
the maximum purchasable quantity at the time of writing, closely approximates a
normal distribution with equivalent mean and standard deviation.

According to the Central Limit Theorem, as sample sizes increase, the sample
means should distribute normally, converging on the population mean and
standard deviation. In the largest simulation run on a local M1 chip-based
Apple machine, 100,000 trials consisted of 40 packs with 5 cards each, totaling
20,000,000 cards. The resulting Probability Density Function (PDF) of the ADV
approximates that of a normal distribution with matching mean and standard
deviation.

Below, the empirical PDFs from the simulations are juxtaposed against the
normal distribution's PDF. The fit's precision is not definitive, but the
bell-shaped curve is consistent across all three PDFs. The normal distribution
employs the simulation's mean and standard deviation.

```python
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

# Given simulation data
mu: float = dust_avg_all_trials  # Empirical mean
li_std: float = np.std(dust_avg_t)  # Empirical standard deviation
packs: int = 40
trials: int = 100_000

# Normal distribution range setup
dist: norm = norm(mu, li_std)
low_end: int = int(np.ceil(dist.ppf(0.0001)))
high_end: int = int(np.ceil(dist.ppf(0.9999)))
x_vals: np.ndarray = np.arange(low_end, high_end, 0.5)
y_vals: np.ndarray = dist.pdf(x_vals)

# Plotting the histogram and PDF
fig, ax = plt.subplots(figsize=(18.5, 12.5))
plt.title(f"Histogram from packs,trials: {packs}, {trials:,}", fontsize=14)
ax.hist(dust_avg_t, density=True, bins=200, label="LI Density Dust per Trial")
ax.hist(np_mu_trial, density=True, bins=200, label="NVI Density Dust per Trial")
ax.plot(x_vals, y_vals, color="#850859", alpha=0.7, label="PDF-Normal with μ and σ from Simulation")
ax.set_ylabel("Relative Frequency")
ax.set_xlabel("Average Dust Per Trial")
ax.set_xlim(low_end, high_end)
ax.legend(loc="best", fontsize=10)
plt.show()

print(f"The current mu, sigma have values {mu:.2f} and {li_std:.2f} respectively.")
