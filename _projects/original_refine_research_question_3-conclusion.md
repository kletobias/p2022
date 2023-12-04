## Research Question 3 - Conclusion

> 3\. Show that the distribution of the ADV converges against a normal
distribution $$N(\mu_{N},\,\sigma_{N})$$ with $$\mu_{N}\,=\hat{\mu}$$ and
$$\sigma_{N}\,=\,\hat{\sigma}$$, $$\hat{\mu}$$ and $$\hat{\sigma}$$ being the
empirical mean and standard deviation from the simulation, as stated by the
central limit theorem.<br>

A histogram showing the relative frequency of dust means per trial over the
entire series. Smaller sample sizes, as well as fewer packs per trial, result in
less Bell Curve shaped PDFs, with the PDF for simulations with 40 packs (the
maximum one can buy at a time, at the time of writing) per trial and 100,000
trials per simulation approaching the shape of a Normal Distribution with the
same mean and standard deviation as found in the simulation sample.

Sample means become normally distributed as sample sizes tend to infinity,
according to the Central Limit Theorem. Therefore, it should be converging
towards the population mean standard deviation. The largest sample size, limited
by the local M1 chip apple machine, was 100,000 trials each with 40 packs and 5
cards per pack. In the largest simulation, 20,000,000 cards are drawn. As shown
below, the PDF of the ADV approximates the PDF of a Normal Distribution with the
same mean and standard deviation.

The plot, where the empirical PDFs from the simulations are compared with each
other and with the PDF of the Normal distribution described above. This
distribution has identical mean and standard deviation, the goodness of fit of
the Normal distribution remains somewhat unclear. The plot does show that the
overall bell curve-like shape is shared by all three PDFs. The Normal
distribution uses the mean and standard deviation from LI, which are identical
to the ones from the NVI for all plots.



```python
mu = dust_avg_all_trials
li_std = np.std(dust_avg_t)
dist = norm(mu, li_std)
low_end = int(np.ceil(dist.ppf(0.0001)))
high_end = int(np.ceil(dist.ppf(0.9999)))
x_vals = [x for x in np.arange(low_end, high_end, 0.5)]
y_vals = [dist.pdf(x) for x in x_vals]

print(f"The current mu, sigma have values {mu} and {li_std} respectively.")

fig, ax = plt.subplots(1, 1, figsize=(18.5, 12.5))
plt.title(f"Histogram from packs,trials: {packs},{trials:,}", fontsize=14)
ax.hist(dust_avg_t, density=True, bins=200, label="LI Density Dust per Trial")
ax.hist(np_mu_trial, density=True, bins=200, label="NVI Density Dust per Trial")
ax.plot(
    x_vals,
    y_vals,
    color="#850859",
    alpha=0.7,
    label="PDF-Normal w/ $μ$ and $σ$ from Simulation",
)
# ax.axvline(x=np.min(dust_avg_t), color="y", linestyle="dotted", label="min of μ")
# ax.axvline(x=np.max(dust_avg_t), color="r", linestyle="dotted", label="max of μ")
ax.set_ylabel("Relative Frequency")
ax.set_xlabel("Average Dust Per Trial")
ax.set_xlim(low_end, high_end)
ax.legend(loc="best", fontsize=10)
plt.show()
```

    The current mu, sigma have values 257.0840275 and 34.91854175261681 respectively.



    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects_hearthstone-euro-to-in-game-currency-conversion/output_33_1.png" title="" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
        Figure 3: The PDFs of both implementations are overlaid and compared to
        the PDF of a Normal distribution with the same mean and standard
        deviation.
</div>
    


The ECDF plots for both implementations plotted together with the CDF of the
theoretical Normal distribution described above, gives a clearer picture as to
where the differences are found between the theoretical Normal distribution and
the inseparable simulation ECDF functions.


```python
# create the ECDF plot for the average dust per trial Random Variable
ecdf = ECDF(dust_avg_t)
ecdf_np = ECDF(np_mu_trial)
# The first value in ecdf.x is -inf, which pyplot does not like.
# So the first value in ecdf.x and ecdf.y is dropped to keep their lengths the same.
ecdf.x = ecdf.x[1:]
ecdf.y = ecdf.y[1:]
ecdf_np.x = ecdf_np.x[1:]
ecdf_np.y = ecdf_np.y[1:]

# using mu and sigma from above as parameters for the theoretical distribution
dist2 = norm.rvs(mu, li_std, 20000000)
dist2s = np.sort(dist2)
ecdf_n = np.arange(1, len(dist2s) + 1) / len(dist2s)
plt.close("all")
fig, ax = plt.subplots(1, 1, figsize=(12, 10), constrained_layout=True)
plt.title(
    f"Theoretical CDF from μ, σ from packs,trials:\n{packs},{trials:,} and ECDF ",
    fontsize=14,
)
ax.plot(dist2s, ecdf_n, color="blue", alpha=0.8, label="Theoretical CDF")
ax.plot(ecdf.x, ecdf.y, color="#FC5A50", alpha=0.8, label="ECDF for-loop")
ax.plot(ecdf_np.x, ecdf_np.y, color="#08877d", alpha=0.8, label="ECDF Numpy")
ax.set_ylabel("CDF")
ax.set_xlabel("Average Dust per Trial")
ax.set_xlim(130, 410)
plt.legend(loc="best")
plt.show()
```


    
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects_hearthstone-euro-to-in-game-currency-conversion/output_35_0.png" title="Figure 4" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
        Figure 4: Using the Normal distribution from Figure 3, as well as the
        empirical distributions found by the simulations, their ECDF functions
        are compared to the CDF of the Normal distribution. The plot shows in
        detail where the three functions differ.
</div>
    
