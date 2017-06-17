import numpy as np
import matplotlib.pyplot as plt
import corner

# Corner Plot of the Posterior Sample
posterior_sample = np.loadtxt(fname = "posterior_sample.txt")
corner.corner(posterior_sample, labels = [r"$\beta_{0}$", r"$\beta_{1}$", 
r"$\beta_{2}$", r"$\beta_{3}$", r"$\sigma$"],
quantiles = [0.16, 0.5, 0.84], show_titles = True, title_kwargs = {"fontsize": 12})
plt.savefig('Corner_Plot.png')
plt.show()
