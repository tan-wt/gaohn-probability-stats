import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt


def plot_discrete_pmf(low, high, title, stats_dist=None, lw=20):

    if stats_dist is None:
        discrete = stats.randint(low, high + 1)
    else:
        discrete = stats_dist

    x = np.arange(low - 1.0, high + 1.0)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.set_xlim(low - 1, high + 1)
    ax.set_xlabel("Outcomes", fontsize=16)
    ax.set_ylabel("Probability Mass Function (pmf)", fontsize=16)
    ax.vlines(x, 0, discrete.pmf(x), colors="darkred", lw=lw, alpha=0.6)
    ax.set_ylim(0, np.max(discrete.pmf(x)) + 0.03)

    plt.title(title, fontsize=20)

    plt.show()
