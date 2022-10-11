import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from typing import *


def seed_all(seed: int = 1992) -> None:
    """Seed all random number generators."""
    print(f"Using Seed Number {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)  # set PYTHONHASHSEED env var
    np.random.seed(seed)  # for numpy pseudo-random generator
    random.seed(seed)  #  set fixed value for python built-in pseudo-random generator


def plot_discrete_pmf(low, high, title, stats_dist=None, lw=20, **kwargs) -> None:
    """Plot the PMF of a discrete distribution.

    Args:
        low (int): Lower bound of the distribution.
        high (int): Upper bound of the distribution.
        title (str): Title of the plot.
        stats_dist (scipy.stats.rv_discrete, optional): A scipy discrete distribution. Defaults to None.
        lw (int, optional): Line width of the plot. Defaults to 20.
        **kwargs: Keyword arguments to pass to the stats_dist.pmf function.
    """
    x = np.arange(low, high + 1)
    if stats_dist:
        y = stats_dist.pmf(x, **kwargs)
    else:
        y = np.ones(len(x)) / len(x)

    plt.stem(
        x,
        y,
        linefmt="C0-",
        markerfmt="C0o",
        use_line_collection=True,
        basefmt="C0-",
        label="PMF",
    )
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("P(X=x)")
    plt.xlim(low - 1, high + 1)
    plt.xticks(np.arange(low, high + 1, 1))
    plt.legend()
    plt.show()
