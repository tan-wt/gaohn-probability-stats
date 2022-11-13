import random
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# pylint: disable=too-many-arguments,too-many-locals
def plot_continuous_pdf_and_cdf(
    stats_dist: Callable,
    low: float,
    high: float,
    title="Uniform Distribution",
    lw: int = 3,
    alpha: float = 0.7,
    color: str = "darkred",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (10, 5),
) -> None:
    """Plots the PDF and CDF of a continuous distribution."""
    x = np.linspace(low, high, 5000)
    X = stats_dist  # symbolic X to represent the distribution

    pdf = X.pdf(x)
    cdf = X.cdf(x)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    if xlim is not None:
        axes[0].set_xlim(*xlim)

    if ylim is not None:
        axes[0].set_ylim(*ylim)

    axes[0].set_xlabel("x", fontsize=16)
    axes[0].set_ylabel("pdf(x)", fontsize=16)

    axes[0].plot(x, pdf, color=color, lw=lw, alpha=alpha)

    axes[0].set_title(f"PDF of {title}", fontsize=16)

    if xlim is not None:
        axes[1].set_xlim(*xlim)

    axes[1].set_ylim(
        0, 1.1
    )  # this can be hardcoded since the CDF is always between 0 and 1

    axes[1].set_xlabel("x", fontsize=16)
    axes[1].set_ylabel("cdf(x)", fontsize=16)

    axes[1].plot(x, cdf, color=color, lw=lw, alpha=alpha)

    axes[1].set_title(f"CDF of {title}", fontsize=16)

    plt.show()


def seed_all(seed: int = 1992) -> None:
    """Seed all random number generators."""
    print(f"Using Seed Number {seed}")

    # os.environ["PYTHONHASHSEED"] = str(seed)  # set PYTHONHASHSEED env var
    np.random.seed(seed)  # numpy pseudo-random generator
    random.seed(seed)  # built-in pseudo-random generator


def true_pmf(x: float, population: np.ndarray) -> float:
    """PMF of the true population: map X(\S) to a probability.

    Note:
        The PMF is completely determined if we know the true distribution.
    """
    return np.sum(population == x) / len(population)


def empirical_pmf(x: float, sample: np.ndarray):
    """Empirical distribution of the sample."""
    return np.sum(sample == x) / len(sample)


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


if __name__ == "__main__":
    seed_all()

    # Standard Normal Distribution
    mean, sigma = 0, 1
    X = stats.norm(mean, sigma)

    plot_continuous_pdf_and_cdf(
        X, -5, 5, title="Normal Distribution $\mu=0, \sigma=1$", figsize=(15, 5)
    )
