import random
from typing import Callable, Optional, Tuple, List, Union
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# plt.rcParams["figure.dpi"] = 600
# plt.rcParams["savefig.dpi"] = 600

# pylint: disable=too-many-arguments
def plot_empirical_hist_distribution(
    distribution: np.ndarray,
    bins: List[float],
    ax: Optional[plt.Axes] = None,
    title="Uniform Distribution",
    linewidth: int = 2,
    alpha: float = 0.5,
    color: str = "#0504AA",
    density: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    xlabel: str = "x",
    ylabel: str = "P[X = x]",
    xticks: Optional[List[float]] = None,
) -> matplotlib.container.BarContainer:
    """Takes in a distribution (population or sample values), and plots the empirical distribution."""
    ax = ax or plt.gca()

    _, _, hist = ax.hist(
        distribution,
        bins,
        density=density,
        color=color,
        alpha=alpha,
        edgecolor="black",
        linewidth=linewidth,
    )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)

    if xticks is None:
        ax.set_xticks(bins + 0.5)
    else:
        ax.set_xticks(xticks)

    if xlim is not None:
        ax.set_xlim(*xlim)

    if ylim is not None:
        ax.set_ylim(*ylim)

    return hist


def plot_sum_of_uniform_distribution() -> None:
    """This plot is used to show the sum of multiple uniform distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    # discrete uniform distribution
    low, high = 1, 6 + 1  # [1, 6] for dice roll
    X = stats.randint(low, high)

    n_rolls = 10000
    num_rvs = 100

    list_of_rvs = [X.rvs(size=n_rolls) for _ in range(num_rvs)]
    Z1 = X.rvs(size=n_rolls)  # X
    states_z1 = np.arange(low, high + 1)
    bins_z1 = states_z1 - 0.5  # center the bins

    Z2 = list_of_rvs[0] + list_of_rvs[1]  # X1 + X2
    states_z2 = np.arange(2, 12 + 1)  # [2, 12] since it is sum of 2 variables
    bins_z2 = states_z2 - 0.5  # center the bins

    Z6 = sum(list_of_rvs[:6])  # X1 + X2 + X3 + X4 + X5 + X6
    states_z6 = np.arange(6, 36 + 1)  # [6, 36] since it is sum of 6 variables
    bins_z6 = states_z6 - 0.5  # center the bins

    Z100 = sum(list_of_rvs)  # X1 + X2 + ... + X100
    states_z100 = np.arange(100, 600 + 1)  # [100, 600] since it is sum of 100 variables
    bins_z100 = states_z100 - 0.5  # center the bins
    plot_empirical_hist_distribution(
        Z1, bins=bins_z1, ax=axes[0, 0], title="Distribution of $Z = X$"
    )
    plot_empirical_hist_distribution(
        Z2, bins=bins_z2, ax=axes[0, 1], title="Distribution of $Z = X1 + X2$"
    )
    plot_empirical_hist_distribution(
        Z6,
        bins=bins_z6,
        ax=axes[1, 0],
        title="Distribution of $Z = X1 + X2 + X3 + X4 + X5 + X6$",
    )
    plot_empirical_hist_distribution(
        Z100,
        bins=bins_z100,
        linewidth=0.01,
        xlim=(100, 600),
        xticks=np.arange(100, 600 + 1, 50),
        ax=axes[1, 1],
        title="Distribution of $Z = X1 + X2 + \ldots + X100$",
    )
    plt.show()


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


if __name__ == "__main__":
    seed_all()

    # # Standard Normal Distribution
    # mean, sigma = 0, 1
    # X = stats.norm(mean, sigma)

    # plot_continuous_pdf_and_cdf(
    #     X, -5, 5, title="Normal Distribution $\mu=0, \sigma=1$", figsize=(15, 5)
    # )

    # # Sum of Discrete Uniform Random Variables
    # plot_sum_of_uniform_distribution()
