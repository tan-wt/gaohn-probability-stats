import random
from typing import Callable, Optional, Tuple, List, Union
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


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


def plot_discrete_pmf(
    stats_dist: Callable,
    states: np.ndarray,
    ax: Optional[plt.Axes] = None,
    xlabel: str = "x",
    ylabel: str = "pmf(x)",
    xlim: Optional[Tuple[float, float]] = None,
    xticks: Optional[np.ndarray] = None,
    linefmt: str = "r-",
    markerfmt: str = "C0o",
    basefmt: str = "C0-",
    label: str = "PMF",
    title: str = "PMF",
    **kwargs,
) -> matplotlib.container.StemContainer:
    """Plot the PMF of a discrete distribution."""
    ax = ax or plt.gca()

    low, high = states.min(), states.max()

    x = np.arange(low, high + 1)  # The states of the distribution X
    y = stats_dist.pmf(x)  # The probability of each state
    # print(f"PMF of X: {y}")

    stem = ax.stem(
        x,
        y,
        linefmt=linefmt,
        markerfmt=markerfmt,
        basefmt=basefmt,
        label=label,
        **kwargs,
    )

    if xlim is None:
        ax.set_xlim(low - 1, high + 1)
    else:
        ax.set_xlim(*xlim)

    if xticks is None:
        ax.set_xticks(np.arange(low, high + 1, 1))
    else:
        ax.set_xticks(xticks)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")
    return stem


def plot_bernoulli_pmf(p: float) -> None:
    """Plot the PMF of a Bernoulli distribution.

    TODO: To refine by adding in appropriate arguments."""

    # `X` is now an object that represents a Bernoulli random variable with parameter $p$.
    X = stats.bernoulli(p)

    states = np.asarray([0, 1])  # The states of the distribution X

    fig, ax = plt.subplots(1)

    plot_discrete_pmf(X, states=states, title="PMF of Bernoulli($p=0.2$)")

    plt.show()


def plot_binomial_pmf(n: int, p: float) -> None:
    """Plot the PMF of a Binomial distribution."""
    # `X` is now an object that represents a Binomial random variable with parameter $p$ and $n$.

    X = stats.binom(n, p)
    states_x = np.arange(0, n + 1)  # The states of the distribution X
    # if n = 3, then states = [0, 1, 2, 3]

    fig, ax = plt.subplots(1)
    plot_discrete_pmf(
        X,
        states=states_x,
        title=f"PMF of Binomial($n={n}, p={p}$)",
    )

    plt.show()


def plot_multiple_binomial_pmf() -> None:
    """Plot the PMF of a Binomial distribution."""
    # `X` is now an object that represents a Binomial random variable with parameter $p$ and $n$.

    fig, axes = plt.subplots(2, 1, figsize=(30, 20))

    ############# Fix n = 60, vary p #############
    n, p = 60, 0.1
    X1 = stats.binom(n, p)
    states_x1 = np.arange(0, n + 1)  # The states of the distribution X1
    # if n = 3, then states = [0, 1, 2, 3]

    n, p = 60, 0.5
    X2 = stats.binom(n, p)
    states_x2 = np.arange(0, n + 1)  # The states of the distribution X2

    n, p = 60, 0.9
    X3 = stats.binom(n, p)
    states_x3 = np.arange(0, n + 1)  # The states of the distribution X3

    plot_discrete_pmf(
        X1,
        states=states_x1,
        linefmt="r-",
        markerfmt="ro",
        label="$n=60, p=0.1$",
        title=f"PMF of Binomial, Fixed $n={n}$, Varying $p$",
        ax=axes[0],
    )
    plot_discrete_pmf(
        X2,
        states=states_x2,
        linefmt="b-",
        markerfmt="bo",
        label="$n=60, p=0.5$",
        title=f"PMF of Binomial, Fixed $n={n}$, Varying $p$",
        ax=axes[0],
    )
    plot_discrete_pmf(
        X3,
        states=states_x3,
        linefmt="g-",
        markerfmt="go",
        label="$n=60, p=0.9$",
        title=f"PMF of Binomial, Fixed $n={n}$, Varying $p$",
        ax=axes[0],
    )

    ############# Fix p=0.5, vary n #############
    n, p = 5, 0.5
    X1 = stats.binom(n, p)
    states_x1 = np.arange(0, n + 1)  # The states of the distribution X1
    # if n = 3, then states = [0, 1, 2, 3]

    n, p = 50, 0.5
    X2 = stats.binom(n, p)
    states_x2 = np.arange(0, n + 1)  # The states of the distribution X2

    n, p = 100, 0.5
    X3 = stats.binom(n, p)
    states_x3 = np.arange(0, n + 1)  # The states of the distribution X3

    plot_discrete_pmf(
        X1,
        states=states_x1,
        linefmt="r-",
        markerfmt="ro",
        label="$n=5, p=0.5$",
        title=f"PMF of Binomial, Fixed $p={p}$, Varying $n$",
        ax=axes[1],
    )
    plot_discrete_pmf(
        X2,
        states=states_x2,
        linefmt="b-",
        markerfmt="bo",
        label="$n=50, p=0.5$",
        title=f"PMF of Binomial, Fixed $p={p}$, Varying $n$",
        ax=axes[1],
    )
    plot_discrete_pmf(
        X3,
        states=states_x3,
        linefmt="g-",
        markerfmt="go",
        label="$n=100, p=0.5$",
        title=f"PMF of Binomial, Fixed $p={p}$, Varying $n$",
        ax=axes[1],
    )

    plt.show()


if __name__ == "__main__":
    seed_all()

    # Bernoulli PMF
    # plot_bernoulli_pmf(p=0.2)

    # Binomial PMF
    # plot_binomial_pmf(n=3, p=0.2)
    plot_multiple_binomial_pmf()

    # # Standard Normal Distribution
    # mean, sigma = 0, 1
    # X = stats.norm(mean, sigma)

    # plot_continuous_pdf_and_cdf(
    #     X, -5, 5, title="Normal Distribution $\mu=0, \sigma=1$", figsize=(15, 5)
    # )

    # # Sum of Discrete Uniform Random Variables
    # plot_sum_of_uniform_distribution()
