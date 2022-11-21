import random
from typing import Any, Callable, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.container import BarContainer, StemContainer
from scipy import stats
import seaborn as sns
from utils import seed_all
import matplotlib

seed_all(42)
COLOR_INDEXES = list(range(0, 10))
random.shuffle(COLOR_INDEXES)


def plot_discrete_empirical_histogram(  # pylint: disable=too-many-arguments
    distribution: Callable,
    states: np.ndarray,
    size: int = 1000,
    bins: Optional[Union[List[float], np.ndarray]] = None,
    ax: Optional[plt.Axes] = None,
    ax_kwargs: Optional[Dict[str, Any]] = None,
    **hist_kwargs: Dict[str, Any],
) -> plt.Axes:
    """Takes in a distribution (population or sample values), and plots the empirical distribution."""
    ax = ax or plt.gca()

    if ax_kwargs is None:
        ax_kwargs = {}

    if not hist_kwargs:
        hist_kwargs = {
            "edgecolor": "black",
            "linewidth": 2,
            "alpha": 0.5,
            "color": "#0504AA",
            "stat": "probability",  # "density",
            "discrete": True,
        }

    empirical_samples = distribution.rvs(size=size)
    # print(f"Empirical samples: {empirical_samples}")

    # center the bins on the states, for discrete distributions.
    bins = np.arange(0, states.max() + 1.5) - 0.5 if bins is None else bins
    # print(f"Bins: {bins}")

    # Matplotlib version.
    # _, _, hist = ax.hist(
    #     empirical_samples,
    #     bins,
    #     **hist_kwargs,
    # )
    hist = sns.histplot(empirical_samples, bins=bins, ax=ax, **hist_kwargs)

    ax.set_title(
        ax_kwargs.get("title", "Empirical Histogram/Distribution"), fontsize=16
    )
    ax.set_xlabel(ax_kwargs.get("xlabel", "x"), fontsize=12)
    ax.set_ylabel(ax_kwargs.get("ylabel", "relative frequency"), fontsize=12)
    ax.set_xticks(ax_kwargs.get("xticks", bins + 0.5))
    ax.set_xlim(*ax_kwargs.get("xlim", (min(bins), max(bins))))
    # ax.set_ylim(*ax_kwargs.get("ylim", (0, 1)))
    ax.legend(loc="best")
    return hist


def plot_discrete_pmf(  # pylint: disable=too-many-arguments
    distribution: Callable,
    states: np.ndarray,
    ax: Optional[plt.Axes] = None,
    ax_kwargs: Optional[Dict[str, Any]] = None,
    **stem_kwargs: Dict[str, Any],
) -> StemContainer:
    """Plot the PMF of a discrete distribution.

    Args:
        distribution (Callable): A function that takes in a state and returns the probability of that state.
            Typically called from a scipy.stats distribution object.
        states (np.ndarray): The states of the distribution.
        ax (Optional[plt.Axes], optional): The axes to plot on. Defaults to None.
        plt_kwargs (Optional[Dict[str, Any]], optional): Keyword arguments for ax.
        **stem_kwargs (Dict[str, Any]): Keyword arguments to pass to the stem.
    """
    ax = ax or plt.gca()

    if ax_kwargs is None:
        ax_kwargs = {}

    if not stem_kwargs:
        stem_kwargs = {
            "linefmt": "b-",
            "markerfmt": "bo",
            "basefmt": "C7-",
            "label": f"PMF of {distribution.__name__}",
        }

    low, high = states.min(), states.max()
    x = np.arange(low, high + 1)  # The states of the distribution X
    y = distribution.pmf(x)  # The probability of each state

    stem = ax.stem(
        x,
        y,
        **stem_kwargs,
    )
    ax.set_title(ax_kwargs.get("title", "PMF"), fontsize=16)
    ax.set_xlabel(ax_kwargs.get("xlabel", "x"), fontsize=12)
    ax.set_ylabel(ax_kwargs.get("ylabel", "pmf(x)"), fontsize=12)
    ax.set_xlim(*ax_kwargs.get("xlim", (low - 1, high + 1)))
    ax.set_xticks(ax_kwargs.get("xticks", np.arange(low, high + 1, 1)))
    ax.legend(loc="best")
    return stem


def plot_discrete_uniform_pmf(
    a: int, b: int, ax: Optional[plt.Axes] = None
) -> StemContainer:
    """Plot the PMF of a Uniform distribution."""
    # X is now an object that represents a Uniform random variable with parameter $a$ and $b$.
    ax = ax or plt.gca()
    X = stats.randint(a, b + 1)
    states = np.arange(a, b + 1)  # Uniform has b - a + 1 states.

    ax_kwargs = {"title": f"PMF of Uniform($a={a}, b={b}$)"}
    stem_kwargs = {
        "linefmt": "b-",
        "markerfmt": "bo",
        "basefmt": "C7-",
        "label": f"PMF of Uniform($a={a}, b={b}$)",
    }

    stem = plot_discrete_pmf(
        X, states=states, ax=ax, ax_kwargs=ax_kwargs, **stem_kwargs
    )
    return stem


def plot_bernoulli_pmf(p: float, ax: Optional[plt.Axes] = None) -> StemContainer:
    """Plot the PMF of a Bernoulli distribution."""
    ax = ax or plt.gca()
    # X is now an object that represents a Bernoulli random variable with parameter $p$.
    X = stats.bernoulli(p)
    states = np.asarray([0, 1])  # Bernoulli only has two states, 0 and 1.

    ax_kwargs = {"title": f"PMF of Bernoulli($p={p}$)"}
    stem_kwargs = {"linefmt": "r-", "markerfmt": "ro", "basefmt": "C7-", "label": "PMF"}

    stem = plot_discrete_pmf(
        X, states=states, ax=ax, ax_kwargs=ax_kwargs, **stem_kwargs
    )
    return stem


def plot_empirical_bernoulli(
    p: float, size: int = 1000, ax: Optional[plt.Axes] = None
) -> Union[BarContainer, plt.Axes]:
    """Plot the empirical distribution of a Bernoulli distribution."""
    ax = ax or plt.gca()
    # X is now an object that represents a Bernoulli random variable with parameter $p$.
    X = stats.bernoulli(p)
    states = np.asarray([0, 1])  # Bernoulli only has two states, 0 and 1.
    bins = np.arange(0, states.max() + 1.5) - 0.5

    ax_kwargs = {
        "title": None,
        "ylabel": None,
    }
    hist_kwargs = {
        "edgecolor": "black",
        "linewidth": 2,
        "alpha": 0.5,
        "color": "#0504AA",
        "density": True,
        "label": "Empirical Histogram",
    }

    hist = plot_discrete_empirical_histogram(
        X,
        states=states,
        size=size,
        bins=bins,
        ax=ax,
        ax_kwargs=ax_kwargs,
        **hist_kwargs,
    )
    return hist


def plot_empirical_binomial(
    p: float, n: int, size: int = 1000, ax: Optional[plt.Axes] = None
) -> Union[BarContainer, plt.Axes]:
    """Plot the empirical distribution of a Bernoulli distribution."""
    ax = ax or plt.gca()
    X = stats.binom(n, p)
    states = np.arange(0, n + 1)  # Binomial has n + 1 states.
    bins = np.arange(0, states.max() + 1.5) - 0.5

    ax_kwargs = {
        "title": None,
        "ylabel": None,
    }
    hist_kwargs = {
        "edgecolor": "black",
        "linewidth": 2,
        "alpha": 0.5,
        "color": "#0504AA",
        "stat": "probability",
        "label": "Empirical Histogram",
    }

    hist = plot_discrete_empirical_histogram(
        X,
        states=states,
        size=size,
        bins=bins,
        ax=ax,
        ax_kwargs=ax_kwargs,
        **hist_kwargs,
    )
    return hist


def plot_binomial_pmfs(
    ns: List[int], ps: List[float], ax: Optional[plt.Axes] = None
) -> StemContainer:
    """Plot the PMFs of multiple Binomial distributions on the same axes.

    Args:
        ns (List[int]): The number of trials for each Binomial distribution.
        ps (List[float]): The probability of success for each Binomial distribution.
        ax (Optional[plt.Axes], optional): The axes to plot on. Defaults to None.

    Returns:
        stem (StemContainer): The stem plot.
    """
    ax = ax or plt.gca()

    for n, p in zip(ns, ps):
        X = stats.binom(n, p)
        states = np.arange(0, n + 1)

        xticks = np.arange(0, n + 1, 5)
        color_index = COLOR_INDEXES.pop()

        ax_kwargs = {"title": f"PMF of Binomial($n={n}, p={p}$)", "xticks": xticks}
        stem_kwargs = {
            "linefmt": f"C{color_index}-",
            "markerfmt": f"C{color_index}o",
            "basefmt": "C7-",
            "label": f"PMF of Binomial($n={n}, p={p}$)",
        }

        stem = plot_discrete_pmf(
            X, states=states, ax=ax, ax_kwargs=ax_kwargs, **stem_kwargs
        )
    return stem


def plot_poisson_pmfs(
    lambdas: List[float], ax: Optional[plt.Axes] = None
) -> StemContainer:
    """Plot the PMFs of multiple Poisson distributions on the same axes."""
    ax = ax or plt.gca()

    for lam in lambdas:
        X = stats.poisson(lam)
        states = np.arange(
            0, 30
        )  # Poisson has infinite states, but we only plot up to 30.

        xticks = np.arange(0, 30, 5)
        color_index = COLOR_INDEXES.pop()

        ax_kwargs = {"title": f"PMF of Poisson($\lambda={lam}$)", "xticks": xticks}
        stem_kwargs = {
            "linefmt": f"C{color_index}-",
            "markerfmt": f"C{color_index}o",
            "basefmt": "C7-",
            "label": f"PMF of Poisson($\lambda={lam}$)",
        }

        stem = plot_discrete_pmf(
            X, states=states, ax=ax, ax_kwargs=ax_kwargs, **stem_kwargs
        )
    return stem


def plot_empirical_poisson(
    lambdas: List[int], size: int = 1000, ax: Optional[plt.Axes] = None
) -> Union[BarContainer, plt.Axes]:
    """Plot the empirical distribution of a Bernoulli distribution."""
    ax = ax or plt.gca()

    for lam in lambdas:
        X = stats.poisson(lam)
        states = np.arange(
            0, 30
        )  # Poisson has infinite states, but we only plot up to 30.

        bins = np.arange(0, states.max() + 1.5) - 0.5

        ax_kwargs = {
            "title": None,
            "ylabel": None,
        }
        hist_kwargs = {
            "edgecolor": "black",
            "linewidth": 2,
            "alpha": 0.5,
            "color": "#0504AA",
            "stat": "probability",
            "label": "Empirical Histogram",
        }

        hist = plot_discrete_empirical_histogram(
            X,
            states=states,
            size=size,
            bins=bins,
            ax=ax,
            ax_kwargs=ax_kwargs,
            **hist_kwargs,
        )
    return hist


def plot_geometric_pmfs(
    ps: List[float], ax: Optional[plt.Axes] = None
) -> StemContainer:
    """Plot the PMFs of multiple Geometric distributions on the same axes."""
    ax = ax or plt.gca()

    for p in ps:
        X = stats.geom(p)
        states = np.arange(
            1, 10
        )  # Geometric has infinite states, but we only plot up to 10.

        xticks = np.arange(0, 10, 1)
        color_index = COLOR_INDEXES.pop()

        ax_kwargs = {"title": f"PMF of Geometric($p={p}$)", "xticks": xticks}
        stem_kwargs = {
            "linefmt": f"C{color_index}-",
            "markerfmt": f"C{color_index}o",
            "basefmt": "C7-",
            "label": f"PMF of Geometric($p={p}$)",
        }

        stem = plot_discrete_pmf(
            X, states=states, ax=ax, ax_kwargs=ax_kwargs, **stem_kwargs
        )
    return stem


if __name__ == "__main__":
    # Uniform PMF
    # _fig, ax = plt.subplots(1, figsize=(8, 6))
    # plot_discrete_uniform_pmf(a=1, b=6, ax=ax)
    # low, high = 1, 6 + 1  # [1, 6] for dice roll
    # X = stats.randint(low, high)
    # Z1 = X.rvs(size=10000)
    # states_z1 = np.arange(low, high + 1)

    # plot_discrete_empirical_histogram(
    #     X,
    #     states=states_z1,
    # )
    # plt.show()

    # _fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    # plot_discrete_uniform_pmf(a=1, b=6, ax=axes[0])
    # low, high = 1, 6 + 1  # [1, 6] for dice roll
    # X = stats.randint(low, high)
    # Z1 = X.rvs(size=10000)
    # states_z1 = np.arange(low, high + 1)

    # plot_discrete_empirical_histogram(
    #     Z1,
    #     states=states_z1,
    #     ax=axes[1],
    # )
    # plt.show()

    # Bernoulli PMF
    # fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.8), sharey=True, dpi=100)
    # plot_bernoulli_pmf(p=0.2, ax=axes[0])
    # plot_empirical_bernoulli(p=0.2, size=100, ax=axes[0])

    # plot_bernoulli_pmf(p=0.2, ax=axes[1])
    # plot_empirical_bernoulli(p=0.2, size=1000, ax=axes[1])

    # fig.supylabel("relative frequency")
    # fig.suptitle("Histogram of Bernoulli($p=0.2$) based on $100$ and $1000$ samples.")
    # plt.show()

    # Binomial PMF
    # p=0.2, n=3
    # _fig, ax = plt.subplots(1, figsize=(12, 8))
    # plot_binomial_pmfs(ns=[3], ps=[0.2], ax=ax)
    # plot_empirical_binomial(n=3, p=0.2, size=1000, ax=ax)
    # plt.show()

    # _fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    # plot_binomial_pmfs(ns=[60, 60, 60], ps=[0.1, 0.5, 0.9], ax=axes[0])
    # plot_binomial_pmfs(ns=[5, 50, 100], ps=[0.5, 0.5, 0.5], ax=axes[1])
    # plt.show()

    # Geometric PMF
    # _fig, ax = plt.subplots(1, figsize=(12, 8))
    # plot_geometric_pmfs(ps=[0.5], ax=ax)
    # plt.show()

    # Poisson PMF
    _fig = plt.figure(figsize=(12, 8))
    lambdas = [5, 10, 20]
    plot_poisson_pmfs(lambdas=lambdas)
    plot_empirical_poisson(lambdas=lambdas)
    plt.show()
