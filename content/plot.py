import random
from typing import Callable, Optional, Tuple, List, Union, Dict, Any
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from utils import seed_all
from matplotlib.container import StemContainer

COLOR_INDEXES = list(range(0, 10))
random.shuffle(COLOR_INDEXES)


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


def plot_bernoulli_pmf(p: float) -> None:
    """Plot the PMF of a Bernoulli distribution."""
    # X is now an object that represents a Bernoulli random variable with parameter $p$.
    X = stats.bernoulli(p)
    states = np.asarray([0, 1])  # Bernoulli only has two states, 0 and 1.

    _fig, ax = plt.subplots(1, figsize=(8, 6))

    ax_kwargs = {"title": f"PMF of Bernoulli($p={p}$)"}
    stem_kwargs = {"linefmt": "r-", "markerfmt": "ro", "basefmt": "C7-", "label": "PMF"}

    plot_discrete_pmf(X, states=states, ax=ax, ax_kwargs=ax_kwargs, **stem_kwargs)
    plt.show()


def plot_binomial_pmf(n: int, p: float) -> None:
    """Plot the PMF of a Binomial distribution."""
    # X is now an object that represents a Binomial random variable with parameter $p$ and $n$.
    X = stats.binom(n, p)
    states = np.arange(0, n + 1)  # Binomial has states from 0 to n, given n trials.

    _fig, ax = plt.subplots(1)

    ax_kwargs = {"title": f"PMF of Binomial($n={n}, p={p}$)"}
    stem_kwargs = {"linefmt": "g-", "markerfmt": "go", "basefmt": "C7-", "label": "PMF"}

    plot_discrete_pmf(X, states=states, ax=ax, ax_kwargs=ax_kwargs, **stem_kwargs)
    plt.show()


def plot_multiple_binomial_pmf(ns: int, ps: List[float], ax=None) -> None:
    """Plot the PMF of multiple Binomial distributions."""
    # _fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    ax = ax or plt.gca()

    for n, p in zip(ns, ps):
        # X is now an object that represents a Binomial random variable with parameter $p$ and $n$.
        X = stats.binom(n, p)
        states = np.arange(0, n + 1)  # Binomial has states from 0 to n, given n trials.
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

    # plt.show()


if __name__ == "__main__":
    seed_all()

    # Bernoulli PMF
    # plot_bernoulli_pmf(p=0.2)

    # Binomial PMF
    # plot_binomial_pmf(n=3, p=0.2)
    _fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    plot_multiple_binomial_pmf(ns=[60, 60, 60], ps=[0.1, 0.5, 0.9], ax=axes[0])
    plot_multiple_binomial_pmf(ns=[5, 50, 100], ps=[0.5, 0.5, 0.5], ax=axes[1])
    plt.show()
