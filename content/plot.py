from __future__ import annotations

import random
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.container import BarContainer, StemContainer
from scipy import stats
from utils import seed_all

T = TypeVar("T", str, int, float)
FigAxParams = Dict[str, Dict[str, T]]
seed_all(42)

# FIXME:
# 1. For example, plot_poisson_pmfs and plot_empirical_poisson plot with different colors.


@dataclass(frozen=False, init=True)
class Plot(ABC):
    """Plot params.

    Args:
        states (np.ndarray): The states of the distribution. For example, if a rv X
            is a binomial distribution with n=10, p=0.5, then the states are
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], the values which X can take.
    """

    states: np.ndarray
    default_ax_params: FigAxParams = field(init=False)  # default
    custom_ax_params: FigAxParams = field(default_factory=dict)
    custom_fig_params: FigAxParams = field(default_factory=dict)
    color_index: List[int] = field(default_factory=lambda: list(range(10)))

    @property
    def low(self) -> int:
        """The lower bound of the states."""
        return self.states.min()

    @property
    def high(self) -> int:
        """The upper bound of the states."""
        return self.states.max()

    @property
    def default_fig_params(self) -> FigAxParams:
        """The default figure params.
        TODO: not used in code yet, to consider.
        """

    def _update_default_ax_params(self) -> None:
        """Update the ax params with the custom_ax_params if any."""
        if self.custom_ax_params:  # if custom_ax_params is not empty
            for ax_attr, ax_params in self.custom_ax_params.items():
                self.default_ax_params.update({ax_attr: ax_params})

    @staticmethod
    def shuffle(container: List[int]) -> None:
        """Randomly shuffle a list in place."""
        random.shuffle(container)


@dataclass(frozen=False, init=True)
class PMF(Plot):
    """PMF plot params."""

    def __post_init__(self) -> None:
        self.default_ax_params = {
            "set_title": {"label": "PMF", "fontsize": 12},
            "set_xlabel": {"xlabel": "x", "fontsize": 12},
            "set_ylabel": {"ylabel": "pmf(x)", "fontsize": 12},
            "set_xlim": {"left": self.low - 1, "right": self.high + 1},
            "set_xticks": {"ticks": np.arange(self.low, self.high + 1, 1)},
            "legend": {"loc": "best"},
        }
        self._update_default_ax_params()
        self.shuffle(self.color_index)


@dataclass(frozen=False, init=True)
class EmpiricalHistogram(Plot):
    # center the bins on the states, for discrete distributions.
    bins: Optional[Union[List[float], np.ndarray]] = None
    size: int = 1000  # number of samples to draw from the distribution

    def __post_init__(self) -> None:
        self.bins = (
            np.arange(0, self.high + 1.5) - 0.5 if self.bins is None else self.bins
        )

        self.default_ax_params = {
            "set_title": {"label": "Empirical Histogram", "fontsize": 12},
            "set_xlabel": {"xlabel": "x", "fontsize": 12},
            "set_ylabel": {"ylabel": "relative frequency", "fontsize": 12},
            "set_xticks": {
                "ticks": self.bins + 0.5
            },  # FIXME: this is not working as expected, if I put this after set_xlim, it shows the "max" tick
            "set_xlim": {"left": min(self.bins), "right": max(self.bins)},
            "legend": {"loc": "best"},
        }
        self._update_default_ax_params()
        self.shuffle(self.color_index)


def plot_discrete_empirical_histogram(
    distribution: Callable,
    plot_params: EmpiricalHistogram,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    **hist_kwargs: Dict[str, Any],
) -> plt.Axes:
    """Takes in a distribution (population or sample values), and plots the empirical distribution."""
    fig = fig or plt.gcf()
    ax = ax or plt.gca()

    if not hist_kwargs:
        hist_kwargs = {
            "edgecolor": "black",
            "linewidth": 2,
            "alpha": 0.5,
            "color": "#0504AA",
            "stat": "probability",  # "density",
            "discrete": True,
        }

    empirical_samples = distribution.rvs(size=plot_params.size)
    # print(f"Empirical samples: {empirical_samples}")

    bins = plot_params.bins
    hist = sns.histplot(empirical_samples, bins=bins, ax=ax, **hist_kwargs)

    # call the set_* methods on the ax
    for ax_attr, ax_params in plot_params.default_ax_params.items():
        # print(f"ax_attr: {ax_attr}, ax_params: {ax_params}")
        getattr(ax, ax_attr)(**ax_params)

    if fig is not None:
        # fig.tight_layout()
        for fig_attr, fig_params in plot_params.custom_fig_params.items():
            getattr(fig, fig_attr)(**fig_params)
    return hist


def plot_discrete_pmf(
    distribution: Callable,
    plot_params: PMF,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    **stem_kwargs: Dict[str, Any],
) -> StemContainer:
    """Plot the PMF of a discrete distribution.

    Args:
        distribution (Callable): A function that takes in a state and returns the probability of that state.
            Typically called from a scipy.stats distribution object.
        plot_params (PMF): A dataclass that contains the default and custom parameters for the ax.
        fig (Optional[plt.Figure]): The figure to plot on. Defaults to None.
        ax (Optional[plt.Axes]): The axes to plot on. Defaults to None.
        **stem_kwargs (Dict[str, Any]): Keyword arguments to pass to the stem.
    """
    fig = fig or plt.gcf()
    ax = ax or plt.gca()

    if not stem_kwargs:  # if stem_kwargs is empty
        stem_kwargs = {
            "linefmt": "b-",
            "markerfmt": "bo",
            "basefmt": "C7-",
            "label": f"PMF of {distribution.__name__}",
        }

    states = plot_params.states
    low, high = states.min(), states.max()
    x = np.arange(low, high + 1)  # The states of the distribution X
    y = distribution.pmf(x)  # The probability of each state

    stem = ax.stem(
        x,
        y,
        **stem_kwargs,
    )

    # call the set_* methods on the ax
    for ax_attr, ax_params in plot_params.default_ax_params.items():
        # print(f"ax_attr: {ax_attr}, ax_params: {ax_params}")
        getattr(ax, ax_attr)(**ax_params)

    if fig is not None:
        # fig.tight_layout()
        for fig_attr, fig_params in plot_params.custom_fig_params.items():
            getattr(fig, fig_attr)(**fig_params)

    return stem


def plot_discrete_uniform_pmf(
    a: int, b: int, fig: Optional[plt.Figure] = None, ax: Optional[plt.Axes] = None
) -> StemContainer:
    """Plot the PMF of a Uniform distribution."""
    # X is now an object that represents a Uniform random variable with parameter $a$ and $b$.
    fig = fig or plt.gcf()
    ax = ax or plt.gca()

    X = stats.randint(a, b + 1)
    states = np.arange(a, b + 1)  # Uniform has b - a + 1 states.

    plot_params = PMF(
        states=states,
        custom_ax_params={"set_title": {"label": f"PMF of Uniform($a={a}, b={b}$)"}},
    )

    stem_kwargs = {
        "linefmt": "b-",
        "markerfmt": "bo",
        "basefmt": "C7-",
        "label": f"PMF of Uniform($a={a}, b={b}$)",
    }

    stem = plot_discrete_pmf(X, plot_params=plot_params, fig=fig, ax=ax, **stem_kwargs)
    return stem


def plot_empirical_discrete_uniform(
    a: int,
    b: int,
    size: int = 5000,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
) -> Union[BarContainer, plt.Axes]:
    """Plot the empirical distribution of a Discrete Uniform distribution."""
    fig = fig or plt.gcf()
    ax = ax or plt.gca()

    X = stats.randint(a, b + 1)
    states = np.arange(a, b + 1)  # Uniform has b - a + 1 states.

    bins = np.arange(a - 0.5, b + 1.5, 1)

    plot_params = EmpiricalHistogram(
        states=states,
        bins=bins,
        size=size,
        custom_ax_params={
            "set_title": {
                "label": f"Empirical Histogram Uniform($a={a}, b={b}$) of {size} samples."
            },
        },
    )

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
        plot_params=plot_params,
        fig=fig,
        ax=ax,
        **hist_kwargs,
    )
    return hist


def plot_bernoulli_pmf(
    p: float, fig: Optional[plt.Figure] = None, ax: Optional[plt.Axes] = None
) -> StemContainer:
    """Plot the PMF of a Bernoulli distribution."""
    fig = fig or plt.gcf()
    ax = ax or plt.gca()

    # X is now an object that represents a Bernoulli random variable with parameter $p$.
    X = stats.bernoulli(p)
    states = np.asarray([0, 1])  # Bernoulli only has two states, 0 and 1.

    plot_params = PMF(
        states=states,
        custom_ax_params={"set_title": {"label": f"PMF of Bernoulli($p={p}$)"}},
    )
    stem_kwargs = {"linefmt": "r-", "markerfmt": "ro", "basefmt": "C7-", "label": "PMF"}

    stem = plot_discrete_pmf(X, plot_params=plot_params, fig=fig, ax=ax, **stem_kwargs)
    return stem


def plot_empirical_bernoulli(
    p: float,
    size: int = 1000,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
) -> Union[BarContainer, plt.Axes]:
    """Plot the empirical distribution of a Bernoulli distribution."""
    fig = fig or plt.gcf()
    ax = ax or plt.gca()

    # X is now an object that represents a Bernoulli random variable with parameter $p$.
    X = stats.bernoulli(p)
    states = np.asarray([0, 1])  # Bernoulli only has two states, 0 and 1.
    bins = np.arange(0, states.max() + 1.5) - 0.5

    plot_params = EmpiricalHistogram(
        states=states,
        bins=bins,
        size=size,
        custom_ax_params={"set_title": {"label": None}, "set_ylabel": {"ylabel": None}},
        custom_fig_params={
            "supylabel": {"t": "relative frequency", "fontsize": 12},
            "suptitle": {
                "t": "Histogram of Bernoulli($p=0.2$) based on $100$ and $1000$ samples.",
                "fontsize": 12,
            },
        },
    )

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
        plot_params=plot_params,
        fig=fig,
        ax=ax,
        **hist_kwargs,
    )
    return hist


def plot_binomial_pmfs(
    ns: List[int],
    ps: List[float],
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
) -> StemContainer:
    """Plot the PMFs of multiple Binomial distributions on the same axes.

    Args:
        ns (List[int]): The number of trials for each Binomial distribution.
        ps (List[float]): The probability of success for each Binomial distribution.
        ax (Optional[plt.Axes], optional): The axes to plot on. Defaults to None.

    Returns:
        stem (StemContainer): The stem plot.
    """
    fig = fig or plt.gcf()
    ax = ax or plt.gca()

    for n, p in zip(ns, ps):
        X = stats.binom(n, p)
        states = np.arange(0, n + 1)

        xticks = np.arange(0, n + 1, 5)

        plot_params = PMF(
            states=states,
            custom_ax_params={
                "set_title": {
                    "label": f"PMF of Binomial($n={ns}, p={ps}$)",
                },
                "set_xticks": {"ticks": xticks},
            },
        )

        color_index = plot_params.color_index.pop()

        stem_kwargs = {
            "linefmt": f"C{color_index}-",
            "markerfmt": f"C{color_index}o",
            "basefmt": "C7-",
            "label": f"PMF of Binomial($n={n}, p={p}$)",
        }

        stem = plot_discrete_pmf(
            X, plot_params=plot_params, fig=fig, ax=ax, **stem_kwargs
        )
    return stem


def plot_empirical_binomial(
    p: float,
    n: int,
    size: int = 1000,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
) -> Union[BarContainer, plt.Axes]:
    """Plot the empirical distribution of a Bernoulli distribution."""
    fig = fig or plt.gcf()
    ax = ax or plt.gca()

    X = stats.binom(n, p)
    states = np.arange(0, n + 1)  # Binomial has n + 1 states.
    bins = np.arange(0, states.max() + 1.5) - 0.5

    plot_params = EmpiricalHistogram(
        states=states,
        bins=bins,
        size=size,
        custom_ax_params={"set_title": {"label": None}, "set_ylabel": {"ylabel": None}},
        custom_fig_params={
            "supylabel": {"t": "relative frequency", "fontsize": 12, "x": 0.07},
            "suptitle": {
                "t": f"Histogram of Binomial($p={p}$) based on {size} samples.",
                "fontsize": 12,
                "y": 0.92,
            },
        },
    )
    color_index = plot_params.color_index.pop()

    hist_kwargs = {
        "edgecolor": "black",
        "linewidth": 2,
        "alpha": 0.5,
        "color": f"C{color_index}",
        "stat": "probability",
        "label": "Empirical Histogram",
    }

    hist = plot_discrete_empirical_histogram(
        X,
        plot_params=plot_params,
        fig=fig,
        ax=ax,
        **hist_kwargs,
    )
    return hist


def plot_poisson_pmfs(
    lambdas: List[float],
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
) -> StemContainer:
    """Plot the PMFs of multiple Poisson distributions on the same axes."""
    fig = fig or plt.gcf()
    ax = ax or plt.gca()

    for lam in lambdas:
        X = stats.poisson(lam)
        states = np.arange(
            0, 30
        )  # Poisson has infinite states, but we only plot up to 30.

        xticks = np.arange(0, 30, 5)

        plot_params = PMF(
            states=states,
            custom_ax_params={
                "set_title": {
                    "label": f"PMF of Poisson($\lambda \in {lambdas}$)",
                    "fontsize": 12,
                },
                "set_xticks": {"ticks": xticks},
            },
        )
        color_index = plot_params.color_index.pop()

        stem_kwargs = {
            "linefmt": f"C{color_index}-",
            "markerfmt": f"C{color_index}o",
            "basefmt": "C7-",
            "label": f"PMF of Poisson($\lambda={lam}$)",
        }

        stem = plot_discrete_pmf(
            X, plot_params=plot_params, fig=fig, ax=ax, **stem_kwargs
        )
    return stem


def plot_empirical_poisson(
    lambdas: List[int],
    size: int = 1000,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
) -> Union[BarContainer, plt.Axes]:
    """Plot the empirical distribution of a Bernoulli distribution."""
    fig = fig or plt.gcf()
    ax = ax or plt.gca()

    for lam in lambdas:
        X = stats.poisson(lam)
        states = np.arange(
            0, 30
        )  # Poisson has infinite states, but we only plot up to 30.

        bins = np.arange(0, states.max() + 1.5) - 0.5

        plot_params = EmpiricalHistogram(
            states=states,
            bins=bins,
            size=size,
            custom_ax_params={
                "set_title": {
                    "label": f"Empirical Histogram of Poisson($\lambda$) with {size} samples.",
                    "fontsize": 12,
                },
            },
        )
        color_index = plot_params.color_index.pop()

        hist_kwargs = {
            "edgecolor": "black",
            "linewidth": 2,
            "alpha": 0.5,
            "color": f"C{color_index}",
            "stat": "probability",
            "label": f"Empirical of Poisson($\lambda={lam}$)",
        }

        hist = plot_discrete_empirical_histogram(
            X,
            plot_params=plot_params,
            fig=fig,
            ax=ax,
            **hist_kwargs,
        )
    return hist


def plot_geometric_pmfs(
    ps: List[float],
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
) -> StemContainer:
    """Plot the PMFs of multiple Geometric distributions on the same axes."""
    fig = fig or plt.gcf()
    ax = ax or plt.gca()

    for p in ps:
        X = stats.geom(p)
        states = np.arange(
            1, 10
        )  # Geometric has infinite states, but we only plot up to 10.

        xticks = np.arange(0, 10, 1)

        plot_params = PMF(
            states=states,
            custom_ax_params={
                "set_title": {
                    "label": f"PMF of Geometric($p={p}$)",
                    "fontsize": 16,
                },
                "set_xticks": {"ticks": xticks},
            },
        )
        color_index = plot_params.color_index.pop()
        stem_kwargs = {
            "linefmt": f"C{color_index}-",
            "markerfmt": f"C{color_index}o",
            "basefmt": "C7-",
            "label": f"PMF of Geometric($p={p}$)",
        }

        stem = plot_discrete_pmf(X, plot_params=plot_params, fig=fig, **stem_kwargs)
    return stem


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

    axes[0, 0].set_title(label="Distribution of $Z = X$")
    hist_z1 = sns.histplot(
        Z1,
        bins=bins_z1,
        ax=axes[0, 0],
        stat="probability",
    )
    axes[0, 1].set_title(label="Distribution of $Z = X_1 + X_2$")
    hist_z2 = sns.histplot(
        Z2,
        bins=bins_z2,
        ax=axes[0, 1],
        stat="probability",
    )

    axes[1, 0].set_title(
        label="Distribution of $Z = X_1 + X_2 + X_3 + X_4 + X_5 + X_6$"
    )
    hist_z6 = sns.histplot(Z6, bins=bins_z6, ax=axes[1, 0], stat="probability")

    axes[1, 1].set_title(label="Distribution of $Z = X_1 + X_2 + ... + X_{100}$")
    hist_z100 = sns.histplot(Z100, bins=bins_z100, ax=axes[1, 1], stat="probability")

    plt.show()


if __name__ == "__main__":
    # Uniform PMF
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    low, high = 1, 6  # [1, 6] for dice roll
    plot_discrete_uniform_pmf(low, high, fig=fig, ax=axes[0])
    plot_empirical_discrete_uniform(low, high, fig=fig, ax=axes[1])
    plt.show()

    plot_sum_of_uniform_distribution()

    # Bernoulli PMF
    # fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.8), sharey=True, dpi=100)
    # plot_bernoulli_pmf(p=0.2, ax=axes[0])
    # plot_empirical_bernoulli(p=0.2, size=100, ax=axes[0])

    # plot_bernoulli_pmf(p=0.2, ax=axes[1])
    # plot_empirical_bernoulli(p=0.2, size=1000, ax=axes[1])
    # plt.show()

    # Binomial PMF
    # p=0.2, n=3
    _fig, ax = plt.subplots(1, figsize=(12, 8))
    plot_binomial_pmfs(ns=[10], ps=[0.5], ax=ax)
    plot_empirical_binomial(n=10, p=0.5, size=5000, ax=ax)
    plt.show()

    _fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    plot_binomial_pmfs(ns=[60, 60, 60], ps=[0.1, 0.5, 0.9], ax=axes[0])
    plot_binomial_pmfs(ns=[5, 50, 100], ps=[0.5, 0.5, 0.5], ax=axes[1])
    plt.show()

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

    _fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=125)
    lambdas = [5, 10, 20]
    plot_poisson_pmfs(lambdas=lambdas, ax=axes[0])
    plot_empirical_poisson(lambdas=lambdas, ax=axes[1])
    plt.show()
