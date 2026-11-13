import numpy as np
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
import matplotlib.ticker as mticker
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib as mpl
import legwork as lw

from copy import copy
import const


plt.rc('font', family='serif')
plt.rcParams['text.usetex'] = False
fs = 24

# update various fontsizes to match
params = {'figure.figsize': (12, 8),
          'legend.fontsize': 0.7*fs,
          'legend.title_fontsize': 0.8*fs,
          'axes.labelsize': fs,
          'xtick.labelsize': 0.9 * fs,
          'ytick.labelsize': 0.9 * fs,
          'axes.linewidth': 1.1,
          'xtick.major.size': 7,
          'xtick.minor.size': 4,
          'ytick.major.size': 7,
          'ytick.minor.size': 4}
plt.rcParams.update(params)


class LogDecadeMinorLocator(mticker.Locator):
    """minor tick locator for axes holding log10(x) data, placing ticks at 2-9 in each decade"""
    def __call__(self):
        lo, hi = sorted(self.axis.get_view_interval())
        decades = np.arange(np.floor(lo), np.ceil(hi) + 1)
        return (decades[:, None] + np.log10(np.arange(2, 10))[None, :]).ravel()


def fake_log_axis(ax, axis="x"):
    """make an axis of log10(x) data read like a standard matplotlib log axis

    Parameters
    ----------
    ax : `matplotlib Axis`
        Axis to adjust
    axis : `str`, optional
        Which axis to adjust, either "x" or "y", by default "x"

    Returns
    -------
    ax : `matplotlib Axis`
        Adjusted axis
    """
    a = ax.xaxis if axis == "x" else ax.yaxis

    a.set_major_locator(mticker.MultipleLocator(1))
    _log_fmt = mticker.LogFormatterSciNotation()
    _log_fmt.set_axis(a)
    a.set_major_formatter(mticker.FuncFormatter(lambda x, pos: _log_fmt(10**x, pos)))

    a.set_minor_locator(LogDecadeMinorLocator())
    a.set_minor_formatter(mticker.NullFormatter())

    # match the tick lengths matplotlib uses for a real log axis
    ax.tick_params(axis=axis, which="major", length=6)
    ax.tick_params(axis=axis, which="minor", length=3)

    return ax

class MirroredKDE(gaussian_kde):
    """ KDE class that mirrors data at boundaries to account for bounded support """

    def __init__(self, data, weights=None, lower_bound=None, upper_bound=None,
                 bw_method=None, bw_adjust=None):
        """ instantiate class in similar way to scipy but with some additions """
        super().__init__(data, weights=weights, bw_method=bw_method)

        # also store the lower and upper bounds
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

        # allow adjustment of the default bandwidth similar to seaborn
        if bw_adjust is not None:
            self.set_bandwidth(self.factor * bw_adjust)

    def evaluate(self, x_vals=None, x_min=None, x_max=None, x_count=200):
        """ evaluate the kde taking into account the boundaries """

        # only return x_vals when they aren't supplied
        return_x_vals = x_vals is None

        if x_vals is None:
            if x_min is None:
                x_min = np.min(self.dataset)
            if x_max is None:
                x_max = np.max(self.dataset)
            x_vals = np.linspace(x_min, x_max, x_count)

        # make a copy of the data before I mirror anything
        unmirrored_x_vals = np.copy(x_vals)

        # evaluate the kde at the original x values
        kde_vals = super().evaluate(x_vals)

        # if either bound is present then mirror the data and
        # add the evaluated kde for the mirrored data to the original
        if self._lower_bound is not None:
            x_vals = 2.0 * self._lower_bound - x_vals
            kde_vals += super().evaluate(x_vals)
            x_vals = unmirrored_x_vals

        if self._upper_bound is not None:
            x_vals = 2.0 * self._upper_bound - x_vals
            kde_vals += super().evaluate(x_vals)
            x_vals = unmirrored_x_vals

        if return_x_vals:
            return x_vals, kde_vals
        else:
            return kde_vals


def bootstrapped_kde(variable, weights, ax, seeds=None, bw_adjust=None, normalisation=1,
                     lower_bound=None, upper_bound=None,
                     bootstraps=200, x_min=None, x_max=None, x_count=200, log_scale=(False, False),
                     color="tab:blue", label=None, **kwargs):
    """Create a bootstrapped weighted KDE plot.

    Parameters
    ----------
    variable : `float/array`
        Variable that you want to make a KDE of.
    weights : 'float/array'
        Weights associated with each variable (see all to 1 for unweighted)
    seeds : `int/array`
        Seeds that make the binaries in COMPAS
    ax : `matplotlib Axis`
        Axis on which to plot
    bw_adjust : `float`, optional
        Factor by which to adjust the bandwidth, by default None
    bootstraps : `int`, optional
        How many bootstraps to do, by default 200
    x_count : `int`, optional
        How many x values to evaluate at, by default 500
    log_scale : `tuple`, optional
        Whether each axis should be log scaled, by default (False, False)
    color : `str`, optional
        Colour for the KDE, by default "tab:blue"
    label : `str`, optional
        Label for the plotted KDE, by default None

    Returns
    -------
    ax : `matplotlib Axis`
        Axis on which KDE is plotted
    """

    if seeds is None:
        seeds = np.arange(len(variable))

    # store the KDE values for each bootstrap
    kde_vals = np.zeros((bootstraps, x_count))

    if x_min is None:
        x_min = np.min(variable)
    if x_max is None:
        x_max = np.max(variable)

    # decide on x values to evaluate at (based on log scaling)
    if log_scale[0]:
        print("WARNING: I think this doesn't work", variable)
        x_vals = np.logspace(np.log10(x_min), np.log10(x_max), x_count)
    else:
        x_vals = np.linspace(x_min, x_max, x_count)

    sorted_order = np.argsort(seeds)
    sorted_seeds = seeds[sorted_order]

    # perform bootstrapping
    for i in range(bootstraps):
        _, starts, counts = np.unique(sorted_seeds, return_counts=True, return_index=True)
        res = np.split(sorted_order, starts[1:])
        inds = np.array([np.random.choice(r) if len(r) > 1 else r[0] for r in res])

        print(inds)
        print(inds.shape)

        loop_variable = variable[inds]
        loop_weights = weights[inds] * counts

        # record indices to sample from
        indices = np.arange(len(loop_variable))

        # sample indices
        boot_index = np.random.choice(indices, size=len(indices), replace=True)

        kde = MirroredKDE(loop_variable[boot_index], weights=loop_weights[boot_index],
                          lower_bound=lower_bound, upper_bound=upper_bound, bw_adjust=bw_adjust)
        kde_vals[i] = kde.evaluate(x_vals) * normalisation

    # calculate 1- and 2- sigma percentiles
    percentiles = np.percentile(kde_vals, [15.89, 84.1, 2.27, 97.725], axis=0)

    # plot uncertainties as filled areas
    ax.fill_between(x_vals, percentiles[2], percentiles[3], alpha=0.15, color=color, **kwargs)
    ax.fill_between(x_vals, percentiles[0], percentiles[1], alpha=0.3, color=color, **kwargs)

    # plot the regular kde
    ax.plot(x_vals, np.median(kde_vals, axis=0), color=color, label=label, **kwargs)

    # adjust scales if needed
    if log_scale[0]:
        ax.set_xscale("log")
    if log_scale[1]:
        ax.set_yscale("log")

    return ax


def bootstrapped_kde_fast(variable, weights, ax, seeds=None, bw_adjust=None, normalisation=1,
                     lower_bound=None, upper_bound=None,
                     bootstraps=200, x_min=None, x_max=None, x_count=200, log_scale=(False, False),
                     color="tab:blue", label=None, rng=None, fill_uncertainties=True, **kwargs):
    """Create a bootstrapped weighted KDE plot.

    Bootstrapping is performed by reweighting a single precomputed kernel matrix rather than
    refitting a KDE for each resample, which requires a bandwidth fixed across bootstraps.

    Parameters
    ----------
    variable : `float/array`
        Variable that you want to make a KDE of.
    weights : `float/array`
        Weights associated with each variable (set all to 1 for unweighted)
    ax : `matplotlib Axis`
        Axis on which to plot
    seeds : `int/array`, optional
        Seeds that make the binaries in COMPAS, by default None
    bw_adjust : `float`, optional
        Factor by which to adjust the bandwidth, by default None
    rng : `numpy.random.Generator`, optional
        Random generator to use, by default None

    Returns
    -------
    ax : `matplotlib Axis`
        Axis on which KDE is plotted
    """
    rng = np.random.default_rng(rng)

    variable = np.asarray(variable, dtype=float)
    weights = np.asarray(weights, dtype=float)
    seeds = np.arange(len(variable)) if seeds is None else np.asarray(seeds)

    x_min = variable.min() if x_min is None else x_min
    x_max = variable.max() if x_max is None else x_max
    x_vals = (np.logspace(np.log10(x_min), np.log10(x_max), x_count) if log_scale[0]
              else np.linspace(x_min, x_max, x_count))

    # group rows by seed once, this is invariant across bootstraps
    order = np.argsort(seeds, kind="stable")
    _, starts, counts = np.unique(seeds[order], return_index=True, return_counts=True)
    n_seeds = len(starts)

    # single bandwidth for all bootstraps (Scott's rule on the full weighted sample)
    w_norm = weights / weights.sum()
    n_eff = 1 / np.sum(w_norm**2)
    mean = np.average(variable, weights=weights)
    sigma = np.sqrt(np.average((variable - mean)**2, weights=weights))
    h = sigma * n_eff**(-0.2) * (1.0 if bw_adjust is None else bw_adjust)

    # kernel matrix: contribution of every row to every x value, with reflections at the bounds
    K = np.exp(-0.5 * ((x_vals[None, :] - variable[:, None]) / h)**2)
    if lower_bound is not None:
        K += np.exp(-0.5 * ((x_vals[None, :] - (2 * lower_bound - variable[:, None])) / h)**2)
    if upper_bound is not None:
        K += np.exp(-0.5 * ((x_vals[None, :] - (2 * upper_bound - variable[:, None])) / h)**2)
    K /= h * np.sqrt(2 * np.pi)

    # pick one row per seed, for every bootstrap at once
    pick = order[starts + (rng.random((bootstraps, n_seeds)) * counts).astype(int)]

    # resampling seeds with replacement is a multinomial draw over the seed groups
    multiplicity = rng.multinomial(n_seeds, np.full(n_seeds, 1 / n_seeds), size=bootstraps)

    # express both resampling steps as a weight matrix over the original rows
    W = np.zeros((bootstraps, len(variable)))
    np.put_along_axis(W, pick, weights[pick] * counts * multiplicity, axis=1)
    W /= W.sum(axis=1, keepdims=True)

    kde_vals = (W @ K) * normalisation

    # calculate 1- and 2-sigma percentiles
    percentiles = np.percentile(kde_vals, [15.89, 84.1, 2.27, 97.725], axis=0)

    if not isinstance(ax, (list, tuple, np.ndarray)):
        ax = [ax]

    for a in ax:
        if fill_uncertainties:
            a.fill_between(x_vals, percentiles[2], percentiles[3], alpha=0.15, color=color, **kwargs)
            a.fill_between(x_vals, percentiles[0], percentiles[1], alpha=0.3, color=color, **kwargs)
        a.plot(x_vals, np.median(kde_vals, axis=0), color=color, label=label, **kwargs)

        if log_scale[0]:
            a.set_xscale("log")
        if log_scale[1]:
            a.set_yscale("log")

    return ax


def nice_transparent_hist(ax, data, bins, label, colour, density=True, lw=2, alpha=0.4, cumulative=False, **kwargs):
    ax.hist(data, bins=bins, color=colour, lw=lw, histtype='step', density=density, label=label, cumulative=cumulative, **kwargs)
    ax.hist(data, bins=bins, color=colour, alpha=alpha, density=density, cumulative=cumulative, **kwargs)


def estimate_scale_height_cdf(z, weights=None, R=None, Rlims=(7.5, 8.5), verbose=False):
    """Estimate the scale height of a distribution given z-positions using the cumulative distribution function (CDF).
    This method does not assume a model, but instead just finds the z-value at which the CDF reaches 1 - 1/e ~ 0.63, which corresponds to the scale height for an exponential distribution."""
    z = np.abs(z)
    if R is not None:
        R = R.to(u.kpc).value if hasattr(R, 'unit') else R
        mask = (R >= Rlims[0]) & (R < Rlims[1])
        if verbose:
            print(len(z), "objects before Rlims")
        z = z[mask]
        if verbose:
            print(len(z), "objects in Rlims")

    if hasattr(z, 'unit'):
        z = z.to(u.kpc).value

    if weights is None:
        weights = np.ones_like(z)

    # calculate empirical CDF without any binning
    order = np.argsort(z)
    sorted_z = z[order]
    # cdf = np.arange(1, len(sorted_z) + 1) / len(sorted_z)
    cdf = np.cumsum(weights[order])
    cdf /= cdf[-1]

    scale_height = sorted_z[cdf >= (1 - 1 / np.e)][0]

    return scale_height


def bootstrap_cdf(data, n_samples=25_000, n_bootstraps=10, n_bins=250, weights=None, norm=False, fig=None, ax=None,
                  reversed_cdf=True,
                  xlim=(0, np.inf), colour=None, label=None, med_kwargs={}, fill_kwargs={}):
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    if weights is None:
        weights = np.ones(len(data))

    default_med_kwargs = {'color': colour, 'lw': 3, 'label': label}
    default_fill_kwargs = {'color': colour, 'alpha': 0.15, 'lw': 1}
    med_kwargs_comb = default_med_kwargs.copy()
    med_kwargs_comb.update(med_kwargs)
    fill_kwargs_comb = default_fill_kwargs.copy()
    fill_kwargs_comb.update(fill_kwargs)

    sample_inds = np.random.choice(len(data), size=(n_samples, n_bootstraps), replace=True)
    samples = data[sample_inds]
    sample_weights = weights[sample_inds]
    order = np.argsort(samples, axis=0)

    # apply order to samples and weights to get the CDFs, can't just immediately mask
    # ensure that shapes are the same after resorting!
    ordered_samples = np.take_along_axis(samples, order, axis=0)
    ordered_weights = np.take_along_axis(sample_weights, order, axis=0)

    cdfs = np.cumsum(ordered_weights, axis=0)
    if reversed_cdf:
        cdfs = cdfs.max() - cdfs
    if norm:
        cdfs /= cdfs.max(axis=0)

    # bin the CDFs to make them have a consistent x-axis for plotting
    bin_edges = np.linspace(*xlim, n_bins)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    binned_cdfs = np.zeros((len(bin_centres), n_bootstraps))
    for i in range(n_bootstraps):
        binned_cdfs[:, i] = np.interp(bin_centres, ordered_samples[:, i], cdfs[:, i])

    med = np.median(binned_cdfs, axis=1)
    low = np.percentile(binned_cdfs, 25, axis=1)
    high = np.percentile(binned_cdfs, 75, axis=1)

    ax.fill_between(bin_centres, low, high, **fill_kwargs_comb)
    ax.plot(bin_centres, med, **med_kwargs_comb)
    ax.set_yscale('log')

    return fig, ax, ordered_samples, bin_centres, med, low, high
