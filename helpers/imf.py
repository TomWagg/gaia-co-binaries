import numpy as np

DEFAULT_MASS_CUTOFFS = [0.08, 0.5, 150.0]
DEFAULT_SLOPES = [1.3, 2.3]


def _normalisation_constants(mass_cutoffs, slopes):
    """Compute the per-segment normalisation constants for a broken power law IMF.

    The IMF is a piecewise power law, zeta(m) ~ b_i * m^(-slopes[i]) on the interval
    [mass_cutoffs[i], mass_cutoffs[i + 1]]. The constants ``b_i`` are chosen so that the
    IMF is continuous at each cutoff and integrates to 1 over the full mass range.

    Parameters
    ----------
    mass_cutoffs : list of `float`, length N
        Masses at which the slope of the IMF changes (must be increasing).
    slopes : list of `float`, length N - 1
        Slope of the IMF on each segment, ``slopes[i]`` applies between
        ``mass_cutoffs[i]`` and ``mass_cutoffs[i + 1]``.

    Returns
    -------
    b : `np.ndarray`, length N - 1
        Normalisation constant for each segment.
    """
    m = np.asarray(mass_cutoffs, dtype=float)
    a = np.asarray(slopes, dtype=float)
    n_seg = len(a)
    assert len(m) == n_seg + 1, "mass_cutoffs must have exactly one more element than slopes"

    # continuity: b[i + 1] = b[i] * m[i + 1] ** (-(a[i] - a[i + 1]))
    # express each b[i] as c[i] * b[0] with c[0] = 1
    c = np.ones(n_seg)
    for i in range(n_seg - 1):
        c[i + 1] = c[i] * m[i + 1] ** (-(a[i] - a[i + 1]))

    # normalisation: b[0] * sum_i c[i] * integral of m^(-a[i]) over segment i == 1
    integral = np.sum(c * (m[1:] ** (1 - a) - m[:-1] ** (1 - a)) / (1 - a))
    b0 = 1 / integral
    return b0 * c


def _cdf_at_cutoffs(mass_cutoffs, slopes, b):
    """Cumulative fraction of stellar mass at each cutoff (i.e. the CDF evaluated at the cutoffs)."""
    m = np.asarray(mass_cutoffs, dtype=float)
    a = np.asarray(slopes, dtype=float)
    # contribution of each segment to the CDF
    seg = b * (m[1:] ** (1 - a) - m[:-1] ** (1 - a)) / (1 - a)
    # F[0] = 0 at the lowest cutoff, then accumulate
    return np.concatenate([[0.0], np.cumsum(seg)])


def IMF(m, mass_cutoffs=DEFAULT_MASS_CUTOFFS, slopes=DEFAULT_SLOPES):
    """Calculate the fraction of stellar mass between m and m + dm for an N-part broken power law.

        zeta(m) ~ m^(-slopes[i])   for mass_cutoffs[i] <= m < mass_cutoffs[i + 1]

    Parameters
    ----------
    m : `float` or `np.ndarray`
        Mass(es) at which to evaluate the IMF.
    mass_cutoffs : list of `float`, length N
        Masses at which the slope of the IMF changes.
    slopes : list of `float`, length N - 1
        Slope of the IMF on each segment.

    Returns
    -------
    imf_vals : `float` or `np.ndarray`
        IMF evaluated at the given mass(es).
    """
    m_cut = np.asarray(mass_cutoffs, dtype=float)
    a = np.asarray(slopes, dtype=float)
    b = _normalisation_constants(mass_cutoffs, slopes)

    m_arr = np.atleast_1d(np.asarray(m, dtype=float))
    vals = np.zeros_like(m_arr)
    for i in range(len(a)):
        mask = (m_arr >= m_cut[i]) & (m_arr < m_cut[i + 1])
        vals[mask] = b[i] * m_arr[mask] ** (-a[i])

    return vals[0] if np.isscalar(m) or np.ndim(m) == 0 else vals


def CDF_IMF(m, mass_cutoffs=DEFAULT_MASS_CUTOFFS, slopes=DEFAULT_SLOPES):
    """Calculate the fraction of stellar mass between 0 and m for an N-part broken power law.

        F(m) ~ int_0^m zeta(m) dm

    Parameters
    ----------
    m : `float` or `np.ndarray`
        Mass(es) at which to evaluate the CDF.
    mass_cutoffs : list of `float`, length N
        Masses at which the slope of the IMF changes.
    slopes : list of `float`, length N - 1
        Slope of the IMF on each segment.

    Returns
    -------
    cdf : `float` or `np.ndarray`
        Cumulative fraction of stellar mass below the given mass(es).
    """
    m_cut = np.asarray(mass_cutoffs, dtype=float)
    a = np.asarray(slopes, dtype=float)
    b = _normalisation_constants(mass_cutoffs, slopes)
    F = _cdf_at_cutoffs(mass_cutoffs, slopes, b)

    m_arr = np.atleast_1d(np.asarray(m, dtype=float))
    cdf = np.zeros_like(m_arr)
    for i in range(len(a)):
        mask = (m_arr >= m_cut[i]) & (m_arr < m_cut[i + 1])
        cdf[mask] = F[i] + b[i] / (1 - a[i]) * (m_arr[mask] ** (1 - a[i]) - m_cut[i] ** (1 - a[i]))
    # everything at or above the top cutoff has accumulated the full mass
    cdf[m_arr >= m_cut[-1]] = 1.0

    return cdf[0] if np.isscalar(m) or np.ndim(m) == 0 else cdf


def inverse_CDF_IMF(U, mass_cutoffs=DEFAULT_MASS_CUTOFFS, slopes=DEFAULT_SLOPES):
    """Calculate the inverse CDF for an N-part broken power law (for inverse-transform sampling).

    Parameters
    ----------
    U : `float` or `np.ndarray`
        Fraction(s) between 0 and 1.
    mass_cutoffs : list of `float`, length N
        Masses at which the slope of the IMF changes.
    slopes : list of `float`, length N - 1
        Slope of the IMF on each segment.

    Returns
    -------
    masses : `float` or `np.ndarray`
        Mass(es) corresponding to the given CDF fraction(s).
    """
    m_cut = np.asarray(mass_cutoffs, dtype=float)
    a = np.asarray(slopes, dtype=float)
    b = _normalisation_constants(mass_cutoffs, slopes)
    F = _cdf_at_cutoffs(mass_cutoffs, slopes, b)

    U_arr = np.atleast_1d(np.asarray(U, dtype=float))
    masses = np.zeros_like(U_arr)
    for i in range(len(a)):
        mask = (U_arr > F[i]) & (U_arr <= F[i + 1])
        masses[mask] = np.power((1 - a[i]) / b[i] * (U_arr[mask] - F[i]) + m_cut[i] ** (1 - a[i]),
                                1 / (1 - a[i]))

    return masses[0] if np.isscalar(U) or np.ndim(U) == 0 else masses
