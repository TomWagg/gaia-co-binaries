import numpy as np
from scipy.integrate import quad
from copy import copy

from imf import IMF

def calculate_f_sample(
        m1_min, m2_min, f_bin,
        m1_min_imf=0.08, m1_max=150.0
    ):
    """Calculate the fraction of initial distributions that are sampled

    sample scenario: binary-only sample: m1 > m1_min, companions with m2 = m1 * q (q ~ U(0,1))
                     retained only if m2 > m2_min.
    full scenario:   full mixed population of singles and binaries down to
                     the imf floor m1 > m2_min

    Parameters
    ----------
    m1_min : `float`
        Lower primary-mass limit for the sample. Assumed to be >= m2_min.
    m2_min : `float`
        Minimum secondary mass
    f_bin : `callable`
        f_bin(m) -> binary fraction in [0, 1] at primary mass m
    zeta : `callable`, optional
        Initial mass function, zeta(m)
    m1_min_imf : `float`, optional
        Lower mass limit of the IMF. Default is 0.08 Msun.
    m1_max : `float`, optional
        Upper mass limit of the IMF. Default is 150 Msun.

    Returns
    -------
    f_sample : `float`
        Fraction of the initial distribution that is sampled.
    """
    if m1_min < m2_min:
        raise ValueError("This code assumes m1_min >= m2_min")
    
    if isinstance(f_bin, float):
        f_bin_val = copy(f_bin)
        f_bin = lambda m: f_bin_val
    
    def mass_sample_integrand(m):
        return IMF(m) * f_bin(m) * (1.5 * m - m2_min - m2_min ** 2 / (2.0 * m))
    
    def mass_full_integrand(m):
        return IMF(m) * (m + f_bin(m) * (0.5 * m - m2_min - m2_min ** 2 / (2.0 * m)))
    
    mass_sample, _ = quad(mass_sample_integrand, m1_min, m1_max)
    mass_full, _ = quad(mass_full_integrand, m1_min_imf, m1_max)

    f_sample = mass_sample / mass_full
    return f_sample, mass_sample, mass_full
