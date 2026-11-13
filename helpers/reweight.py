import numpy as np


def sana_pdf(log_p, lo=0.15, hi=5.5):
    # p(log P) ∝ (log P)^-0.55, truncated to [lo, hi] (lo can be a per-system array)
    norm = (hi**0.45 - lo**0.45) / 0.45
    return np.where((log_p >= lo) & (log_p <= hi), log_p**-0.55 / norm, 0.0)

def loguniform_pdf(log_p, lo=0.15, hi=5.5):
    return np.where((log_p >= lo) & (log_p <= hi), 1.0 / (hi - lo), 0.0)

def reweight_to_loguniform(p=None, w=None, log_p=None, lo=0.15, hi=5.5):
    if w is None or log_p is None:
        w = p.initC["weights"].values
        log_p = np.log10(p.initC["porb"].values)
    return w * loguniform_pdf(log_p, lo, hi) / sana_pdf(log_p, lo, hi)

def sana_ecc_pdf(e, lo=1e-9, hi=0.9999):
    # p(e) ~ e^-0.45
    norm = (hi**0.55 - lo**0.55) / 0.55
    return np.where((e >= lo) & (e <= hi), e**-0.45 / norm, 0.0)

def thermal_ecc_pdf(e, lo=1e-9, hi=0.9999):
    # p(e) = 2e
    norm = hi**2 - lo**2
    return np.where((e >= lo) & (e <= hi), 2 * e / norm, 0.0)

def reweight_to_thermal(p=None, w=None, e=None, lo=1e-9, hi=0.9999):
    if w is None or e is None:
        w = p.initC["weights"].values
        e = p.initC["ecc"].values
    return w * thermal_ecc_pdf(e, lo, hi) / sana_ecc_pdf(e, lo, hi)