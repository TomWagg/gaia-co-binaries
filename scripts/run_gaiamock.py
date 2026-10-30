#!/usr/bin/env python
"""Run gaiamock's full astrometric cascade on a saved cogsworth Population.

    python run_gaiamock.py test_pop.h5 --dr dr4 --out test_pop_gaiamock.parquet
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

RSUN_AU = 0.004650467260
ZSUN = 0.0142      # Asplund+2009
LOGG_SUN = 4.438   # cgs
BH = 14            # COSMIC kstar of a black hole
G_BAND = "Gaia_G_EDR3"
DR4_A0_SNR, DR4_PLX_SNR = 15.0, 5.0  # gaiamock's *_looser_cuts thresholds


# --- population -> observables -------------------------------------------------------------
def load_population(path, *, dust=True) -> pd.DataFrame:
    """Present-day bound BH + luminous-star rows of a cogsworth Population, with phase space.

    Adds cogsworth's photometry: apparent G (``g_cogsworth``; Bayestar19 dust unless ``dust`` is
    False) and absolute G at A_V = 0 (``mg_cogsworth``). For a bound binary ``_1`` is the system
    total, here the star alone since the BH has M_abs = inf. Saved observables are used as they
    are, so ``dust`` only applies when they are computed.
    """
    import cogsworth

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = cogsworth.pop.load(str(path))
        try:
            obs = p.observables
        except ValueError:  # none saved with the file
            obs = None
        if obs is None or f"{G_BAND}_app_1" not in obs:
            obs = p.get_observables(filters=[G_BAND], assume_mw_galactocentric=True,
                                    ignore_extinction=not dust, silence_bounds_warning=True)
        else:
            print("using the observables saved in the population file")
    fr = p.final_bpp.reset_index(drop=True)
    fr["g_cogsworth"], fr["mg_cogsworth"] = obs[f"{G_BAND}_app_1"].values, obs[f"{G_BAND}_abs_1"].values
    fr[["x", "y", "z"]] = p.final_pos.to_value("kpc")
    fr[["vx", "vy", "vz"]] = p.final_vel.to_value("km/s")
    k1, k2 = fr["kstar_1"], fr["kstar_2"]
    keep = (((k1 == BH) & (k2 < 10)) | ((k2 == BH) & (k1 < 10))) & (fr["sep"] > 0)
    return fr[keep].reset_index(drop=True)


def to_icrs(fr: pd.DataFrame) -> dict:
    import astropy.coordinates as coord
    import astropy.units as u

    gc = coord.Galactocentric(x=fr["x"].values * u.kpc, y=fr["y"].values * u.kpc, z=fr["z"].values * u.kpc,
                              v_x=fr["vx"].values * u.km / u.s, v_y=fr["vy"].values * u.km / u.s,
                              v_z=fr["vz"].values * u.km / u.s)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        icrs = gc.transform_to(coord.ICRS())
    d = icrs.distance.to_value(u.pc)
    return dict(ra_deg=icrs.ra.deg, dec_deg=icrs.dec.deg, dist_pc=d, parallax_mas=1e3 / d,
                pmra=icrs.pm_ra_cosdec.to_value(u.mas / u.yr), pmdec=icrs.pm_dec.to_value(u.mas / u.yr))


def gaiamock_g(m_g, dist_pc, ra_deg, dec_deg, dust=True) -> tuple[np.ndarray, np.ndarray]:
    """gaiamock's recipe (``simulate_many_realizations_of_a_single_binary``): G = M_G + 5 log10(d / 10 pc)
    + A_G with A_G = 2.80 E(B-V), E(B-V) from mwdust Combined19. gaiamock takes M_G as given; here it is
    cogsworth's (MIST BC_G at A_V = 0). Returns (G, A_G).
    """
    import astropy.units as u
    import mwdust
    from astropy.coordinates import SkyCoord

    a_g = np.zeros_like(m_g)
    if dust:
        gal = SkyCoord(ra=np.asarray(ra_deg) * u.deg, dec=np.asarray(dec_deg) * u.deg).galactic
        a_g = 2.80 * np.asarray(mwdust.Combined19()(gal.l.deg, gal.b.deg, np.asarray(dist_pc) / 1e3), float)
    return m_g + 5 * np.log10(dist_pc / 10) + a_g, a_g


def build_sources(path, *, seed=0, n=None, dust=True, recompute_g=False, g_min=13.0, g_max=21.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fr = load_population(path, dust=dust and not recompute_g)
    if n is not None and n < len(fr):
        fr = fr.iloc[np.sort(rng.choice(len(fr), n, replace=False))].reset_index(drop=True)
    bh1 = fr["kstar_1"].values == BH
    lum_of = lambda c: np.where(bh1, fr[f"{c}_2"], fr[f"{c}_1"]).astype(float)  # noqa: E731
    m_lum, m_bh = lum_of("mass"), np.where(bh1, fr["mass_1"], fr["mass_2"]).astype(float)
    pos = to_icrs(fr)
    logg = np.log10(m_lum) - 2 * np.log10(lum_of("rad")) + LOGG_SUN
    feh = np.log10(np.clip(fr["metallicity"].values, 1e-8, None) / ZSUN)
    if not recompute_g:
        g = fr["g_cogsworth"].values
        a_g = g - fr["mg_cogsworth"].values - 5 * np.log10(pos["dist_pc"] / 10)
    else:
        g, a_g = gaiamock_g(fr["mg_cogsworth"].values, pos["dist_pc"], pos["ra_deg"], pos["dec_deg"], dust=dust)
    period = 365.25 * np.sqrt((fr["sep"].values * RSUN_AU) ** 3 / (m_lum + m_bh))
    src = pd.DataFrame(dict(
        bin_num=fr["bin_num"].values, weight=fr["weights"].values, **pos,
        m_lum_msun=m_lum, m_bh_msun=m_bh, period_day=period, eccentricity=fr["ecc"].values,
        tp_day=rng.uniform(0, 1, len(fr)) * period, omega=rng.uniform(0, 2 * np.pi, len(fr)),
        inc_deg=np.degrees(np.arccos(rng.uniform(-1, 1, len(fr)))), w=rng.uniform(0, 2 * np.pi, len(fr)),
        teff_k=lum_of("teff"), lum_lsun=lum_of("lum"), logg=logg, feh=feh, a_g=a_g,
        g_mag=g,
    ))
    return src[src["g_mag"].between(g_min, g_max)].reset_index(drop=True)


_C = None

def _cascade(row: dict, dr: str, seed: int) -> list:
    global _C
    import gaiamock

    if _C is None:
        _C = gaiamock.read_in_C_functions()
    np.random.seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return list(gaiamock.run_full_astrometric_cascade(
            ra=row["ra_deg"], dec=row["dec_deg"], parallax=row["parallax_mas"], pmra=row["pmra"],
            pmdec=row["pmdec"], m1=row["m_lum_msun"], m2=row["m_bh_msun"], period=row["period_day"],
            Tp=row["tp_day"], ecc=row["eccentricity"], omega=row["omega"], inc_deg=row["inc_deg"], w=row["w"],
            phot_g_mean_mag=row["g_mag"], f=0.0, data_release=dr, c_funcs=_C))


def decode(res) -> pd.DataFrame:
    """gaiamock's 23-element result rows -> named columns + publication labels."""
    r = np.asarray(res, float)
    s0 = r[:, 0]
    orb = s0 > 0
    sol = np.select([s0 == 0, s0 == -1, s0 == -9, s0 == -7], ["unobservable", "single_star", "accel9", "accel7"],
                    "orbital")
    ruwe = np.select([s0 == -1, s0 == -9, s0 == -7, orb], [r[:, 1], r[:, 12], r[:, 8], r[:, 22]], np.nan)
    nan = lambda j: np.where(orb, r[:, j], np.nan)  # noqa: E731
    out = pd.DataFrame(dict(
        solution_type=sol, ruwe=ruwe,
        accel_significance=np.where((s0 == -9) | (s0 == -7), r[:, 1], np.nan),
        fit_parallax_mas=nan(0), fit_parallax_err_mas=nan(1), fit_period_day=nan(10), fit_period_err_day=nan(11),
        fit_ecc=nan(14), fit_ecc_err=nan(15), fit_inc_deg=nan(16), fit_a0_mas=nan(17), fit_a0_err_mas=nan(18),
        n_vis_periods=nan(19), n_obs=nan(20), f2=np.select([orb, s0 == -9, s0 == -7], [r[:, 21], r[:, 13], r[:, 9]],
                                                           np.nan),
    ))
    with np.errstate(divide="ignore", invalid="ignore"):
        a0_snr = out["fit_a0_mas"] / out["fit_a0_err_mas"]
        plx_snr = out["fit_parallax_mas"] / out["fit_parallax_err_mas"]
        p = out["fit_period_day"]
        out["published_dr3"] = (orb & (a0_snr > 5) & (plx_snr > 2e4 / p) & (a0_snr > 158 / np.sqrt(p))
                                & (out["fit_ecc_err"] < 0.079 * np.log(p) - 0.244))
        out["published_dr4_loose"] = orb & (a0_snr > DR4_A0_SNR) & (plx_snr > DR4_PLX_SNR)
    return out


def run(src: pd.DataFrame, *, dr="dr4", seed=0, n_jobs=-1) -> pd.DataFrame:
    from joblib import Parallel, delayed

    rows = src.to_dict("records")
    res = Parallel(n_jobs=n_jobs, batch_size=64)(delayed(_cascade)(r, dr, seed + i) for i, r in enumerate(rows))
    return pd.concat([src, decode(res)], axis=1)


def summary(out: pd.DataFrame) -> str:
    w = out["weight"]
    lines = [f"N = {len(out)} sources (Σw = {w.sum():.4g})", f"{'':<22}{'N':>8}{'Σw':>12}"]
    for name, m in [*((s, out["solution_type"] == s) for s in
                      ("unobservable", "single_star", "accel7", "accel9", "orbital")),
                    ("published_dr3", out["published_dr3"]), ("published_dr4_loose", out["published_dr4_loose"])]:
        lines.append(f"{name:<22}{m.sum():>8}{w[m].sum():>12.4g}")
    return "\n".join(lines)


def _gaiamock_commit() -> str:
    import gaiamock

    try:
        return subprocess.run(["git", "-C", str(Path(gaiamock.__file__).parent), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"

def _selfcheck():
    rows = [[0] * 23, [-1, 1.1] + [-1] * 21, [-9, 30] + [-9] * 10 + [2.5, 3] + [-9] * 9,
            [-7, 20] + [-7] * 6 + [1.9, 4] + [-7] * 13,
            [2.0, 0.05, *[0.0] * 8, 300.0, 1.0, 0, 0, 0.3, 0.01, 60, 1.5, 0.05, 20, 60, 1.0, 3.0]]
    d = decode(rows)
    assert d["solution_type"].tolist() == ["unobservable", "single_star", "accel9", "accel7", "orbital"]
    assert np.allclose(d["ruwe"].values[1:], [1.1, 2.5, 1.9, 3.0]) and np.isnan(d["ruwe"][0])
    assert d["published_dr4_loose"].tolist() == [False] * 4 + [True]  # a0/σ = 30, ϖ/σ = 40
    # end to end: a 15 Msun BH on a 300 d orbit around a G = 14 star at 1 kpc is a clear DR4 orbit
    row = dict(ra_deg=120.0, dec_deg=-30.0, parallax_mas=1.0, pmra=-5.0, pmdec=3.0, m_lum_msun=1.0,
               m_bh_msun=15.0, period_day=300.0, tp_day=50.0, eccentricity=0.3, omega=1.0, inc_deg=60.0, w=2.0,
               g_mag=14.0)
    d = decode([_cascade(row, "dr4", 0)])
    assert d["published_dr4_loose"][0] and abs(d["fit_period_day"][0] / 300 - 1) < 0.02, d.T
    print("selfcheck OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("population", nargs="?", help="saved cogsworth Population (.h5)")
    ap.add_argument("--out", help="output parquet (default: <population>_gaiamock_<dr>.parquet)")
    ap.add_argument("--dr", default="dr4", choices=["dr3", "dr4", "dr5"])
    ap.add_argument("--seed", type=int, default=0, help="orientation draws and per-source noise")
    ap.add_argument("-n", type=int, default=None, help="random subset of n binaries (before the G cut)")
    ap.add_argument("--g-min", type=float, default=13.0, help="bright limit (gaiamock's noise model is for G > 13)")
    ap.add_argument("--g-max", type=float, default=21.0, help="faint limit (no Gaia astrometry beyond)")
    ap.add_argument("--recompute-g", action="store_true",
                    help="G from gaiamock's recipe (cogsworth M_G + A_G = 2.80 E(B-V) Combined19) instead of cogsworth's G")
    ap.add_argument("--no-dust", action="store_true", help="A_V = 0 (when G is computed, by either route)")
    ap.add_argument("--n-jobs", type=int, default=-1)
    a = ap.parse_args()

    if a.population is None:
        ap.error("population file required")
    out_path = a.out or f"{Path(a.population).with_suffix('')}_gaiamock_{a.dr}.parquet"

    t0 = time.time()
    src = build_sources(a.population, seed=a.seed, n=a.n, dust=not a.no_dust, recompute_g=a.recompute_g, g_min=a.g_min, g_max=a.g_max)
    print(f"{len(src)} BH + luminous-star binaries with {a.g_min} <= G <= {a.g_max} ({time.time() - t0:.0f} s)")
    out = run(src, dr=a.dr, seed=a.seed, n_jobs=a.n_jobs)
    out.to_parquet(out_path)
    Path(f"{out_path}.json").write_text(json.dumps(dict(
        args=vars(a), gaiamock_commit=_gaiamock_commit(), n_rows=len(out),
        g_mag=("gaiamock recipe: cogsworth M_G(A_V = 0) + " + ("no dust" if a.no_dust else "A_G = 2.80 E(B-V) Combined19"))
        if a.recompute_g else "cogsworth observables (saved, or get_observables: Bayestar19 unless --no-dust)",
        written=datetime.datetime.now().isoformat(timespec="seconds")), indent=2))
    print(summary(out))
    print(f"wrote {out_path} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
