#!/usr/bin/env python
"""Run harv's orbit detection on a saved cogsworth Population 

python run_harv.py test_pop.h5 --dr dr4 --out test_pop_harv.parquet
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import run_gaiamock as rg

ECC_A, ECC_B = 0.867, 3.03        # Kipping (2013) Beta, harv's default eccentricity prior
P_MIN_DAY, P_MAX_DAY = 10.0, 1e4  # gaiamock's orbital search range
SIGMA_A0_AU = 2.0                 # TI prior scale at P = 1 yr (dark-companion photocentre orbits: ~1.3 AU rms)
N_MC = 4000
TI = ("ti_A", "ti_B", "ti_F", "ti_G")
PAD_TO, PAD_ERR = 16, 1e8         # pad epoch counts to a multiple of 16 (fewer JAX compiles); padded rows' error, mas


# --- harv model ------------------------------------------------------------------------------
def make_harv(*, seed=0, n_mc=N_MC, accjerk=False, sigma_a0_au=SIGMA_A0_AU):
    """Model, priors and the jitted evaluators. Returns a dict of callables."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import equinox as eqx
    import jax.numpy as jnp
    import numpyro.distributions as dist
    from harv import GaiaAstrometryData
    from harv.distributions import QD
    from harv.models.astrometry import GaiaAstrometryModel
    from harv.models.extensions import MonomialTrend
    from harv.models.parameterizations.gaia import ThieleInnesGaiaAstrometry
    from harv.models.priors.callables import PeriodDependentSemiMajorAxisPrior
    from jax.scipy.special import logsumexp as logsumexp_j
    from unxt import Q

    model = GaiaAstrometryModel(parameterization=ThieleInnesGaiaAstrometry(), extensions=(
        (MonomialTrend(order=2, astrometry=True, obs_unit="mas", time_unit="yr"),) if accjerk else ()))
    broad = {"ra0": QD(dist.Normal(0.0, 1e4), "mas"), "dec0": QD(dist.Normal(0.0, 1e4), "mas"),
             "pmra": QD(dist.Normal(0.0, 1e4), "mas/yr"), "pmdec": QD(dist.Normal(0.0, 1e4), "mas/yr"),
             "parallax": QD(dist.Normal(0.0, 1e3), "mas")}
    if accjerk:  # acceleration (dt^2) and jerk (dt^3); harv's MonomialTrend names them trend_*
        broad |= {f"trend_{c}_{k}": QD(dist.Normal(0.0, 1e4), "mas") for c in ("ra", "dec") for k in (1, 2)}
    ti_prior = PeriodDependentSemiMajorAxisPrior(sigma_a0=Q(sigma_a0_au, "AU"), period_ref=Q(1.0, "yr"))
    lp = broad | {k: ti_prior for k in TI}
    lp0 = broad | {k: QD(dist.Normal(0.0, 1e-9), "mas") for k in TI}  # orbit pinned to zero

    class AtParallax(eqx.Module):
        """Feed the TI prior a fixed parallax (harv only passes one when parallax is sampled, ours is marginalised)."""
        inner: eqx.Module
        plx_mas: jax.Array

        def __call__(self, params):
            return self.inner(params | {"parallax": Q(self.plx_mas, "mas")})

    def lp_at(p):
        return {k: AtParallax(v, jnp.asarray(p, float)) if k in TI else v for k, v in lp.items()}

    def lw(ni, d, p):
        return model.log_prob(ni, d, linear_priors=lp_at(p))

    nl0 = {"period": Q(365.25, "day"), "eccentricity": 0.1, "phase_peri": 0.0}  # ignored: TI pinned to 0
    lnz0 = jax.jit(lambda d: model.log_prob(nl0, d, linear_priors=lp0))

    @jax.jit
    def stats(n, d, p):  # ln B, ESS, P of the best draw, MC standard error of ln Z(orbit)
        w = jax.vmap(lambda ni: lw(ni, d, p))(n)
        ls1, log_ess = logsumexp_j(w), 2 * logsumexp_j(w) - logsumexp_j(2 * w)
        return jnp.stack([ls1 - jnp.log(w.shape[0]) - lnz0(d), jnp.exp(log_ess), n["period"].value[jnp.argmax(w)],
                          jnp.sqrt(jnp.maximum(0.0, jnp.exp(-log_ess) - 1.0 / w.shape[0]))])

    def draw(n, s):
        rng = np.random.default_rng(s)
        return {"period": Q(np.exp(rng.uniform(np.log(P_MIN_DAY), np.log(P_MAX_DAY), n)), "day"),
                "eccentricity": jnp.asarray(rng.beta(ECC_A, ECC_B, n)), "phase_peri": jnp.asarray(rng.uniform(0, 1, n))}

    def data(epochs, pad=True):
        t, psi, pf, pos, err = (np.asarray(a, float) for a in epochs)
        if pad and (k := -len(t) % PAD_TO):  # zero-weight rows: their constant normalisation cancels in ln B
            t, psi, pf, pos = (np.concatenate([a, np.zeros(k)]) for a in (t, psi, pf, pos))
            err = np.concatenate([err, np.full(k, PAD_ERR)])
        return GaiaAstrometryData(time=Q(t, "yr"), al_position=Q(pos, "mas"), al_position_err=Q(err, "mas"),
                                  scan_angle=Q(psi, "rad"), parallax_factor=jnp.asarray(pf), time_ref=Q(0.0, "yr"))
    nl = draw(n_mc, seed)
    return dict(data=data,
                ln_bf=lambda d, p, n=nl: tuple(float(x) for x in np.asarray(stats(n, d, p))))


# --- epochs ------------------------------------------------------------------------------------
_C = None


def epochs(row: dict, dr: str, seed: int, a0_mas=None):
    """gaiamock's mock epochs, seeded as run_gaiamock._cascade (same draws). ``a0_mas=0``: no orbit (null)."""
    global _C
    import gaiamock

    if _C is None:
        _C = gaiamock.read_in_C_functions()
    np.random.seed(seed)
    common = dict(ra=row["ra_deg"], dec=row["dec_deg"], parallax=row["parallax_mas"], pmra=row["pmra"],
                  pmdec=row["pmdec"], period=row["period_day"], Tp=row["tp_day"], ecc=row["eccentricity"],
                  omega=row["omega"], inc=np.radians(row["inc_deg"]), w=row["w"], phot_g_mean_mag=row["g_mag"],
                  data_release=dr, c_funcs=_C)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if a0_mas is not None:
            return gaiamock.predict_astrometry_binary_in_terms_of_a0(a0_mas=a0_mas, **common)
        return gaiamock.predict_astrometry_luminous_binary(m1=row["m_lum_msun"], m2=row["m_bh_msun"], f=0.0, **common)


def five_par(ep):
    """gaiamock's observability gate, and the 5-par RUWE and parallax (floored at its error) of the epochs."""
    import gaiamock

    t, pos = np.asarray(ep[0]), np.asarray(ep[3])
    if len(pos) < 13 or np.sum(np.diff(np.sort(t) * 365.25) > 4) + 1 < 12:
        return False, np.nan, np.nan
    ruwe, mu, sig = gaiamock.check_ruwe(*(np.asarray(a, float) for a in ep))
    return True, float(ruwe), float(max(mu[-1], sig[-1]))


COLS = ("ln_bf", "ess", "best_period_day", "lnz_mcse")


def run(src: pd.DataFrame, h, *, dr="dr4", seed=0) -> pd.DataFrame:
    out = {k: np.full(len(src), np.nan) for k in (*COLS, "ruwe", "plx_5par_mas")}
    out["n_epochs"] = np.zeros(len(src), int)
    t0 = time.time()
    for i, row in enumerate(src.to_dict("records")):
        if i % 2000 == 0:
            print(f"{i}/{len(src)} sources, {time.time() - t0:.0f} s", flush=True)
        ep = epochs(row, dr, seed + i)
        ok, out["ruwe"][i], plx = five_par(ep)
        if not ok:
            continue
        out["n_epochs"][i], out["plx_5par_mas"][i] = len(ep[0]), plx
        for k, v in zip(COLS, h["ln_bf"](h["data"](ep), plx)):
            out[k][i] = v
    return pd.concat([src, pd.DataFrame(out)], axis=1)


def summary(out: pd.DataFrame) -> str:
    w, b = out["weight"], out["ln_bf"]
    lines = [f"N = {len(out)} sources (Σw = {w.sum():.4g}), observable {(out.n_epochs > 0).sum()}, "
             f"ESS < 50 at ln B > 5: {((out.ess < 50) & (b > 5)).sum()}", f"{'':<14}{'N':>8}{'Σw':>10}{'RUWE < 1.4':>12}"]
    for t in (5, 10, 20):
        m = b > t
        lines.append(f"ln B > {t:<7}{m.sum():>8}{w[m].sum():>10.4g}{(m & (out.ruwe < 1.4)).sum():>12}")
    return "\n".join(lines)


# --- self-check ----------------------------------------------------------------------------------
def _selfcheck():
    import gaiamock

    row = dict(ra_deg=120.0, dec_deg=-30.0, parallax_mas=1.0, pmra=-5.0, pmdec=3.0, m_lum_msun=1.0, m_bh_msun=15.0,
               period_day=300.0, tp_day=50.0, eccentricity=0.3, omega=1.0, inc_deg=60.0, w=2.0, g_mag=17.0)
    ep = epochs(row, "dr4", 0)
    # same epochs as run_gaiamock's cascade (P < 1e4 d: no blending rerun)
    assert np.allclose(rg._cascade(row, "dr4", 0), gaiamock.fit_full_astrometric_cascade(*ep, c_funcs=_C))
    h = make_harv()
    _, _, plx = five_par(ep)
    bf, ess, p, _ = h["ln_bf"](h["data"](ep), plx)
    assert bf > 20 and abs(p / 300 - 1) < 0.02, (bf, ess, p)
    # padding leaves ln B unchanged
    assert abs(bf - h["ln_bf"](h["data"](ep, pad=False), plx)[0]) < 1e-6
    # no orbit: no detection
    ep0 = epochs(row, "dr4", 1, a0_mas=0.0)
    _, _, plx0 = five_par(ep0)
    bf0 = h["ln_bf"](h["data"](ep0), plx0)[0]
    assert bf0 < 5, bf0
    print(f"selfcheck OK (ln B = {bf:.0f}, P = {p:.1f} d; null ln B = {bf0:.1f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("population", nargs="?", help="saved cogsworth Population (.h5)")
    ap.add_argument("--out", help="output parquet (default: <population>_harv_<dr>.parquet)")
    ap.add_argument("--dr", default="dr4", choices=["dr3", "dr4", "dr5"])
    ap.add_argument("--seed", type=int, default=0, help="orientations, per-source noise, prior draws")
    ap.add_argument("-n", type=int, default=None, help="random subset of n binaries (before the G cut)")
    ap.add_argument("--g-min", type=float, default=13.0,
                    help="bright limit (gaiamock adds unreported noise at G < 13, which harv would read as signal)")
    ap.add_argument("--g-max", type=float, default=21.0)
    ap.add_argument("--recompute-g", action="store_true", help="G from gaiamock's recipe (see run_gaiamock.py)")
    ap.add_argument("--no-dust", action="store_true")
    ap.add_argument("--accjerk", action="store_true",
                    help="add acceleration + jerk to both models (default: orbit + 5-par vs 5-par)")
    ap.add_argument("--sigma-a0-au", type=float, default=SIGMA_A0_AU, help="TI prior scale at P = 1 yr")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return _selfcheck()
    if a.population is None:
        ap.error("population file required")
    out_path = a.out or f"{Path(a.population).with_suffix('')}_harv_{a.dr}.parquet"

    t0 = time.time()
    src = rg.build_sources(a.population, seed=a.seed, n=a.n, dust=not a.no_dust, recompute_g=a.recompute_g,
                           g_min=a.g_min, g_max=a.g_max)
    print(f"{len(src)} BH + luminous-star binaries with {a.g_min} <= G <= {a.g_max}")
    h = make_harv(seed=a.seed, accjerk=a.accjerk, sigma_a0_au=a.sigma_a0_au)
    out = run(src, h, dr=a.dr, seed=a.seed)
    out.to_parquet(out_path)
    print(summary(out))
    print(f"wrote {out_path} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
