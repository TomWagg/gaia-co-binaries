#!/usr/bin/env python
import os
import time
import argparse
import numpy as np

from cosmic.sample.stroopwafel import AdaptiveSampler, ParameterSpace, Parameter
from cosmic.sample.stroopwafel.rejection import default_reject
from cosmic.utils import parse_inifile

params = ParameterSpace([
    # --- orbital / stellar ---
    Parameter('mass_1',       5.0,        150.0,      dist='kroupa'),
    Parameter('q',            0.01,       1.0,        dist='uniform'),
    Parameter('porb',         10**(0.15), 10**(5.5),  dist='sana'),   # ~1.4 d to ~316 000 d
    Parameter('ecc',          1e-9,       0.99999999, dist='sana_ecc'),
    # --- primary natal kick magnitude only ---
    Parameter('natal_kick_1', 0.1,        5000.0,     dist='disberg'),
])

# extend the default_reject function to also reject systems with secondary mass < 0.1 Msun
def reject_systems(binary_params, SSEDict=None):
    return default_reject(binary_params, SSEDict=SSEDict, min_secondary_mass=0.1)

# ------------------------------------------------------------------
# Derived quantities
# ------------------------------------------------------------------
def derive_params(sampled):
    """Provide the secondary mass from the sampled primary mass and mass ratio.

    A binary is defined by {mass_1, mass_2, porb, ecc, metallicity}.  Here
    mass_1, porb, ecc, and metallicity are sampled directly, so only mass_2
    needs deriving.

    Parameters
    ----------
    sampled : dict
        Maps each sampled parameter name to its (N,) array of physical values.

    Returns
    -------
    dict
        ``{'mass_2': ...}`` -- the one required parameter not sampled here.
    """
    return {'mass_2': sampled['mass_1'] * sampled['q'],
            "metallicity": np.full_like(sampled['mass_1'], 0.002)}

# ------------------------------------------------------------------
# Hit definition: BH (kstar=14) + normal star (kstar 0–9), still bound
# ------------------------------------------------------------------
_STAR_KSTARS = set(range(10))   # kstar 0–9: non-degenerate stars

def is_bh_star(bpp):
    """Identify binaries that are in a bound BH + normal-star phase for ≥ 100 Myr.

    Parameters
    ----------
    bpp : pandas.DataFrame
        COSMIC binary population parameters output.

    Returns
    -------
    n_hits : int
        Number of distinct binaries satisfying the criterion.
    hit_bin_nums : numpy.ndarray
        Integer bin_num values of those binaries.
    """
    bh_star_mask = (
        (   (bpp.kstar_1 == 14) & bpp.kstar_2.isin(_STAR_KSTARS))
        | (bpp.kstar_1.isin(_STAR_KSTARS) & (bpp.kstar_2 == 14))
    ) & (bpp.sep > 0)

    bh_star_rows = bpp.loc[bh_star_mask]
    if len(bh_star_rows) == 0:
        return 0, np.array([], dtype=int)

    # Group by bin_num (the natural key) so that min/max tphys are aligned
    # on the same index.  drop_duplicates would give rows with different
    # integer-row indices that pandas would NOT align correctly on subtraction.
    phase_start = bh_star_rows.groupby('bin_num')['tphys'].min()  # Series indexed by bin_num
    phase_end   = bh_star_rows.groupby('bin_num')['tphys'].max()  # Series indexed by bin_num
    duration    = phase_end - phase_start                          # tphys in Myr

    long_enough_bin_nums = duration[duration >= 100.0].index.values
    return len(long_enough_bin_nums), long_enough_bin_nums

# ------------------------------------------------------------------
# Helper: run one sampler and return (result, elapsed_seconds)
# ------------------------------------------------------------------
def run_sampler(mc_only, BSEDict, SSEDict, seed):
    label = "Monte Carlo" if mc_only else "STROOPWAFEL"
    print(f"\n{'='*60}")
    print(f"  Running {label}  (seed={seed})")
    print(f"{'='*60}")

    sw = AdaptiveSampler(
        parameter_space=params,
        total_systems=args.num_systems,
        batch_size=args.batch_size,
        BSEDict=BSEDict,
        SSEDict=SSEDict,
        is_interesting=is_bh_star,
        derive_params=derive_params,
        reject_systems=reject_systems,
        nproc=args.num_cores,
        n_generations=args.n_generations,
        mc_only=mc_only,
        seed=seed,
    )

    t0 = time.time()
    result = sw.run()
    elapsed = time.time() - t0

    return result, elapsed

# ------------------------------------------------------------------
# Helper: summarise one result
# ------------------------------------------------------------------
def print_summary(label, result, elapsed):
    raw_hits = int(np.sum(result.is_hit))
    hit_rate = result.hit_rate
    uncertainty = result.hit_rate_uncertainty

    print(f"\n--- {label} summary ---")
    print(f"  Systems evolved  : {len(result.weights):,}")
    print(f"  Raw hits found   : {raw_hits:,}")
    print(f"  Weighted hit rate: {hit_rate:.6e} ± {uncertainty:.6e}")
    print(f"  Wall-clock time  : {elapsed:.1f} s")

# ------------------------------------------------------------------
# Helper: side-by-side comparison
# ------------------------------------------------------------------
def print_comparison(mc_result, mc_elapsed, sw_result, sw_elapsed):
    mc_rate  = mc_result.hit_rate
    mc_unc   = mc_result.hit_rate_uncertainty
    sw_rate  = sw_result.hit_rate
    sw_unc   = sw_result.hit_rate_uncertainty

    print(f"\n{'='*60}")
    print("  Efficiency comparison")
    print(f"{'='*60}")
    print(f"{'':30s}  {'Monte Carlo':>15s}  {'STROOPWAFEL':>15s}")
    print(f"  {'Systems evolved':<28s}  {len(mc_result.weights):>15,}  {len(sw_result.weights):>15,}")
    print(f"  {'Raw hits found':<28s}  {int(np.sum(mc_result.is_hit)):>15,}  {int(np.sum(sw_result.is_hit)):>15,}")
    print(f"  {'Weighted hit rate':<28s}  {mc_rate:>15.4e}  {sw_rate:>15.4e}")
    print(f"  {'Uncertainty (1σ)':<28s}  {mc_unc:>15.4e}  {sw_unc:>15.4e}")
    print(f"  {'Wall-clock time (s)':<28s}  {mc_elapsed:>15.1f}  {sw_elapsed:>15.1f}")

    if sw_unc > 0 and mc_unc > 0:
        # How many MC systems would give the same precision as SW?
        equivalent_mc = len(mc_result.weights) * (mc_unc / sw_unc) ** 2
        speedup = equivalent_mc / len(sw_result.weights)
        print(f"\n  STROOPWAFEL is ~{speedup:.1f}x more statistically efficient:")
        print(f"  Monte Carlo would need ~{equivalent_mc:,.0f} systems to match")
        print(f"  STROOPWAFEL's precision on {len(sw_result.weights):,} systems.")

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('-i', '--ini-file', type=str, default='/mnt/home/twagg/projects/gaia-bhs/simulations/params.ini',
                        help='COSMIC ini file (default: params.ini)')
    parser.add_argument('-o', '--output-dir', type=str, default='/mnt/home/twagg/projects/gaia-bhs/data',
                        help='Output directory (default: data)')
    parser.add_argument('-s', '--sim-name', type=str, default='test',
                        help='Simulation name (default: test)')
    parser.add_argument('--num_systems', type=int, default=50000,
                        help='Total systems to evolve per method (default: 50000)')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Systems per COSMIC call (default: 1000)')
    parser.add_argument('--num_cores', type=int, default=8,
                        help='CPU cores for COSMIC (default: 8)')
    parser.add_argument('--n_generations', type=int, default=1,
                        help='STROOPWAFEL refinement generations (default: 1)')
    parser.add_argument('--mc_only', action='store_true',
                        help='Run Monte Carlo baseline only, skip STROOPWAFEL')
    parser.add_argument('--sw_only', action='store_true',
                        help='Run STROOPWAFEL only, skip Monte Carlo baseline')
    parser.add_argument('--seed', type=int, default=117,
                        help='Base random seed; MC uses seed, STROOPWAFEL uses seed+1')
    args = parser.parse_args()

    BSEDict, SSEDict, _, _, _, _ = parse_inifile(args.ini_file)

    mc_result = mc_elapsed = None
    sw_result = sw_elapsed = None

    if not args.sw_only:
        mc_result, mc_elapsed = run_sampler(mc_only=True, BSEDict=BSEDict, SSEDict=SSEDict, seed=args.seed)
        print_summary("Monte Carlo", mc_result, mc_elapsed)
        mc_result.save(os.path.join(args.output_dir, f"{args.sim_name}_mc.h5"))

    if not args.mc_only:
        sw_result, sw_elapsed = run_sampler(mc_only=False, BSEDict=BSEDict, SSEDict=SSEDict, seed=args.seed + 1)
        print_summary("STROOPWAFEL", sw_result, sw_elapsed)
        sw_result.save(os.path.join(args.output_dir, f"{args.sim_name}.h5"))

    if mc_result is not None and sw_result is not None:
        print_comparison(mc_result, mc_elapsed, sw_result, sw_elapsed)
