import os.path
import argparse
import time

import pandas as pd
import astropy.units as u
import numpy as np
import h5py as h5

import sys
sys.path.append("/mnt/home/twagg/projects/gaia-bhs/helpers/")
import const
from sfh import SB15_PhiCut
import norm

import gala.potential as gp
import cogsworth
from cosmic.evolve import Evolve
from cosmic.output import load_initC



def evolve_milky_way_instance(
        all_formation_rows, all_kick_infos, all_initCs,
        Z_edges, Z_centres,
        n_per_instance=500_000, oversample_factor=5,
        variation="fiducial", cores=2
    ):
    lap = time.time()

    # draw a random sample, weighted by the Milky Way metallicity distribution
    rand_sample = all_formation_rows.sample(n_per_instance, weights="MW_Z_weight", replace=True)
    rand_kicks = all_kick_infos.loc[rand_sample.index]
    rand_initC = all_initCs.loc[rand_sample.index]

    print(f"  [{time.time() - lap:03.1f}s] Sampled {n_per_instance} systems from the formation rows")
    lap = time.time()

    # work out the target number of systems in each metallicity bin, for galaxy sampling
    target_counts, _ = np.histogram(rand_sample["metallicity"].values, bins=Z_edges)
    sample_size = target_counts.sum() * oversample_factor

    # loop variables
    counts, rnd, sfhs = np.zeros_like(target_counts), 0, []

    print(f"  [{time.time() - lap:03.1f}s] Starting to sample from the SFH to get {target_counts.sum()} systems in the right metallicity bins...")
    lap = time.time()

    # continue until we have enough systems in each metallicity bin
    while np.sum(counts) < np.sum(target_counts):
        # find the maximum metallicity that doesn't have a full bin
        max_Z = Z_centres[counts < target_counts].max()
    
        # max Z is below 1e-3 then switch to the phi cut version
        if max_Z <= 5e-4:
            print(f"    Using phi cut of 3.0 for max_Z <= 5e-4, with max_Z = {max_Z}")
            SB15 = SB15_PhiCut(phi_cut=3.0, potential=gp.MilkyWayPotential(version="v2"))
        elif max_Z <= 1e-3:
            print(f"    Using phi cut of 2.0 for max_Z <= 1e-3, with max_Z = {max_Z}")
            SB15 = SB15_PhiCut(phi_cut=2.0, potential=gp.MilkyWayPotential(version="v2"))
        else:
            print(f"    Using phi cut of 0.0 for max_Z > 1e-3, with max_Z = {max_Z}")
            SB15 = SB15_PhiCut(phi_cut=0.0, potential=gp.MilkyWayPotential(version="v2"))
        # sample from the SFH, clipping metallicities to the range of the COSMIC sim
        SB15.sample(int(sample_size))
        SB15.Z[SB15.Z < 1e-4] = 1e-4
        SB15.Z[SB15.Z > 0.03] = 0.03

        # go through each metallicity bin and add systems to the SFH until we have enough
        for i in range(len(target_counts)):
            if counts[i] == target_counts[i]:
                continue

            matches = (Z_edges[i] <= SB15.Z.value) & (SB15.Z.value < Z_edges[i+1])
            if matches.sum() == 0:
                continue
            elif matches.sum() > target_counts[i] - counts[i]:
                # if we have more matches than we need, randomly select a subset of the matches to add
                matches[np.random.choice(np.where(matches)[0], matches.sum() - (target_counts[i] - counts[i]), replace=False)] = False

            sfhs.append(SB15[matches])
            counts[i] = min(target_counts[i], counts[i] + matches.sum())

        print(f"    After iteration {rnd + 1}, we have {np.sum(counts)} systems in the right metallicity bins.")
        rnd += 1

    print(f"  [{time.time() - lap:03.1f}s] Finished sampling from SFH after {rnd} iterations. Sampled {np.sum(counts)} systems.")
    lap = time.time()

    # start a new population with the random sample
    p = cogsworth.pop.Population(
        len(rand_sample),
        # unused but required for cogsworth
        sfh_model=cogsworth.sfh.SandersBinney2015(potential=gp.MilkyWayPotential(version='v2')),
        ini_file=f"/mnt/home/twagg/projects/gaia-bhs/settings/{variation}.ini",
        galactic_potential=gp.MilkyWayPotential(version='v2'),
        processes=cores, store_entire_orbits=False,
        # make retries more often, but not quite so large a decrease in timestep
        # (avoids more failures without tanking runtime)
        orbit_integration_retry_settings={
            "timestep_multiplier": 0.25,
            "max_retries": 4
        },
        error_file_path=None
    )

    # stick together all of the samples galaxy SFHs
    p._initial_galaxy = cogsworth.sfh.concat(*sfhs)

    # reset the bin_nums for the random sample and kick info
    rand_sample.index = np.arange(len(rand_sample))
    rand_sample["bin_num"] = rand_sample.index.values
    rand_kicks.index = np.arange(len(rand_kicks)) // 2
    rand_kicks["bin_num"] = rand_kicks.index.values
    rand_initC.index = np.arange(len(rand_initC))
    rand_initC["bin_num"] = rand_initC.index.values

    # these are all zero by definition, but we need to set them explicitly to avoid errors in cogsworth
    rand_kicks[['disrupted', 'delta_vsysx_2', 'delta_vsysy_2', 'delta_vsysz_2']] = 0.0

    # save both to the population
    p._initial_binaries = rand_initC.copy()
    p._initial_binaries["MW_Z_weight"] = rand_sample["MW_Z_weight"].values
    p._initial_binaries["metallicity"] = rand_sample["metallicity"].values

    p._kick_info = rand_kicks

    # bpp shouldn't have weights or metallicity and needs evol_type, let's just pick 15
    bpp = rand_sample.copy()
    bpp.drop(columns=["MW_Z_weight", "metallicity"], inplace=True)
    bpp["evol_type"] = 15
    p._bpp = bpp

    print(f"  [{time.time() - lap:03.1f}s] Finished setting up the population with {len(p)} systems.")
    lap = time.time()

    # step 1: retain only systems that are currently undergoing RLOF
    # --------------------------------------------------------------

    # t_start and t_end in formation rows are in Myr, p.initial_galaxy.tau is the lookback time
    is_bh_star = (
        (p.initial_galaxy.tau >= p.bpp["t_start"].values * u.Myr) &
        (p.initial_galaxy.tau <= p.bpp["t_end"].values * u.Myr)
    )
    p_bh_star = p[is_bh_star]

    # step 2: re-evolve systems with COSMIC to get their current orbital/secondary star parameters
    # --------------------------------------------------------------------------------------------
    ibt = rand_initC.loc[p_bh_star.bpp["bin_num"]].copy()
    ibt["tphysf"] = p_bh_star.initial_galaxy.tau.to(u.Myr).value

    # need to evolve each unique metallicity separately, since COSMIC doesn't support multiple metallicities in one run for METISSE
    unique_Z = np.sort(p_bh_star.initC["metallicity"].unique())

    re_ev_start = time.time()
    re_ev_bpp_list = []
    for Z_ind, Z in enumerate(unique_Z):
        lap = time.time()
        ibt_Z = ibt[ibt["metallicity"] == Z]
        re_ev_bpp_Z, _, _, _ = Evolve.evolve(ibt_Z, nproc=cores, progress_bar=True)
        # keep only the last row of the bpp (present state)
        re_ev_bpp_Z = re_ev_bpp_Z.drop_duplicates(subset="bin_num", keep="last")
        re_ev_bpp_list.append(re_ev_bpp_Z)    
        print(f"      [{time.time() - lap:03.1f}s] Evolved {len(ibt_Z)} systems with metallicity {Z:.6f} ({Z_ind + 1}/{len(unique_Z)} bins)")

    re_ev_bpp = pd.concat(re_ev_bpp_list, ignore_index=True)

    # sort by bin_num to match the order of p_bh_star.bpp
    re_ev_bpp.sort_values("bin_num", inplace=True)

    # copy over the cols COSMIC didn't produce
    extra_cols = ["weights", "t_start", "t_end", "bin_num"]
    for col in extra_cols:
        re_ev_bpp[col] = p_bh_star.bpp[col].values
    p_bh_star._bpp = re_ev_bpp[p_bh_star.bpp.columns]

    print(f"  [{time.time() - re_ev_start:03.1f}s] Evolved the {len(p_bh_star)} systems to their current orbital parameters with COSMIC.")
    lap = time.time()

    # step 3: evolve the loud systems through the Milky Way and calculate their SNRs at true distances
    p_bh_star.perform_galactic_evolution(progress_bar=False)
    print(f"  [{time.time() - lap:03.1f}s] Finished integrating the Galactic orbits of those systems through the Milky Way")
    lap = time.time()

    # et voila, a detectable fraction for this Milky Way instance
    _add_dummy_stats(p_bh_star)

    total_weights_all = p.bpp["weights"].values.sum()

    return p_bh_star, total_weights_all


def _add_dummy_stats(p):
    p._mass_binaries = 0.0
    p._mass_singles = 0.0
    p._n_singles_req = 0
    p._n_bin_req = 0
    return p


def main():
    parser = argparse.ArgumentParser(description="Evolve a Milky Way instance and calculate the number of DCOs several times.")
    parser.add_argument("-N", "--n_per_instance", type=int, default=1_000_000, help="Number of systems to sample per Milky Way instance.")
    parser.add_argument("-n", "--n_instances", type=int, default=10, help="Number of times to repeat the evolution.")
    parser.add_argument("-s", "--suffix", type=str, default="", help="Suffix to add to the output files.")
    parser.add_argument("-v", "--variation", type=str, default="metisse", help="Variation of the model to use (default: fiducial).")
    parser.add_argument('-c', '--cores', type=int, default=2, help='Number of cores to use for parallel processing (default: 2).')

    args = parser.parse_args()

    args.folder = os.path.join("/mnt/ceph/users/twagg/gaia-bhs", args.variation)
    args.output_folder = os.path.join(args.folder, "present_day_files")

    os.makedirs(args.output_folder, exist_ok=True)

    full_start = time.time()
    all_formation_rows = pd.read_hdf(os.path.join(args.folder, "BH_LMS.h5"), key="formation_rows")
    print(f"Loaded {len(all_formation_rows)} formation rows in {time.time() - full_start:.2f} seconds.")

    lap = time.time()
    all_kick_infos = pd.read_hdf(os.path.join(args.folder, "BH_LMS.h5"), key="kick_info")
    print(f"Loaded {len(all_kick_infos)} kick infos in {time.time() - lap:.2f} seconds.")

    lap = time.time()
    all_initCs = load_initC(os.path.join(args.folder, "BH_LMS.h5"), key="initC", settings_key="initC_settings")
    print(f"Loaded {len(all_initCs)} initC in {time.time() - lap:.2f} seconds.")

    SB15 = cogsworth.sfh.SandersBinney2015(potential=gp.MilkyWayPotential(version='v2'))

    WEIGHTS_PATH = f"/mnt/home/twagg/projects/gaia-bhs/data/MW_Z_weights_{"SSE" if args.variation == "sse" else "METISSE"}.npy"
    if os.path.isfile(WEIGHTS_PATH):
        MW_weights = np.load(WEIGHTS_PATH)
    else:
        SB15.sample(10_000_000)
        np.save(WEIGHTS_PATH, norm.metallicity_weights(const.SSE_Z_GRID if args.variation == "sse" else const.METISSE_Z_GRID, SB15.Z.value, clip_edges=True))

    Z_centres = const.SSE_Z_GRID if args.variation == "sse" else const.METISSE_Z_GRID

    all_formation_rows["MW_Z_weight"] = 0.0
    for Z in Z_centres:
        all_formation_rows.loc[all_formation_rows["metallicity"] == Z, "MW_Z_weight"] = MW_weights[np.digitize(Z, Z_centres) - 1]

    p_mws = []
    total_weight_list = []
    for inst in range(args.n_instances):
        print(f"Running Milky Way instance {inst + 1}/{args.n_instances}...")
        start = time.time()
        p_mw, weights_all = evolve_milky_way_instance(
            all_formation_rows, all_kick_infos, all_initCs,
            n_per_instance=args.n_per_instance,
            oversample_factor=1.75,
            variation=args.variation,
            cores=args.cores,
            Z_edges=const.SSE_Z_EDGES if args.variation == "sse" else const.METISSE_Z_EDGES,
            Z_centres=Z_centres
        )
        if p_mw is not None:
            p_mw.bpp["MW_instance"] = inst
            p_mws.append(p_mw)
        total_weight_list.append(weights_all)

        print(f"  Instance {inst + 1} finished in {time.time() - start:.2f} seconds.")

    p_mw_all = cogsworth.pop.concat(*p_mws)

    sfh = cogsworth.sfh.StarFormationHistory()
    for var in ["_x", "_y", "_z", "_v_x", "_v_y", "_v_z", "_tau", "_Z"]:
        setattr(sfh, var, getattr(p_mw_all.initial_galaxy, var))
    p_mw_all._initial_galaxy = sfh

    p_mw_all.save(os.path.join(args.output_folder, f"BH_LMS_present_day_{args.suffix}.h5"), overwrite=True)
    with h5.File(os.path.join(args.output_folder, f"BH_LMS_present_day_{args.suffix}.h5"), "a") as f:
        f.attrs["total_weight_all"] = np.sum(total_weight_list)

    print(f"Saved population to {args.output_folder}.")
    print(f"Total time for {args.n_instances} instances: {time.time() - full_start:.2f} seconds.\n\n\n")


if __name__ == "__main__":
    main()

# python sample_present_day_population.py -N 1000000 -n 1 -s "test_new" -v "metisse" -c 28