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
import norm
import binfrac

import gala.potential as gp
import cogsworth

def instrinic_counts(output_folder, f_sample_kwargs={}, m1_min_NS=6.0):
    """Calculate the intrinsic counts of BH-LMS systems for each metallicity

    Parameters
    ----------
    output_folder : `str`
        Path to the folder containing the output files from create_bh_lms_population.py
    """
    f_sample = {"BH": None, "NS": None}
    f_sample["BH"] = norm.calculate_f_sample(**f_sample_kwargs)[0]
    f_sample_kwargs["m1_min"] = m1_min_NS
    f_sample["NS"] = norm.calculate_f_sample(**f_sample_kwargs)[0]
    print(f_sample)
    M_SF_MW = 1e11

    p_Z = {}
    base_path = "/mnt/home/twagg/projects/gaia-bhs/data/"
    SB15 = cogsworth.sfh.SandersBinney2015(potential=gp.MilkyWayPotential(version='v2'))
    for k, path in zip(["sse", "metisse"], ["MW_Z_weights_SSE.npy", "MW_Z_weights_METISSE.npy"]):
        if not os.path.isfile(os.path.join(base_path, path)):
            if len(SB15) == 0:
                SB15.sample(10_000_000)
            np.save(
                os.path.join(base_path, path),
                norm.metallicity_weights(const.METISSE_Z_GRID if k == "metisse" else const.SSE_Z_GRID,
                                         SB15.Z.value, clip_edges=True)
            )
        p_Z[k] = np.load(os.path.join(base_path, path))

    n_intrinsics = {"BH": [], "NS": []}

    for variation in const.VARIATIONS:
        # if "raghavan" not in variation:
        # if variation not in ["olejak_20_new_winds_and_cbm", "olejak_20_new_winds_and_cbm_kicks"]:
            # n_intrinsics["BH"].append(np.nan)
            # n_intrinsics["NS"].append(np.nan)
            continue
        print(variation)
        grid = "sse" if variation == "sse" else "metisse"
        grid_vals = const.SSE_Z_GRID if variation == "sse" else const.METISSE_Z_GRID
        n_met = len(grid_vals)
        eta = np.zeros(n_met)

        for co_type in ["BH", "NS"]:
            print("  co_type:", co_type)

            if not os.path.isfile(os.path.join(output_folder, variation, f"{co_type}_LMS.h5")):
                n_intrinsics[co_type].append(np.nan)
                continue

            # formation_rows = pd.read_hdf(os.path.join(output_folder, variation, f"{co_type}_LMS.h5"), key="formation_rows")
            # initC = pd.read_hdf(os.path.join(output_folder, variation, "BH_LMS.h5"), key="initC")
            # p_alive = np.clip((formation_rows["t_end"] - formation_rows["t_start"]).values / 12000, 0, 1)

            for i in range(n_met):
                print("    metallicity index:", i)
                file_path = os.path.join(output_folder, variation, f"stroopwafel_files/{co_type}_LMS_Z_IND_{i}.h5")
                if not os.path.exists(file_path):
                    print(f"File {file_path} does not exist, skipping.")
                    continue

                with h5.File(file_path, "r") as f:
                    # print(f["stroopwafel"].attrs["param_names"])
                    weights = f["stroopwafel"]["weights"][...]
                    is_hit = f["stroopwafel"]["is_hit"][...]
                    m_1 = f["stroopwafel"]["samples"][:, 0]
                    m_2 = m_1 * f["stroopwafel"]["samples"][:, 1]

                total_mass = m_1 + m_2
                f_bin = f_sample_kwargs.get("f_bin", binfrac.offner_multiplicity)
                if callable(f_bin):
                    f_bin = f_bin(m_1)

                #  p_alive[initC["metallicity"] == grid_vals[i]]
                eta[i] = f_sample[co_type] * np.sum(weights[is_hit] * f_bin[is_hit]) / np.sum(weights * f_bin * total_mass)
                # print(f"    eta[{i}] = {eta[i]:1.3e}")

            n_intrinsic = M_SF_MW * np.sum(p_Z[grid] * eta)
            print(f"  n_intrinsic[{variation}] = {n_intrinsic:1.3e}")
            n_intrinsics[co_type].append(n_intrinsic)

    df = pd.DataFrame({
        "variation": const.VARIATIONS.keys(),
        "n_intrinsic_BH": n_intrinsics["BH"],
        "n_intrinsic_NS": n_intrinsics["NS"]
    })
    df.to_hdf(os.path.join(output_folder, "n_intrinsic.h5"), key="n_intrinsic", mode="w")
    print(df)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the intrinsic counts of BH-LMS systems for each metallicity")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Path to the folder containing the output files from create_bh_lms_population.py")
    parser.add_argument("--m1_min", type=float, default=10.0,
                        help="Minimum mass of the primary star (in solar masses)")
    parser.add_argument("--m1_min_NS", type=float, default=6.0,
                        help="Minimum mass of the primary star in NS systems (in solar masses)")
    parser.add_argument("--m1_max", type=float, default=150.0,
                        help="Maximum mass of the primary star (in solar masses)")
    parser.add_argument("--m2_min", type=float, default=0.1,
                        help="Minimum mass of the secondary star (in solar masses)")
    parser.add_argument("--m1_min_imf", type=float, default=0.08,
                        help="Minimum mass for the IMF integration (in solar masses)")
    parser.add_argument("--f_bin", type=float, default=None,
                        help="Binary fraction (can be a float or a callable function)")

    args = parser.parse_args()

    if args.f_bin is None:
        args.f_bin = binfrac.offner_multiplicity

    f_sample_kwargs = {
        "m1_min": args.m1_min,
        "m1_max": args.m1_max,
        "m2_min": args.m2_min,
        "m1_min_imf": args.m1_min_imf,
        "f_bin": args.f_bin
    }

    instrinic_counts(args.output_folder, f_sample_kwargs=f_sample_kwargs, m1_min_NS=args.m1_min_NS)

# python get_n_intrinsic.py --output_folder /mnt/ceph/users/twagg/gaia-bhs