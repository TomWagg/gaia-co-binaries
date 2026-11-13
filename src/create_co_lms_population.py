import argparse
from cosmic.sample.stroopwafel import AdaptiveSampler, ParameterSpace, Parameter
from cosmic.sample.stroopwafel.distributions import TruncatedNormal, Log10
from cosmic.sample.stroopwafel.rejection import default_reject
from cosmic.utils import parse_inifile
from functools import partial
from time import time
import numpy as np
import os

import sys
sys.path.append("../helpers")
import const


def is_co_lms(bpp, kstar):
    bh_star_mask = (
        ((bpp.kstar_1 == kstar) & (bpp.kstar_2 < 10)) | ((bpp.kstar_1 < 10) & (bpp.kstar_2 == kstar))
    ) & (bpp.sep > 0)

    bh_star_rows = bpp.loc[bh_star_mask]
    if len(bh_star_rows) == 0:
        return 0, np.array([], dtype=int)

    # work out the duration with groupings
    phase_start = bh_star_rows.groupby('bin_num')['tphys'].min()
    phase_end = bh_star_rows.groupby('bin_num')['tphys'].max()
    duration = phase_end - phase_start

    long_enough_bin_nums = duration[duration >= 100.0].index.values
    return len(long_enough_bin_nums), long_enough_bin_nums


def is_bh_lms(bpp):
    return is_co_lms(bpp, 14)

def is_ns_lms(bpp):
    return is_co_lms(bpp, 13)


def create_co_lms_population(
        co, metallicity_ind, metallicity_grid, inifile_path, total_systems, batch_size, nproc, output_path, mc_only=False,
        ecc_model="sana_ecc", porb_model="sana"
    ):

    metallicity = const.METISSE_Z_GRID[metallicity_ind] if metallicity_grid == "metisse" else const.SSE_Z_GRID[metallicity_ind]

    start_message = f"Starting {co}LMS population generation with these parameters:"
    start_message += f"\nMetallicity: {metallicity:.2e} (from {metallicity_grid} grid, index {metallicity_ind})"
    start_message += f"\nINI file: {inifile_path}"
    start_message += f"\nTotal systems: {total_systems}"
    start_message += f"\nBatch size: {batch_size}"
    start_message += f"\nNumber of processes: {nproc}"
    start_message += f"\nOutput path: {output_path}"
    if mc_only:
        start_message += "\nOnly running with MONTE CARLO sampling"
    print(start_message)

    start = time()

    def derive_params(sampled):
        """Provide binary parameters not drawn from the ParameterSpace."""
        return {'mass_2': sampled['mass_1'] * sampled['q'], "metallicity": metallicity}

    if porb_model == "raghavan10":
        # log normal orbital periods in days with mean_logP = 5.03 and sigma_logP = 2.28 between 0 and 9 following Raghavan+2010 
        porb_param = Parameter(
            'porb', 10**0, 10**9,
            dist=TruncatedNormal(mu=5.03, scale=2.28, transform=Log10()),
        )
    else:
        porb_param = Parameter('porb', 10**(0.15), 10**(5.5), dist=porb_model)

    param_list = [
        Parameter('mass_1', 6 if co.lower() == "ns" else 10.0, 150.0, dist='kroupa'),
        Parameter('q', 0.0, 1.0, dist='uniform'),
        porb_param,
        Parameter('ecc', 1e-9, 0.9999, dist=ecc_model),
        Parameter('natal_kick_1', 0.1, 5000.0, dist='disberg'),
    ]

    params = ParameterSpace(param_list)

    BSEDict, SSEDict, _, _, _, _ = parse_inifile(inifile_path)

    sampler = AdaptiveSampler(
        parameter_space=params,
        total_systems=total_systems,
        batch_size=batch_size,
        BSEDict=BSEDict,
        SSEDict=SSEDict,
        is_interesting=is_bh_lms if co.lower() == "bh" else is_ns_lms,
        derive_params=derive_params,
        reject_systems=partial(default_reject, SSEDict=SSEDict, min_secondary_mass=0.1),
        nproc=nproc,
        n_generations=1,
        seed=117,
        only_save_hit_tables=not mc_only,
        mc_only=mc_only
    )
    result = sampler.run()

    print(f"Finished generating {total_systems} systems for metallicity {metallicity:.2e}.")
    print(f"Time taken: {time() - start:.2f} seconds.")

    result.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Create a population of DCOs using adaptive sampling.")
    parser.add_argument("--co", type=str, choices=["BH", "NS"], required=True, help="Which compact object to generate: 'bh' or 'ns'.")
    parser.add_argument("--metallicity-ind", type=int, required=True, help="Index of the metallicity in the grid to use.")
    parser.add_argument("--metallicity-grid", type=str, choices=["metisse", "sse"], required=True, help="Which metallicity grid to use: 'metisse' or 'sse'.")
    parser.add_argument("--inifile", type=str, required=True, help="Path to the ini file with simulation parameters.")
    parser.add_argument("--total_systems", type=int, required=True, help="Total number of systems to generate.")
    parser.add_argument("--batch_size", type=int, required=True, help="Number of systems to generate per batch.")
    parser.add_argument("--nproc", type=int, required=True, help="Number of processes to use for parallelization.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the generated population data.")
    parser.add_argument("--mc-only", action="store_true", help="Whether to just do Monte Carlo sampling")
    parser.add_argument("--ecc-model", type=str, default="sana_ecc", help="Eccentricity distribution model to use.")
    parser.add_argument("--porb-model", type=str, default="sana", help="Orbital period distribution model to use.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    create_co_lms_population(
        co=args.co,
        metallicity_ind=args.metallicity_ind,
        metallicity_grid=args.metallicity_grid,
        inifile_path=args.inifile,
        total_systems=args.total_systems,
        batch_size=args.batch_size,
        nproc=args.nproc,
        output_path=args.output_path,
        mc_only=args.mc_only,
        ecc_model=args.ecc_model,
        porb_model=args.porb_model
    )

if __name__ == "__main__":
    main()

# python create_co_lms_population.py --co BH --metallicity-ind 16 --metallicity-grid metisse --inifile /mnt/home/twagg/projects/gaia-bhs/settings/metisse.ini --total_systems 200000 --batch_size 10000 --nproc 32 --output_path /mnt/ceph/users/twagg/gaia-bhs/metisse/stroopwafel_files/mc_only_test.h5