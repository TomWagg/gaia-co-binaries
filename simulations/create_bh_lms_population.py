import argparse
from cosmic.sample.stroopwafel import AdaptiveSampler, ParameterSpace, Parameter
from cosmic.utils import parse_inifile
from time import time
import numpy as np


def is_bh_lms(bpp):
    bh_star_mask = (
        ((bpp.kstar_1 == 14) & (bpp.kstar_2 < 10)) | ((bpp.kstar_1 < 10) & (bpp.kstar_2 == 14))
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


def create_bh_lms_population(
        metallicity, inifile_path, total_systems, batch_size, nproc, output_path
    ):

    start = time()

    def derive_params(sampled):
        """Provide binary parameters not drawn from the ParameterSpace."""
        return {'mass_2': sampled['mass_1'] * sampled['q'], "metallicity": metallicity}

    param_list = [
        Parameter('mass_1', 10.0, 150.0, dist='kroupa'),
        Parameter('q', 0.0, 1.0, dist='uniform'),
        Parameter('porb', 10**(0.15), 10**(5.5),  dist='sana'),
        Parameter('ecc', 1e-9, 0.9999, dist='sana_ecc'),
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
        is_interesting=is_bh_lms,
        derive_params=derive_params,
        reject_systems="default",
        nproc=nproc,
        n_generations=1,
        seed=117,
        only_save_hit_tables=True
    )
    result = sampler.run()

    print(f"Finished generating {total_systems} systems for metallicity {metallicity:.2e}.")
    print(f"Time taken: {time() - start:.2f} seconds.")

    result.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Create a population of DCOs using adaptive sampling.")
    parser.add_argument("--metallicity", type=float, required=True, help="Metallicity of the population.")
    parser.add_argument("--inifile", type=str, required=True, help="Path to the ini file with simulation parameters.")
    parser.add_argument("--total_systems", type=int, required=True, help="Total number of systems to generate.")
    parser.add_argument("--batch_size", type=int, required=True, help="Number of systems to generate per batch.")
    parser.add_argument("--nproc", type=int, required=True, help="Number of processes to use for parallelization.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the generated population data.")
    args = parser.parse_args()

    create_bh_lms_population(
        metallicity=args.metallicity,
        inifile_path=args.inifile,
        total_systems=args.total_systems,
        batch_size=args.batch_size,
        nproc=args.nproc,
        output_path=args.output_path
    )

if __name__ == "__main__":
    main()
