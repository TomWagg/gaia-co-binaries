from cosmic.output import load_initC, save_initC
import h5py as h5
import os
import pandas as pd
import numpy as np
from time import time
import argparse


def postprocess(folder):
    input_folder = os.path.join(folder, "stroopwafel_files")

    start_time = time()
    print(f"Processing...")

    files = []
    for file in os.listdir(input_folder):
        if file.startswith(f"BH_LMS_Z_"):
            files.append(file)

    formation_rows_list = []
    initC_list = []

    files = np.sort(files)
    for file in files:
        file_start_time = time()
        print(f"\n\nStarting {file}")

        # load initC and bpp
        initC = load_initC(f"{input_folder}/{file}", key="initC")
        bpp = pd.read_hdf(f"{input_folder}/{file}", key="bpp")

        print(f"    [{time() - file_start_time:.2f}s] Loaded initC and bpp")
        lap = time()

        # find the BH-star formation rows
        bh_star_mask = (((bpp["kstar_1"] == 14) & (bpp["kstar_2"] < 10)) | ((bpp["kstar_1"] < 10) & (bpp["kstar_2"] == 14))) & (bpp["sep"] > 0)

        # if there are none then skip
        if bh_star_mask.sum() == 0:
            print(f"  No BH-star systems found in this file.")
        else:
            print(f"    [{time() - lap:.2f}s] Found {bh_star_mask.sum()} BH-star systems in this file.")
            lap = time()

            # otherwise record the formation rows and their metallicities
            bh_star_rows = bpp[bh_star_mask]
            formation_rows = bh_star_rows.drop_duplicates(subset="bin_num", keep="first")[["tphys", "mass_1", "mass_2", "kstar_1", "kstar_2", "sep", "ecc", "bin_num", "rad_1", "rad_2", "lum_1", "lum_2", "teff_1", "teff_2"]]
            formation_rows["metallicity"] = initC["metallicity"].iloc[0]

            # also work out the duration they'll spend in this phase
            final_bh_star_time = bh_star_rows.drop_duplicates(subset="bin_num", keep="last")["tphys"]
            formation_rows["t_end"] = final_bh_star_time.values
            formation_rows.rename(columns={"tphys": "t_start"}, inplace=True)

            # read in the AIS weights from STROOPWAFEL
            with h5.File(f"{input_folder}/{file}", "r") as f:
                formation_rows["weights"] = f["stroopwafel"]["weights"][...][f["stroopwafel"]["is_hit"][...]]

            print(f"    [{time() - lap:.2f}s] Found formation rows and weights")
            lap = time()
            initC_list.append(initC.loc[formation_rows.index])

            formation_rows_list.append(formation_rows)

        print(f"  [{time() - file_start_time:.2f}s] Finishing processing {file}")

    all_initC = pd.concat(initC_list, ignore_index=True)
    save_initC(f"{folder}/BH_LMS.h5", all_initC)

    all_formation_rows = pd.concat(formation_rows_list, ignore_index=True)
    all_formation_rows.to_hdf(f"{folder}/BH_LMS.h5", key="formation_rows")

    kick_infos = [pd.read_hdf(f"{input_folder}/{file}", key="kick_info")[
        ["tphys", "star", "delta_vsysx_1", "delta_vsysy_1", "delta_vsysz_1", "bin_num"]
    ] for file in os.listdir(input_folder) if file.startswith(f"BH_LMS_Z_")]
    all_kick_infos = pd.concat(kick_infos, ignore_index=True)
    all_kick_infos.index = all_kick_infos.index.values // 2
    all_kick_infos.to_hdf(f"{folder}/BH_LMS.h5", key="kick_info")

    print(f"Finished processing BH_LMS in {time() - start_time:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Postprocess DCOS")
    parser.add_argument("-f", "--folder", type=str, required=True, help="Folder containing the STROOPWAFEL files")
    args = parser.parse_args()

    postprocess(args.folder)


if __name__ == "__main__":
    main()

# python postprocess.py -f /mnt/ceph/users/twagg/gaia-bhs/fiducial