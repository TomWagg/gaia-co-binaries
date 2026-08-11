import os

TEMPLATE = r"""#!/bin/bash
## Job Name
#SBATCH --job-name=gaia_bhlms_VARIATION
#SBATCH --partition=cca,gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=2:00:00
#SBATCH -o /mnt/home/twagg/projects/gaia-bhs/slurm/stroopwafel_runs/logs/bhlms_VARIATION_%a_%A.out
#SBATCH -e /mnt/home/twagg/projects/gaia-bhs/slurm/stroopwafel_runs/logs/bhlms_VARIATION_%a_%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twagg@flatironinstitute.org
#SBATCH --export=all

source /mnt/home/twagg/.bashrc
conda activate cosmic

# create a log-spaced array between 1e-4 and 0.03 with 50 bins
METS=($(python -c "import numpy as np; print(' '.join(map(str, np.logspace(-4, np.log10(0.03), 50).round(5))))"))

# select a metallicity based on the SLURM_ARRAY_TASK_ID
MET=${METS[$SLURM_ARRAY_TASK_ID]}

echo "Starting BHLMS simulation with metallicity: $MET, for variation: VARIATION"

mkdir -p /mnt/ceph/users/twagg/gaia-bhs/VARIATION/stroopwafel_files

# run the distributed underworld simulation
python /mnt/home/twagg/projects/gaia-bhs/simulations/create_bh_lms_population.py \
    --metallicity $MET \
    --inifile /mnt/home/twagg/projects/gaia-bhs/settings/params_VARIATION.ini \
    --total_systems 2000000 \
    --batch_size 25000 \
    --nproc 64 \
    --output_path /mnt/ceph/users/twagg/gaia-bhs/VARIATION/stroopwafel_files/BH_LMS_Z_$MET.h5
"""

os.makedirs("jobs", exist_ok=True)
for variation in ["fiducial"]:
    with open(os.path.join("jobs", f"BH-LMS_{variation}.slurm"), "w") as f:
        f.write(TEMPLATE.replace("VARIATION", variation))
