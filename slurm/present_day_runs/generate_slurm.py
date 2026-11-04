import os
import sys
sys.path.append("/mnt/home/twagg/projects/gaia-bhs/helpers/")
import const

SUBMITTER_TEMPLATE = r"""#!/bin/bash
## job name
#SBATCH --job-name=gaia_COlms_pd_VARIATION
#SBATCH --partition=cca
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=32
#SBATCH --constraint=genoa
#SBATCH --time=00:45:00
#SBATCH -o /mnt/home/twagg/projects/gaia-bhs/slurm/present_day_runs/logs/COlms_VARIATION_disBatch_%A.out
#SBATCH -e /mnt/home/twagg/projects/gaia-bhs/slurm/present_day_runs/logs/COlms_VARIATION_disBatch_%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twagg@flatironinstitute.org
#SBATCH --export=all

module load disBatch

mkdir -p /mnt/ceph/users/twagg/gaia-bhs/VARIATION/present_day_files

disBatch -c 32 -p /mnt/home/twagg/projects/gaia-bhs/slurm/present_day_runs/logs/ Tasks_CO
"""

# the script appends ../helpers to the path, so run it from src
TASK_TEMPLATE = r"""#DISBATCH PREFIX source /mnt/home/twagg/.bashrc ; conda activate cogsworth ; cd /mnt/home/twagg/projects/gaia-bhs/src ; python -u sample_present_day_population.py -N 1000000 -n 1 -s ${DISBATCH_REPEAT_INDEX} -v VARIATION -c 32 -t CO
#DISBATCH SUFFIX &> /mnt/home/twagg/projects/gaia-bhs/slurm/present_day_runs/logs/COlms_VARIATION_${DISBATCH_REPEAT_INDEX}.log
#DISBATCH REPEAT 10 start 0"""

for variation in const.VARIATIONS:
    job_dir = os.path.join("jobs", variation)
    os.makedirs(job_dir, exist_ok=True)

    for co_type in ["BH", "NS"]:

        with open(os.path.join(job_dir, f"job_{co_type}.slurm"), "w") as f:
            f.write(SUBMITTER_TEMPLATE.replace("VARIATION", variation).replace("CO", co_type))

        task = TASK_TEMPLATE.replace("VARIATION", variation).replace("CO", co_type)
        with open(os.path.join(job_dir, f"Tasks_{co_type}"), "w") as f:
            f.write(task)
