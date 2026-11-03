import os
import sys
sys.path.append("/mnt/home/twagg/projects/gaia-bhs/helpers/")
import const

SUBMITTER_TEMPLATE = r"""#!/bin/bash
## job name
#SBATCH --job-name=gaia_bhlms_pd_VARIATION
#SBATCH --partition=cca
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=32
#SBATCH --constraint=genoa
#SBATCH --time=00:45:00
#SBATCH -o /mnt/home/twagg/projects/gaia-bhs/slurm/present_day_runs/logs/bhlms_VARIATION_disBatch_%A.out
#SBATCH -e /mnt/home/twagg/projects/gaia-bhs/slurm/present_day_runs/logs/bhlms_VARIATION_disBatch_%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twagg@flatironinstitute.org
#SBATCH --export=all

module load disBatch

mkdir -p /mnt/ceph/users/twagg/gaia-bhs/VARIATION/present_day_files

disBatch -c 32 -p /mnt/home/twagg/projects/gaia-bhs/slurm/present_day_runs/logs/ Tasks
"""

# the script appends ../helpers to the path, so run it from src
TASK_TEMPLATE = r"""#DISBATCH PREFIX source /mnt/home/twagg/.bashrc ; conda activate cogsworth ; cd /mnt/home/twagg/projects/gaia-bhs/src ; python -u sample_present_day_population.py -N 1000000 -n 1 -s ${DISBATCH_REPEAT_INDEX} -v VARIATION -c 32
#DISBATCH SUFFIX &> /mnt/home/twagg/projects/gaia-bhs/slurm/present_day_runs/logs/bhlms_VARIATION_${DISBATCH_REPEAT_INDEX}.log
#DISBATCH REPEAT 10 start 0"""

for variation in const.VARIATIONS:
    job_dir = os.path.join("jobs", variation)
    os.makedirs(job_dir, exist_ok=True)

    with open(os.path.join(job_dir, "job.slurm"), "w") as f:
        f.write(SUBMITTER_TEMPLATE.replace("VARIATION", variation))

    task = TASK_TEMPLATE.replace("VARIATION", variation)
    with open(os.path.join(job_dir, "Tasks"), "w") as f:
        f.write(task)