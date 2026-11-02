import os
import sys
sys.path.append("/mnt/home/twagg/projects/gaia-bhs/helpers/")
import const

# each task is one metallicity, run with NPROC processes
NPROC = 64
# number of tasks (metallicities) to run at once
N_CONCURRENT = 50

# pass a different task file as the first argument to override, e.g. `sbatch --ntasks=1 job.slurm Tasks_test`
SUBMITTER_TEMPLATE = r"""#!/bin/bash
## job name
#SBATCH --job-name=gaia_bhlms_VARIATION
#SBATCH --partition=cca
#SBATCH --ntasks=NTASKS
#SBATCH --cpus-per-task=NPROC
#SBATCH --time=03:00:00
#SBATCH -o /mnt/home/twagg/projects/gaia-bhs/slurm/stroopwafel_runs/logs/bhlms_VARIATION_disBatch_%A.out
#SBATCH -e /mnt/home/twagg/projects/gaia-bhs/slurm/stroopwafel_runs/logs/bhlms_VARIATION_disBatch_%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twagg@flatironinstitute.org
#SBATCH --export=all

module load disBatch

mkdir -p /mnt/ceph/users/twagg/gaia-bhs/VARIATION/stroopwafel_files

disBatch -c NPROC -p /mnt/home/twagg/projects/gaia-bhs/slurm/stroopwafel_runs/logs/ ${1:-Tasks}
"""

# the script appends ../helpers to the path, so run it from src
TASK_TEMPLATE = r"""#DISBATCH PREFIX source /mnt/home/twagg/.bashrc ; conda activate cosmic ; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ; cd /mnt/home/twagg/projects/gaia-bhs/src ; python -u create_bh_lms_population.py --metallicity-ind ${DISBATCH_REPEAT_INDEX} --metallicity-grid Z_GRID --inifile /mnt/home/twagg/projects/gaia-bhs/settings/VARIATION.ini --total_systems 2000000 --batch_size 25000 --nproc NPROC --output_path /mnt/ceph/users/twagg/gaia-bhs/VARIATION/stroopwafel_files/BH_LMS_Z_IND_${DISBATCH_REPEAT_INDEX}.h5
#DISBATCH SUFFIX &> /mnt/home/twagg/projects/gaia-bhs/slurm/stroopwafel_runs/logs/bhlms_VARIATION_Z_IND_${DISBATCH_REPEAT_INDEX}.log
#DISBATCH REPEAT N_MET start START"""

for variation in const.VARIATIONS:
    grid = "sse" if variation == "sse" else "metisse"
    n_met = len(const.SSE_Z_GRID) if variation == "sse" else len(const.METISSE_Z_GRID)
    job_dir = os.path.join("jobs", variation)
    os.makedirs(job_dir, exist_ok=True)

    with open(os.path.join(job_dir, "job.slurm"), "w") as f:
        f.write(SUBMITTER_TEMPLATE.replace("VARIATION", variation).replace("NPROC", str(NPROC))
                .replace("NTASKS", str(min(N_CONCURRENT, n_met))))

    task = TASK_TEMPLATE.replace("VARIATION", variation).replace("Z_GRID", grid).replace("NPROC", str(NPROC))
    with open(os.path.join(job_dir, "Tasks"), "w") as f:
        f.write(task.replace("N_MET", str(n_met)).replace("START", "0") + "\n")

    # a single metallicity for testing
    with open(os.path.join(job_dir, "Tasks_test"), "w") as f:
        f.write(task.replace("N_MET", "1").replace("START", "0") + "\n")
