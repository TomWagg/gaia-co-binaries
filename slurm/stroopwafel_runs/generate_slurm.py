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
#SBATCH --job-name=gaia_COLMS_VARIATION
#SBATCH --partition=cca
#SBATCH --ntasks=NTASKS
#SBATCH --cpus-per-task=NPROC
#SBATCH --time=02:00:00
#SBATCH -o /mnt/home/twagg/projects/gaia-bhs/slurm/stroopwafel_runs/logs/COLMS_VARIATION_disBatch_%A.out
#SBATCH -e /mnt/home/twagg/projects/gaia-bhs/slurm/stroopwafel_runs/logs/COLMS_VARIATION_disBatch_%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twagg@flatironinstitute.org
#SBATCH --export=all

module load disBatch

mkdir -p /mnt/ceph/users/twagg/gaia-bhs/VARIATION/stroopwafel_files

disBatch -c NPROC -p /mnt/home/twagg/projects/gaia-bhs/slurm/stroopwafel_runs/logs/ ${1:-Tasks_CO}
"""

# the script appends ../helpers to the path, so run it from src
TASK_TEMPLATE = r"""#DISBATCH PREFIX source /mnt/home/twagg/.bashrc ; conda activate cosmic ; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ; cd /mnt/home/twagg/projects/gaia-bhs/src ; python -u create_co_lms_population.py --metallicity-ind ${DISBATCH_REPEAT_INDEX} --metallicity-grid Z_GRID --inifile /mnt/home/twagg/projects/gaia-bhs/settings/SETTINGS.ini --total_systems 2000000 --batch_size 25000 --nproc NPROC --output_path /mnt/ceph/users/twagg/gaia-bhs/VARIATION/stroopwafel_files/CO_LMS_Z_IND_${DISBATCH_REPEAT_INDEX}.h5 --co CO --ecc-model ECCMODEL --porb-model PORBMODEL
#DISBATCH SUFFIX &> /mnt/home/twagg/projects/gaia-bhs/slurm/stroopwafel_runs/logs/COLMS_VARIATION_Z_IND_${DISBATCH_REPEAT_INDEX}.log
#DISBATCH REPEAT N_MET start START"""

def generate_slurm_submission(variation, co_type, settings=None, ecc_model="sana_ecc", porb_model="sana"):
    if settings is None:
        settings = variation
    grid = "sse" if variation == "sse" else "metisse"
    n_met = len(const.SSE_Z_GRID) if variation == "sse" else len(const.METISSE_Z_GRID)
    job_dir = os.path.join("jobs", variation)
    os.makedirs(job_dir, exist_ok=True)

    with open(os.path.join(job_dir, f"job_{co_type}.slurm"), "w") as f:
        f.write(SUBMITTER_TEMPLATE.replace("VARIATION", variation).replace("NPROC", str(NPROC))
                .replace("NTASKS", str(min(N_CONCURRENT, n_met))).replace("CO", co_type))

    task = TASK_TEMPLATE.replace("VARIATION", variation).replace("Z_GRID", grid)
    task = task.replace("NPROC", str(NPROC)).replace("SETTINGS", settings).replace("CO", co_type)
    task = task.replace("ECCMODEL", ecc_model).replace("PORBMODEL", porb_model)
    with open(os.path.join(job_dir, f"Tasks_{co_type}"), "w") as f:
        f.write(task.replace("N_MET", str(n_met)).replace("START", "0") + "\n")

    # a single metallicity for testing
    with open(os.path.join(job_dir, f"Tasks_test_{co_type}"), "w") as f:
        f.write(task.replace("N_MET", "1").replace("START", "0") + "\n")

for variation in const.VARIATIONS:
    for co_type in ["BH", "NS"]:
        settings = const.VARIATIONS[variation]["settings"] if "settings" in const.VARIATIONS[variation] else None
        ecc_model = const.VARIATIONS[variation]["ecc_model"] if "ecc_model" in const.VARIATIONS[variation] else "sana_ecc"
        porb_model = const.VARIATIONS[variation]["porb_model"] if "porb_model" in const.VARIATIONS[variation] else "sana"
        generate_slurm_submission(variation, co_type, settings=settings, ecc_model=ecc_model, porb_model=porb_model)
