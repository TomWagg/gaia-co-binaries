#!/bin/bash
# Submit present-day population jobs for a list of variations.
# Edit CO_TYPE and VARIATIONS below as needed, then run: bash submit_jobs.sh

CO_TYPE="BH"    # BH or NS

VARIATIONS=(
    # alpha_10
    metisse
    new_cbm
    new_winds
    new_winds_and_cbm
    new_winds_and_cbm_kicks
    olejak_20
    olejak_20_new_cbm
    olejak_20_new_winds
    olejak_20_new_winds_and_cbm
    olejak_20_new_winds_and_cbm_kicks
    olejak_40
    sse
)

JOBS_DIR="/mnt/home/twagg/projects/gaia-bhs/slurm/present_day_runs/jobs"

for var in "${VARIATIONS[@]}"; do
    job_dir="${JOBS_DIR}/${var}"
    job_file="job_${CO_TYPE}.slurm"
    if [[ ! -f "${job_dir}/${job_file}" ]]; then
        echo "WARNING: ${job_dir}/${job_file} not found, skipping"
        continue
    fi
    # cd so the relative Tasks_* file in the slurm script resolves correctly
    echo -n "${var}: "
    (cd "${job_dir}" && sbatch "${job_file}")
    sleep 0.3   # be gentle on the scheduler
done
