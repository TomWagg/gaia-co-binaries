#!/usr/bin/env bash
# Run src/postprocess.py over a queue of folders, at most MAX_JOBS at a time.
# As soon as one finishes, the next folder in the queue starts.
#
# Usage:  bash scripts/run_postprocess.sh
# The whole queue detaches with nohup, so it is safe to log out.
# Per-folder output goes to $LOG_DIR/<folder name>_<CO>.log; queue progress to $LOG_DIR/queue.log

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
FOLDERS=(
    # /mnt/ceph/users/twagg/gaia-bhs/sse
    # /mnt/ceph/users/twagg/gaia-bhs/metisse
    # /mnt/ceph/users/twagg/gaia-bhs/new_winds
    # /mnt/ceph/users/twagg/gaia-bhs/new_cbm
    # /mnt/ceph/users/twagg/gaia-bhs/new_winds_and_cbm
    # /mnt/ceph/users/twagg/gaia-bhs/new_winds_and_cbm_kicks
    /mnt/ceph/users/twagg/gaia-bhs/new_winds_and_cbm_raghavan10
    # /mnt/ceph/users/twagg/gaia-bhs/olejak_20
    # /mnt/ceph/users/twagg/gaia-bhs/olejak_40
    # /mnt/ceph/users/twagg/gaia-bhs/olejak_20_new_winds
    # /mnt/ceph/users/twagg/gaia-bhs/olejak_20_new_cbm
    # /mnt/ceph/users/twagg/gaia-bhs/olejak_20_new_winds_and_cbm
    # /mnt/ceph/users/twagg/gaia-bhs/olejak_20_new_winds_and_cbm_kicks
    /mnt/ceph/users/twagg/gaia-bhs/olejak_20_new_winds_and_cbm_raghavan10
)
CO="BH"          # BH or NS
MAX_JOBS=3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POSTPROCESS="$SCRIPT_DIR/../src/postprocess.py"
LOG_DIR="$SCRIPT_DIR/../logs/postprocess"
# ---------------------------------------------------------------------------

mkdir -p "$LOG_DIR"

# Relaunch this script under nohup in the background, then return immediately
if [[ -z "$_POSTPROCESS_QUEUE_DETACHED" ]]; then
    _POSTPROCESS_QUEUE_DETACHED=1 nohup bash "${BASH_SOURCE[0]}" "$@" > "$LOG_DIR/queue.log" 2>&1 &
    echo "Queue started in background (PID $!). Progress: $LOG_DIR/queue.log"
    exit 0
fi

if [[ ${#FOLDERS[@]} -eq 0 ]]; then
    echo "No folders in queue - add some to FOLDERS."
    exit 1
fi

echo "$(date '+%F %T') Starting queue of ${#FOLDERS[@]} folder(s), max $MAX_JOBS at once (CO=$CO)"

failed=0
for folder in "${FOLDERS[@]}"; do
    # Block until a slot is free
    while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
        wait -n || failed=$((failed + 1))
    done

    log="$LOG_DIR/$(basename "$folder")_${CO}.log"
    echo "$(date '+%F %T') START $folder  (log: $log)"
    (
        python -u "$POSTPROCESS" -f "$folder" -c "$CO" > "$log" 2>&1
        status=$?
        echo "$(date '+%F %T') DONE  $folder  (exit $status)"
        exit $status
    ) &
done

# Wait for the remaining jobs
while (( $(jobs -rp | wc -l) > 0 )); do
    wait -n || failed=$((failed + 1))
done

echo "$(date '+%F %T') All done. $failed job(s) failed."
