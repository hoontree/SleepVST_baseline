#!/usr/bin/env bash
# Run v2 respiratory proxy extraction on all KVSS videos.
#
# Usage:
#   bash scripts/run_v2_extraction.sh              # all videos, 60 workers
#   bash scripts/run_v2_extraction.sh --workers 30  # custom worker count
#   bash scripts/run_v2_extraction.sh --records A2019-EM-01-0001,A2020-EM-01-0038
#   bash scripts/run_v2_extraction.sh --no-skip      # re-process already done epochs
#
# Outputs go to:  data/resp_proxy_video_epochs_v2/{record_id}/epoch_{k}/
#   epoch_{k}_movement.npy   — z-scored respiratory proxy (T,)
#   epoch_{k}_quality.npz    — SNR, breathing_rate, ok flag, max_point

set -euo pipefail
cd "$(dirname "$0")/.."

# ── defaults ──────────────────────────────────────────────────────────────────
WORKERS=60
RECORDS=""
SKIP_EXISTING="true"
LOG_SUFFIX=""

# ── parse args ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers)   WORKERS="$2";       shift 2 ;;
        --records)   RECORDS="$2";       shift 2 ;;
        --no-skip)   SKIP_EXISTING="false"; shift ;;
        --suffix)    LOG_SUFFIX="_$2";   shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── build hydra overrides ──────────────────────────────────────────────────────
OVERRIDES=(
    "multiprocessing.num_workers=${WORKERS}"
    "skip_existing=${SKIP_EXISTING}"
    "log.name=respiratory_extraction_v2${LOG_SUFFIX}"
)

if [[ -n "$RECORDS" ]]; then
    # Write a temp file_list and point config at it
    TMPLIST=$(mktemp /tmp/v2_record_list_XXXX.txt)
    echo "$RECORDS" | tr ',' '\n' > "$TMPLIST"
    OVERRIDES+=("video.file_list=${TMPLIST}")
    echo "Processing subset: ${RECORDS}"
    trap "rm -f ${TMPLIST}" EXIT
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " v2 Respiratory Extraction"
echo " workers   : ${WORKERS}"
echo " skip done : ${SKIP_EXISTING}"
echo " output    : data/resp_proxy_video_epochs_v2/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python -m src.cli_extract_respiratory \
    --config-name preprocess/respiratory_v2 \
    "${OVERRIDES[@]}"
