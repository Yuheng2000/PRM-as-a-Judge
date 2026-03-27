#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash eval/run_eval.sh
# Optional env vars:
#   CUDA_VISIBLE_DEVICES=0,1
#   PRM_PATH=/path/to/your-prm-checkpoint
#   GRM_PATH=/path/to/your-prm-checkpoint  # backward-compatible alias
#   EVAL_MODE=backward
#   FRAME_INTERVAL=10
#   BATCH_SIZE=4
#   TP_SIZE=2

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PRM_PATH="${PRM_PATH:-${GRM_PATH:-}}"
EVAL_MODE="${EVAL_MODE:-backward}"
FRAME_INTERVAL="${FRAME_INTERVAL:-10}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TP_SIZE="${TP_SIZE:-1}"

if [[ -z "${PRM_PATH}" ]]; then
  echo "[ERROR] Please set PRM_PATH, e.g.:"
  echo "  PRM_PATH=/path/to/your-prm-checkpoint bash eval/run_eval.sh"
  exit 1
fi

export CUDA_VISIBLE_DEVICES

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] PRM_PATH=${PRM_PATH}"
echo "[INFO] benchmark=demo_cases eval_mode=${EVAL_MODE}"

python "${SCRIPT_DIR}/run_judge.py" \
  --benchmark demo_cases \
  --videos-root "${SCRIPT_DIR}/videos" \
  --tasks-root "${SCRIPT_DIR}/tasks" \
  --goals-root "${SCRIPT_DIR}/goals" \
  --output-root "${SCRIPT_DIR}/results" \
  --frame-interval "${FRAME_INTERVAL}" \
  --batch-size "${BATCH_SIZE}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --eval-mode "${EVAL_MODE}" \
  --prm-path "${PRM_PATH}" \
  --visualize

echo "[INFO] Evaluation finished."
