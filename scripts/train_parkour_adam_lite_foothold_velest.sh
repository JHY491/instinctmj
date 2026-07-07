#!/usr/bin/env bash
#
# Instinct-Parkour-Adam-Lite-Foothold-VelEst-v0 training script.
# SSR imagined foothold guidance + AMP + velocity-only MLP estimator.

set -e

TASK_ID="Instinct-Parkour-Adam-Lite-Foothold-VelEst-v0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/instinct_rl"
NUM_ENVS=""
BACKGROUND=false
CUDA_VISIBLE_DEVICES_VAL="${CUDA_VISIBLE_DEVICES:-}"
GPU_IDS_ARG=""
RESUME=false
LOAD_RUN_ARG=""
CHECKPOINT_ARG=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--num-envs)
      NUM_ENVS="$2"
      shift 2
      ;;
    -b|--background)
      BACKGROUND=true
      shift
      ;;
    --gpu|--cuda-visible-devices)
      CUDA_VISIBLE_DEVICES_VAL="$2"
      shift 2
      ;;
    --gpu-ids|--gpus)
      GPU_IDS_ARG="$2"
      shift 2
      ;;
    --resume)
      if [[ $# -gt 1 && "$2" != -* && "$2" == *.pt ]]; then
        CHECKPOINT_ARG="$2"
        shift 2
      else
        shift
      fi
      RESUME=true
      ;;
    --load-run|--load_run)
      LOAD_RUN_ARG="$2"
      shift 2
      ;;
    --checkpoint|--pt)
      CHECKPOINT_ARG="$2"
      RESUME=true
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

TRAIN_CMD=()
if command -v instinct-train &>/dev/null; then
  TRAIN_CMD=(instinct-train)
else
  TRAIN_CMD=(python3 -m instinct_mj.scripts.instinct_rl.train)
fi

TRAIN_CMD+=("$TASK_ID")
[[ -n "$NUM_ENVS" ]] && TRAIN_CMD+=(--num-envs "$NUM_ENVS")
if [[ -n "$GPU_IDS_ARG" ]]; then
  TRAIN_CMD+=(--gpu-ids "$GPU_IDS_ARG")
elif [[ -z "$CUDA_VISIBLE_DEVICES_VAL" ]]; then
  TRAIN_CMD+=(--gpu-ids all)
fi
[[ "$RESUME" == true ]] && TRAIN_CMD+=(--agent.resume true)
[[ -n "$LOAD_RUN_ARG" ]] && TRAIN_CMD+=(--agent.load_run "$LOAD_RUN_ARG")
[[ -n "$CHECKPOINT_ARG" ]] && TRAIN_CMD+=(--agent.load_checkpoint "$CHECKPOINT_ARG")
TRAIN_CMD+=("${EXTRA_ARGS[@]}")

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG_FILE="${LOG_DIR}/train_parkour_adam_lite_foothold_velest_${TIMESTAMP}.log"

filter_output() {
  grep --line-buffered -vE \
    "opt\.ccd_iterations, currently set to [0-9]+, needs to be increased|contact match overflow: please increase Option\.contact_sensor_maxmatch to [0-9]+|height field collision overflow|decrease the number of hfield rows/cols or modify size of colliding geom"
}

if [[ "$BACKGROUND" == true ]]; then
  echo "[INFO] Starting training in background: $TASK_ID"
  [[ -n "$NUM_ENVS" ]] && echo "[INFO] num envs: $NUM_ENVS"
  if [[ -n "${CUDA_VISIBLE_DEVICES_VAL}" ]]; then
    echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES_VAL}"
  else
    echo "[INFO] CUDA_VISIBLE_DEVICES: unset, using --gpu-ids all"
  fi
  echo "[INFO] log: $LOG_FILE"
  PID_FILE="${LOG_FILE%.log}.pid"
  (
    cd "$REPO_ROOT"
    export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
    if [[ -n "${CUDA_VISIBLE_DEVICES_VAL}" ]]; then
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VAL}" \
        nohup "${TRAIN_CMD[@]}" > >(filter_output >> "$LOG_FILE") 2> >(filter_output >> "$LOG_FILE") &
    else
      nohup "${TRAIN_CMD[@]}" > >(filter_output >> "$LOG_FILE") 2> >(filter_output >> "$LOG_FILE") &
    fi
    echo $!
  ) > "$PID_FILE"
  PID=$(cat "$PID_FILE")
  echo "[INFO] PID: $PID"
else
  cd "$REPO_ROOT"
  export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
  if [[ -n "${CUDA_VISIBLE_DEVICES_VAL}" ]]; then
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VAL}" "${TRAIN_CMD[@]}" 2>&1 | filter_output
  else
    "${TRAIN_CMD[@]}" 2>&1 | filter_output
  fi
fi
