#!/usr/bin/env bash
#
# Instinct-Parkour-Adam-Lite-Foothold-Mixed-Play-v0：与 VelEst-DeepCNN 同栈，mixed 地形 play / 导出 ONNX。
#
# 用法:
#   ./scripts/play_parkour_adam_lite_foothold_mixed.sh --load-run <run>
#   ./scripts/play_parkour_adam_lite_foothold_mixed.sh --viewer viser --load-run <run>
#   ./scripts/play_parkour_adam_lite_foothold_mixed.sh --gpu 0 --load-run <run>
#   ./scripts/play_parkour_adam_lite_foothold_mixed.sh --checkpoint logs/instinct_rl/adam_lite_foothold_mixed/<run>/model_10000.pt
#   ./scripts/play_parkour_adam_lite_foothold_mixed.sh --load-run <run> --export-onnx
#
# 日志根目录: logs/instinct_rl/adam_lite_foothold_mixed/
#

set -e

TASK_ID="Instinct-Parkour-Adam-Lite-Foothold-Mixed-Play-v0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CUDA_VISIBLE_DEVICES_VAL="${CUDA_VISIBLE_DEVICES:-}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu|--cuda-visible-devices)
      CUDA_VISIBLE_DEVICES_VAL="$2"
      shift 2
      ;;
    --checkpoint|--pt)
      EXTRA_ARGS+=("--checkpoint-file" "$2")
      shift 2
      ;;
    --export-onnx)
      EXTRA_ARGS+=("--export-onnx")
      shift
      if [[ $# -gt 0 && "$1" =~ ^(True|False|true|false)$ ]]; then
        EXTRA_ARGS+=("$1")
        shift
      fi
      ;;
    --onnx-output-dir)
      EXTRA_ARGS+=("--onnx-output-dir" "$2")
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

PLAY_CMD=()
if command -v instinct-play &>/dev/null; then
  PLAY_CMD=(instinct-play)
else
  PLAY_CMD=(python3 -m instinct_mj.scripts.instinct_rl.play)
fi
PLAY_CMD+=("$TASK_ID")
PLAY_CMD+=("${EXTRA_ARGS[@]}")

filter_output() {
  grep --line-buffered -vE \
    "opt\.ccd_iterations, currently set to [0-9]+, needs to be increased|contact match overflow: please increase Option\.contact_sensor_maxmatch to [0-9]+|height field collision overflow|decrease the number of hfield rows/cols or modify size of colliding geom" \
    | grep --line-buffered -viE \
      'warning|UserWarning|DeprecationWarning|FutureWarning|PendingDeprecationWarning|RuntimeWarning|VisibleDeprecationWarning|ResourceWarning|SyntaxWarning|BytesWarning|UnicodeWarning|ImportWarning|EncodingWarning|\[W[0-9]{4}\]|^W[0-9]{4}[[:space:]]|: WARN:|CUDA Warning|ExperimentalWarning'
}

cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore

if [[ -n "${CUDA_VISIBLE_DEVICES_VAL}" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VAL}" \
    "${PLAY_CMD[@]}" > >(filter_output) 2> >(filter_output >&2)
else
  "${PLAY_CMD[@]}" > >(filter_output) 2> >(filter_output >&2)
fi
