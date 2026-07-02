#!/usr/bin/env bash
#
# Instinct-Parkour-Adam-Lite-VelEst-Play-v0：透传参数给 instinct-play，并抑制 / 过滤告警输出。
#
# 用法（仅 --load-run）:
#   ./scripts/play_parkour_adam_lite_velest.sh --load-run <run>
#   ./scripts/play_parkour_adam_lite_velest.sh --viewer viser --load-run <run>
#   ./scripts/play_parkour_adam_lite_velest.sh --gpu 0 --load-run <run>   # 仅使用第 0 张卡
#
# 说明:
#   - 设置 PYTHONWARNINGS=ignore，屏蔽 Python warnings 模块发出的告警。
#   - stdout/stderr 经 filter：与 train_parkour_adam_lite_velest.sh 相同的 MuJoCo 刷屏行过滤，
#     再过滤常见 Python / PyTorch 等 warning 行。
#

set -e

TASK_ID="Instinct-Parkour-Adam-Lite-VelEst-Play-v0"
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
    --export-onnx)
      EXTRA_ARGS+=("--export-onnx")
      shift
      # 与部分文档中的 tyro 写法一致：允许可选的 True/False，避免被当作多余位置参数。
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

# 与 scripts/train_parkour_adam_lite_velest.sh 一致：MuJoCo 刷屏（可走 stdout/stderr；torchrunx 行前缀不影响行内匹配）。
# 再叠加 play 侧对通用 warning 文案的过滤。
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
