#!/usr/bin/env bash
#
# Instinct-Parkour-Adam-Lite-v0 训练脚本
# 支持指定环境数量、后台运行，其余参数透传给 instinct-train。
#
# 用法:
#   ./scripts/train_parkour_adam_lite.sh                    # 前台：未设置 CUDA 时用节点上全部 GPU（torchrunx）
#   ./scripts/train_parkour_adam_lite.sh --gpu 0          # 仅第 0 张卡（单进程）
#   ./scripts/train_parkour_adam_lite.sh -n 4096          # 前台，4096 环境
#   ./scripts/train_parkour_adam_lite.sh -n 2048 -b       # 后台，2048 环境
#   ./scripts/train_parkour_adam_lite.sh --gpu 1 -n 4096  # 指定第 1 张 GPU
#   ./scripts/train_parkour_adam_lite.sh --gpu 0,1        # 限定可见 0,1：不传 --gpu-ids，由 train.py 按可见卡数自动起多卡
#   已 export CUDA_VISIBLE_DEVICES=0,1 时直接运行脚本，行为与上一行相同
#   ./scripts/train_parkour_adam_lite.sh --gpu-ids "[0,1]" # 显式透传 instinct-train（与未设置 CUDA 时「物理卡下标」语义一致）
#   ./scripts/train_parkour_adam_lite.sh --num-envs 8192 --background
#
# 断点续训（默认不 resume，需显式加 --resume 或直接指定 .pt）:
#   ./scripts/train_parkour_adam_lite.sh --resume --load-run 2026-03-24_11-19-28 --checkpoint model_29500.pt
#   ./scripts/train_parkour_adam_lite.sh --checkpoint logs/instinct_rl/adam_lite_parkour_amp_vision/2026-03-24_11-19-28/model_29500.pt
#   ./scripts/train_parkour_adam_lite.sh --resume logs/instinct_rl/adam_lite_parkour_amp_vision/2026-03-24_11-19-28/model_29500.pt
#   ./scripts/train_parkour_adam_lite.sh --resume --load-run /abs/path/to/run_dir --checkpoint model_29500.pt
#

set -e

TASK_ID="Instinct-Parkour-Adam-Lite-v0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/instinct_rl"
NUM_ENVS=""
BACKGROUND=false
# 默认沿用当前 shell 的 CUDA_VISIBLE_DEVICES；未设置则为空，由下方逻辑配合 train.py 使用全卡或自动多卡。
CUDA_VISIBLE_DEVICES_VAL="${CUDA_VISIBLE_DEVICES:-}"
GPU_IDS_ARG=""
RESUME=false
LOAD_RUN_ARG=""
CHECKPOINT_ARG=""
EXTRA_ARGS=()

# 解析 -n/--num-envs、-b/--background、GPU、续训相关选项，其余参数原样传递
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

# 构建 instinct-train 命令
TRAIN_CMD=()
if command -v instinct-train &>/dev/null; then
  TRAIN_CMD=(instinct-train)
else
  # 未安装 entry point 时从仓库根目录以模块方式运行
  TRAIN_CMD=(python3 -m instinct_mj.scripts.instinct_rl.train)
fi

TRAIN_CMD+=("$TASK_ID")
[[ -n "$NUM_ENVS" ]] && TRAIN_CMD+=(--num-envs "$NUM_ENVS")
# 与 instinct_mj.scripts.instinct_rl.train.launch_training 对齐：
# - 显式 --gpu-ids / --gpus：原样传给 instinct-train。
# - 仅设置了 CUDA_VISIBLE_DEVICES（本脚本的 --gpu 或继承自环境）：不传 --gpu-ids，由 Python 根据可见卡数自动多卡（>=2 时 torchrunx）。
# - 未限定可见设备：传 --gpu-ids all，按机器上全部 CUDA 设备起多卡。
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
LOG_FILE="${LOG_DIR}/train_parkour_adam_lite_${TIMESTAMP}.log"
# 过滤已知的 MuJoCo 刷屏警告。
# 这些告警在不同版本/路径下可能走 stdout 或 stderr，所以两边都过滤。
filter_output() {
  # MuJoCo 刷屏；多卡时 torchrunx 会在行前加 INFO:torchrunx.<host>:，hfield 文案须行内匹配（勿用 ^ 锚行首）。
  grep --line-buffered -vE \
    "opt\.ccd_iterations, currently set to [0-9]+, needs to be increased|contact match overflow: please increase Option\.contact_sensor_maxmatch to [0-9]+|height field collision overflow|decrease the number of hfield rows/cols or modify size of colliding geom"
}

if [[ "$BACKGROUND" == true ]]; then
  echo "[INFO] 后台启动训练: $TASK_ID"
  [[ -n "$NUM_ENVS" ]] && echo "[INFO] 环境数量: $NUM_ENVS"
  if [[ -n "${CUDA_VISIBLE_DEVICES_VAL}" ]]; then
    echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES_VAL}"
  else
    echo "[INFO] CUDA_VISIBLE_DEVICES: (未设置，训练入口使用 --gpu-ids all，占满节点可见 CUDA 设备)"
  fi
  echo "[INFO] 日志: $LOG_FILE"
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
  echo "[INFO] PID: $PID (已写入 $PID_FILE)"
  echo "[INFO] 查看日志: tail -f $LOG_FILE"
else
  cd "$REPO_ROOT"
  export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
  if [[ -n "${CUDA_VISIBLE_DEVICES_VAL}" ]]; then
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VAL}" \
      "${TRAIN_CMD[@]}" > >(filter_output) 2> >(filter_output >&2)
  else
    "${TRAIN_CMD[@]}" > >(filter_output) 2> >(filter_output >&2)
  fi
fi
