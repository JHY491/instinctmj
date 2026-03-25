#!/usr/bin/env bash
#
# Instinct-Parkour-Simple-Adam-SP-v0 训练脚本
# 支持指定环境数量、后台运行，其余参数透传给 instinct-train。
#
# 用法:
#   ./scripts/train_parkour_adam_sp.sh                    # 前台，默认环境数
#   ./scripts/train_parkour_adam_sp.sh -n 4096            # 前台，4096 环境
#   ./scripts/train_parkour_adam_sp.sh -n 2048 -b         # 后台，2048 环境
#   ./scripts/train_parkour_adam_sp.sh --gpu 0            # 指定使用第 0 张 GPU
#   ./scripts/train_parkour_adam_sp.sh --gpu 1 -n 4096     # 指定使用第 1 张 GPU
#   ./scripts/train_parkour_adam_sp.sh --num-envs 8192 --background
#
# 断点续训（默认不 resume，需显式加 --resume；传给 instinct-train 为 agent dot override）:
#   ./scripts/train_parkour_adam_sp.sh --resume --load-run 2026-03-24_11-19-28 --checkpoint model_29500.pt
# 等价于:
#   instinct-train ... --agent.resume true \
#     --agent.load-run 2026-03-24_11-19-28 \
#     --agent.load-checkpoint model_29500.pt
# （脚本内使用 --agent.load_run / --agent.load_checkpoint，因 dataclass 字段为下划线，与上式等价。）
#   ./scripts/train_parkour_adam_sp.sh --resume --load-run /abs/path/to/run_dir --checkpoint model_29500.pt
#

set -e

TASK_ID="Instinct-Parkour-Simple-Adam-SP-v0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/instinct_rl"
NUM_ENVS=""
BACKGROUND=false
CUDA_VISIBLE_DEVICES_VAL="0"
RESUME=false
LOAD_RUN_ARG=""
CHECKPOINT_ARG=""
EXTRA_ARGS=()

# 解析 -n/--num-envs、-b/--background、--gpu、续训相关选项，其余参数原样传递
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
    --resume)
      RESUME=true
      shift
      ;;
    --load-run|--load_run)
      LOAD_RUN_ARG="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT_ARG="$2"
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
[[ "$RESUME" == true ]] && TRAIN_CMD+=(--agent.resume true)
[[ -n "$LOAD_RUN_ARG" ]] && TRAIN_CMD+=(--agent.load_run "$LOAD_RUN_ARG")
[[ -n "$CHECKPOINT_ARG" ]] && TRAIN_CMD+=(--agent.load_checkpoint "$CHECKPOINT_ARG")
TRAIN_CMD+=("${EXTRA_ARGS[@]}")

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG_FILE="${LOG_DIR}/train_parkour_adam_sp_${TIMESTAMP}.log"
# 过滤已知的 MuJoCo 刷屏警告。
# 这些告警在不同版本/路径下可能走 stdout 或 stderr，所以两边都过滤。
filter_output() {
  grep --line-buffered -vE \
    "^Warning: opt\.ccd_iterations, currently set to [0-9]+, needs to be increased\.$|^contact match overflow: please increase Option\.contact_sensor_maxmatch to [0-9]+$|^height field collision overflow.*$|^[[:space:]]*decrease the number of hfield rows/cols or modify size of colliding geom[[:space:]]*$"
}

if [[ "$BACKGROUND" == true ]]; then
  echo "[INFO] 后台启动训练: $TASK_ID"
  [[ -n "$NUM_ENVS" ]] && echo "[INFO] 环境数量: $NUM_ENVS"
  echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES_VAL}"
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
