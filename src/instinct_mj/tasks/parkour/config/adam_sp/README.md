# Adam SP Parkour

基于 PND Adam SP（23 DOF）的 parkour 任务配置，在 `InstinctMJ` + `mjlab` 上运行。

## 任务 ID

| 任务 ID | 用途 | 说明 |
|--------|------|------|
| `Instinct-Parkour-Simple-Adam-SP-v0` | 训练 | 仅噪声地形，无视觉 / 无 AMP / 无 MoE |
| `Instinct-Parkour-Simple-Adam-SP-Play-v0` | 回放 | 同上，用于查看/评测 |

后续可扩展：Target（无 AMP）、Target-Amp 等任务 ID，与 G1 对应任务类似。

## 配置结构

- **`env_cfgs.py`**  
  基础 flat 环境：`adam_sp_flat_parkour_env_cfg(play, has_state_estimation)`，供 simple/target 等 parkour 配置在其上叠加地形、传感器、奖励等。

- **`adam_sp_parkour_simple_cfg.py`**  
  Simple parkour 环境与 final 配置：
  - `instinct_adam_sp_parkour_simple_env_cfg(play=False)`：噪声地形、足端接触/射线、速度命令、课程等。
  - `instinct_adam_sp_parkour_simple_final_cfg(play=False)`：在之上加 play 时的 viewer 等最终配置。

- **`agents/instinct_rl_ppo_simple_cfg.py`**  
  PPO 训练配置：`adam_parkour_simple_ppo_runner_cfg()`，ActorCritic + PPO，无 AMP/MoE。

- **`__init__.py`**  
  将上述 env_cfg 与 runner_cfg 注册为 `Instinct-Parkour-Simple-Adam-SP-v0` / `Instinct-Parkour-Simple-Adam-SP-Play-v0`。

## 训练

```bash
instinct-train Instinct-Parkour-Simple-Adam-SP-v0
```

可选常用参数：`--num-envs`、`--video` 等，与项目其他任务一致。

**脚本方式（可指定环境数量、后台运行）**：在仓库根目录执行

```bash
# 前台，默认环境数
./scripts/train_parkour_adam_sp.sh

# 指定环境数量（例如 4096）
./scripts/train_parkour_adam_sp.sh -n 4096
./scripts/train_parkour_adam_sp.sh --num-envs 2048

# 后台运行，日志写入 logs/instinct_rl/
./scripts/train_parkour_adam_sp.sh -n 4096 -b
./scripts/train_parkour_adam_sp.sh --num-envs 8192 --background
```

## 回放

加载已训练 run 进行回放（`--load-run` 建议用绝对路径或明确 run 名）：

```bash
instinct-play Instinct-Parkour-Simple-Adam-SP-Play-v0 --load-run <run_name> --viewer viser
```

随机策略可视化（不加载 checkpoint）：

```bash
instinct-play Instinct-Parkour-Simple-Adam-SP-Play-v0 --agent random
```

## 导出与 ONNX

导出 ONNX：

```bash
instinct-play Instinct-Parkour-Simple-Adam-SP-Play-v0 --load-run <run_name> --export-onnx
```

使用已导出的 ONNX 推理回放：

```bash
instinct-play Instinct-Parkour-Simple-Adam-SP-Play-v0 --load-run <run_name> --use-onnx
```

## 机器人说明

- **模型**：Adam SP 23 DOF（`instinct_mj.assets.adam_sp`），腿 12 + 腰 3 + 臂 8，无手腕。
- **足端**：`left_ankle_roll_link` / `right_ankle_roll_link` 用于接触与高度射线。
- **地形**：Simple 使用 noise-only 地形（见 `parkour_env_cfg.NOISE_ONLY_TERRAINS_CFG`），课程按跟踪速度提升难度。

更多环境与奖励细节见 `adam_sp_parkour_simple_cfg.py` 内注释与参数。
