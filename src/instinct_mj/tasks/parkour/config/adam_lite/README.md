# Adam Lite Parkour

基于 PND Adam Lite 的 parkour / foothold 任务配置，在 `InstinctMJ` + `mjlab` 上运行。  
当前主线是复现 [SSR](https://ssr-humanoid.github.io/) 的 imagined foothold guidance：去掉 CReF 分析式落脚 reward，落脚引导只由 SSR `r^f`（独立 foothold critic）提供，并保留 AMP 风格先验。

## 推荐任务（DeepCNN / Mixed）

| 任务 ID | 用途 | 栈 | 地形 | 日志目录 |
|--------|------|----|------|---------|
| `Instinct-Parkour-Adam-Lite-Foothold-VelEst-DeepCNN-v0` | 训练 | DeepCNN + VelEst + SSR privileged critic + AMP | 楼梯（stairs-only） | `logs/instinct_rl/adam_lite_foothold_velest_deepcnn/` |
| `Instinct-Parkour-Adam-Lite-Foothold-VelEst-DeepCNN-Play-v0` | 回放 | 同上 | 同上 | 同上 |
| `Instinct-Parkour-Adam-Lite-Foothold-Mixed-v0` | 训练 | 上表 DeepCNN/VelEst 栈 + **MoE×4 actor** | mixed 多地形 | `logs/instinct_rl/adam_lite_foothold_mixed/` |
| `Instinct-Parkour-Adam-Lite-Foothold-Mixed-Play-v0` | 回放 | 同上 | 同上 | 同上 |

### VelEst-DeepCNN

- **深度编码器**：SSR 对齐 CNN，`channels=[32,64,128]`，输入 `[10,36,36]`（10 帧堆叠 depth）→ 128-dim latent；仅挂在 actor。
- **Critic**：SSR asymmetric，使用 privileged heightmap / limb state，**无** depth encoder。
- **VelEst**：MLP 估计 `base_lin_vel`，注入 actor；监督目标来自 critic 侧真值。
- **算法**：`FootholdWasabiEstimatorPPO`；advantage 混合 `(task, foothold, style)=(1.0, 0.25, 1.0)`。
- **环境**：与 Foothold-VelEst 观测栈一致，仅 `run_name` / policy CNN 更深。

配置入口：

- Env：`adam_lite_foothold_parkour_cfg.instinct_adam_lite_foothold_deep_final_cfg`
- Runner：`agents/instinct_rl_amp_cfg.adam_lite_foothold_velest_deep_amp_runner_cfg`

### Foothold-Mixed

在 VelEst-DeepCNN 训练栈上换 mixed 地形，并启用 **MoE×4 actor**（critic 仍为 SSR privileged MLP）。

- **地形比例**（`FOOTHOLD_MIXED_TERRAINS_CFG`）：
  - `one_sided_stairs` 0.45
  - `continuous_box_gaps` 0.25
  - `perlin_rough` / `perlin_rough_stand` / `continuous_slope` 各 0.10
- **Virtual edges**：仅 stairs / box_gaps；slope 与 plane 不挂 edge cylinders。
- **指令**：按子地形设置 `velocity_ranges`；`perlin_rough_stand` 为零前进速度、随机 yaw。
- **终止**：`continuous_box_gaps` 上启用 `feet_on_pit_bottom`。
- **算法**：`terrain_level_gate=3`；坡度等平滑地形由 per-env mask 关闭 `r^f`（与 SSR 一致）。

配置入口：

- Env：`adam_lite_foothold_mixed_parkour_cfg.instinct_adam_lite_foothold_mixed_final_cfg`
- Runner：`agents/instinct_rl_amp_cfg.adam_lite_foothold_mixed_amp_runner_cfg`

## 训练 / 回放脚本

仓库根目录：

```bash
# DeepCNN（楼梯）
./scripts/train_parkour_adam_lite_foothold_velest_deep.sh
./scripts/train_parkour_adam_lite_foothold_velest_deep.sh --gpu 0 -n 4096
./scripts/train_parkour_adam_lite_foothold_velest_deep.sh -n 2048 -b

./scripts/play_parkour_adam_lite_foothold_velest_deep.sh \
  --load_run logs/instinct_rl/adam_lite_foothold_velest_deepcnn/<run>/model_18000.pt

# Mixed（多地形 + MoE）
./scripts/train_parkour_adam_lite_foothold_mixed.sh
./scripts/train_parkour_adam_lite_foothold_mixed.sh --gpu 0 -n 4096
./scripts/train_parkour_adam_lite_foothold_mixed.sh -n 2048 -b

./scripts/play_parkour_adam_lite_foothold_mixed.sh \
  --load-run logs/instinct_rl/adam_lite_foothold_mixed/<run>
```

等价 CLI：

```bash
instinct-train Instinct-Parkour-Adam-Lite-Foothold-VelEst-DeepCNN-v0
instinct-train Instinct-Parkour-Adam-Lite-Foothold-Mixed-v0

instinct-play Instinct-Parkour-Adam-Lite-Foothold-VelEst-DeepCNN-Play-v0 --load-run <run>
instinct-play Instinct-Parkour-Adam-Lite-Foothold-Mixed-Play-v0 --load-run <run> --viewer viser
```

常用参数：`--num-envs` / `-n`、`--gpu`、`--background` / `-b`、`--resume`、`--load-run`、`--checkpoint`。

## 其他 foothold 变体（简表）

| 任务 ID | 说明 |
|--------|------|
| `Instinct-Parkour-Adam-Lite-Foothold-v0` | 楼梯 + SSR foothold，浅层 depth CNN，无 VelEst |
| `Instinct-Parkour-Adam-Lite-Foothold-VelEst-v0` | 上表 + MLP 速度估计（浅层 CNN） |
| `Instinct-Parkour-Adam-Lite-Foothold-ContinuousSlope-v0` | 仅 continuous_slope；算法侧全局关闭 `r^f` |
| `Instinct-Parkour-Adam-Lite-Foothold-SquareGaps-v0` | 仅 continuous_box_gaps；`terrain_level_gate=3` |

另有非 foothold 的 parkour 任务（默认 / Three-Slopes / Hurdle / Bridge-L-Corner / Plum-Blossom / VelEst），见 `__init__.py` 注册列表。

## 配置结构

- **`env_cfgs.py`** — flat 底座环境。
- **`adam_lite_parkour_cfg.py`** — 默认 stairs parkour + AMP。
- **`adam_lite_foothold_*.py`** — foothold / VelEst / DeepCNN / Mixed / 单地形变体。
- **`agents/instinct_rl_amp_cfg.py`** — policy / algo / runner（含 DeepCNN、Mixed MoE）。
- **`__init__.py`** — `register_instinct_task` 注册。

## AMP 数据集

运动数据默认在仓库内：

```python
_PARKOUR_DATASET_DIR = os.path.expanduser("src/instinct_mj/motions")
```

楼梯 foothold 使用 `motion_reference.yaml`。若路径或过滤列表不同，改对应 `adam_lite_*_parkour_cfg.py` 中的 `AmassMotionCfg`。

## 机器人说明

- **模型**：Adam Lite（`instinct_mj.assets`），足端 `left_ankle_roll_link` / `right_ankle_roll_link`。
- 更多奖励与传感器细节见各 `*_cfg.py` 内注释。
