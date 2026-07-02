# SSR Imagined Foothold Guidance 复现实现计划（Adam Lite 楼梯任务）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `Instinct-Parkour-Adam-Lite` 楼梯任务上复现 SSR 论文的 imagined foothold guidance（引导点）：一个可学习的落脚点想象网络预测每只脚未来触地点的高斯分布，并把"支撑缺失度"引导 reward 注入 PPO/AMP 训练，改善楼梯边沿落脚安全性。

**Architecture:** 因本仓库 env（mjlab reward manager）与 RL（instinct_rl 网络）分离，引导 reward `r^f` 依赖学习网络，无法写成纯 mjlab reward term，故照搬仓库既有的 AMP 判别器范式——想象网络 + 独立优化器 + 独立 storage 挂在算法 mixin 上，reward 通过 `compute_auxiliary_reward` 注入，NLL 监督目标（未来第一次触地位置）在有序 rollout 缓冲上前向回填。整套改动新建一个兄弟任务 `Instinct-Parkour-Adam-Lite-Foothold-v0`，不改动现有 baseline。

**Tech Stack:** Python, PyTorch, mjlab (MuJoCo 环境层), instinct_rl (PPO/WasabiPPO/Estimator 算法与网络), pytest。

---

## 背景与关键事实（实现者必读）

论文机制（SSR Sec 3.2 / Appendix A.3, Table 6/8/9）：

- 落脚点想象模型：MLP `[256,128]`，输入 privileged state + action，输出每脚 `(μ̂∈R², σ̂∈R)`（base 系）。
- NLL 监督（式1）：`L = Σ_i (‖p*_i-μ̂_i‖²/(2σ̂_i²) + 2·logσ̂_i)`，`p*` = 该脚未来第一次有效触地点（base 系）。terrain-level 门控：低于设定课程等级不启用。
- 支撑缺失度 ρ（式2）：一块 `22.5cm×10cm` 鞋底 patch、每 `2.5cm` 采样一次的高度图，`ρ(p)=1 - mean_k 1[h^f(p)-h_k(p) < ε_h]`，`ε_h=0.03`。想象点用 `h^f(p)=max_k h_k(p)`。
- 引导 reward（式3）：`r^f = exp(-(Σ_i ρ̃_i)² / σ_f)`，`σ_f=0.0625`；支撑脚 `ρ̃=ρ(当前脚)`，摆动脚 `ρ̃=E_{p~q}[ρ(p)]`（想象分布下期望）。权重 `w_f=0.25`。

仓库既有事实（已核对）：

- 训练环路 `instinct_rl/runners/on_policy_runner.py`：
  - `alg_class = getattr(algorithms, alg_cfg["class_name"])`，`alg = alg_class(actor_critic, device=..., **alg_cfg)` → **新算法类必须在 `instinct_rl/algorithms/__init__.py` 导出；算法 cfg 的所有字段（除 class_name）以 kwargs 透传到算法 `__init__`**。
  - `alg.init_storage(num_envs, num_steps_per_env, obs_format, num_actions, num_rewards)`。
  - rollout：`alg.process_env_step(rewards, dones, infos, obs, critic_obs)`；`infos["observations"]` 是 `obs_pack`，**含所有观测组**（`_pack_observations` 打包每个组，非 policy/critic 组保持原始未归一化）。
  - 学习：`losses, stats = alg.update(iter)`。
- `PPO.process_env_step`（`instinct_rl/algorithms/ppo.py:151`）调用 `compute_auxiliary_reward(infos["observations"])`，把返回的每个 `{key: (N,1)}` 以 `self.transition.rewards += getattr(self, key+"_coef",1.0) * v * self.auxiliary_reward_per_env_reward_coefs` 累加进 reward（AMP 的 `discriminator_reward` 正是这样注入）。
- `WasabiAlgoMixin`（`instinct_rl/algorithms/wasabi.py`）示范了"独立网络（discriminator）+ 独立优化器 + 独立 storage（AmpStorage）+ 覆写 `init_storage/process_env_step/compute_auxiliary_reward/update/state_dict/load_state_dict`"的完整范式。**本计划的 `FootholdAlgoMixin` 结构与之一一对应。**
- 当前算法组合：`EstimatorWasabiPPO`（`estimator.py`）= `EstimatorAlgoMixin` + `WasabiPPO`；`WasabiEstimatorPPO`（`algorithms/__init__.py:41`）= `WasabiAlgoMixin` + `EstimatorPPO`。
- 楼梯任务 = `Instinct-Parkour-Adam-Lite-v0`，env 工厂 `instinct_adam_lite_parkour_env_cfg`（`tasks/parkour/config/adam_lite/adam_lite_parkour_cfg.py`），使用 `STAIRS_ONLY_TERRAINS_CFG`。已有传感器 `left/right_foot_sole_scanner`（site `left_foot_scan`/`right_foot_scan`，Adam Lite XML 已定义这些 site）。
- 现成可复用逻辑：`feet_unsupported`（`tasks/parkour/mdp/rewards.py:545`）已实现"鞋底 scan → 支撑缺失比例"，ρ 计算移植它。
- 观测函数签名：`envs_mdp.height_scan(env, sensor_name: str, offset=0.0, miss_value=None) -> (N, num_rays)`（`mjlab/.../envs/mdp/observations.py`）。`RayCastSensor.data.hit_pos_w[...,2]` 是命中点世界高度，`data.distances` <0 表示未命中。

**Phase 划分**：Phase 1（Task 0–7）单 critic，`r^f` 直接加到唯一 reward，跑通并验证。Phase 2（Task 8，可选）拆多 critic 对齐 SSR。先做 Phase 1。

---

## Contracts（跨任务共享接口，务必一致）

**C1. `foothold` 观测组的 term 顺序与维度**（`concatenate_terms=False`，`obs_pack["foothold"]` 是按此顺序 flatten 拼接的 `(N, D)`）：

| # | term 名 | 维度 | 含义 |
|---|---|---|---|
| 1 | `foot_region_scan_l` | `(REGION_RAYS,)` | 左脚区域高度图（世界高度 z），行主序 |
| 2 | `foot_region_scan_r` | `(REGION_RAYS,)` | 右脚区域高度图 |
| 3 | `foot_contact` | `(2,)` | [left, right] 接触 0/1 |
| 4 | `foot_pos_b` | `(4,)` | [lx, ly, rx, ry] base 系 |
| 5 | `foot_vel_b` | `(4,)` | [lvx, lvy, rvx, rvy] base 系 |
| 6 | `base_lin_vel` | `(3,)` | base 系线速度 |
| 7 | `foot_pos_w` | `(4,)` | [lx, ly, rx, ry] 世界系（供 GT 回填） |
| 8 | `base_pose_w` | `(3,)` | [x, y, yaw] 世界系（供 GT 回填） |
| 9 | `terrain_level` | `(1,)` | 该 env 当前地形等级（门控） |
| 10 | `last_action` | `(num_actions,)` | 最近应用动作 a_t（想象输入） |

`REGION_RAYS` 与网格形状 `(REGION_H, REGION_W)` 在 Task 1 探测确定，写入算法 cfg 字段 `region_grid=(REGION_H, REGION_W)`。

**C2. 想象网络输入特征向量**（顺序固定，用于训练与 reward 两处，必须一致）：
`feat = cat([foot_region_scan_l, foot_region_scan_r, foot_contact, foot_pos_b, foot_vel_b, base_lin_vel, last_action], dim=-1)`
维度 `IMAG_IN = 2*REGION_RAYS + 2 + 4 + 4 + 3 + num_actions`。

**C3. 想象网络输出**：`mu (N,2,2)`（每脚 base 系 xy），`log_sigma (N,2)`（clamp 到 `[-4,1]`）。

**C4. `foothold` storage 每步存**：`feat (IMAG_IN,)`、`foot_contact (2,)`、`foot_pos_w (4,)`、`base_pose_w (3,)`、`terrain_level (1,)`、`dones (1,)`。

**C5. 常量（SSR）**：`SOLE_SIZE=(0.225,0.10)`、`SOLE_RES=0.025`（→ patch 网格 `SOLE_H=10, SOLE_W=5`，50 点）、`EPSILON_H=0.03`、`SIGMA_F=0.0625`、想象 MLP `[256,128]`/ELU、想象优化器 Adam `lr=5e-4`、`w_f=0.25`。

---

## Task 0: 克隆楼梯任务为独立可训练任务

先得到一个与楼梯 baseline 完全等价、能训练的新任务，后续改动都叠在它上面。

**Files:**
- Create: `src/instinct_mj/tasks/parkour/config/adam_lite/adam_lite_foothold_parkour_cfg.py`
- Modify: `src/instinct_mj/tasks/parkour/config/adam_lite/agents/instinct_rl_amp_cfg.py`（追加 runner cfg）
- Modify: `src/instinct_mj/tasks/parkour/config/adam_lite/__init__.py`（注册任务）

- [ ] **Step 1: 新建 env cfg 文件，复用楼梯工厂**

`adam_lite_foothold_parkour_cfg.py`：

```python
"""Adam Lite foothold parkour: 楼梯任务 + SSR imagined foothold guidance 的环境侧改动。"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.viewer.viewer_config import ViewerConfig

from instinct_mj.tasks.parkour.config.adam_lite.adam_lite_parkour_cfg import (
    instinct_adam_lite_parkour_env_cfg,
)


def instinct_adam_lite_foothold_parkour_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    """楼梯 env + foothold 传感器/观测组（后续 Task 1 在此叠加）。"""
    cfg = instinct_adam_lite_parkour_env_cfg(play=play)
    setattr(cfg, "run_name", "AdamLiteFoothold")
    return cfg


def instinct_adam_lite_foothold_parkour_final_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = instinct_adam_lite_foothold_parkour_env_cfg(play=play)
    if play:
        cfg.viewer = ViewerConfig(
            lookat=(0.0, 0.75, 0.0),
            distance=4.123105625617661,
            elevation=-14.036243467926479,
            azimuth=180.0,
            origin_type=ViewerConfig.OriginType.WORLD,
            entity_name=None,
        )
    return cfg
```

- [ ] **Step 2: 追加 runner cfg**（`agents/instinct_rl_amp_cfg.py` 末尾，先沿用现有 VelEst 策略与 AMP 算法，暂不接 foothold 算法）

```python
@dataclass(kw_only=True)
class AdamLiteFootholdParkourRunnerCfg(AdamSPParkourVisionAmpRunnerCfg):
    experiment_name: str = "adam_lite_foothold_parkour"
    policy: InstinctRlActorCriticCfg = field(default_factory=lambda: VelEstPolicyCfg())


def adam_lite_foothold_parkour_amp_runner_cfg() -> InstinctRlOnPolicyRunnerCfg:
    """Foothold 任务 runner（Task 6 会把 algorithm 换成 FootholdAmpAlgoCfg）。"""
    return AdamLiteFootholdParkourRunnerCfg()
```

- [ ] **Step 3: 注册任务**（`__init__.py` 末尾）

```python
def _foothold_parkour_cfg(play: bool):
  from .adam_lite_foothold_parkour_cfg import instinct_adam_lite_foothold_parkour_final_cfg
  return instinct_adam_lite_foothold_parkour_final_cfg(play=play)


def _foothold_runner_cfg():
  from .agents.instinct_rl_amp_cfg import adam_lite_foothold_parkour_amp_runner_cfg
  return adam_lite_foothold_parkour_amp_runner_cfg()


register_instinct_task(
  task_id="Instinct-Parkour-Adam-Lite-Foothold-v0",
  env_cfg_factory=lambda: _foothold_parkour_cfg(play=False),
  play_env_cfg_factory=lambda: _foothold_parkour_cfg(play=True),
  instinct_rl_cfg_factory=_foothold_runner_cfg,
)

register_instinct_task(
  task_id="Instinct-Parkour-Adam-Lite-Foothold-Play-v0",
  env_cfg_factory=lambda: _foothold_parkour_cfg(play=True),
  play_env_cfg_factory=lambda: _foothold_parkour_cfg(play=True),
  instinct_rl_cfg_factory=_foothold_runner_cfg,
)
```

- [ ] **Step 4: 验证任务注册**

Run: `python -m instinct_mj.scripts.list_envs | grep Foothold`
Expected: 输出 `Instinct-Parkour-Adam-Lite-Foothold-v0` 与 `-Play-v0`。

- [ ] **Step 5: Commit**

```bash
git add src/instinct_mj/tasks/parkour/config/adam_lite/
git commit -m "feat(parkour): clone adam lite stairs task as foothold task scaffold"
```

---

## Task 1: 足部区域高度图传感器 + foothold 观测组 + 观测函数

**Files:**
- Create: `src/instinct_mj/tasks/parkour/mdp/foothold_obs.py`（新观测函数）
- Modify: `src/instinct_mj/tasks/parkour/mdp/__init__.py`（re-export）
- Modify: `src/instinct_mj/tasks/parkour/config/adam_lite/adam_lite_foothold_parkour_cfg.py`

- [ ] **Step 1: 新建观测函数文件** `foothold_obs.py`

```python
"""foothold 观测组用到的 privileged 观测函数（base 系脚状态、世界系脚/base 位姿、地形等级）。"""

from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import SceneEntityCfg
from mjlab.utils.math import quat_apply_inverse


def _foot_ids(asset_cfg: SceneEntityCfg) -> list[int]:
    return list(asset_cfg.body_ids)  # [left_ankle_roll, right_ankle_roll]


def foot_pos_b(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """两脚位置在 base 系 xy，(N,4)=[lx,ly,rx,ry]。"""
    asset: Entity = env.scene[asset_cfg.name]
    ids = _foot_ids(asset_cfg)
    root_pos = asset.data.root_link_pos_w                    # (N,3)
    root_quat = asset.data.root_link_quat_w                  # (N,4)
    out = []
    for i in ids:
        rel_w = asset.data.body_link_pos_w[:, i, :] - root_pos
        rel_b = quat_apply_inverse(root_quat, rel_w)
        out.append(rel_b[:, :2])
    return torch.cat(out, dim=-1)


def foot_vel_b(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """两脚线速度在 base 系 xy，(N,4)。"""
    asset: Entity = env.scene[asset_cfg.name]
    ids = _foot_ids(asset_cfg)
    root_quat = asset.data.root_link_quat_w
    out = []
    for i in ids:
        vel_b = quat_apply_inverse(root_quat, asset.data.body_link_lin_vel_w[:, i, :])
        out.append(vel_b[:, :2])
    return torch.cat(out, dim=-1)


def foot_pos_w_xy(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """两脚位置世界系 xy，(N,4)=[lx,ly,rx,ry]（供 GT 回填）。"""
    asset: Entity = env.scene[asset_cfg.name]
    ids = _foot_ids(asset_cfg)
    out = [asset.data.body_link_pos_w[:, i, :2] for i in ids]
    return torch.cat(out, dim=-1)


def base_xy_yaw_w(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """base 世界系 [x, y, yaw]，(N,3)（供 GT 回填的世界→base 变换）。"""
    asset: Entity = env.scene[asset_cfg.name]
    xy = asset.data.root_link_pos_w[:, :2]
    yaw = asset.data.heading_w.unsqueeze(-1)
    return torch.cat([xy, yaw], dim=-1)


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str, contact_force_threshold: float = 1.0) -> torch.Tensor:
    """两脚接触 0/1，(N,2)=[left,right]。与 rewards.py 中接触判定一致。"""
    from mjlab.sensor import ContactSensor
    sensor: ContactSensor = env.scene[sensor_name]
    force = torch.max(torch.linalg.vector_norm(sensor.data.force_history, dim=-1), dim=2)[0]
    return (force > contact_force_threshold).float()


def terrain_level(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """该 env 当前地形等级，(N,1) float。"""
    level = env.scene.terrain.terrain_levels                  # (N,) long
    return level.unsqueeze(-1).float()
```

> 注：`quat_apply_inverse`、`heading_w`、`terrain_levels`、`body_link_lin_vel_w` 若命名与本仓库不一致，以 `rewards.py`（`feet_close_xy_gauss` 用 `heading_w`；`feet_at_plane` 用 `body_link_pos_w`）和 `terrain_importer` 的实际属性为准，探测后修正。Step 3 的探测会暴露任何错名。

- [ ] **Step 2: re-export** —— 在 `tasks/parkour/mdp/__init__.py` 追加
```python
from .foothold_obs import (
    base_xy_yaw_w,
    foot_contact,
    foot_pos_b,
    foot_pos_w_xy,
    foot_vel_b,
    terrain_level,
)
```

- [ ] **Step 3: 在 foothold env cfg 中加区域扫描器 + foothold 观测组**

修改 `instinct_adam_lite_foothold_parkour_env_cfg`，在拿到 `cfg` 后追加（import 见代码顶部注释）：

```python
    # --- 需要的 import（文件顶部）---
    # from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg
    # from mjlab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
    # import mjlab.envs.mdp as envs_mdp
    # import instinct_mj.tasks.parkour.mdp as parkour_mdp

    ankle = SceneEntityCfg("robot", body_names=("left_ankle_roll_link", "right_ankle_roll_link"))

    region_scanners = (
        RayCastSensorCfg(
            name="left_foot_region_scanner",
            frame=ObjRef(type="site", name="left_foot_scan", entity="robot"),
            pattern=GridPatternCfg(resolution=0.025, size=(0.5, 0.3)),
            ray_alignment="yaw", max_distance=10.0, debug_vis=False,
        ),
        RayCastSensorCfg(
            name="right_foot_region_scanner",
            frame=ObjRef(type="site", name="right_foot_scan", entity="robot"),
            pattern=GridPatternCfg(resolution=0.025, size=(0.5, 0.3)),
            ray_alignment="yaw", max_distance=10.0, debug_vis=False,
        ),
    )
    cfg.scene.sensors = tuple(cfg.scene.sensors) + region_scanners

    foothold_terms = {
        "foot_region_scan_l": ObservationTermCfg(
            func=envs_mdp.height_scan, params={"sensor_name": "left_foot_region_scanner"}),
        "foot_region_scan_r": ObservationTermCfg(
            func=envs_mdp.height_scan, params={"sensor_name": "right_foot_region_scanner"}),
        "foot_contact": ObservationTermCfg(
            func=parkour_mdp.foot_contact, params={"sensor_name": "contact_forces"}),
        "foot_pos_b": ObservationTermCfg(func=parkour_mdp.foot_pos_b, params={"asset_cfg": ankle}),
        "foot_vel_b": ObservationTermCfg(func=parkour_mdp.foot_vel_b, params={"asset_cfg": ankle}),
        "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
        "foot_pos_w": ObservationTermCfg(func=parkour_mdp.foot_pos_w_xy, params={"asset_cfg": ankle}),
        "base_pose_w": ObservationTermCfg(func=parkour_mdp.base_xy_yaw_w),
        "terrain_level": ObservationTermCfg(func=parkour_mdp.terrain_level),
        "last_action": ObservationTermCfg(func=envs_mdp.last_action),
    }
    cfg.observations["foothold"] = ObservationGroupCfg(
        terms=foothold_terms, concatenate_terms=False, enable_corruption=False,
    )
```

同时在 play 分支里给两个 region scanner 打开 `debug_vis=True`（复用现有 play 分支的 `for sensor_cfg in cfg.scene.sensors` 循环，把名字加进去）。

- [ ] **Step 4: 探测观测维度与网格形状**

Run:
```bash
python - <<'PY'
import gymnasium, torch, instinct_mj  # noqa
from instinct_mj.tasks.registry import make_env_cfg  # 若无此 helper，用 list_envs 里同款构造
# 用 play cfg，小 env 数
from instinct_mj.tasks.parkour.config.adam_lite.adam_lite_foothold_parkour_cfg import (
    instinct_adam_lite_foothold_parkour_final_cfg as F)
# 依据仓库实际的 env 构造入口实例化 env（参考 scripts/instinct_rl/train 的构造）
PY
```
若无现成脚本入口，改用最小训练命令跑 1 iter 并在 `on_policy_runner` 打印：
```bash
python -m instinct_mj.scripts.instinct_rl.train Instinct-Parkour-Adam-Lite-Foothold-v0 \
  --run-name probe 2>&1 | grep -iE "foothold|region" | head
```
目标：确认 `obs_format["foothold"]` 各 term 维度符合 C1；记录 `foot_region_scan_l` 的 ray 数 `REGION_RAYS` 与网格 `(REGION_H, REGION_W)`（由 `GridPatternCfg(resolution=0.025, size=(0.5,0.3))` 决定；把该值写进 Task 6 的 cfg 字段）。

Expected: 训练能启动（说明 foothold 组能被 `_pack_observations` 打包、`get_obs_format` 暴露），且能打印/读到 foothold 各维度。

- [ ] **Step 5: Commit**

```bash
git add src/instinct_mj/tasks/parkour/mdp/ src/instinct_mj/tasks/parkour/config/adam_lite/adam_lite_foothold_parkour_cfg.py
git commit -m "feat(parkour): add foot-region scanners and foothold privileged obs group"
```

---

## Task 2: 落脚点想象网络模块

**Files:**
- Create: `src/instinct-rl/instinct_rl/modules/foothold_imagination.py`
- Modify: `src/instinct-rl/instinct_rl/modules/__init__.py`
- Test: `src/instinct-rl/tests/test_foothold_imagination.py`

- [ ] **Step 1: 写失败测试**

```python
import torch
from instinct_rl.modules.foothold_imagination import FootholdImaginationModel


def test_imagination_shapes_and_sigma_clamp():
    model = FootholdImaginationModel(input_size=32, hidden_sizes=(64, 32))
    x = torch.randn(8, 32)
    mu, log_sigma = model(x)
    assert mu.shape == (8, 2, 2)
    assert log_sigma.shape == (8, 2)
    assert torch.all(log_sigma <= 1.0 + 1e-6) and torch.all(log_sigma >= -4.0 - 1e-6)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest src/instinct-rl/tests/test_foothold_imagination.py -v`
Expected: FAIL（`No module named ... foothold_imagination`）。

- [ ] **Step 3: 实现模块**

```python
"""落脚点想象网络：privileged state + action -> 每脚未来接触点高斯 (mu_xy, log_sigma)。

对应 SSR Table 8 的 Foothold Imagination Model（MLP [256,128]）。它不属于 actor_critic，
由 FootholdAlgoMixin 持有并用独立优化器训练（与 AMP discriminator 同构）。
"""

import torch
import torch.nn as nn

from instinct_rl.modules.mlp import MlpModel


class FootholdImaginationModel(nn.Module):
    def __init__(self, input_size: int, hidden_sizes=(256, 128), nonlinearity: str = "ELU",
                 log_sigma_min: float = -4.0, log_sigma_max: float = 1.0):
        super().__init__()
        self.net = MlpModel(
            input_size=input_size,
            output_size=2 * 3,  # 每脚 (mu_x, mu_y, log_sigma)
            hidden_sizes=list(hidden_sizes),
            nonlinearity=nonlinearity,
        )
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max

    def forward(self, x: torch.Tensor):
        out = self.net(x).view(-1, 2, 3)
        mu = out[..., :2]
        log_sigma = out[..., 2].clamp(self.log_sigma_min, self.log_sigma_max)
        return mu, log_sigma
```

> `MlpModel` 的构造参数以 `instinct_rl/modules/mlp.py` 实际签名为准（`input_size/output_size/hidden_sizes/nonlinearity` 是仓库通用命名，见 `state_estimator.py` 用法）。

- [ ] **Step 4: 导出** —— `modules/__init__.py` 追加 `from .foothold_imagination import FootholdImaginationModel`，并加入 `__all__`。

- [ ] **Step 5: 运行测试确认通过**

Run: `pytest src/instinct-rl/tests/test_foothold_imagination.py -v`
Expected: PASS。

- [ ] **Step 6: Commit**

```bash
git add src/instinct-rl/instinct_rl/modules/foothold_imagination.py src/instinct-rl/instinct_rl/modules/__init__.py src/instinct-rl/tests/test_foothold_imagination.py
git commit -m "feat(rl): add FootholdImaginationModel (per-foot Gaussian foothold head)"
```

---

## Task 3: 支撑缺失度 ρ 与引导 reward（纯函数，可测）

把 SSR 式2/式3 落成不依赖网络之外状态的纯函数，供 Task 5 的 mixin 调用，并单测。

**Files:**
- Create: `src/instinct-rl/instinct_rl/algorithms/foothold_math.py`
- Test: `src/instinct-rl/tests/test_foothold_math.py`

- [ ] **Step 1: 写失败测试**

```python
import torch
from instinct_rl.algorithms.foothold_math import (
    support_deficiency_at_center, expected_support_deficiency, guidance_reward,
)

REGION_H, REGION_W, REGION_RES = 21, 13, 0.025
SOLE_H, SOLE_W = 10, 5
EPS_H = 0.03


def _flat_region(value, n=4):
    # n 个 env，全平地高度=0 的区域图
    return torch.full((n, REGION_H * REGION_W), value)


def test_flat_ground_zero_deficiency():
    scan = _flat_region(0.0)                     # 地面 z=0
    sole_z = torch.zeros(4)                       # 鞋底 z=0
    rho = support_deficiency_at_center(
        scan, sole_z, region_grid=(REGION_H, REGION_W), region_res=REGION_RES,
        sole_grid=(SOLE_H, SOLE_W), epsilon_h=EPS_H)
    assert torch.allclose(rho, torch.zeros(4), atol=1e-6)


def test_deep_gap_full_deficiency():
    scan = _flat_region(-1.0)                    # 地面在鞋底下方 1m
    sole_z = torch.zeros(4)
    rho = support_deficiency_at_center(
        scan, sole_z, region_grid=(REGION_H, REGION_W), region_res=REGION_RES,
        sole_grid=(SOLE_H, SOLE_W), epsilon_h=EPS_H)
    assert torch.allclose(rho, torch.ones(4), atol=1e-6)


def test_guidance_reward_monotonic():
    # rho 越大 reward 越小；rho=0 -> 1
    r0 = guidance_reward(torch.zeros(4, 2), sigma_f=0.0625)
    r1 = guidance_reward(torch.full((4, 2), 0.5), sigma_f=0.0625)
    assert torch.allclose(r0, torch.ones(4), atol=1e-6)
    assert torch.all(r1 < r0)
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest src/instinct-rl/tests/test_foothold_math.py -v`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 实现**

```python
"""SSR 式2（支撑缺失度 rho）/ 式3（引导 reward）的纯张量实现。

区域高度图约定：height_scan 输出的每个 ray 是命中点世界高度 z，按行主序 reshape 成
(H, W) 网格；网格中心对齐足部扫描 site（即当前脚 xy）。sole patch 是网格中心处一块
SOLE_H x SOLE_W 的子窗口；对想象点，用中心相对偏移取子窗口。
"""

import torch


def _reshape_region(scan: torch.Tensor, region_grid) -> torch.Tensor:
    H, W = region_grid
    return scan.view(scan.shape[0], H, W)


def _sole_patch(region_hw: torch.Tensor, center_ij: torch.Tensor, sole_grid) -> torch.Tensor:
    """从 (N,H,W) 网格按每 env 的中心格 (i,j) 取 (N, SOLE_H*SOLE_W)。越界夹紧。"""
    N, H, W = region_hw.shape
    sh, sw = sole_grid
    di = torch.arange(sh, device=region_hw.device) - sh // 2
    dj = torch.arange(sw, device=region_hw.device) - sw // 2
    ii = (center_ij[:, 0:1] + di.view(1, -1)).clamp(0, H - 1)          # (N, sh)
    jj = (center_ij[:, 1:2] + dj.view(1, -1)).clamp(0, W - 1)          # (N, sw)
    ii = ii[:, :, None].expand(N, sh, sw).reshape(N, -1)
    jj = jj[:, None, :].expand(N, sh, sw).reshape(N, -1)
    batch = torch.arange(N, device=region_hw.device)[:, None].expand(N, sh * sw)
    return region_hw[batch, ii, jj]                                    # (N, sh*sw)


def _deficiency(patch: torch.Tensor, sole_z: torch.Tensor, epsilon_h: float) -> torch.Tensor:
    # rho = 1 - mean_k 1[sole_z - h_k < epsilon_h]
    supported = (sole_z.unsqueeze(-1) - patch < epsilon_h).float()
    return 1.0 - supported.mean(dim=-1)


def support_deficiency_at_center(scan, sole_z, region_grid, region_res, sole_grid, epsilon_h):
    """支撑脚：patch 取网格正中心。scan:(N,H*W), sole_z:(N,)."""
    region_hw = _reshape_region(scan, region_grid)
    N, H, W = region_hw.shape
    center = torch.tensor([[H // 2, W // 2]], device=scan.device).expand(N, 2)
    patch = _sole_patch(region_hw, center, sole_grid)
    return _deficiency(patch, sole_z, epsilon_h)


def support_deficiency_at_point(scan, offset_xy, region_grid, region_res, sole_grid, epsilon_h):
    """想象点：patch 中心 = 网格中心 + offset_xy(米, scanner 系)。
    scan:(N,H*W), offset_xy:(N,2). sole_z 用 patch 内最大高度（SSR: h^f=max_k h_k）。"""
    region_hw = _reshape_region(scan, region_grid)
    N, H, W = region_hw.shape
    # 网格行=x(前后)、列=y(左右)：以行主序为 (i沿x, j沿y)。offset->格数
    di = torch.round(offset_xy[:, 0] / region_res).long()
    dj = torch.round(offset_xy[:, 1] / region_res).long()
    center = torch.stack([H // 2 + di, W // 2 + dj], dim=-1).clamp(
        torch.tensor([0, 0], device=scan.device),
        torch.tensor([H - 1, W - 1], device=scan.device))
    patch = _sole_patch(region_hw, center, sole_grid)
    sole_z = patch.max(dim=-1)[0]
    return _deficiency(patch, sole_z, epsilon_h)


def expected_support_deficiency(scan, mu_xy, log_sigma, region_grid, region_res,
                                sole_grid, epsilon_h, num_samples: int = 8):
    """摆动脚：E_{p~N(mu,sigma^2 I)}[rho(p)]，离散高斯采样近似（SSR 式3）。
    mu_xy:(N,2) scanner 系相对偏移；log_sigma:(N,)."""
    sigma = torch.exp(log_sigma).unsqueeze(-1)                         # (N,1)
    eps = torch.randn(mu_xy.shape[0], num_samples, 2, device=scan.device)
    samples = mu_xy.unsqueeze(1) + sigma.unsqueeze(-1) * eps           # (N,S,2)
    rho = []
    for s in range(num_samples):
        rho.append(support_deficiency_at_point(
            scan, samples[:, s, :], region_grid, region_res, sole_grid, epsilon_h))
    return torch.stack(rho, dim=-1).mean(dim=-1)                       # (N,)


def guidance_reward(rho_tilde: torch.Tensor, sigma_f: float) -> torch.Tensor:
    """SSR 式3: r^f = exp(-(sum_i rho_tilde_i)^2 / sigma_f). rho_tilde:(N,2)."""
    s = rho_tilde.sum(dim=-1)
    return torch.exp(-(s ** 2) / sigma_f)
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest src/instinct-rl/tests/test_foothold_math.py -v`
Expected: PASS（3 个用例）。

- [ ] **Step 5: Commit**

```bash
git add src/instinct-rl/instinct_rl/algorithms/foothold_math.py src/instinct-rl/tests/test_foothold_math.py
git commit -m "feat(rl): add support-deficiency and guidance-reward math (SSR eq2/eq3)"
```

---

## Task 4: FootholdStorage（未来触地点 GT 回填）

这是 NLL 监督的核心——在有序 `[T,N]` 缓冲上前向回填"下一次触地点"，处理 episode 边界与右删失。

**Files:**
- Create: `src/instinct-rl/instinct_rl/storage/foothold_storage.py`
- Modify: `src/instinct-rl/instinct_rl/storage/__init__.py`
- Test: `src/instinct-rl/tests/test_foothold_storage.py`

- [ ] **Step 1: 写失败测试**（手工构造接触序列，断言目标与 mask）

```python
import torch
from instinct_rl.storage.foothold_storage import compute_next_touchdown_targets


def test_next_touchdown_single_env_one_foot():
    # T=5, N=1, 1 只脚：contact 序列 0,0,1,0,1 -> 触地边沿在 t=2 和 t=4
    contact = torch.tensor([[0.], [0.], [1.], [0.], [1.]]).view(5, 1, 1)   # (T,N,1)
    pos_w = torch.tensor([[0.], [0.], [2.], [0.], [4.]]).view(5, 1, 1)     # x 坐标, y=0
    pos_w = torch.cat([pos_w, torch.zeros_like(pos_w)], dim=-1)            # (T,N,1,2)? -> 见下
    # 用 base=原点, yaw=0，则 base 系 = 世界系
    base_pose = torch.zeros(5, 1, 3)
    dones = torch.zeros(5, 1, 1)
    target, valid = compute_next_touchdown_targets(
        contact=contact, foot_pos_w=pos_w, base_pose_w=base_pose, dones=dones, horizon=5)
    # t=0,1,2 的下一次触地在 t=2 (x=2)；t=3,4 的在 t=4 (x=4)
    assert torch.allclose(target[0, 0, 0], torch.tensor([2., 0.]), atol=1e-5)
    assert torch.allclose(target[2, 0, 0], torch.tensor([2., 0.]), atol=1e-5)
    assert torch.allclose(target[3, 0, 0], torch.tensor([4., 0.]), atol=1e-5)
    assert valid[:, 0, 0].tolist() == [1, 1, 1, 1, 1]


def test_right_censored_masked():
    # 全程摆动，无触地 -> 全部 invalid
    contact = torch.zeros(4, 1, 1)
    pos_w = torch.zeros(4, 1, 1, 2)
    base_pose = torch.zeros(4, 1, 3)
    dones = torch.zeros(4, 1, 1)
    _, valid = compute_next_touchdown_targets(contact, pos_w, base_pose, dones, horizon=4)
    assert valid.sum() == 0


def test_done_boundary_blocks_future():
    # t=1 done；t=2 触地不应作为 t<=1 的目标
    contact = torch.tensor([[0.], [0.], [1.]]).view(3, 1, 1)
    pos_w = torch.tensor([[0.], [0.], [9.]]).view(3, 1, 1)
    pos_w = torch.cat([pos_w, torch.zeros_like(pos_w)], dim=-1)
    base_pose = torch.zeros(3, 1, 3)
    dones = torch.tensor([[0.], [1.], [0.]]).view(3, 1, 1)
    _, valid = compute_next_touchdown_targets(contact, pos_w, base_pose, dones, horizon=3)
    assert valid[0, 0, 0] == 0 and valid[1, 0, 0] == 0 and valid[2, 0, 0] == 1
```

> `foot_pos_w` 约定形状 `(T,N,2,2)`（脚, xy）。测试里单脚，reshape 成 `(T,N,1,2)`；函数需对脚维通用。修正测试第一例的 `pos_w` 构造为 `(5,1,1,2)`：`pos_w = torch.stack([x, zeros], dim=-1)` 其中 x 形状 `(5,1,1)`。

- [ ] **Step 2: 运行确认失败**

Run: `pytest src/instinct-rl/tests/test_foothold_storage.py -v`
Expected: FAIL。

- [ ] **Step 3: 实现 storage + 回填函数**

```python
"""落脚点想象的独立缓冲（仿 AmpStorage）：存 privileged 特征与几何量，
在有序 [T,N] 上前向回填"下一次触地点"作为 NLL 目标（SSR 式1）。"""

from collections import namedtuple

import torch


def compute_next_touchdown_targets(contact, foot_pos_w, base_pose_w, dones, horizon):
    """
    Args:
        contact:    (T,N,F) in {0,1}
        foot_pos_w: (T,N,F,2) 世界系 xy
        base_pose_w:(T,N,3)  [x,y,yaw] 世界系
        dones:      (T,N,1)
    Returns:
        target_b: (T,N,F,2) 目标触地点在【查询步 t 的 base 系】
        valid:    (T,N,F)   1 表示 horizon 内、同 episode 内找到触地边沿
    说明：触地边沿定义为 contact[t']==1 且 contact[t'-1]==0；t'=0 视为非边沿(False)。
    target[t] = 首个 t'>=t 的触地边沿处 foot_pos_w；跨越 dones 边界不算。
    """
    T, N, F = contact.shape
    device = contact.device
    # 触地边沿
    prev = torch.zeros_like(contact)
    prev[1:] = contact[:-1]
    edge = ((contact == 1) & (prev == 0)).float()                     # (T,N,F)

    target_w = torch.zeros(T, N, F, 2, device=device)
    valid = torch.zeros(T, N, F, device=device)

    future_pos = torch.zeros(N, F, 2, device=device)
    future_valid = torch.zeros(N, F, device=device)
    for t in reversed(range(T)):
        # dones[t]==1: t 是本 episode 末步，来自 t+1 的 carry 属于下一 episode，切断
        keep = (dones[t] == 0).float().view(N, 1)                      # (N,1)
        future_valid = future_valid * keep
        future_pos = future_pos * keep.unsqueeze(-1)

        e = edge[t]                                                    # (N,F)
        this_valid = torch.clamp(e + future_valid, max=1.0)
        this_pos = torch.where(e.unsqueeze(-1) > 0, foot_pos_w[t], future_pos)
        target_w[t] = this_pos
        valid[t] = this_valid
        future_valid = this_valid
        future_pos = this_pos

    # 世界 -> 查询步 base 系
    bx = base_pose_w[..., 0:1].unsqueeze(2)                            # (T,N,1,1)
    by = base_pose_w[..., 1:2].unsqueeze(2)
    yaw = base_pose_w[..., 2:3].unsqueeze(2)                           # (T,N,1,1)
    dx = target_w[..., 0:1] - bx
    dy = target_w[..., 1:2] - by
    c = torch.cos(-yaw); s = torch.sin(-yaw)
    px = c * dx - s * dy
    py = s * dx + c * dy
    target_b = torch.cat([px, py], dim=-1)                            # (T,N,F,2)
    return target_b, valid


class FootholdStorage:
    class Transition:
        def __init__(self):
            self.feat = None
            self.foot_contact = None
            self.foot_pos_w = None
            self.base_pose_w = None
            self.terrain_level = None
            self.dones = None

        def clear(self):
            self.__init__()

    MiniBatch = namedtuple("MiniBatch", ["feat", "target", "valid", "terrain_level"])

    def __init__(self, num_envs, num_transitions_per_env, feat_size, num_feet=2, device="cpu"):
        self.device = device
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.num_feet = num_feet
        T, N = num_transitions_per_env, num_envs
        self.feat = torch.zeros(T, N, feat_size, device=device)
        self.foot_contact = torch.zeros(T, N, num_feet, device=device)
        self.foot_pos_w = torch.zeros(T, N, num_feet, 2, device=device)
        self.base_pose_w = torch.zeros(T, N, 3, device=device)
        self.terrain_level = torch.zeros(T, N, 1, device=device)
        self.dones = torch.zeros(T, N, 1, device=device)
        self.step = 0
        self._target = None
        self._valid = None

    def add_transitions(self, tr: "FootholdStorage.Transition"):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Foothold buffer overflow")
        self.feat[self.step].copy_(tr.feat)
        self.foot_contact[self.step].copy_(tr.foot_contact)
        self.foot_pos_w[self.step].copy_(tr.foot_pos_w.view(self.num_envs, self.num_feet, 2))
        self.base_pose_w[self.step].copy_(tr.base_pose_w)
        self.terrain_level[self.step].copy_(tr.terrain_level)
        self.dones[self.step].copy_(tr.dones.view(-1, 1))
        self.step += 1

    def compute_targets(self):
        self._target, self._valid = compute_next_touchdown_targets(
            self.foot_contact, self.foot_pos_w, self.base_pose_w, self.dones,
            horizon=self.num_transitions_per_env)

    def clear(self):
        self.step = 0
        self._target = None
        self._valid = None

    def mini_batch_generator(self, num_mini_batches, num_epochs=5):
        T, N = self.num_transitions_per_env, self.num_envs
        batch = T * N
        mb = batch // num_mini_batches
        feat = self.feat.reshape(batch, -1)
        target = self._target.reshape(batch, self.num_feet, 2)
        valid = self._valid.reshape(batch, self.num_feet)
        level = self.terrain_level.reshape(batch, 1)
        for _ in range(num_epochs):
            idx = torch.randperm(num_mini_batches * mb, device=self.device)
            for i in range(num_mini_batches):
                sel = idx[i * mb:(i + 1) * mb]
                yield FootholdStorage.MiniBatch(feat[sel], target[sel], valid[sel], level[sel])
```

- [ ] **Step 4: 导出** —— `storage/__init__.py` 追加 `from .foothold_storage import FootholdStorage, compute_next_touchdown_targets`。

- [ ] **Step 5: 运行确认通过**（先按 Step 1 注记修正测试里 `pos_w` 形状为 `(T,N,1,2)`）

Run: `pytest src/instinct-rl/tests/test_foothold_storage.py -v`
Expected: PASS（3 个用例）。

- [ ] **Step 6: Commit**

```bash
git add src/instinct-rl/instinct_rl/storage/foothold_storage.py src/instinct-rl/instinct_rl/storage/__init__.py src/instinct-rl/tests/test_foothold_storage.py
git commit -m "feat(rl): add FootholdStorage with next-touchdown target backfill"
```

---

## Task 5: FootholdAlgoMixin + 组合算法类

把想象网络、reward 注入、NLL 训练全部串起来（结构对齐 `WasabiAlgoMixin`）。

**Files:**
- Create: `src/instinct-rl/instinct_rl/algorithms/foothold.py`
- Modify: `src/instinct-rl/instinct_rl/algorithms/__init__.py`

- [ ] **Step 1: 实现 mixin**

```python
"""FootholdAlgoMixin：SSR imagined foothold guidance 的算法插件（仿 WasabiAlgoMixin）。

- 独立网络 FootholdImaginationModel + 独立 Adam 优化器 + 独立 FootholdStorage。
- compute_auxiliary_reward: 用想象输出算 r^f（式3）注入 rollout reward（键名 foothold_reward）。
- update: PPO/AMP 更新后，回填 GT 并用 NLL（式1）训练想象网络。
"""

import torch
import torch.optim as optim

from instinct_rl.algorithms.foothold_math import (
    expected_support_deficiency, guidance_reward, support_deficiency_at_center,
)
from instinct_rl.modules.foothold_imagination import FootholdImaginationModel
from instinct_rl.storage.foothold_storage import FootholdStorage
from instinct_rl.utils.utils import get_subobs_by_components, get_subobs_size


class FootholdAlgoMixin:
    def __init__(
        self,
        *args,
        foothold_state_key="foothold",
        imagination_feature_components=(
            "foot_region_scan_l", "foot_region_scan_r", "foot_contact",
            "foot_pos_b", "foot_vel_b", "base_lin_vel", "last_action",
        ),
        region_grid=(21, 13),
        region_res=0.025,
        sole_grid=(10, 5),
        epsilon_h=0.03,
        sigma_f=0.0625,
        num_swing_samples=8,
        terrain_level_gate=5,
        imagination_hidden_sizes=(256, 128),
        imagination_lr=5e-4,
        foothold_reward_coef=0.25,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.foothold_state_key = foothold_state_key
        self.imagination_feature_components = list(imagination_feature_components)
        self.region_grid = tuple(region_grid)
        self.region_res = region_res
        self.sole_grid = tuple(sole_grid)
        self.epsilon_h = epsilon_h
        self.sigma_f = sigma_f
        self.num_swing_samples = num_swing_samples
        self.terrain_level_gate = terrain_level_gate
        self.imagination_hidden_sizes = tuple(imagination_hidden_sizes)
        self.imagination_lr = imagination_lr
        self.foothold_reward_coef = foothold_reward_coef  # 供 PPO.process_env_step 的 *_coef 机制

    def init_storage(self, num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards=1):
        super().init_storage(num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards)
        self._foothold_segments = obs_format[self.foothold_state_key]
        feat_size = get_subobs_size(self._foothold_segments, self.imagination_feature_components)
        self.imagination = FootholdImaginationModel(
            input_size=feat_size, hidden_sizes=self.imagination_hidden_sizes).to(self.device)
        self.imagination_optimizer = optim.Adam(self.imagination.parameters(), lr=self.imagination_lr)
        self.foothold_storage = FootholdStorage(
            num_envs, num_transitions_per_env, feat_size=feat_size, num_feet=2, device=self.device)
        self.foothold_transition = FootholdStorage.Transition()

    # ---- 从 obs_pack["foothold"] 切片 ----
    def _fh_slice(self, fh, names):
        return get_subobs_by_components(fh, names, self._foothold_segments)

    def _imagination_input(self, fh):
        return self._fh_slice(fh, self.imagination_feature_components)

    def process_env_step(self, rewards, dones, infos, next_obs, next_critic_obs):
        fh = infos["observations"][self.foothold_state_key]
        self.foothold_transition.feat = self._imagination_input(fh).clone()
        self.foothold_transition.foot_contact = self._fh_slice(fh, ["foot_contact"]).clone()
        self.foothold_transition.foot_pos_w = self._fh_slice(fh, ["foot_pos_w"]).clone()
        self.foothold_transition.base_pose_w = self._fh_slice(fh, ["base_pose_w"]).clone()
        self.foothold_transition.terrain_level = self._fh_slice(fh, ["terrain_level"]).clone()
        self.foothold_transition.dones = dones.clone()
        self.foothold_storage.add_transitions(self.foothold_transition)
        self.foothold_transition.clear()
        super().process_env_step(rewards, dones, infos, next_obs, next_critic_obs)

    @torch.no_grad()
    def compute_auxiliary_reward(self, obs_pack):
        aux = super().compute_auxiliary_reward(obs_pack)
        fh = obs_pack[self.foothold_state_key]
        feat = self._imagination_input(fh)
        mu, log_sigma = self.imagination(feat)                          # (N,2,2),(N,2)

        scan_l = self._fh_slice(fh, ["foot_region_scan_l"])
        scan_r = self._fh_slice(fh, ["foot_region_scan_r"])
        contact = self._fh_slice(fh, ["foot_contact"])                  # (N,2)
        foot_pos_b = self._fh_slice(fh, ["foot_pos_b"]).view(-1, 2, 2)  # (N,2,2)
        level = self._fh_slice(fh, ["terrain_level"]).squeeze(-1)       # (N,)

        rho = torch.zeros_like(contact)                                 # (N,2)
        for i, scan in enumerate((scan_l, scan_r)):
            sole_z_stance = foot_pos_b.new_zeros(scan.shape[0])         # 用相对高度网格: 见下注
            rho_stance = support_deficiency_at_center(
                scan, sole_z_stance, self.region_grid, self.region_res, self.sole_grid, self.epsilon_h)
            # 想象点相对 scanner 中心的偏移 = mu(base) - 当前脚(base)
            offset = mu[:, i, :] - foot_pos_b[:, i, :]
            rho_swing = expected_support_deficiency(
                scan, offset, log_sigma[:, i], self.region_grid, self.region_res,
                self.sole_grid, self.epsilon_h, num_samples=self.num_swing_samples)
            rho[:, i] = contact[:, i] * rho_stance + (1.0 - contact[:, i]) * rho_swing

        rf = guidance_reward(rho, self.sigma_f)                         # (N,)
        gate = (level >= self.terrain_level_gate).float()
        aux["foothold_reward"] = (rf * gate).unsqueeze(-1)
        return aux

    def update(self, *args, **kwargs):
        mean_losses, average_stats = super().update(*args, **kwargs)
        self.foothold_storage.compute_targets()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        for mb in self.foothold_storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
            mu, log_sigma = self.imagination(mb.feat)                   # (B,2,2),(B,2)
            sq = ((mb.target - mu) ** 2).sum(dim=-1)                    # (B,2)
            nll = sq / (2.0 * torch.exp(2.0 * log_sigma)) + 2.0 * log_sigma  # (B,2), SSR 式1
            gate = (mb.terrain_level.squeeze(-1) >= self.terrain_level_gate).float().unsqueeze(-1)
            m = mb.valid * gate                                         # (B,2)
            loss = (nll * m).sum() / m.sum().clamp(min=1.0)
            self.imagination_optimizer.zero_grad()
            loss.backward()
            self.imagination_optimizer.step()
            mean_losses["imagination_nll"] = mean_losses.get("imagination_nll", 0.0) + loss.detach() / num_updates
        self.foothold_storage.clear()
        return mean_losses, average_stats

    def state_dict(self):
        sd = super().state_dict()
        sd["imagination"] = self.imagination.state_dict()
        sd["imagination_optimizer"] = self.imagination_optimizer.state_dict()
        return sd

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if "imagination" in state_dict:
            self.imagination.load_state_dict(state_dict["imagination"])
        if "imagination_optimizer" in state_dict:
            self.imagination_optimizer.load_state_dict(state_dict["imagination_optimizer"])
```

> **关于 stance ρ 的 `sole_z`（重要注记）**：`height_scan` 返回的是命中点世界高度 z（绝对）。上面用 `sole_z_stance=0` 是占位——正确做法是让 stance 的鞋底平面 = 当前脚底高度。有两种一致方案，二选一并在实现时固定：
> (a) 给 `foothold` 组再加一个 term `foot_sole_z`（= `body_link_pos_w[foot,2] - foot_height_offset`，见 `feet_unsupported` 的 `foot_height_offset=0.065`），传入 `support_deficiency_at_center` 作为 `sole_z`；
> (b) 或在 `height_scan` 里用 `offset` 参数把区域图转成相对脚底的高度，从而 `sole_z=0` 成立。
> 推荐 (a)，与 `feet_unsupported`（rewards.py:574-576）完全同源。若选 (a)，在 C1/Task1 的 foothold 组追加 `foot_sole_z (2,)` term，并在 mixin 里 `sole_z_stance = self._fh_slice(fh,["foot_sole_z"])[:, i]`，`expected_support_deficiency` 的想象点仍用 patch 内 `max` 作 `h^f`（SSR 规定）。

- [ ] **Step 2: 组合类 + 导出**（`algorithms/__init__.py`）

```python
from .foothold import FootholdAlgoMixin

class FootholdWasabiEstimatorPPO(FootholdAlgoMixin, WasabiEstimatorPPO):
    """SSR: AMP + 速度/地形估计 + imagined foothold guidance。
    MRO: Foothold -> Wasabi -> Estimator -> PPO，保证各自的
    process_env_step/compute_auxiliary_reward/update/state_dict 沿 super 链正确叠加。"""
    pass
```
加入 `__all__`。

- [ ] **Step 3: 冒烟导入测试**

Run: `python -c "from instinct_rl.algorithms import FootholdWasabiEstimatorPPO; print('ok')"`
Expected: 打印 `ok`。

- [ ] **Step 4: Commit**

```bash
git add src/instinct-rl/instinct_rl/algorithms/foothold.py src/instinct-rl/instinct_rl/algorithms/__init__.py
git commit -m "feat(rl): add FootholdAlgoMixin and FootholdWasabiEstimatorPPO (SSR guidance)"
```

---

## Task 6: 算法 Cfg + runner 接线 + 若需 foot_sole_z 补 term

**Files:**
- Modify: `src/instinct_mj/rl/config.py`（新算法 cfg）
- Modify: `src/instinct_mj/rl/__init__.py`（re-export）
- Modify: `.../adam_lite/agents/instinct_rl_amp_cfg.py`（接算法 cfg）
- Modify: `.../adam_lite/adam_lite_foothold_parkour_cfg.py`（若选注记方案 (a)，补 `foot_sole_z` term + 观测函数）

- [ ] **Step 1: 若采用 stance 方案 (a)：补 `foot_sole_z` 观测函数**（`foothold_obs.py`）

```python
def foot_sole_z(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, foot_height_offset: float = 0.065) -> torch.Tensor:
    """两脚鞋底平面世界高度 (N,2)=[left,right]，与 feet_unsupported 一致。"""
    asset: Entity = env.scene[asset_cfg.name]
    ids = _foot_ids(asset_cfg)
    z = torch.stack([asset.data.body_link_pos_w[:, i, 2] for i in ids], dim=-1)
    return z - foot_height_offset
```
在 `mdp/__init__.py` 导出，并在 foothold 组（Task1 Step3）追加 `"foot_sole_z": ObservationTermCfg(func=parkour_mdp.foot_sole_z, params={"asset_cfg": ankle})`（放在 `foot_pos_b` 之后，更新 C1 顺序表）。

- [ ] **Step 2: 新算法 Cfg**（`rl/config.py` 末尾）

```python
@dataclass
class FootholdWasabiPpoAlgorithmCfg(EstimatorWasabiPpoAlgorithmCfg):
    """EstimatorWasabiPPO + SSR imagined foothold guidance。"""

    class_name: str = "FootholdWasabiEstimatorPPO"
    foothold_state_key: str = "foothold"
    region_grid: tuple[int, int] = (21, 13)          # 由 Task1 Step4 探测填入
    region_res: float = 0.025
    sole_grid: tuple[int, int] = (10, 5)
    epsilon_h: float = 0.03
    sigma_f: float = 0.0625
    num_swing_samples: int = 8
    terrain_level_gate: int = 5
    imagination_hidden_sizes: tuple[int, ...] = (256, 128)
    imagination_lr: float = 5.0e-4
    foothold_reward_coef: float = 0.25               # w_f
```
在 `rl/__init__.py` 的 import 与 `__all__` 里加 `FootholdWasabiPpoAlgorithmCfg`。

- [ ] **Step 3: runner 接算法**（`agents/instinct_rl_amp_cfg.py`）

```python
from instinct_mj.rl import FootholdWasabiPpoAlgorithmCfg  # 顶部 import

@dataclass(kw_only=True)
class FootholdAmpAlgoCfg(FootholdWasabiPpoAlgorithmCfg, AmpAlgoCfg):
    """AMP（判别器 reward + estimator）+ foothold guidance。字段取二者并集，class_name 用 Foothold 版。"""
    class_name: str = "FootholdWasabiEstimatorPPO"
```
把 `AdamLiteFootholdParkourRunnerCfg`（Task0）的 `algorithm` 换成：
```python
    algorithm: InstinctRlPpoAlgorithmCfg = field(default_factory=lambda: FootholdAmpAlgoCfg())
```

> `region_grid` 用 Task1 Step4 探测到的真实网格；若探测得到与 (21,13) 不同的值，在 `FootholdWasabiPpoAlgorithmCfg` 与测试常量里同步更新。

- [ ] **Step 4: 配置健全性检查**

Run: `python -c "from instinct_mj.tasks.parkour.config.adam_lite.agents.instinct_rl_amp_cfg import adam_lite_foothold_parkour_amp_runner_cfg as f; import json; print(list(f().algorithm.__dict__.keys()))"`
Expected: 输出含 `foothold_state_key/region_grid/foothold_reward_coef/class_name=FootholdWasabiEstimatorPPO` 等字段。

- [ ] **Step 5: Commit**

```bash
git add src/instinct_mj/rl/ src/instinct_mj/tasks/parkour/config/adam_lite/
git commit -m "feat(parkour): wire foothold algorithm cfg into adam lite foothold runner"
```

---

## Task 7: 端到端冒烟训练与验证（Phase 1）

**Files:** 无新增（验证为主）

- [ ] **Step 1: 短训冒烟（几十 iter）**

Run:
```bash
python -m instinct_mj.scripts.instinct_rl.train Instinct-Parkour-Adam-Lite-Foothold-v0 \
  --run-name foothold_smoke 2>&1 | tee /tmp/foothold_smoke.log
```
观察前 ~50 iter：进程不崩、TensorBoard/日志出现 `imagination_nll`（在 `losses` 里，经 `on_policy_runner.log` 打到 `Loss/imagination_nll`）、`Step/foothold_reward`（`compute_auxiliary_reward` 注入后 PPO 把 `infos["step"]["foothold_reward"]` 记录）。
Expected: 无 shape/维度错误；`imagination_nll` 有限且随迭代下降；`foothold_reward` 在 (0,1]。

- [ ] **Step 2: reward 语义抽查（play）**

Run:
```bash
python -m instinct_mj.scripts.instinct_rl.play Instinct-Parkour-Adam-Lite-Foothold-Play-v0 \
  --load_run <smoke run dir> --load_checkpoint model_<iter>.pt
```
在 viewer 里 region scanner `debug_vis` 应覆盖脚周并含前方落点区域；平地上想象点 ρ≈0、`r^f≈1`；楼梯边沿摆动脚 ρ 上升、`r^f` 下降。
Expected: 上述定性行为成立。

- [ ] **Step 3: checkpoint 存取往返**

短训到产生一个 `model_*.pt` 后 `--resume` 续训一步。
Expected: `imagination`/`imagination_optimizer` 被 `state_dict/load_state_dict` 正确存取，续训不报缺键。

- [ ] **Step 4: 正式训练**

Run:
```bash
python -m instinct_mj.scripts.instinct_rl.train Instinct-Parkour-Adam-Lite-Foothold-v0 \
  --run-name foothold_b
```
与 baseline `Instinct-Parkour-Adam-Lite-v0` 对比楼梯上的边沿接触/跌倒率、`feet_unsupported` 指标（可临时把 `feet_unsupported` 作为**只读诊断** reward 加进两个任务对比，权重设 0 不影响优化）。
Expected: foothold 任务边沿落脚更居中、支撑比例更高。

- [ ] **Step 5: Commit（若有调参改动）**

```bash
git add -A && git commit -m "chore(parkour): tune foothold guidance hyperparameters"
```

---

## Task 8（可选，Phase 2）: 多 critic 对齐 SSR

SSR 用独立 critic 估计 `r^f` 的价值。此任务把 `r^f` 单独路由到第二个 critic。

**Files:**
- Modify: `.../adam_lite/adam_lite_foothold_parkour_cfg.py`（rewards 拆组）
- Modify: `.../agents/instinct_rl_amp_cfg.py`（多 reward 权重）

- [ ] **Step 1: rewards 拆成两组**

把 `cfg.rewards`（当前单 dict）改为：
```python
task_rewards = cfg.rewards           # 现有全部
cfg.rewards = {
    "task": task_rewards,
    "foothold": {                    # 仅作为 r^f 的价值载体
        "foothold_anchor": RewardTermCfg(func=envs_mdp.is_alive, weight=0.0),
    },
}
```
`InstinctRlEnv` 会把它交给 `MultiRewardManager`，`env.num_rewards=2`。

- [ ] **Step 2: 路由 r^f 到第二组 + 混合权重**（`FootholdAmpAlgoCfg`）

```python
    advantage_mixing_weights: tuple[float, ...] = (1.0, 0.25)          # (task, foothold)
    auxiliary_reward_per_env_reward_coefs: tuple[float, ...] = (0.0, 1.0)  # r^f 只进 foothold 组
    discriminator_reward_coef: float = 0.25
```
> `auxiliary_reward_per_env_reward_coefs` 对**所有** auxiliary reward 生效（含 `discriminator_reward`）。若希望 AMP reward 仍进 task 组、只有 foothold 进第二组，需要更细的路由——这超出 `PPO.process_env_step` 现有粒度，需在 `FootholdAlgoMixin.process_env_step` 覆写 reward 累加逻辑（把 `foothold_reward` 单独按 `(0,1)`、`discriminator_reward` 按 `(1,0)` 加）。若不需要该细分，保持默认让两种 aux 同路由即可。

- [ ] **Step 3: 验证多 critic**

Run: 短训，确认日志出现 `value_loss_0/value_loss_1`、`advantage_0/advantage_1`（`ppo.py:285-289` 在多 reward 时输出），且 `Actor MLP` 后打印 `Multiple Critics MLP: 2 in total`（`actor_critic.py:90`）。
Expected: 双 critic 生效，训练稳定。

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat(parkour): add multi-critic routing for foothold guidance reward"
```

---

## Self-Review

**1. Spec coverage（对照 SSR 引导点三要素）：**
- 想象网络（MLP[256,128]→高斯）：Task 2 ✓；输入=privileged+action：C2 + Task5 `_imagination_input` ✓。
- NLL 监督（式1）+ terrain-level 门控：Task 4（GT 回填）+ Task 5.update（NLL + gate）✓。
- 支撑缺失度 ρ（式2）+ 引导 reward（式3）：Task 3 ✓；stance/swing 切换：Task 5.compute_auxiliary_reward ✓。
- `w_f=0.25`：`foothold_reward_coef`（单 critic）/ `advantage_mixing_weights`（多 critic，Task 8）✓。
- 常量 `ε_h=0.03/σ_f=0.0625/sole 22.5×10@2.5cm/lr5e-4`：C5 + 各 cfg ✓。

**2. Placeholder scan：** stance `sole_z` 的正确来源已在 Task 5 注记明确给出方案 (a)（`foot_sole_z` term，Task6 Step1 实现），非 TODO。`region_grid` 具体值依赖真实网格，已给出探测步骤（Task1 Step4）并在多处标注需同步——这是"运行时确定的常量+探测方法"，非占位逻辑。

**3. Type/命名一致性：**
- 观测组键 `"foothold"`、term 名（C1）在 Task1/Task5 一致。
- `FootholdImaginationModel(input_size, hidden_sizes, nonlinearity)` 在 Task2 定义、Task5 调用一致。
- `compute_next_touchdown_targets(contact, foot_pos_w, base_pose_w, dones, horizon)` 在 Task4 定义与测试一致；`FootholdStorage.Transition` 字段（feat/foot_contact/foot_pos_w/base_pose_w/terrain_level/dones）在 Task4 与 Task5.process_env_step 一致。
- `support_deficiency_at_center / expected_support_deficiency / guidance_reward` 在 Task3 定义、Task5 调用签名一致。
- 算法 `class_name="FootholdWasabiEstimatorPPO"` 在 Task5（类）、Task6（cfg）一致；runner 通过 `getattr(algorithms, class_name)` 解析（已核对 `on_policy_runner.py:73`）。
- 需实现时核对的仓库 API（探测暴露）：`quat_apply_inverse`、`heading_w`、`body_link_lin_vel_w`、`terrain.terrain_levels`、`MlpModel` 参数名、`get_subobs_by_components/get_subobs_size` 是否支持 `component_names` 子集切片（`estimator.py`/`encoder_terrain_vel_actor_critic.py` 已在用，签名以那里为准）。

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-02-ssr-imagined-foothold-guidance.md`. 建议先只执行 **Phase 1（Task 0–7）** 跑通验证，再决定是否做 Task 8。

两种执行方式：
1. **Subagent-Driven（推荐）**：每个 Task 派一个 fresh subagent 实现，任务间我做两阶段 review，迭代快。
2. **Inline Execution**：在本会话按 executing-plans 分批执行，带检查点 review。

你想用哪种？
