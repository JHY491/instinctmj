---
name: instinctmj-dev
description: Use when adding parkour tasks, modifying observation/encoder/policy architecture, or debugging config issues in the InstinctMJ + instinct-rl codebase
---

# InstinctMJ Parkour Development Guide

## Project Layout

```
InstinctMJ (tasks, env, sensors)           instinct-rl (modules, algorithms)
──────────────────────────────────────     ───────────────────────────────
tasks/parkour/config/adam_lite/            modules/
  ├── __init__.py           ← registry      ├── parallel_layer.py    ← builds encoders from Cfg
  ├── env_cfgs.py           ← base cfg      ├── conv2d.py            ← Conv2dHeadModel
  ├── rl_cfgs.py            ← runner glue   ├── mlp.py               ← MlpModel
  ├── agents/                               ├── perceptive_encoder.py ← CNN+FiLM+AttnPool
  │   ├── instinct_rl_amp_cfg.py             ├── encoder_actor_critic.py
  │   └── instinct_rl_ppo_cfg.py             ├── actor_critic_recurrent.py
  └── adam_lite_*_parkour_cfg.py ← full env ├── transformer.py
                                             └── state_conditioned_depth_transformer.py
rl/config.py       ← Policy Cfg classes
rl/module_cfg.py   ← Encoder block Cfg classes
rl/__init__.py     ← Re-exports
```

## Adding a New Parkour Task

### 1. Env Config `adam_lite_NEW_parkour_cfg.py`

Copy from the closest existing task. A complete env config function returns a `ManagerBasedRlEnvCfg` and overwrites these sections:

| Section | Typical contents |
|---------|-----------------|
| `cfg.scene.terrain` | `InstinctTerrainImporterCfg(terrain_generator=...)` — pick a preset from `parkour_env_cfg` |
| `cfg.scene.sensors` | Contact sensors, height scanners, volume points, `NoisyGroupedRayCasterCameraCfg`, motion reference |
| `cfg.commands["base_velocity"]` | `PoseVelocityCommandCfg` with per-terrain `velocity_ranges` |
| `policy_terms` | Obs the policy sees. depth: `delayed_visualizable_image` (10 frames). proprio/all others: `history_length=8, flatten_history_dim=True` |
| `critic_terms` | Obs the critic sees. Same shape as policy + `base_lin_vel`. Use depth, not heightmap |
| `amp_policy_terms` / `amp_reference_terms` | Paired proprio + reference states, `history_length=10` |
| `cfg.observations["policy/critic/amp_policy/amp_reference"]` | `ObservationGroupCfg(terms=..., concatenate_terms=False)` |
| `cfg.rewards` | Tracking + regularization + contact penalties |
| `cfg.curriculum` | Terrain level progression |
| `cfg.terminations` | Timeout, terrain bounds, illegal contact, bad orientation, dataset exhausted |
| `cfg.events` | Domain rand (friction, mass, gains, com), reset logic, push robot |

The env factory pattern is:
```python
def instinct_adam_lite_NEW_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = adam_sp_flat_parkour_env_cfg(play=play, has_state_estimation=True)
    # ... mutate cfg ...
    return cfg

def instinct_adam_lite_NEW_final_cfg(*, play: bool = False):
    cfg = instinct_adam_lite_NEW_env_cfg(play=play)
    if play:
        cfg.viewer = ViewerConfig(...)
    return cfg
```

### 2. Agent Config `agents/instinct_rl_amp_cfg.py`

Add a `RunnerCfg` subclass with a unique `experiment_name`:
```python
@dataclass(kw_only=True)
class AdamLiteNewParkourVisionAmpRunnerCfg(AdamSPParkourVisionAmpRunnerCfg):
    experiment_name: str = "adam_lite_new_parkour"

def adam_lite_new_parkour_amp_runner_cfg():
    return AdamLiteNewParkourVisionAmpRunnerCfg()
```

### 3. Registry `__init__.py`

Two lambdas + `register_instinct_task` call:
```python
def _new_parkour_cfg(play: bool):
    from .adam_lite_NEW_parkour_cfg import instinct_adam_lite_NEW_final_cfg
    return instinct_adam_lite_NEW_final_cfg(play=play)

def _new_runner_cfg():
    from .agents.instinct_rl_amp_cfg import adam_lite_new_parkour_amp_runner_cfg
    return adam_lite_new_parkour_amp_runner_cfg()

register_instinct_task(
    task_id="Instinct-Parkour-Adam-Lite-New-v0",
    env_cfg_factory=lambda: _new_parkour_cfg(play=False),
    play_env_cfg_factory=lambda: _new_parkour_cfg(play=True),
    instinct_rl_cfg_factory=_new_runner_cfg,
)
# Also register the -Play-v0 variant
```

### 4. Terrain Preset `parkour_env_cfg.py` / `terrains/presets.py`

Add `NEW_TERRAINS_CFG` + `NEW_TERRAINS_CFG_PLAY`, re-export from `parkour_env_cfg.py`.

---

## Terrain System

### Architecture

```
terrains/
  terrain_importer_cfg.py    ← InstinctTerrainImporterCfg (entry point)
  terrain_generator_cfg.py   ← FiledTerrainGeneratorCfg (sizes, rows, cols, curriculum)
  height_field/
    hf_terrains_cfg.py       ← Perlin* TerrainCfg classes (30+ terrain types)
    utils.py                 ← heightfield → mesh conversion
  trimesh/                   ← STL/OBJ mesh terrains (motion-matched, etc.)
  virtual_obstacle/          ← edge cylinders, mesh obstacles for contact sensing
```

### Terrain Preset Pattern

Defined in `tasks/parkour/config/terrains/presets.py`. All presets follow:

```python
PRESET_CFG = FiledTerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),        # per-tile size (x, y) in meters
    border_width=3.0,        # flat border around each tile
    num_rows=10,             # rows in terrain grid
    num_cols=20,             # columns in terrain grid
    horizontal_scale=0.07,   # hfield cell spacing (m)
    vertical_scale=0.005,    # hfield height quantization (m)
    slope_threshold=1.0,     # max slope before clipping
    curriculum=True,         # enable terrain level progression
    sub_terrains={
        "terrain_key": PerlinXxxTerrainCfg(proportion=0.15, ...),
        ...
    },
)

PRESET_CFG_PLAY = copy.deepcopy(PRESET_CFG)
PRESET_CFG_PLAY.num_rows = 4   # smaller grid for play/viewer
PRESET_CFG_PLAY.num_cols = 10
```

Key fields on `FiledTerrainGeneratorCfg`:

| Field | Purpose |
|-------|---------|
| `size` | `(x, y)` meters per terrain column |
| `num_rows × num_cols` | Grid dimensions. 10×20 = 200 tiles for training, 4×10 for play |
| `horizontal_scale` | hfield cell size in meters (0.07 = 7cm resolution) |
| `sub_terrains` | Dict of terrain types, keyed by name. `proportion` sums to 1.0 |
| `curriculum=True` | Enables terrain-level progression via the curriculum manager |

### Sub-Terrain Types (all in `hf_terrains_cfg.py`)

Each terrain class inherits `HfTerrainBaseCfg` + optional `WallTerrainCfgMixin` (adds boundary walls):

| Category | Classes | Key params |
|----------|---------|-----------|
| **Plane** | `PerlinPlaneTerrainCfg` | `noise_scale`, `noise_frequency`, `fractal_*` |
| **Stairs** | `PerlinPyramidStairsTerrainCfg`, `PerlinInvertedPyramidStairsTerrainCfg`, `PerlinOneSidedStairsTerrainCfg`, `PerlinSpiralStairsTerrainCfg` | `step_height_range`, `step_width`, `platform_width`, `perlin_cfg` |
| **Slopes** | `PerlinContinuousSlopeTerrainCfg`, `PerlinCrossSlopeTerrainCfg`, `PerlinSymmetricalSlopeTerrainCfg`, `PerlinSlopeTerrainCfg` | `slope_range`, `perlin_cfg` |
| **Hurdles** | `PerlinContinuousHurdleTerrainCfg`, `PerlinDiscreteObstaclesTerrainCfg` | obstacle height/depth ranges |
| **Bridges** | `PerlinCurvedBridgeTerrainCfg` | curve radius, width |
| **Corners** | `PerlinNarrowLCornerTerrainCfg` | corridor width, turn direction |
| **Gaps** | `PerlinSquareGapTerrainCfg` | `gap_distance_range`, `gap_depth` |
| **Waves** | `PerlinWaveTerrainCfg` | wave amplitude, frequency |
| **Poles** | `PerlinSlalomPoleTerrainCfg` | pole spacing, radius |
| **Tilt** | `PerlinTiltTerrainCfg`, `PerlinTiltedRampTerrainCfg` | tilt angle |

Common parameters across all types: `proportion`, `border_width`, `wall_prob`, `wall_height`, `flat_patch_sampling`.

### Flat Patch Sampling

Each sub-terrain defines named flat patches for robot spawning:

```python
flat_patch_sampling={
    "target": FlatPatchSamplingCfg(
        num_patches=50,
        patch_radius=[0.05, 0.10, 0.15, 0.20],
        max_height_diff=0.05,
    ),
}
```

`FixedSequenceFlatPatchSamplingCfg` (for bridges, L-corners) uses explicit `fixed_xy_points` for centerline-guided spawning.

### How Tasks Use Terrain

**Stairs task** (with virtual obstacles):
```python
terrain_gen = copy.deepcopy(STAIRS_ONLY_TERRAINS_CFG)
edge_obstacle_cfg = GreedyconcatEdgeCylinderCfg(cylinder_radius=0.03, ...)
cfg.scene.terrain = InstinctTerrainImporterCfg(
    terrain_type="hacked_generator",
    terrain_generator=terrain_gen,
    virtual_obstacle_source="mesh",
    virtual_obstacle_hfield_height_threshold=0.04,  # stairs need threshold for edge detection
    virtual_obstacles={"edges": edge_obstacle_cfg},
)
```

**Slopes task** (no virtual obstacles):
```python
terrain_gen = copy.deepcopy(THREE_SLOPES_TERRAINS_CFG)
cfg.scene.terrain = InstinctTerrainImporterCfg(
    terrain_generator=terrain_gen,
    virtual_obstacle_hfield_height_threshold=None,  # slopes: no virtual obstacles
    virtual_obstacles={},  # empty dict
)
```

**Hurdle task** (with virtual obstacles, different cylinder radius):
```python
terrain_gen = copy.deepcopy(MIXED_HURDLE_TERRAINS_CFG)
hurdle_edge_obstacle_cfg = GreedyconcatEdgeCylinderCfg(cylinder_radius=0.05, ...)
cfg.scene.terrain = InstinctTerrainImporterCfg(
    virtual_obstacle_hfield_height_threshold=None,  # hurdles may have perlin ground below
    virtual_obstacles={"edges": hurdle_edge_obstacle_cfg},
)
```

### Velocity Ranges Per Terrain

Each task maps terrain keys to velocity ranges:

```python
velocity_ranges={
    "perlin_rough_stand": {"lin_vel_x": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
    "continuous_slope":  {"lin_vel_x": (0.3, 2.0), "ang_vel_z": (-1.0, 1.0)},
    "one_sided_stairs":  {"lin_vel_x": (0.3, 1.0), "ang_vel_z": (-0.5, 0.5)},
},
```

Keys in `velocity_ranges` must match `sub_terrains` keys in the terrain preset.

### Single-Terrain Presets

For individual terrain types (play/debug), use `_with_single_terrain`:
```python
CONTINUOUS_SLOPE_TERRAIN_CFG = _with_single_terrain(PARKOUR_TERRAIN_CATALOG_CFG, "continuous_slope")
```

### Adding a New Terrain Type

1. Define the `HfTerrainBaseCfg` subclass in `hf_terrains_cfg.py` (pure heightfield generation)
2. Define the `Perlin*` wrapper in `hf_terrains_cfg.py` (adds perlin noise + walls)
3. Add to a terrain preset in `terrains/presets.py`
4. Add `velocity_ranges` entry in the env config
5. Re-export from `parkour_env_cfg.py`

---

## Command System

### PoseVelocityCommand (current default)

Parkour tasks use `PoseVelocityCommandCfg` — a terrain-aware velocity command generator that maps position targets to velocity commands via flat patches on the terrain surface.

**Config pattern** in env config:
```python
cfg.commands = {
    "base_velocity": PoseVelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(8.0, 12.0),     # re-sample every 8-12s
        velocity_control_stiffness=2.0,         # pos error → lin vel gain
        heading_control_stiffness=2.0,          # heading error → ang vel gain
        rel_standing_envs=0.05,                 # 5% envs stand still
        only_positive_lin_vel_x=True,           # never go backwards
        lin_vel_threshold=0.0,                  # deadband for lin vel
        ang_vel_threshold=0.0,                  # deadband for ang vel
        target_dis_threshold=0.4,               # arrival radius (m)
        ranges=PoseVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),               # global ranges (for random_velocity_terrain)
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
        ),
        random_velocity_terrain=["perlin_rough_stand",],   # randomize vel on these
        velocity_ranges={                        # per-terrain speed limits
            "perlin_rough_stand": {"lin_vel_x": (0.0, 0.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
            "perlin_rough":       {"lin_vel_x": (0.3, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-0.5, 0.5)},
            "one_sided_stairs":   {"lin_vel_x": (0.3, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-0.5, 0.5)},
        },
    ),
}
```

### Key parameters

| Parameter | Purpose |
|-----------|---------|
| `resampling_time_range` | How often velocity targets are regenerated |
| `velocity_control_stiffness` | Position error → linear velocity gain (body frame) |
| `heading_control_stiffness` | Heading error → angular velocity gain |
| `rel_standing_envs` | Fraction of envs that get zero velocity (standing still) |
| `velocity_ranges` | Per-terrain-type velocity limits, keys must match `sub_terrains` names |
| `random_velocity_terrain` | Terrain types where velocity is randomized (using `ranges` or `velocity_ranges` limits) |
| `target_dis_threshold` | Distance to target below which velocity is zeroed |
| `only_positive_lin_vel_x` | Never command backward motion |

### Observation integration

The velocity command feeds into observations via `generated_commands`:
```python
"velocity_commands": ObservationTermCfg(
    func=envs_mdp.generated_commands,
    params={"command_name": "base_velocity"},
),
```
Output: 3-dim `(lin_vel_x, lin_vel_y, ang_vel_z)` in body frame.

### Flat-patch targets

The command system reads `terrain.flat_patches["target"]` — per (terrain_level, terrain_type) a set of `(x, y, z)` positions on flat ground. These are defined by the terrain's `flat_patch_sampling` configuration (see Terrain System section).

**Standard mode** (most terrains): Random flat patch per episode.

**Fixed-sequence mode** (bridges, L-corners): Patches advance when robot arrives, enabling sequential traversal. Enabled via:
```python
flat_patch_advance_on_arrival=True,          # advance to next patch on arrival
flat_patch_arrival_xy_dist=0.4,             # arrival distance threshold
flat_patch_arrival_hold_steps=3,            # consecutive steps within threshold
flat_patch_resample_velocity_on_advance=True, # re-sample velocity on patch advance
```
Used with `FixedSequenceFlatPatchSamplingCfg` in terrain presets.

### Guided waypoints

For terrains requiring a fixed navigation path:
```python
guided_waypoints_local={
    "terrain_key": [(x1, y1, z1), (x2, y2, z2), ...],
},
guided_arrival_xy_dist=0.35,
guided_arrival_hold_steps=3,
guided_loop_waypoints=True,                 # wrap back to first waypoint
guided_resample_velocity_on_advance=True,
```

### Per-terrain velocity ranges pattern

The `velocity_ranges` dict maps terrain keys → speed limits. Keys must match `sub_terrains` keys in the terrain preset. `random_velocity_terrain` lists terrain types where velocities are randomized (typically standing terrains for yaw randomization).

**Stairs task** velocities: standing (0, 0, ±1.0) / walking (0.3-1.0, 0, ±0.5)
**Slopes task** velocities: standing (0, 0, ±1.0) / continuous_slope (0.3-2.0, 0, ±1.0) / cross_slope (0.3-1.0, 0, ±1.0)
**Hurdle task** velocities: standing (0, 0, ±1.0) / walking (0.3-1.0, 0, ±0.5) / slalom (0.3-1.5, 0, 0)

---

## Policy Architecture

### Available Cfg classes (all in `rl/config.py`)

| Cfg class | `class_name` | Encoder | RNN | AMP uses? |
|-----------|-------------|:------:|:---:|:--------:|
| `InstinctRlActorCriticCfg` | `ActorCritic` | ❌ | ❌ | PPO |
| `InstinctRlEncoderActorCriticCfg` | `EncoderActorCritic` | ✅ | ❌ | **AMP default** |
| `InstinctRlActorCriticRecurrentCfg` | `ActorCriticRecurrent` | ❌ | ✅ GRU | — |
| `InstinctRlEncoderActorCriticRecurrentCfg` | `EncoderActorCriticRecurrent` | ✅ | ✅ GRU | — |
| `InstinctRlEncoderMoEActorCriticCfg` | `EncoderMoEActorCritic` | ✅ | ❌ | MoE variant |
| `InstinctRlVaeActorCriticCfg` | `VaeActor` | ✅ VAE | ❌ | — |
| `EstimatorActorCriticCfg` | `EstimatorActorCritic` | ✅ estimator | ❌ | — |
| `EstimatorActorCriticRecurrentCfg` | `EstimatorActorCriticRecurrent` | ✅ estimator | ✅ | — |

### Current AMP config uses

```python
class PolicyCfg(InstinctRlEncoderActorCriticCfg):
    actor_hidden_dims = (512, 512, 256, 128)
    critic_hidden_dims = (512, 512, 256, 128)
    activation = "elu"
    encoder_configs = {"depth_encoder": DepthEncoderConv2dCfg()}
    critic_encoder_configs = {"depth_encoder": DepthEncoderConv2dCfg()}
```

Critic has its **own independent** depth encoder (not shared, not None).

---

## Observation Patterns

### Depth: `delayed_visualizable_image` (current default)

```python
"depth_image": ObservationTermCfg(
    func=instinct_envs_mdp.delayed_visualizable_image,
    params={
        "data_type": "distance_to_image_plane_noised_history",
        "sensor_cfg": SceneEntityCfg("camera"),
        "history_skip_frames": 5,         # spacing between output frames
        "num_output_frames": 10,          # total frames in output
        "delayed_frame_ranges": (0, 1),   # random offset in [0,1] from latest
    },
),
```
Output shape: `(N, 10, H, W)`. `Conv2dHead` treats T=10 as input channels → CNN mixes frames across channels (order-insensitive).

### Proprio: `history_length` + `flatten_history_dim`

```python
"joint_pos": ObservationTermCfg(
    func=envs_mdp.joint_pos_rel,
    params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=".*")},
    noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
    history_length=8,            # stack 8 historical steps
    flatten_history_dim=True,    # flatten (N,8*D) not (N,8,D)
),
```
All proprio, commands, and actions use `history_length=8`. AMP observations use `history_length=10`.

### Observation function parameter convention

| Function | Sensor arg | Type |
|----------|:----------:|------|
| `visualizable_image` / `delayed_visualizable_image` | `sensor_cfg` | `SceneEntityCfg("name")` |
| `height_scan` | `sensor_name` | `str` |
| `generated_commands` | `command_name` | `str` |
| `joint_pos_rel` / `joint_vel_rel` | `asset_cfg` | `SceneEntityCfg(name, joint_names=...)` |
| `base_lin_vel` / `base_ang_vel` | `asset_cfg` (optional) | `SceneEntityCfg("robot")` |

---

## Depth Camera Configuration

Current camera on the branch:
```python
NoisyGroupedRayCasterCameraCfg(
    name="camera",
    frame=ObjRef(type="body", name="waist_pitch_link", entity="robot"),
    pattern=PinholeCameraPatternCfg(width=64, height=36, fovy=57.74),
    offset=NoisyGroupedRayCasterCameraCfg.OffsetCfg(
        pos=(0.0958, 0.0475, 0.0095),
        rot=(0.9063078, 0.0, 0.4226183, 0.0),
        convention="world",
    ),
    data_types=["distance_to_image_plane"],
    noise_pipeline={
        "crop_and_resize": CropAndResizeCfg(crop_region=(18, 0, 16, 16)),
        "gaussian_blur": GaussianBlurNoiseCfg(kernel_size=3, sigma=1),
        "depth_normalization": DepthNormalizationCfg(depth_range=(0.0, 2.5), normalize=True),
    },
    data_histories={"distance_to_image_plane_noised": 47},
    min_distance=0.1,
    max_distance=2.5,
),
```
- No explicit `update_period` set → defaults to per-physics-step capture
- `data_histories=47` provides the buffer that `delayed_visualizable_image` reads from
- `data_type` on the sensor is `"distance_to_image_plane"`; the `_noised` and `_history` suffixes are added by the noise pipeline and history buffer

---

## Encoder Configuration

### Conv2dHeadCfg (current parkour default)

```python
class DepthEncoderConv2dCfg(InstinctRlConv2dHeadCfg):
    output_size: int = 128
    channels: list = [4]            # 10 depth frames → 4 channels
    kernel_sizes: list = [3]
    strides: list = [1]
    paddings: list = [1]
    hidden_sizes: list = [256, 256] # MLP after flatten
    nonlinearity: str = "ReLU"
    use_maxpool: bool = True
    component_names: list = ["depth_image"]
    takeout_input_components: bool = True
```

`component_names` selects which obs components this block consumes. `takeout_input_components=True` removes the raw component from downstream (replaced by the encoder output).

### Other available encoder types (register in `ParallelLayer._build_one_block`)

| `class_name` | Model class | Input |
|-------------|------------|-------|
| `MlpModel` | `MlpModel` | Flat vector |
| `Conv2dHeadModel` | `Conv2dHeadModel` | Single image `(C,H,W)` |
| `TransformerHeadModel` | `TransformerHeadModel` | Sequence |
| `StateConditionedDepthTransformerHeadModel` | State-conditioned transformer | depth + state |
| `PerceptiveEncoder` | `PerceptiveEncoder` | depth + dynamics |

---

## AMP Architecture

Current AMP uses `WasabiPPO` (PPO + discriminator):

```
Policy obs (has depth) → Encoder → MLP → action
                                ↓
AMP reference obs → MLP → discriminator → AMP reward

Critic obs (has depth + base_lin_vel) → Encoder → MLP → value
```

The discriminator takes 10-step windows of proprio (no depth):
```python
Discriminator(hidden_sizes=[1024, 512], nonlinearity="ReLU")
```
Policy and reference share the same obs structure (`amp_policy_terms` ↔ `amp_reference_terms`), each with `history_length=10`.

---

## Pitfalls

| Symptom | Fix |
|---------|-----|
| `height_scan() got unexpected keyword argument 'sensor_cfg'` | `height_scan` takes `sensor_name` (str), not `sensor_cfg` (SceneEntityCfg) |
| `visualizable_image` doesn't accept `history_skip_frames` | `visualizable_image` is for single-frame output; use `delayed_visualizable_image` for multi-frame |
| `EncoderActorCritic` doesn't have `rnn_type` | Use `InstinctRlEncoderActorCriticRecurrentCfg` |
| `num_output_frames=1` + `history_skip_frames` has no effect | `frame_offset` arange is `[0]` regardless of skip |
| forgot to register `-Play-v0` variant | Every task needs both `-v0` and `-Play-v0` |

## Training

```bash
# List tasks
python -m instinct_mj.scripts.list_envs

# Train (auto multi-GPU if CUDA_VISIBLE_DEVICES > 1)
python -m instinct_mj.scripts.instinct_rl.train Instinct-Parkour-Adam-Lite-v0 \
  --run-name experiment_name

# Resume
python -m instinct_mj.scripts.instinct_rl.train Instinct-Parkour-Adam-Lite-v0 --resume
```

Logs: `logs/instinct_rl/{experiment_name}/{timestamp}_{run_name}/`
