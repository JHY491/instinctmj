"""Adam Lite foothold continuous-slope: SSR stack with r^f disabled (rho_tilde=0)."""

from __future__ import annotations

import copy
from typing import Any, cast

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.viewer.viewer_config import ViewerConfig

from instinct_mj.tasks.parkour.config.adam_lite.adam_lite_foothold_parkour_cfg import (
    attach_foothold_stack,
)
from instinct_mj.tasks.parkour.config.adam_lite.adam_lite_three_slopes_parkour_cfg import (
    instinct_adam_lite_three_slopes_parkour_env_cfg,
)
from instinct_mj.tasks.parkour.config.parkour_env_cfg import (
    CONTINUOUS_SLOPE_ONLY_TERRAINS_CFG,
    CONTINUOUS_SLOPE_ONLY_TERRAINS_CFG_PLAY,
)
from instinct_mj.tasks.parkour.mdp.commands import PoseVelocityCommandCfg


_CONTINUOUS_SLOPE_VELOCITY_RANGES = {
    "perlin_rough_stand": {
        "lin_vel_x": (0.0, 0.0),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (-1.0, 1.0),
    },
    "continuous_slope": {
        "lin_vel_x": (0.3, 2.0),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (-1.0, 1.0),
    },
}


def instinct_adam_lite_foothold_continuous_slope_env_cfg(
    *,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Three-slopes 底座裁成 continuous_slope only，再挂 foothold 栈（引导由算法侧关闭）。"""
    cfg = instinct_adam_lite_three_slopes_parkour_env_cfg(play=play)
    setattr(cfg, "run_name", "AdamLiteFootholdContinuousSlope")

    terrain_gen = copy.deepcopy(
        CONTINUOUS_SLOPE_ONLY_TERRAINS_CFG_PLAY if play else CONTINUOUS_SLOPE_ONLY_TERRAINS_CFG
    )
    cfg.scene.terrain.terrain_generator = terrain_gen
    cfg.scene.terrain.virtual_obstacles = {}
    cfg.scene.terrain.virtual_obstacle_hfield_height_threshold = None

    base_velocity_cmd = cast(PoseVelocityCommandCfg, cfg.commands["base_velocity"])
    base_velocity_cmd.velocity_ranges = copy.deepcopy(_CONTINUOUS_SLOPE_VELOCITY_RANGES)
    base_velocity_cmd.random_velocity_terrain = ["perlin_rough_stand"]
    # continuous_slope 用普通 FlatPatchSampling，无需序列推进。
    base_velocity_cmd.flat_patch_advance_on_arrival = False

    # 无 CReF 项；drop 空元组即可。
    return attach_foothold_stack(cfg, drop_reward_names=())


def instinct_adam_lite_foothold_continuous_slope_final_cfg(
    *,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    cfg = instinct_adam_lite_foothold_continuous_slope_env_cfg(play=play)
    if play:
        terrain_generator = cast(Any, cfg.scene.terrain.terrain_generator)
        terrain_generator.num_rows = 10
        terrain_generator.num_cols = 20
        cfg.viewer = ViewerConfig(
            lookat=(0.0, 0.75, 0.0),
            distance=4.123105625617661,
            elevation=-14.036243467926479,
            azimuth=180.0,
            origin_type=ViewerConfig.OriginType.WORLD,
            entity_name=None,
        )
    return cfg
