"""Adam Lite foothold mixed: VelEst-DeepCNN env stack + mixed terrain only."""

from __future__ import annotations

import copy
from typing import Any, cast

import instinct_mj.tasks.parkour.mdp as parkour_mdp
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import SceneEntityCfg, TerminationTermCfg
from mjlab.viewer.viewer_config import ViewerConfig

from instinct_mj.tasks.parkour.config.adam_lite.adam_lite_foothold_velest_parkour_cfg import (
    instinct_adam_lite_foothold_velest_env_cfg,
)
from instinct_mj.tasks.parkour.config.parkour_env_cfg import (
    FOOTHOLD_MIXED_TERRAINS_CFG,
    FOOTHOLD_MIXED_TERRAINS_CFG_PLAY,
)
from instinct_mj.tasks.parkour.mdp.commands import PoseVelocityCommandCfg
from instinct_mj.terrains.virtual_obstacle.edge_cylinder_cfg import GreedyconcatEdgeCylinderCfg


_FOOTHOLD_MIXED_VELOCITY_RANGES = {
    "perlin_rough_stand": {
        "lin_vel_x": (0.0, 0.0),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (-1.0, 1.0),
    },
    "perlin_rough": {
        "lin_vel_x": (0.3, 1.0),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (-0.5, 0.5),
    },
    "continuous_slope": {
        "lin_vel_x": (0.3, 2.0),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (-1.0, 1.0),
    },
    "continuous_box_gaps": {
        "lin_vel_x": (0.3, 1.0),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (-0.5, 0.5),
    },
    "one_sided_stairs": {
        "lin_vel_x": (0.3, 1.0),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (-0.5, 0.5),
    },
}


def instinct_adam_lite_foothold_mixed_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Same as VelEst-DeepCNN env (incl. base_lin_vel_est); only terrain/commands/pit differ."""
    cfg = instinct_adam_lite_foothold_velest_env_cfg(play=play)
    setattr(cfg, "run_name", "AdamLiteFootholdMixed")

    terrain_gen = copy.deepcopy(FOOTHOLD_MIXED_TERRAINS_CFG_PLAY if play else FOOTHOLD_MIXED_TERRAINS_CFG)
    cfg.scene.terrain.terrain_generator = terrain_gen

    virtual_obstacles = {}
    if not play:
        virtual_obstacles["edges"] = GreedyconcatEdgeCylinderCfg(
            cylinder_radius=0.03,
            min_points=2,
            component_workers=0,
            merge_collinear_gap=0.09,
            merge_collinear_angle_threshold=30.0,
            merge_collinear_line_distance=0.04,
        )
    cfg.scene.terrain.virtual_obstacles = virtual_obstacles
    cfg.scene.terrain.virtual_obstacle_hfield_height_threshold = None if play else 0.04

    base_velocity_cmd = cast(PoseVelocityCommandCfg, cfg.commands["base_velocity"])
    base_velocity_cmd.velocity_ranges = copy.deepcopy(_FOOTHOLD_MIXED_VELOCITY_RANGES)
    base_velocity_cmd.random_velocity_terrain = ["perlin_rough_stand"]
    base_velocity_cmd.flat_patch_advance_on_arrival = True
    base_velocity_cmd.flat_patch_arrival_xy_dist = 0.4
    base_velocity_cmd.flat_patch_arrival_hold_steps = 3

    cfg.terminations["feet_on_pit_bottom"] = TerminationTermCfg(
        func=parkour_mdp.feet_on_pit_bottom,
        params={
            "contact_sensor_name": "contact_forces",
            "left_sole_scanner_name": "left_foot_sole_scanner",
            "right_sole_scanner_name": "right_foot_sole_scanner",
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
            ),
            "foot_height_offset": 0.065,
            "pit_ground_height_max": -0.04,
            "active_terrain_names": ("continuous_box_gaps",),
        },
    )
    return cfg


def instinct_adam_lite_foothold_mixed_final_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = instinct_adam_lite_foothold_mixed_env_cfg(play=play)
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
