"""Adam Lite foothold parkour: 楼梯任务 + SSR imagined foothold guidance 的环境侧改动。"""

from __future__ import annotations

import copy
from typing import Any, cast

import instinct_mj.tasks.parkour.mdp as parkour_mdp
import mjlab.envs.mdp as envs_mdp
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.viewer.viewer_config import ViewerConfig

from instinct_mj.tasks.parkour.config.adam_lite.adam_lite_parkour_cfg import (
    instinct_adam_lite_parkour_env_cfg,
)


def instinct_adam_lite_foothold_parkour_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    """楼梯 env + foothold 传感器/观测组。"""
    cfg = instinct_adam_lite_parkour_env_cfg(play=play)
    setattr(cfg, "run_name", "AdamLiteFoothold")
    cfg.observations["policy"].terms["depth_image"].params["num_output_frames"] = 10
    depth_params = cfg.observations["policy"].terms["depth_image"].params
    depth_history_length = (
        (depth_params["num_output_frames"] - 1) * depth_params["history_skip_frames"]
        + 1
        + depth_params["delayed_frame_ranges"][1]
    )
    for sensor_cfg in cfg.scene.sensors:
        if sensor_cfg.name == "camera":
            sensor_cfg.data_histories["distance_to_image_plane_noised"] = depth_history_length
    cfg.observations["critic"].terms = {
        term_name: term_cfg
        for term_name, term_cfg in cfg.observations["critic"].terms.items()
        if term_name != "height_map_priv"
    }
    cfg.observations["critic"].terms["depth_image"] = copy.deepcopy(cfg.observations["policy"].terms["depth_image"])

    ankle = SceneEntityCfg("robot", body_names=("left_ankle_roll_link", "right_ankle_roll_link"))

    region_scanners = (
        RayCastSensorCfg(
            name="left_foot_region_scanner",
            frame=ObjRef(type="site", name="left_foot_scan", entity="robot"),
            pattern=GridPatternCfg(resolution=0.025, size=(0.5, 0.3)),
            ray_alignment="yaw",
            max_distance=10.0,
            debug_vis=play,
        ),
        RayCastSensorCfg(
            name="right_foot_region_scanner",
            frame=ObjRef(type="site", name="right_foot_scan", entity="robot"),
            pattern=GridPatternCfg(resolution=0.025, size=(0.5, 0.3)),
            ray_alignment="yaw",
            max_distance=10.0,
            debug_vis=play,
        ),
    )
    cfg.scene.sensors = tuple(cfg.scene.sensors) + region_scanners

    foothold_terms = {
        "foot_region_scan_l": ObservationTermCfg(
            func=parkour_mdp.foot_region_hit_z,
            params={"sensor_name": "left_foot_region_scanner"},
        ),
        "foot_region_scan_r": ObservationTermCfg(
            func=parkour_mdp.foot_region_hit_z,
            params={"sensor_name": "right_foot_region_scanner"},
        ),
        "foot_contact": ObservationTermCfg(
            func=parkour_mdp.foot_contact,
            params={"sensor_name": "contact_forces"},
        ),
        "foot_pos_b": ObservationTermCfg(
            func=parkour_mdp.foot_pos_b,
            params={"asset_cfg": ankle},
        ),
        "foot_sole_z": ObservationTermCfg(
            func=parkour_mdp.foot_sole_z,
            params={"asset_cfg": ankle},
        ),
        "foot_vel_b": ObservationTermCfg(
            func=parkour_mdp.foot_vel_b,
            params={"asset_cfg": ankle},
        ),
        "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
        "foot_pos_w": ObservationTermCfg(
            func=parkour_mdp.foot_pos_w_xy,
            params={"asset_cfg": ankle},
        ),
        "base_pose_w": ObservationTermCfg(func=parkour_mdp.base_xy_yaw_w),
        "terrain_level": ObservationTermCfg(func=parkour_mdp.terrain_level),
        "last_action": ObservationTermCfg(func=envs_mdp.last_action),
    }
    cfg.observations["foothold"] = ObservationGroupCfg(
        terms=foothold_terms,
        concatenate_terms=False,
        enable_corruption=False,
    )
    cfg.observations["critic"].terms.update(
        {
            "foot_contact": ObservationTermCfg(
                func=parkour_mdp.foot_contact,
                params={"sensor_name": "contact_forces"},
            ),
            "foot_pos_b": ObservationTermCfg(
                func=parkour_mdp.foot_pos_b,
                params={"asset_cfg": ankle},
            ),
            "foot_sole_z": ObservationTermCfg(
                func=parkour_mdp.foot_sole_z,
                params={"asset_cfg": ankle},
            ),
            "foot_vel_b": ObservationTermCfg(
                func=parkour_mdp.foot_vel_b,
                params={"asset_cfg": ankle},
            ),
        }
    )

    # Three reward groups match SSR: locomotion task, foothold guidance, and AMP style.
    # Keep the analytic CReF stair shaping (301cefc baseline) as the dominant, always-on
    # foot-placement signal; SSR imagined-foothold guidance is layered on additively via the
    # second critic so the task can only improve over, not regress from, that baseline.
    task_rewards = dict(cfg.rewards)
    cfg.rewards = {
        "task": task_rewards,
        "foothold": {
            "foothold_anchor": RewardTermCfg(func=envs_mdp.is_alive, weight=0.0),
        },
        "style": {
            "style_anchor": RewardTermCfg(func=envs_mdp.is_alive, weight=0.0),
        },
    }

    return cfg


def instinct_adam_lite_foothold_parkour_final_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = instinct_adam_lite_foothold_parkour_env_cfg(play=play)
    if play:
        # 父级 play 会把 num_rows 缩到 4（level 最高 3），达不到 SSR foothold gate=5。
        # 与训练 STAIRS_ONLY_TERRAINS_CFG 对齐，才能在 play 里评测后期楼梯难度。
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
