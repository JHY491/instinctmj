"""Adam Lite foothold parkour: 楼梯任务 + SSR imagined foothold guidance 的环境侧改动。"""

from __future__ import annotations

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


_FOOTHOLD_REGION_RES = 0.025
# This ray grid is a terrain cache around the current foot, not the support
# patch itself. SSR evaluates each imagined point with the sole-sized patch
# configured as sole_grid=(5, 10) in FootholdWasabiPpoAlgorithmCfg.
_FOOTHOLD_TERRAIN_CACHE_SIZE = (0.5, 0.3)

# SSR Table 12: H^b is 18×10, H^f is 2×10×5 (sole 22.5cm×10cm @ 2.5cm).
_SSR_BASE_HEIGHTMAP_SIZE = (1.7, 0.9)
_SSR_BASE_HEIGHTMAP_RES = 0.1
_SSR_FOOT_HEIGHTMAP_SIZE = (0.225, 0.10)
_SSR_FOOT_HEIGHTMAP_RES = 0.025


def attach_foothold_stack(
    cfg: ManagerBasedRlEnvCfg,
    *,
    drop_reward_names: tuple[str, ...] = ("cref_stair_foothold",),
) -> ManagerBasedRlEnvCfg:
    """挂 region scanner、foothold 观测组、SSR critic privileged、三组 reward。

    Critic 对齐 SSR Sec 3.1 / Table 12：proprio + v_t + v^f + c + p^h + p^f + H^b + H^f。
    不向 critic 喂 depth（depth 仅给 actor）。
    """
    cfg.observations["policy"].terms["depth_image"].params["num_output_frames"] = 10
    depth_params = cfg.observations["policy"].terms["depth_image"].params
    depth_history_length = (
        (depth_params["num_output_frames"] - 1) * depth_params["history_skip_frames"]
        + 1
        + depth_params["delayed_frame_ranges"][1]
    )
    for sensor_cfg in cfg.scene.sensors:
        if sensor_cfg.name == "camera":
            # SSR 论文对齐: depth image 36×36，64×36 居中裁切 (top, bottom, left, right)
            sensor_cfg.noise_pipeline["crop_and_resize"].crop_region = (0, 0, 14, 14)
            sensor_cfg.data_histories["distance_to_image_plane_noised"] = depth_history_length
        elif sensor_cfg.name == "base_heightmap_scanner":
            sensor_cfg.pattern = GridPatternCfg(
                resolution=_SSR_BASE_HEIGHTMAP_RES,
                size=_SSR_BASE_HEIGHTMAP_SIZE,
            )

    # Critic keeps privileged height maps; drop any deploy-time depth copy if present.
    cfg.observations["critic"].terms.pop("depth_image", None)
    if "height_map_priv" not in cfg.observations["critic"].terms:
        cfg.observations["critic"].terms["height_map_priv"] = ObservationTermCfg(
            func=envs_mdp.height_scan,
            params={"sensor_name": "base_heightmap_scanner"},
        )

    ankle = SceneEntityCfg("robot", body_names=("left_ankle_roll_link", "right_ankle_roll_link"))
    # Adam Lite 无 wrist；用肘端作为 SSR p^h 的远端臂位置代理。
    elbow = SceneEntityCfg("robot", body_names=("left_elbow_link", "right_elbow_link"))

    region_scanners = (
        RayCastSensorCfg(
            name="left_foot_region_scanner",
            frame=ObjRef(type="site", name="left_foot_scan", entity="robot"),
            pattern=GridPatternCfg(resolution=_FOOTHOLD_REGION_RES, size=_FOOTHOLD_TERRAIN_CACHE_SIZE),
            ray_alignment="yaw",
            max_distance=10.0,
            debug_vis=False,
        ),
        RayCastSensorCfg(
            name="right_foot_region_scanner",
            frame=ObjRef(type="site", name="right_foot_scan", entity="robot"),
            pattern=GridPatternCfg(resolution=_FOOTHOLD_REGION_RES, size=_FOOTHOLD_TERRAIN_CACHE_SIZE),
            ray_alignment="yaw",
            max_distance=10.0,
            debug_vis=False,
        ),
        RayCastSensorCfg(
            name="left_foot_heightmap_scanner",
            frame=ObjRef(type="site", name="left_foot_scan", entity="robot"),
            pattern=GridPatternCfg(resolution=_SSR_FOOT_HEIGHTMAP_RES, size=_SSR_FOOT_HEIGHTMAP_SIZE),
            ray_alignment="yaw",
            max_distance=10.0,
            debug_vis=False,
        ),
        RayCastSensorCfg(
            name="right_foot_heightmap_scanner",
            frame=ObjRef(type="site", name="right_foot_scan", entity="robot"),
            pattern=GridPatternCfg(resolution=_SSR_FOOT_HEIGHTMAP_RES, size=_SSR_FOOT_HEIGHTMAP_SIZE),
            ray_alignment="yaw",
            max_distance=10.0,
            debug_vis=False,
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
        # 1 on stairs/gap/flat；0 on continuous_slope（SSR 斜坡关 r^f）
        "foothold_guidance_active": ObservationTermCfg(
            func=parkour_mdp.foothold_guidance_active,
            params={"disable_terrain_names": ("continuous_slope",)},
        ),
        "last_action": ObservationTermCfg(func=envs_mdp.last_action),
    }
    cfg.observations["foothold"] = ObservationGroupCfg(
        terms=foothold_terms,
        concatenate_terms=False,
        enable_corruption=False,
    )
    # SSR critic privileged extras (proprio + base_lin_vel + H^b already present).
    cfg.observations["critic"].terms.update(
        {
            "foot_lin_vel_b": ObservationTermCfg(
                func=parkour_mdp.link_lin_vel_b_flat,
                params={"asset_cfg": ankle},
            ),
            "foot_contact": ObservationTermCfg(
                func=parkour_mdp.foot_contact,
                params={"sensor_name": "contact_forces"},
            ),
            "hand_pos_b": ObservationTermCfg(
                func=parkour_mdp.link_pos_b_flat,
                params={"asset_cfg": elbow},
            ),
            "foot_pos_b": ObservationTermCfg(
                func=parkour_mdp.link_pos_b_flat,
                params={"asset_cfg": ankle},
            ),
            "height_map_foot_l": ObservationTermCfg(
                func=envs_mdp.height_scan,
                params={"sensor_name": "left_foot_heightmap_scanner"},
            ),
            "height_map_foot_r": ObservationTermCfg(
                func=envs_mdp.height_scan,
                params={"sensor_name": "right_foot_heightmap_scanner"},
            ),
        }
    )
    # Drop pre-SSR critic-only sole helpers if any leftover from older stacks.
    cfg.observations["critic"].terms.pop("foot_sole_z", None)
    cfg.observations["critic"].terms.pop("foot_vel_b", None)

    # Three reward groups match SSR: locomotion task, foothold guidance, and AMP style.
    # Drop analytic CReF — foothold guidance comes only from SSR imagined foothold r^f.
    drop = set(drop_reward_names)
    task_rewards = {name: term for name, term in dict(cfg.rewards).items() if name not in drop}
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


def instinct_adam_lite_foothold_parkour_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    """楼梯 env + foothold 传感器/观测组。"""
    cfg = instinct_adam_lite_parkour_env_cfg(play=play)
    setattr(cfg, "run_name", "AdamLiteFoothold")
    return attach_foothold_stack(cfg)


def instinct_adam_lite_foothold_deep_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Foothold VelEst + SSR 对齐深度 CNN 的 env（观测与 velest 一致，run_name 区分实验）。"""
    from instinct_mj.tasks.parkour.config.adam_lite.adam_lite_foothold_velest_parkour_cfg import (
        instinct_adam_lite_foothold_velest_env_cfg,
    )

    cfg = instinct_adam_lite_foothold_velest_env_cfg(play=play)
    setattr(cfg, "run_name", "AdamLiteFootholdDeepCNN")
    return cfg


def instinct_adam_lite_foothold_deep_final_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = instinct_adam_lite_foothold_deep_env_cfg(play=play)
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


def instinct_adam_lite_foothold_parkour_final_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = instinct_adam_lite_foothold_parkour_env_cfg(play=play)
    if play:
        # 父级 play 会把 num_rows 缩到 4（level 最高 3），刚好卡在 foothold gate=3 边界；
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
