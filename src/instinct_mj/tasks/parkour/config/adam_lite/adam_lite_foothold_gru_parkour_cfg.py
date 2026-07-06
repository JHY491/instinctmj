"""Adam Lite foothold GRU parkour: foothold baseline + SSR-style depth CNN + proprio GRU encoder obs."""

from __future__ import annotations

import copy
from typing import Any, cast

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.viewer.viewer_config import ViewerConfig

from instinct_mj.tasks.parkour.config.adam_lite.adam_lite_foothold_parkour_cfg import (
    instinct_adam_lite_foothold_parkour_env_cfg,
)

_GRU_PROPRIO_HISTORY_LENGTH = 8
_GRU_PROPRIO_TERM_MAP = (
    ("base_ang_vel", "gru_base_ang_vel"),
    ("projected_gravity", "gru_projected_gravity"),
    ("velocity_commands", "gru_velocity_commands"),
    ("joint_pos", "gru_joint_pos"),
    ("joint_vel", "gru_joint_vel"),
    ("actions", "gru_actions"),
)


def _depth_history_length(depth_term: Any) -> int:
    depth_params = depth_term.params
    return (
        (depth_params["num_output_frames"] - 1) * depth_params["history_skip_frames"]
        + 1
        + depth_params["delayed_frame_ranges"][1]
    )


def configure_gru_policy_observations(cfg: ManagerBasedRlEnvCfg) -> None:
    """Add GRU proprio history terms and switch policy depth to a single latest frame."""
    policy_terms = cfg.observations["policy"].terms
    critic_terms = cfg.observations["critic"].terms

    depth_term = policy_terms["depth_image"]
    depth_term.params["num_output_frames"] = 1
    depth_term.params["history_skip_frames"] = 1

    depth_history_length = _depth_history_length(depth_term)
    if "depth_image" in critic_terms:
        depth_history_length = max(depth_history_length, _depth_history_length(critic_terms["depth_image"]))
    for sensor_cfg in cfg.scene.sensors:
        if sensor_cfg.name == "camera":
            sensor_cfg.data_histories["distance_to_image_plane_noised"] = depth_history_length

    for src_name, gru_name in _GRU_PROPRIO_TERM_MAP:
        gru_term = copy.deepcopy(policy_terms[src_name])
        gru_term.history_length = _GRU_PROPRIO_HISTORY_LENGTH
        gru_term.flatten_history_dim = False
        policy_terms[gru_name] = gru_term


def instinct_adam_lite_foothold_gru_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Foothold env with GRU-specific policy observations layered on top."""
    cfg = instinct_adam_lite_foothold_parkour_env_cfg(play=play)
    setattr(cfg, "run_name", "AdamLiteFootholdGRU")
    configure_gru_policy_observations(cfg)
    return cfg


def instinct_adam_lite_foothold_gru_final_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = instinct_adam_lite_foothold_gru_env_cfg(play=play)
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
