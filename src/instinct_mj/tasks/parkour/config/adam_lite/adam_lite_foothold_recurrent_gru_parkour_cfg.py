"""Adam Lite foothold recurrent GRU parkour: single-frame obs + cross-step encoder memory."""

from __future__ import annotations

from typing import Any, cast

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.viewer.viewer_config import ViewerConfig

from instinct_mj.tasks.parkour.config.adam_lite.adam_lite_foothold_parkour_cfg import (
    instinct_adam_lite_foothold_parkour_env_cfg,
)

_RECURRENT_PROPRIO_TERM_NAMES = (
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos",
    "joint_vel",
    "actions",
)


def _depth_history_length(depth_term: Any) -> int:
    depth_params = depth_term.params
    return (
        (depth_params["num_output_frames"] - 1) * depth_params["history_skip_frames"]
        + 1
        + depth_params["delayed_frame_ranges"][1]
    )


def configure_recurrent_gru_policy_observations(cfg: ManagerBasedRlEnvCfg) -> None:
    """Single-frame proprio + latest depth; temporal context lives in the recurrent encoder."""
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

    for term_name in _RECURRENT_PROPRIO_TERM_NAMES:
        policy_terms[term_name].history_length = 1
        policy_terms[term_name].flatten_history_dim = True


def instinct_adam_lite_foothold_recurrent_gru_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Foothold env with single-frame policy obs for cross-step recurrent GRU encoder."""
    cfg = instinct_adam_lite_foothold_parkour_env_cfg(play=play)
    setattr(cfg, "run_name", "AdamLiteFootholdGRURecurrent")
    configure_recurrent_gru_policy_observations(cfg)
    return cfg


def instinct_adam_lite_foothold_recurrent_gru_final_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = instinct_adam_lite_foothold_recurrent_gru_env_cfg(play=play)
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
