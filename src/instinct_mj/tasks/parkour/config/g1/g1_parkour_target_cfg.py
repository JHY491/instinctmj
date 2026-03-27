"""G1 parkour task config without AMP-specific components."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.viewer.viewer_config import ViewerConfig

from .g1_parkour_target_amp_cfg import instinct_g1_parkour_amp_env_cfg


def instinct_g1_parkour_env_cfg(
    *,
    play: bool = False,
    shoe: bool = True,
) -> ManagerBasedRlEnvCfg:
    """Build the base G1 parkour environment configuration without AMP."""
    cfg = instinct_g1_parkour_amp_env_cfg(play=play, shoe=shoe)

    cfg.scene.sensors = tuple(sensor_cfg for sensor_cfg in cfg.scene.sensors if sensor_cfg.name != "motion_reference")
    cfg.observations.pop("amp_policy", None)
    cfg.observations.pop("amp_reference", None)
    cfg.terminations.pop("dataset_exhausted", None)
    cfg.run_name = "G1ParkourTarget" + ("_shoe" if shoe else "")

    return cfg


def instinct_g1_parkour_final_cfg(
    *,
    play: bool = False,
    shoe: bool = True,
) -> ManagerBasedRlEnvCfg:
    """Create the final G1 parkour env configuration without AMP."""
    cfg = instinct_g1_parkour_env_cfg(play=play, shoe=shoe)

    if play:
        cfg.viewer = ViewerConfig(
            lookat=(0.0, 0.75, 0.0),
            distance=4.123105625617661,
            elevation=-14.036243467926479,
            azimuth=180.0,
            origin_type=ViewerConfig.OriginType.WORLD,
            entity_name=None,
        )
        cfg.viewer.origin_type = ViewerConfig.OriginType.WORLD
        cfg.viewer.entity_name = None
        cfg.viewer.body_name = None

    return cfg
