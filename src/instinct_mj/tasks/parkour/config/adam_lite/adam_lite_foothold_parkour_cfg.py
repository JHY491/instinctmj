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
