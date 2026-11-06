"""G1 plum-blossom foothold parkour: 梅花桩任务 + SSR imagined foothold guidance。"""

from __future__ import annotations

from typing import Any, cast

import mjlab.envs.mdp as envs_mdp
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import ObservationTermCfg, SceneEntityCfg
from mjlab.viewer.viewer_config import ViewerConfig

from instinct_mj.tasks.parkour.config.adam_lite.adam_lite_foothold_parkour_cfg import (
    attach_foothold_stack,
)
from instinct_mj.tasks.parkour.config.g1.g1_plum_blossom_parkour_cfg import (
    instinct_g1_plum_blossom_parkour_env_cfg,
)

_VEL_EST_TARGET_NAME = "base_lin_vel_est"


def _base_lin_vel_est_term() -> ObservationTermCfg:
    return ObservationTermCfg(
        func=envs_mdp.base_lin_vel,
        history_length=1,
        flatten_history_dim=True,
        noise=None,
    )


def instinct_g1_plum_blossom_foothold_env_cfg(*, play: bool = False, use_cref: bool = False) -> ManagerBasedRlEnvCfg:
    """梅花桩 env + foothold 传感器/观测组。"""
    cfg = instinct_g1_plum_blossom_parkour_env_cfg(play=play)
    setattr(cfg, "run_name", "G1PlumBlossomFoothold")
    cfg = attach_foothold_stack(
        cfg,
        drop_reward_names=() if use_cref else ("cref_stepping_stone_foothold",),
        # G1 有完整腕链；SSR p^h 用腕末端而非肘（Adam Lite 无 wrist 才用肘端）。
        hand_body_names=("left_wrist_yaw_link", "right_wrist_yaw_link"),
    )
    # G1's scan sites are 4 cm ahead of the ankle. Use the fixed sole-center
    # bodies for both imagination inputs and touchdown targets, so mu and the
    # terrain cache share an origin. Their world z already is the sole plane;
    # the shared Adam default (ankle z - 6.5 cm) is invalid for G1.
    sole = SceneEntityCfg("robot", body_names=("LL_FOOT", "LR_FOOT"))
    if use_cref:
        reward_name = "cref_stepping_stone_foothold"
        cref = cfg.rewards["task"].pop(reward_name)
        # Preserve the original task-channel scale under the foothold advantage weight (0.25).
        cref.weight = 6.0
        cref.params.update(
            asset_cfg=sole,
            foot_height_offset=0.0,
            sole_scanner_names=("left_foot_sole_scanner", "right_foot_sole_scanner"),
        )
        cfg.rewards["foothold"] = {reward_name: cref}
        cfg.observations.pop("foothold")
        cfg.scene.sensors = tuple(sensor for sensor in cfg.scene.sensors if not sensor.name.endswith("_region_scanner"))
    else:
        foothold_terms = cfg.observations["foothold"].terms
        for name in ("foot_pos_b", "foot_pos_w", "foot_vel_b", "foot_sole_z"):
            foothold_terms[name].params["asset_cfg"] = sole
        foothold_terms["foot_sole_z"].params["foot_height_offset"] = 0.0
    # HIMLoco 式速度估计槽（同 Adam foothold-velest）：
    # - policy 侧该槽在 act() 时被 EncoderVelActorCritic 用 MLP 估计值覆盖（真值不泄漏给策略）；
    # - critic 侧保留真值，作为 estimator_loss 的监督 target（target_from_critic=True）。
    cfg.observations["policy"].terms[_VEL_EST_TARGET_NAME] = _base_lin_vel_est_term()
    cfg.observations["critic"].terms[_VEL_EST_TARGET_NAME] = _base_lin_vel_est_term()
    return cfg


def instinct_g1_plum_blossom_foothold_final_cfg(*, play: bool = False, use_cref: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = instinct_g1_plum_blossom_foothold_env_cfg(play=play, use_cref=use_cref)
    if play:
        # 父级 play 会把 num_rows 缩到 4（难度只有 row 0-3）；恢复课程全 10 行，
        # 才能在 play 里评测后期梅花桩难度（stake 收窄到 0.3m、孔深 -0.3m）。
        terrain_generator = cast(Any, cfg.scene.terrain.terrain_generator)
        terrain_generator.num_rows = 10
        cfg.viewer = ViewerConfig(
            lookat=(0.0, 0.75, 0.0),
            distance=4.123105625617661,
            elevation=-14.036243467926479,
            azimuth=180.0,
            origin_type=ViewerConfig.OriginType.WORLD,
            entity_name=None,
        )
    return cfg
