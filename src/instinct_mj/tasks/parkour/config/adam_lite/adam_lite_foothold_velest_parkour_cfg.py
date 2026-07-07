"""Adam Lite foothold stage-2: imagined foothold guidance + MLP velocity estimator."""

from __future__ import annotations

import mjlab.envs.mdp as envs_mdp
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import ObservationTermCfg

from instinct_mj.tasks.parkour.config.adam_lite.adam_lite_foothold_parkour_cfg import (
    instinct_adam_lite_foothold_parkour_final_cfg,
)


_VEL_EST_TARGET_NAME = "base_lin_vel_est"


def _base_lin_vel_est_term() -> ObservationTermCfg:
    return ObservationTermCfg(
        func=envs_mdp.base_lin_vel,
        history_length=1,
        flatten_history_dim=True,
        noise=None,
    )


def instinct_adam_lite_foothold_velest_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = instinct_adam_lite_foothold_parkour_final_cfg(play=play)
    setattr(cfg, "run_name", "AdamLiteFootholdVelEst")

    # The actor receives this slot only after EncoderVelActorCritic replaces it with
    # the MLP velocity estimate; the critic copy is the supervised target.
    cfg.observations["policy"].terms[_VEL_EST_TARGET_NAME] = _base_lin_vel_est_term()
    cfg.observations["critic"].terms[_VEL_EST_TARGET_NAME] = _base_lin_vel_est_term()
    return cfg


def instinct_adam_lite_foothold_velest_final_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    return instinct_adam_lite_foothold_velest_env_cfg(play=play)
