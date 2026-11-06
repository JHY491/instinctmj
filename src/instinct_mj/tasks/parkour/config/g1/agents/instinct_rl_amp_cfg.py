"""Instinct-RL AMP config for G1 plum-blossom foothold parkour.

算法栈 = Adam Lite Foothold-Mixed 的三合一（SSR imagined foothold 引导 r^f +
AMP 风格判别器 + 速度估计头监督，terrain_level_gate=3），但 actor 为单头
（无 MoE）：DeepCNN 深度编码 + HIMLoco 式 base_lin_vel 估计器。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from instinct_mj.rl import (
    InstinctRlActorCriticCfg,
    InstinctRlNormalizerCfg,
    InstinctRlOnPolicyRunnerCfg,
    InstinctRlPpoAlgorithmCfg,
)
from instinct_mj.tasks.config.rl_utils import default_policy_critic_normalizers
from instinct_mj.tasks.parkour.config.adam_lite.agents.instinct_rl_amp_cfg import (
    AdamSPParkourVisionAmpRunnerCfg,
    AmpAlgoCfg,
    FootholdMixedAmpAlgoCfg,
    FootholdVelEstDeepVisionPolicyCfg,
)


@dataclass(kw_only=True)
class G1FootholdVisionPolicyCfg(FootholdVelEstDeepVisionPolicyCfg):
    # Smaller initial joint-target noise for G1's narrow ankle ranges.
    init_noise_std: float = 0.5


@dataclass(kw_only=True)
class G1FootholdAmpAlgoCfg(FootholdMixedAmpAlgoCfg):
    terrain_level_gate: int = 3
    # G1 collision sole is about 20 x 7 cm, centered 4 cm ahead of the ankle.
    # Rows are lateral and columns longitudinal, at 2.5 cm resolution.
    sole_grid: tuple[int, int] = (3, 9)
    entropy_coef: float = 0.002


@dataclass(kw_only=True)
class G1PlumBlossomFootholdRunnerCfg(AdamSPParkourVisionAmpRunnerCfg):
    experiment_name: str = "g1_plum_blossom_foothold"
    policy: InstinctRlActorCriticCfg = field(default_factory=lambda: G1FootholdVisionPolicyCfg())
    algorithm: InstinctRlPpoAlgorithmCfg = field(default_factory=lambda: G1FootholdAmpAlgoCfg())
    normalizers: dict[str, InstinctRlNormalizerCfg] = field(default_factory=lambda: default_policy_critic_normalizers())


def g1_plum_blossom_foothold_amp_runner_cfg() -> InstinctRlOnPolicyRunnerCfg:
    """G1 梅花桩 foothold 任务 runner（Mixed 算法栈，单 actor 无 MoE）。"""
    return G1PlumBlossomFootholdRunnerCfg()


@dataclass(kw_only=True)
class G1CrefAmpAlgoCfg(AmpAlgoCfg):
    entropy_coef: float = 0.002
    advantage_mixing_weights: tuple[float, ...] = (1.0, 0.25, 1.0)
    auxiliary_reward_per_env_reward_coefs_by_key: dict[str, tuple[float, ...]] = field(
        default_factory=lambda: {"discriminator_reward": (0.0, 0.0, 1.0)}
    )


@dataclass(kw_only=True)
class G1PlumBlossomCrefRunnerCfg(G1PlumBlossomFootholdRunnerCfg):
    experiment_name: str = "g1_plum_blossom_cref"
    algorithm: InstinctRlPpoAlgorithmCfg = field(default_factory=G1CrefAmpAlgoCfg)
