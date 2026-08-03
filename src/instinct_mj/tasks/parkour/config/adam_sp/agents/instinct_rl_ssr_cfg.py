"""Instinct-RL SSR config for Adam SP stairs parkour: EncoderActorCritic + FootholdWasabiPPO."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from instinct_mj.rl import (
    InstinctRlActorCriticCfg,
    InstinctRlConv2dHeadCfg,
    InstinctRlEncoderActorCriticCfg,
    InstinctRlNormalizerCfg,
    InstinctRlOnPolicyRunnerCfg,
    InstinctRlPpoAlgorithmCfg,
)
from instinct_mj.tasks.config.rl_utils import default_policy_critic_normalizers


@dataclass(kw_only=True)
class DepthEncoderConv2dCfg(InstinctRlConv2dHeadCfg):
    output_size: int | None = 128
    channels: list = field(default_factory=lambda: [4])
    kernel_sizes: list = field(default_factory=lambda: [3])
    strides: list = field(default_factory=lambda: [1])
    hidden_sizes: list = field(default_factory=lambda: [256, 256])
    paddings: list = field(default_factory=lambda: [1])
    nonlinearity: str | None = "ReLU"
    use_maxpool: bool = True
    component_names: list = field(default_factory=lambda: ["depth_image"])


@dataclass(kw_only=True)
class PolicyCfg(InstinctRlEncoderActorCriticCfg):
    init_noise_std: float = 1.0
    actor_hidden_dims: tuple[int, ...] = (512, 512, 256, 128)
    critic_hidden_dims: tuple[int, ...] = (512, 512, 256, 128)
    activation: str = "elu"
    encoder_configs: dict[str, Any] | None = field(
        default_factory=lambda: {"depth_encoder": DepthEncoderConv2dCfg()}
    )
    critic_encoder_configs: dict[str, Any] | str | None = field(
        default_factory=lambda: {"depth_encoder": DepthEncoderConv2dCfg()}
    )


@dataclass(kw_only=True)
class SsrAlgoCfg(InstinctRlPpoAlgorithmCfg):
    """AMP + imagined foothold guidance (SSR Sec. 3.2), without Sec. 3.3 symmetry."""

    class_name: str = "FootholdWasabiPPO"

    # Wasabi / AMP
    actor_state_key: str | None = "amp_policy"
    reference_state_key: str | None = "amp_reference"
    discriminator_kwargs: dict[str, Any] | None = field(
        default_factory=lambda: {
            "hidden_sizes": [1024, 512],
            "nonlinearity": "ReLU",
        }
    )
    discriminator_reward_coef: float | None = 0.25
    discriminator_reward_type: str | None = "quad"
    discriminator_loss_func: str | None = "MSELoss"
    discriminator_gradient_penalty_coef: float | None = 5.0
    discriminator_optimizer_class_name: str | None = "AdamW"
    discriminator_weight_decay_coef: float | None = 3e-4
    discriminator_logit_weight_decay_coef: float | None = 0.04
    discriminator_optimizer_kwargs: dict[str, Any] | None = field(
        default_factory=lambda: {
            "lr": 1.0e-4,
            "betas": [0.9, 0.999],
        }
    )

    # FootholdAlgoMixin (defaults match instinct_rl.algorithms.foothold)
    foothold_state_key: str = "foothold"
    region_grid: tuple[int, int] = (13, 21)
    region_res: float = 0.025
    sole_grid: tuple[int, int] = (5, 10)
    epsilon_h: float = 0.03
    sigma_f: float = 0.0625
    num_swing_samples: int = 8
    terrain_level_gate: int = 5
    imagination_hidden_sizes: tuple[int, ...] = (256, 128)
    imagination_lr: float = 5e-4
    foothold_reward_coef: float = 0.25

    # PPO
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.006
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0


@dataclass(kw_only=True)
class AdamSPParkourSsrRunnerCfg(InstinctRlOnPolicyRunnerCfg):
    num_steps_per_env: int = 24
    policy_observation_group: str = "policy"
    critic_observation_group: str = "critic"
    max_iterations: int = 30000
    save_interval: int = 500
    log_interval: int = 10
    experiment_name: str = "adam_sp_parkour_stairs_ssr"
    resume: bool = False
    load_run: str = "^(?!_play$).*"
    empirical_normalization: bool = False
    policy: InstinctRlActorCriticCfg = field(default_factory=lambda: PolicyCfg())
    algorithm: InstinctRlPpoAlgorithmCfg = field(default_factory=lambda: SsrAlgoCfg())
    normalizers: dict[str, InstinctRlNormalizerCfg] = field(
        default_factory=lambda: default_policy_critic_normalizers()
    )


def adam_parkour_ssr_runner_cfg() -> InstinctRlOnPolicyRunnerCfg:
    """Adam SP stairs parkour runner: FootholdWasabiPPO + depth encoder."""
    return AdamSPParkourSsrRunnerCfg()
