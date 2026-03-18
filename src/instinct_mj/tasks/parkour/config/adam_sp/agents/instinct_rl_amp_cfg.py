"""Instinct-RL AMP config for Adam SP parkour: ActorCritic + WasabiPPO (AMP)."""

from __future__ import annotations

from dataclasses import dataclass, field

from instinct_mj.rl import (
    InstinctRlActorCriticCfg,
    InstinctRlOnPolicyRunnerCfg,
    InstinctRlPpoAlgorithmCfg,
)
from instinct_mj.tasks.config.rl_utils import default_policy_critic_normalizers


@dataclass(kw_only=True)
class AmpAlgoCfg(InstinctRlPpoAlgorithmCfg):
    """AMP algorithm config using WasabiPPO with discriminator."""

    class_name: str = "WasabiPPO"
    discriminator_kwargs: dict = field(
        default_factory=lambda: {
            "hidden_sizes": [1024, 512],
            "nonlinearity": "ReLU",
        }
    )
    discriminator_reward_coef: float = 0.4
    discriminator_reward_type: str = "quad"
    discriminator_loss_func: str = "MSELoss"
    discriminator_gradient_penalty_coef: float = 5.0
    discriminator_optimizer_class_name: str = "AdamW"
    discriminator_weight_decay_coef: float = 3e-4
    discriminator_logit_weight_decay_coef: float = 0.04
    discriminator_optimizer_kwargs: dict = field(
        default_factory=lambda: {
            "lr": 1.0e-4,
            "betas": [0.9, 0.999],
        }
    )
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


def adam_parkour_simple_amp_runner_cfg() -> InstinctRlOnPolicyRunnerCfg:
    """Adam SP simple parkour AMP runner config (no vision, no MoE)."""
    return InstinctRlOnPolicyRunnerCfg(
        policy=InstinctRlActorCriticCfg(
            class_name="ActorCritic",
            init_noise_std=1.0,
            actor_hidden_dims=(256, 128, 128),
            critic_hidden_dims=(256, 128, 128),
            activation="elu",
        ),
        algorithm=AmpAlgoCfg(),
        normalizers=default_policy_critic_normalizers(),
        num_steps_per_env=24,
        max_iterations=30000,
        save_interval=100,
        log_interval=10,
        experiment_name="adam_sp_parkour_simple_amp",
        run_name="",
        policy_observation_group="policy",
        critic_observation_group="critic",
        resume=False,
        load_run="^(?!_play$).*",
    )
