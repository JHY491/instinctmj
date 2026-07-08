"""Instinct-RL AMP config for Adam SP parkour: ActorCritic + WasabiPPO (AMP)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from instinct_mj.rl import (
    EncoderTerrainActorCriticCfg,
    EncoderTerrainVelActorCriticCfg,
    EncoderVelActorCriticCfg,
    EstimatorWasabiPpoAlgorithmCfg,
    FootholdWasabiPpoAlgorithmCfg,
    InstinctRlActorCriticCfg,
    InstinctRlConv2dHeadCfg,
    InstinctRlDepthProprioGruEncoderCfg,
    InstinctRlDepthProprioRecurrentGruEncoderCfg,
    InstinctRlEncoderActorCriticCfg,
    InstinctRlEncoderRecurrentActorCriticCfg,
    InstinctRlEncoderMoEActorCriticCfg,
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
    component_names: list = field(
        default_factory=lambda: [
            "depth_image",
        ]
    )


@dataclass(kw_only=True)
class EncoderConfigs:
    depth_encoder: DepthEncoderConv2dCfg = field(default_factory=lambda: DepthEncoderConv2dCfg())


@dataclass(kw_only=True)
class PolicyCfg(EncoderTerrainActorCriticCfg):
    init_noise_std: float = 1.0
    actor_hidden_dims: tuple[int, ...] = (512, 512, 256, 128)
    critic_hidden_dims: tuple[int, ...] = (512, 512, 256, 128)
    activation: str = "elu"
    encoder_configs: dict[str, Any] | None = field(
        default_factory=lambda: {"depth_encoder": DepthEncoderConv2dCfg()}
    )
    # Critic consumes the privileged heightmap directly, so it needs no depth encoder.
    critic_encoder_configs: dict[str, Any] | str | None = None
    estimator_target_components: list[str] = field(default_factory=lambda: ["height_map_priv"])
    terrain_latent_component: str = "parallel_latent_0_depth_encoder"
    terrain_predictor_configs: dict[str, Any] = field(
        default_factory=lambda: {"hidden_sizes": [256, 128], "nonlinearity": "ReLU"}
    )


@dataclass(kw_only=True)
class FootholdVisionPolicyCfg(InstinctRlEncoderActorCriticCfg):
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
class DepthEncoderDeepCfg(InstinctRlConv2dHeadCfg):
    """SSR 论文对齐的 Depth CNN: channels [32,64,128], stride 2× 下采样。

    输入: [10, 36, 36] (10 帧堆叠, 36×36 depth 裁切后) →
    CNN 输出: [128, 5, 5] → MLP 压缩到 128-dim.
    """
    output_size: int | None = 128
    channels: list = field(default_factory=lambda: [32, 64, 128])
    kernel_sizes: list = field(default_factory=lambda: [5, 3, 3])
    strides: list = field(default_factory=lambda: [2, 2, 2])
    hidden_sizes: list = field(default_factory=lambda: [512, 256])
    paddings: list = field(default_factory=lambda: [2, 1, 1])
    nonlinearity: str | None = "ELU"
    use_maxpool: bool = False
    component_names: list = field(
        default_factory=lambda: [
            "depth_image",
        ]
    )
    takeout_input_components: bool = True


@dataclass(kw_only=True)
class FootholdVelEstDeepVisionPolicyCfg(EncoderVelActorCriticCfg):
    """VelEst + SSR 对齐的深度 CNN (无 GRU)。"""
    init_noise_std: float = 1.0
    actor_hidden_dims: tuple[int, ...] = (512, 512, 256, 128)
    critic_hidden_dims: tuple[int, ...] = (512, 512, 256, 128)
    activation: str = "elu"
    encoder_configs: dict[str, Any] | None = field(
        default_factory=lambda: {"depth_encoder": DepthEncoderDeepCfg()}
    )
    critic_encoder_configs: dict[str, Any] | str | None = field(
        default_factory=lambda: {"depth_encoder": DepthEncoderDeepCfg()}
    )
    vel_estimator_obs_components: list[str] = field(
        default_factory=lambda: [
            "base_ang_vel",
            "projected_gravity",
            "velocity_commands",
            "joint_pos",
            "joint_vel",
            "actions",
        ]
    )
    vel_target_components: list[str] = field(default_factory=lambda: ["base_lin_vel_est"])
    vel_estimator_configs: dict[str, Any] = field(
        default_factory=lambda: {"hidden_sizes": [512, 256, 128], "nonlinearity": "ReLU"}
    )


@dataclass(kw_only=True)
class FootholdVelEstVisionPolicyCfg(EncoderVelActorCriticCfg):
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
    vel_estimator_obs_components: list[str] = field(
        default_factory=lambda: [
            "base_ang_vel",
            "projected_gravity",
            "velocity_commands",
            "joint_pos",
            "joint_vel",
            "actions",
        ]
    )
    vel_target_components: list[str] = field(default_factory=lambda: ["base_lin_vel_est"])
    vel_estimator_configs: dict[str, Any] = field(
        default_factory=lambda: {"hidden_sizes": [512, 256, 128], "nonlinearity": "ReLU"}
    )


@dataclass(kw_only=True)
class FootholdGruVisionPolicyCfg(InstinctRlEncoderActorCriticCfg):
    init_noise_std: float = 1.0
    actor_hidden_dims: tuple[int, ...] = (512, 512, 256, 128)
    critic_hidden_dims: tuple[int, ...] = (512, 512, 256, 128)
    activation: str = "elu"
    encoder_configs: dict[str, Any] | None = field(
        default_factory=lambda: {"depth_proprio_gru": InstinctRlDepthProprioGruEncoderCfg()}
    )
    critic_encoder_configs: dict[str, Any] | str | None = field(
        default_factory=lambda: {"depth_encoder": DepthEncoderConv2dCfg()}
    )


@dataclass(kw_only=True)
class VelEstPolicyCfg(EncoderTerrainVelActorCriticCfg):
    init_noise_std: float = 1.0
    actor_hidden_dims: tuple[int, ...] = (512, 512, 256, 128)
    critic_hidden_dims: tuple[int, ...] = (512, 512, 256, 128)
    activation: str = "elu"
    encoder_configs: dict[str, Any] | None = field(
        default_factory=lambda: {"depth_encoder": DepthEncoderConv2dCfg()}
    )
    critic_encoder_configs: dict[str, Any] | str | None = None
    estimator_target_components: list[str] = field(default_factory=lambda: ["height_map_priv"])
    terrain_latent_component: str = "parallel_latent_0_depth_encoder"
    terrain_predictor_configs: dict[str, Any] = field(
        default_factory=lambda: {"hidden_sizes": [256, 128], "nonlinearity": "ReLU"}
    )
    vel_estimator_obs_components: list[str] = field(
        default_factory=lambda: [
            "base_ang_vel",
            "projected_gravity",
            "velocity_commands",
            "joint_pos",
            "joint_vel",
            "actions",
        ]
    )
    vel_target_components: list[str] = field(default_factory=lambda: ["base_lin_vel"])
    vel_estimator_configs: dict[str, Any] = field(
        default_factory=lambda: {"hidden_sizes": [256, 128], "nonlinearity": "ReLU"}
    )


@dataclass(kw_only=True)
class MoEPolicyCfg(InstinctRlEncoderMoEActorCriticCfg):
    init_noise_std: float = 1.0
    num_moe_experts: int = 4
    actor_hidden_dims: list = field(default_factory=lambda: [256, 128, 64])
    critic_hidden_dims: list = field(default_factory=lambda: [256, 128, 64])
    activation: str = "elu"
    encoder_configs: object = field(default_factory=lambda: EncoderConfigs())
    critic_encoder_configs: object = field(default_factory=lambda: EncoderConfigs())


@dataclass(kw_only=True)
class AmpAlgoCfg(EstimatorWasabiPpoAlgorithmCfg):
    """AMP algorithm config using EstimatorWasabiPPO with discriminator."""

    class_name: str = "EstimatorWasabiPPO"
    estimator_loss_coef: float = 1.0
    target_from_critic: bool = True
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
class FootholdAmpAlgoCfg(FootholdWasabiPpoAlgorithmCfg):
    """AMP discriminator + foothold guidance, without terrain/velocity estimator heads."""

    class_name: str = "FootholdWasabiPPO"
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
    # (task, foothold, style). Keep AMP style at full advantage weight so the human-like
    # motion prior stays as strong as in the 301cefc baseline; only the SSR foothold critic
    # (index 1) is down-weighted to w_f=0.25.
    advantage_mixing_weights: tuple[float, ...] = (1.0, 0.25, 1.0)
    auxiliary_reward_per_env_reward_coefs_by_key: dict[str, tuple[float, ...]] = field(
        default_factory=lambda: {
            "discriminator_reward": (0.0, 0.0, 1.0),
            "foothold_reward": (0.0, 1.0, 0.0),
        }
    )


@dataclass(kw_only=True)
class FootholdVelEstAmpAlgoCfg(FootholdAmpAlgoCfg):
    """Foothold + AMP + velocity-only estimator supervision."""

    class_name: str = "FootholdWasabiEstimatorPPO"
    estimator_loss_coef: float = 1.0
    target_from_critic: bool = True
    estimator_loss_func: str = "mse_loss"
    estimator_loss_kwargs: dict[str, Any] | None = None


@dataclass(kw_only=True)
class AdamSPParkourVisionAmpRunnerCfg(InstinctRlOnPolicyRunnerCfg):
    num_steps_per_env: int = 24
    policy_observation_group: str = "policy"
    critic_observation_group: str = "critic"
    max_iterations: int = 30000
    save_interval: int = 500
    log_interval: int = 10
    experiment_name: str = "adam_lite_parkour_amp_vision"
    resume: bool = False
    load_run: str = "^(?!_play$).*"
    empirical_normalization: bool = False
    policy: InstinctRlActorCriticCfg = field(default_factory=lambda: PolicyCfg())
    algorithm: InstinctRlPpoAlgorithmCfg = field(default_factory=lambda: AmpAlgoCfg())
    normalizers: dict[str, InstinctRlNormalizerCfg] = field(default_factory=lambda: default_policy_critic_normalizers())


def adam_parkour_amp_runner_cfg() -> InstinctRlOnPolicyRunnerCfg:
    """Adam SP parkour AMP runner config with depth vision encoder."""
    return AdamSPParkourVisionAmpRunnerCfg()


@dataclass(kw_only=True)
class AdamLiteThreeSlopesParkourVisionAmpRunnerCfg(AdamSPParkourVisionAmpRunnerCfg):
    experiment_name: str = "adam_lite_slopes_parkour"


def adam_lite_three_slopes_parkour_amp_runner_cfg() -> InstinctRlOnPolicyRunnerCfg:
    """AMP runner for three-slope parkour (separate log dir from stairs/default parkour)."""
    return AdamLiteThreeSlopesParkourVisionAmpRunnerCfg()


@dataclass(kw_only=True)
class AdamLiteHurdleParkourVisionAmpRunnerCfg(AdamSPParkourVisionAmpRunnerCfg):
    experiment_name: str = "adam_lite_hurdle_parkour"


def adam_lite_hurdle_parkour_amp_runner_cfg() -> InstinctRlOnPolicyRunnerCfg:
    """AMP runner for continuous hurdle parkour."""
    return AdamLiteHurdleParkourVisionAmpRunnerCfg()


@dataclass(kw_only=True)
class AdamLiteBridgeLCornerParkourVisionAmpRunnerCfg(AdamSPParkourVisionAmpRunnerCfg):
    experiment_name: str = "adam_lite_chained_straight_bridge_parkour"


def adam_lite_bridge_l_corner_parkour_amp_runner_cfg() -> InstinctRlOnPolicyRunnerCfg:
    """AMP runner for curved bridge + narrow L-corner parkour."""
    return AdamLiteBridgeLCornerParkourVisionAmpRunnerCfg()


@dataclass(kw_only=True)
class AdamLitePlumBlossomParkourVisionAmpRunnerCfg(AdamSPParkourVisionAmpRunnerCfg):
    experiment_name: str = "adam_lite_plum_blossom_parkour"


def adam_lite_plum_blossom_parkour_amp_runner_cfg() -> InstinctRlOnPolicyRunnerCfg:
    """AMP runner for plum-blossom stepping-stakes parkour."""
    return AdamLitePlumBlossomParkourVisionAmpRunnerCfg()


@dataclass(kw_only=True)
class AdamLiteParkourVelEstAmpRunnerCfg(AdamSPParkourVisionAmpRunnerCfg):
    experiment_name: str = "adam_lite_parkour_velest"
    policy: InstinctRlActorCriticCfg = field(default_factory=lambda: VelEstPolicyCfg())


def adam_lite_parkour_velest_amp_runner_cfg() -> InstinctRlOnPolicyRunnerCfg:
    """AMP runner with a HIMLoco-style base-velocity estimator injected into the actor."""
    return AdamLiteParkourVelEstAmpRunnerCfg()


@dataclass(kw_only=True)
class AdamLiteFootholdParkourRunnerCfg(AdamSPParkourVisionAmpRunnerCfg):
    experiment_name: str = "adam_lite_foothold_parkour"
    policy: InstinctRlActorCriticCfg = field(default_factory=lambda: FootholdVisionPolicyCfg())
    algorithm: InstinctRlPpoAlgorithmCfg = field(default_factory=lambda: FootholdAmpAlgoCfg())


def adam_lite_foothold_parkour_amp_runner_cfg() -> InstinctRlOnPolicyRunnerCfg:
    """Foothold 任务 runner（algorithm=FootholdAmpAlgoCfg）。"""
    return AdamLiteFootholdParkourRunnerCfg()


@dataclass(kw_only=True)
class AdamLiteFootholdVelEstParkourRunnerCfg(AdamSPParkourVisionAmpRunnerCfg):
    experiment_name: str = "adam_lite_foothold_velest_parkour"
    policy: InstinctRlActorCriticCfg = field(default_factory=lambda: FootholdVelEstVisionPolicyCfg())
    algorithm: InstinctRlPpoAlgorithmCfg = field(default_factory=lambda: FootholdVelEstAmpAlgoCfg())


def adam_lite_foothold_velest_parkour_amp_runner_cfg() -> InstinctRlOnPolicyRunnerCfg:
    """Foothold stage-2 runner with a velocity-only MLP estimator."""
    return AdamLiteFootholdVelEstParkourRunnerCfg()


@dataclass(kw_only=True)
class AdamLiteFootholdVelEstDeepRunnerCfg(AdamLiteFootholdVelEstParkourRunnerCfg):
    experiment_name: str = "adam_lite_foothold_velest_deepcnn"
    policy: InstinctRlActorCriticCfg = field(default_factory=lambda: FootholdVelEstDeepVisionPolicyCfg())


def adam_lite_foothold_velest_deep_amp_runner_cfg() -> InstinctRlOnPolicyRunnerCfg:
    """Foothold VelEst + SSR 对齐的深度 CNN 编码器。"""
    return AdamLiteFootholdVelEstDeepRunnerCfg()


@dataclass(kw_only=True)
class AdamLiteFootholdGruVisionAmpRunnerCfg(AdamLiteFootholdParkourRunnerCfg):
    experiment_name: str = "adam_lite_foothold_gru"
    policy: InstinctRlActorCriticCfg = field(default_factory=lambda: FootholdGruVisionPolicyCfg())


def adam_lite_foothold_gru_amp_runner_cfg() -> InstinctRlOnPolicyRunnerCfg:
    """Foothold GRU 任务 runner（depth CNN + proprio GRU encoder）。"""
    return AdamLiteFootholdGruVisionAmpRunnerCfg()


@dataclass(kw_only=True)
class FootholdRecurrentGruVisionPolicyCfg(InstinctRlEncoderRecurrentActorCriticCfg):
    init_noise_std: float = 1.0
    actor_hidden_dims: tuple[int, ...] = (512, 512, 256, 128)
    critic_hidden_dims: tuple[int, ...] = (512, 512, 256, 128)
    activation: str = "elu"
    encoder_configs: dict[str, Any] | None = field(
        default_factory=lambda: {"depth_proprio_recurrent_gru": InstinctRlDepthProprioRecurrentGruEncoderCfg()}
    )
    critic_encoder_configs: dict[str, Any] | str | None = field(
        default_factory=lambda: {"depth_encoder": DepthEncoderConv2dCfg()}
    )


@dataclass(kw_only=True)
class AdamLiteFootholdRecurrentGruVisionAmpRunnerCfg(AdamLiteFootholdParkourRunnerCfg):
    experiment_name: str = "adam_lite_foothold_gru_recurrent"
    policy: InstinctRlActorCriticCfg = field(default_factory=lambda: FootholdRecurrentGruVisionPolicyCfg())


def adam_lite_foothold_recurrent_gru_amp_runner_cfg() -> InstinctRlOnPolicyRunnerCfg:
    """Foothold recurrent GRU 任务 runner（cross-step depth+proprio GRU encoder）。"""
    return AdamLiteFootholdRecurrentGruVisionAmpRunnerCfg()
