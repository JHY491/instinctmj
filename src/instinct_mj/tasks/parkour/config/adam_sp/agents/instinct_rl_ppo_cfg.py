from dataclasses import dataclass, field

from instinct_mj.rl import (
    InstinctRlConv2dHeadCfg,
    InstinctRlEncoderMoEActorCriticCfg,
    InstinctRlOnPolicyRunnerCfg,
    InstinctRlPpoAlgorithmCfg,
)


@dataclass(kw_only=True)
class DepthEncoderConv2dCfg(InstinctRlConv2dHeadCfg):
    output_size: int = 128
    channels: list = field(default_factory=lambda: [4])
    kernel_sizes: list = field(default_factory=lambda: [3])
    strides: list = field(default_factory=lambda: [1])
    hidden_sizes: list = field(default_factory=lambda: [256, 256])
    paddings: list = field(default_factory=lambda: [1])
    nonlinearity: str = "ReLU"
    use_maxpool: bool = True
    component_names: list = field(
        default_factory=lambda: [
            "depth_image",
        ]
    )


@dataclass(kw_only=True)
class EncoderConfigs:
    depth_encoder: object = field(default_factory=lambda: DepthEncoderConv2dCfg())


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
class PpoAlgoCfg(InstinctRlPpoAlgorithmCfg):
    class_name: str = "PPO"
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
class G1ParkourTargetPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    num_steps_per_env: int = 24
    policy_observation_group: str = "policy"
    critic_observation_group: str = "critic"
    max_iterations: int = 30000
    save_interval: int = 100
    experiment_name: str = "g1_parkour_target"
    resume: bool = False
    load_run: str = "^(?!_play$).*"
    empirical_normalization: bool = False
    policy: object = field(default_factory=lambda: MoEPolicyCfg())
    algorithm: object = field(default_factory=lambda: PpoAlgoCfg())


def g1_parkour_ppo_runner_cfg() -> InstinctRlOnPolicyRunnerCfg:
    return G1ParkourTargetPPORunnerCfg()
