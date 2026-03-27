"""Instinct-RL config for G1 simple parkour: ActorCritic + PPO (no MoE, no AMP)."""

from __future__ import annotations

from instinct_mj.rl import InstinctRlActorCriticCfg, InstinctRlOnPolicyRunnerCfg, InstinctRlPpoAlgorithmCfg
from instinct_mj.tasks.config.rl_utils import default_policy_critic_normalizers


def g1_parkour_simple_ppo_runner_cfg() -> InstinctRlOnPolicyRunnerCfg:
    return InstinctRlOnPolicyRunnerCfg(
        policy=InstinctRlActorCriticCfg(
            class_name="ActorCritic",
            init_noise_std=1.0,
            actor_hidden_dims=(256, 128, 128),
            critic_hidden_dims=(256, 128, 128),
            activation="elu",
        ),
        algorithm=InstinctRlPpoAlgorithmCfg(
            class_name="PPO",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.006,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        normalizers=default_policy_critic_normalizers(),
        num_steps_per_env=24,
        max_iterations=30000,
        save_interval=100,
        log_interval=10,
        experiment_name="g1_parkour_simple",
        run_name="",
        policy_observation_group="policy",
        critic_observation_group="critic",
        resume=False,
        load_run="^(?!_play$).*",
    )
