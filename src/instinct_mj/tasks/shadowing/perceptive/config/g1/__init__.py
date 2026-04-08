"""Register Instinct Mj perceptive G1 tasks."""

from instinct_mj.tasks.registry import register_instinct_task


def _shadowing_cfg():
    from .perceptive_shadowing_cfg import G1PerceptiveShadowingEnvCfg
    return G1PerceptiveShadowingEnvCfg()


def _shadowing_play_cfg():
    from .perceptive_shadowing_cfg import G1PerceptiveShadowingEnvCfg_PLAY
    return G1PerceptiveShadowingEnvCfg_PLAY()


def _vae_cfg():
    from .perceptive_vae_cfg import G1PerceptiveVaeEnvCfg
    return G1PerceptiveVaeEnvCfg()


def _vae_play_cfg():
    from .perceptive_vae_cfg import G1PerceptiveVaeEnvCfg_PLAY
    return G1PerceptiveVaeEnvCfg_PLAY()


def _shadowing_runner_cfg():
    from .rl_cfgs import g1_perceptive_shadowing_instinct_rl_cfg
    return g1_perceptive_shadowing_instinct_rl_cfg()


def _vae_runner_cfg():
    from .rl_cfgs import g1_perceptive_vae_instinct_rl_cfg
    return g1_perceptive_vae_instinct_rl_cfg()


register_instinct_task(
    task_id="Instinct-Perceptive-Shadowing-G1-v0",
    env_cfg_factory=_shadowing_cfg,
    play_env_cfg_factory=_shadowing_play_cfg,
    instinct_rl_cfg_factory=_shadowing_runner_cfg,
)

register_instinct_task(
    task_id="Instinct-Perceptive-Shadowing-G1-Play-v0",
    env_cfg_factory=_shadowing_play_cfg,
    play_env_cfg_factory=_shadowing_play_cfg,
    instinct_rl_cfg_factory=_shadowing_runner_cfg,
)

register_instinct_task(
    task_id="Instinct-Perceptive-Vae-G1-v0",
    env_cfg_factory=_vae_cfg,
    play_env_cfg_factory=_vae_play_cfg,
    instinct_rl_cfg_factory=_vae_runner_cfg,
)

register_instinct_task(
    task_id="Instinct-Perceptive-Vae-G1-Play-v0",
    env_cfg_factory=_vae_play_cfg,
    play_env_cfg_factory=_vae_play_cfg,
    instinct_rl_cfg_factory=_vae_runner_cfg,
)
