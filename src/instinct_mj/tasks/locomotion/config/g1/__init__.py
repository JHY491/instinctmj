"""Register Instinct Mj locomotion G1 tasks."""

from instinct_mj.tasks.registry import register_instinct_task


def _flat_cfg(play: bool):
    from .flat_env_cfg import instinct_g1_locomotion_flat_env_cfg
    return instinct_g1_locomotion_flat_env_cfg(play=play)


def _runner_cfg():
    from .rl_cfgs import g1_locomotion_instinct_rl_cfg
    return g1_locomotion_instinct_rl_cfg()


register_instinct_task(
    task_id="Instinct-Locomotion-Flat-G1-v0",
    env_cfg_factory=lambda: _flat_cfg(play=False),
    play_env_cfg_factory=lambda: _flat_cfg(play=True),
    instinct_rl_cfg_factory=_runner_cfg,
)

register_instinct_task(
    task_id="Instinct-Locomotion-Flat-G1-Play-v0",
    env_cfg_factory=lambda: _flat_cfg(play=True),
    play_env_cfg_factory=lambda: _flat_cfg(play=True),
    instinct_rl_cfg_factory=_runner_cfg,
)
