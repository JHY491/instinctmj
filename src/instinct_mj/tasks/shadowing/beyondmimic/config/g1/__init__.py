"""Register Instinct Mj BeyondMimic G1 tasks."""

from instinct_mj.tasks.registry import register_instinct_task


def _plane_cfg(play: bool):
    from .beyondmimic_plane_cfg import g1_beyondmimic_plane_env_cfg
    return g1_beyondmimic_plane_env_cfg(play=play)


def _runner_cfg():
    from .rl_cfgs import g1_beyondmimic_instinct_rl_cfg
    return g1_beyondmimic_instinct_rl_cfg()


register_instinct_task(
    task_id="Instinct-BeyondMimic-Plane-G1-v0",
    env_cfg_factory=lambda: _plane_cfg(play=False),
    play_env_cfg_factory=lambda: _plane_cfg(play=True),
    instinct_rl_cfg_factory=_runner_cfg,
)

register_instinct_task(
    task_id="Instinct-BeyondMimic-Plane-G1-Play-v0",
    env_cfg_factory=lambda: _plane_cfg(play=True),
    play_env_cfg_factory=lambda: _plane_cfg(play=True),
    instinct_rl_cfg_factory=_runner_cfg,
)
