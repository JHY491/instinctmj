"""Register Instinct Mj parkour G1 tasks."""

# Copyright (c) 2022-2025, The Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from instinct_mj.tasks.registry import register_instinct_task


def _amp_cfg(play: bool, shoe: bool = True):
    from .g1_parkour_target_amp_cfg import instinct_g1_parkour_amp_final_cfg
    return instinct_g1_parkour_amp_final_cfg(play=play, shoe=shoe)


def _target_cfg(play: bool, shoe: bool = True):
    from .g1_parkour_target_cfg import instinct_g1_parkour_final_cfg
    return instinct_g1_parkour_final_cfg(play=play, shoe=shoe)


def _simple_cfg(play: bool, shoe: bool = True):
    from .g1_parkour_simple_cfg import instinct_g1_parkour_simple_final_cfg
    return instinct_g1_parkour_simple_final_cfg(play=play, shoe=shoe)


def _amp_runner_cfg():
    from .agents.instinct_rl_amp_cfg import G1ParkourPPORunnerCfg
    return G1ParkourPPORunnerCfg()


def _target_runner_cfg():
    from .agents.instinct_rl_ppo_cfg import G1ParkourTargetPPORunnerCfg
    return G1ParkourTargetPPORunnerCfg()


def _simple_runner_cfg():
    from .agents.instinct_rl_ppo_simple_cfg import g1_parkour_simple_ppo_runner_cfg
    return g1_parkour_simple_ppo_runner_cfg()


register_instinct_task(
    task_id="Instinct-Parkour-Target-Amp-G1-v0",
    env_cfg_factory=lambda: _amp_cfg(play=False, shoe=True),
    play_env_cfg_factory=lambda: _amp_cfg(play=True, shoe=True),
    instinct_rl_cfg_factory=_amp_runner_cfg,
)


register_instinct_task(
    task_id="Instinct-Parkour-Target-Amp-G1-Play-v0",
    env_cfg_factory=lambda: _amp_cfg(play=True, shoe=True),
    play_env_cfg_factory=lambda: _amp_cfg(play=True, shoe=True),
    instinct_rl_cfg_factory=_amp_runner_cfg,
)


register_instinct_task(
    task_id="Instinct-Parkour-Target-G1-v0",
    env_cfg_factory=lambda: _target_cfg(play=False, shoe=True),
    play_env_cfg_factory=lambda: _target_cfg(play=True, shoe=True),
    instinct_rl_cfg_factory=_target_runner_cfg,
)


register_instinct_task(
    task_id="Instinct-Parkour-Target-G1-Play-v0",
    env_cfg_factory=lambda: _target_cfg(play=True, shoe=True),
    play_env_cfg_factory=lambda: _target_cfg(play=True, shoe=True),
    instinct_rl_cfg_factory=_target_runner_cfg,
)


register_instinct_task(
    task_id="Instinct-Parkour-Simple-G1-v0",
    env_cfg_factory=lambda: _simple_cfg(play=False, shoe=True),
    play_env_cfg_factory=lambda: _simple_cfg(play=True, shoe=True),
    instinct_rl_cfg_factory=_simple_runner_cfg,
)


register_instinct_task(
    task_id="Instinct-Parkour-Simple-G1-Play-v0",
    env_cfg_factory=lambda: _simple_cfg(play=True, shoe=True),
    play_env_cfg_factory=lambda: _simple_cfg(play=True, shoe=True),
    instinct_rl_cfg_factory=_simple_runner_cfg,
)
