"""Register Instinct Mj parkour Adam SP tasks."""

# Copyright (c) 2022-2025, The Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from instinct_mj.tasks.registry import register_instinct_task

from .agents.instinct_rl_amp_cfg import adam_parkour_simple_amp_runner_cfg
from .adam_sp_parkour_simple_cfg import instinct_adam_sp_parkour_simple_final_cfg


register_instinct_task(
    task_id="Instinct-Parkour-Simple-Adam-SP-v0",
    env_cfg_factory=lambda: instinct_adam_sp_parkour_simple_final_cfg(play=False),
    play_env_cfg_factory=lambda: instinct_adam_sp_parkour_simple_final_cfg(play=True),
    instinct_rl_cfg_factory=adam_parkour_simple_amp_runner_cfg,
)


register_instinct_task(
    task_id="Instinct-Parkour-Simple-Adam-SP-Play-v0",
    env_cfg_factory=lambda: instinct_adam_sp_parkour_simple_final_cfg(play=True),
    play_env_cfg_factory=lambda: instinct_adam_sp_parkour_simple_final_cfg(play=True),
    instinct_rl_cfg_factory=adam_parkour_simple_amp_runner_cfg,
)
