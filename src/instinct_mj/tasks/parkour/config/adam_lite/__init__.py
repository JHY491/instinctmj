"""Register Instinct Mj parkour Adam Lite tasks."""

# Copyright (c) 2022-2025, The Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from instinct_mj.tasks.registry import register_instinct_task


def _parkour_cfg(play: bool):
  from .adam_lite_parkour_cfg import instinct_adam_lite_parkour_final_cfg
  return instinct_adam_lite_parkour_final_cfg(play=play)


def _runner_cfg():
  from .agents.instinct_rl_amp_cfg import adam_parkour_amp_runner_cfg
  return adam_parkour_amp_runner_cfg()


register_instinct_task(
  task_id="Instinct-Parkour-Adam-Lite-v0",
  env_cfg_factory=lambda: _parkour_cfg(play=False),
  play_env_cfg_factory=lambda: _parkour_cfg(play=True),
  instinct_rl_cfg_factory=_runner_cfg,
)


register_instinct_task(
  task_id="Instinct-Parkour-Adam-Lite-Play-v0",
  env_cfg_factory=lambda: _parkour_cfg(play=True),
  play_env_cfg_factory=lambda: _parkour_cfg(play=True),
  instinct_rl_cfg_factory=_runner_cfg,
)


def _three_slopes_parkour_cfg(play: bool):
  from .adam_lite_three_slopes_parkour_cfg import instinct_adam_lite_three_slopes_parkour_final_cfg
  return instinct_adam_lite_three_slopes_parkour_final_cfg(play=play)


def _three_slopes_runner_cfg():
  from .agents.instinct_rl_amp_cfg import adam_lite_three_slopes_parkour_amp_runner_cfg
  return adam_lite_three_slopes_parkour_amp_runner_cfg()


register_instinct_task(
  task_id="Instinct-Parkour-Adam-Lite-Three-Slopes-v0",
  env_cfg_factory=lambda: _three_slopes_parkour_cfg(play=False),
  play_env_cfg_factory=lambda: _three_slopes_parkour_cfg(play=True),
  instinct_rl_cfg_factory=_three_slopes_runner_cfg,
)


register_instinct_task(
  task_id="Instinct-Parkour-Adam-Lite-Three-Slopes-Play-v0",
  env_cfg_factory=lambda: _three_slopes_parkour_cfg(play=True),
  play_env_cfg_factory=lambda: _three_slopes_parkour_cfg(play=True),
  instinct_rl_cfg_factory=_three_slopes_runner_cfg,
)


def _hurdle_parkour_cfg(play: bool):
  from .adam_lite_hurdle_parkour_cfg import instinct_adam_lite_hurdle_parkour_final_cfg
  return instinct_adam_lite_hurdle_parkour_final_cfg(play=play)


def _hurdle_runner_cfg():
  from .agents.instinct_rl_amp_cfg import adam_lite_hurdle_parkour_amp_runner_cfg
  return adam_lite_hurdle_parkour_amp_runner_cfg()


register_instinct_task(
  task_id="Instinct-Parkour-Adam-Lite-Hurdle-v0",
  env_cfg_factory=lambda: _hurdle_parkour_cfg(play=False),
  play_env_cfg_factory=lambda: _hurdle_parkour_cfg(play=True),
  instinct_rl_cfg_factory=_hurdle_runner_cfg,
)


register_instinct_task(
  task_id="Instinct-Parkour-Adam-Lite-Hurdle-Play-v0",
  env_cfg_factory=lambda: _hurdle_parkour_cfg(play=True),
  play_env_cfg_factory=lambda: _hurdle_parkour_cfg(play=True),
  instinct_rl_cfg_factory=_hurdle_runner_cfg,
)


def _bridge_l_corner_parkour_cfg(play: bool):
  from .adam_lite_bridge_l_corner_parkour_cfg import instinct_adam_lite_bridge_l_corner_parkour_final_cfg
  return instinct_adam_lite_bridge_l_corner_parkour_final_cfg(play=play)


def _bridge_l_corner_runner_cfg():
  from .agents.instinct_rl_amp_cfg import adam_lite_bridge_l_corner_parkour_amp_runner_cfg
  return adam_lite_bridge_l_corner_parkour_amp_runner_cfg()


register_instinct_task(
  task_id="Instinct-Parkour-Adam-Lite-Bridge-L-Corner-v0",
  env_cfg_factory=lambda: _bridge_l_corner_parkour_cfg(play=False),
  play_env_cfg_factory=lambda: _bridge_l_corner_parkour_cfg(play=True),
  instinct_rl_cfg_factory=_bridge_l_corner_runner_cfg,
)


register_instinct_task(
  task_id="Instinct-Parkour-Adam-Lite-Bridge-L-Corner-Play-v0",
  env_cfg_factory=lambda: _bridge_l_corner_parkour_cfg(play=True),
  play_env_cfg_factory=lambda: _bridge_l_corner_parkour_cfg(play=True),
  instinct_rl_cfg_factory=_bridge_l_corner_runner_cfg,
)
