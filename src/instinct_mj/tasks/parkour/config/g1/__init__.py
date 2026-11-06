"""Register Instinct Mj parkour G1 tasks."""

# Copyright (c) 2022-2025, The Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from instinct_mj.tasks.registry import register_instinct_task


def _plum_blossom_foothold_cfg(play: bool, *, use_cref: bool = False):
  from .g1_plum_blossom_foothold_parkour_cfg import instinct_g1_plum_blossom_foothold_final_cfg
  return instinct_g1_plum_blossom_foothold_final_cfg(play=play, use_cref=use_cref)


def _plum_blossom_foothold_runner_cfg():
  from .agents.instinct_rl_amp_cfg import g1_plum_blossom_foothold_amp_runner_cfg
  return g1_plum_blossom_foothold_amp_runner_cfg()


def _plum_blossom_cref_runner_cfg():
  from .agents.instinct_rl_amp_cfg import G1PlumBlossomCrefRunnerCfg
  return G1PlumBlossomCrefRunnerCfg()


register_instinct_task(
  task_id="Instinct-Parkour-G1-Plum-Blossom-Foothold-v0",
  env_cfg_factory=lambda: _plum_blossom_foothold_cfg(play=False),
  play_env_cfg_factory=lambda: _plum_blossom_foothold_cfg(play=True),
  instinct_rl_cfg_factory=_plum_blossom_foothold_runner_cfg,
)


register_instinct_task(
  task_id="Instinct-Parkour-G1-Plum-Blossom-Foothold-Play-v0",
  env_cfg_factory=lambda: _plum_blossom_foothold_cfg(play=True),
  play_env_cfg_factory=lambda: _plum_blossom_foothold_cfg(play=True),
  instinct_rl_cfg_factory=_plum_blossom_foothold_runner_cfg,
)


register_instinct_task(
  task_id="Instinct-Parkour-G1-Plum-Blossom-CReF-v0",
  env_cfg_factory=lambda: _plum_blossom_foothold_cfg(play=False, use_cref=True),
  play_env_cfg_factory=lambda: _plum_blossom_foothold_cfg(play=True, use_cref=True),
  instinct_rl_cfg_factory=_plum_blossom_cref_runner_cfg,
)
