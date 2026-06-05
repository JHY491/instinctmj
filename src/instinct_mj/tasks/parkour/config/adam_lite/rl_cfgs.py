"""Instinct-RL configs for Adam SP parkour tasks."""

from __future__ import annotations

from instinct_mj.rl import InstinctRlOnPolicyRunnerCfg

from .agents.instinct_rl_amp_cfg import adam_parkour_amp_runner_cfg
from .agents.instinct_rl_ppo_cfg import adam_parkour_ppo_runner_cfg


def adam_sp_parkour_amp_instinct_rl_cfg() -> InstinctRlOnPolicyRunnerCfg:
  return adam_parkour_amp_runner_cfg()


def adam_sp_parkour_ppo_instinct_rl_cfg() -> InstinctRlOnPolicyRunnerCfg:
  return adam_parkour_ppo_runner_cfg()
