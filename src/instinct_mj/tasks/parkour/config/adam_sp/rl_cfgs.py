"""Instinct-RL configs for G1 parkour tasks."""

from __future__ import annotations

from .agents.instinct_rl_amp_cfg import G1ParkourPPORunnerCfg
from .agents.instinct_rl_ppo_cfg import G1ParkourTargetPPORunnerCfg
from .agents.instinct_rl_ppo_simple_cfg import g1_parkour_simple_ppo_runner_cfg


def g1_parkour_amp_instinct_rl_cfg() -> G1ParkourPPORunnerCfg:
    return G1ParkourPPORunnerCfg()


def g1_parkour_instinct_rl_cfg() -> G1ParkourTargetPPORunnerCfg:
    return G1ParkourTargetPPORunnerCfg()


def g1_parkour_simple_instinct_rl_cfg():
    return g1_parkour_simple_ppo_runner_cfg()
