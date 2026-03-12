"""Instinct-RL runner configs for G1 parkour tasks."""

from .instinct_rl_amp_cfg import G1ParkourPPORunnerCfg
from .instinct_rl_ppo_cfg import G1ParkourTargetPPORunnerCfg
from .instinct_rl_ppo_simple_cfg import g1_parkour_simple_ppo_runner_cfg

__all__ = [
    "G1ParkourPPORunnerCfg",
    "G1ParkourTargetPPORunnerCfg",
    "g1_parkour_simple_ppo_runner_cfg",
]
