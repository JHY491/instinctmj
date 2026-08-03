"""Instinct-RL runner configs for Adam SP parkour tasks."""

from .instinct_rl_amp_cfg import adam_parkour_simple_amp_runner_cfg
from .instinct_rl_ppo_simple_cfg import adam_parkour_simple_ppo_runner_cfg
from .instinct_rl_ssr_cfg import adam_parkour_ssr_runner_cfg

__all__ = [
    "adam_parkour_simple_amp_runner_cfg",
    "adam_parkour_simple_ppo_runner_cfg",
    "adam_parkour_ssr_runner_cfg",
]
