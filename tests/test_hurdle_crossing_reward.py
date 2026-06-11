from __future__ import annotations

from types import SimpleNamespace

import torch
from mjlab.managers import RewardTermCfg

from instinct_mj.tasks.parkour.mdp.rewards import hurdle_crossing_reward


class _FakeScene:
    def __init__(self, robot: object, terrain: object, sensors: dict[str, object]) -> None:
        self.terrain = terrain
        self.sensors = sensors
        self._entities = {"robot": robot}

    def __getitem__(self, name: str) -> object:
        return self._entities[name]


class _FakeTerrainGenerator:
    def __init__(self, sub_cfg: object) -> None:
        self._sub_cfg = sub_cfg

    def get_subterrain_cfg(self, row_ids: int, col_ids: int) -> object:
        del row_ids, col_ids
        return self._sub_cfg


def _make_env(*, penetration: float = 0.0) -> object:
    sub_cfg = SimpleNamespace(
        num_hurdles=2,
        hurdle_thickness=0.10,
        hurdle_spacing=1.0,
        hurdle_spacing_range=None,
        difficulty=0.0,
        size=(8.0, 8.0),
    )
    terrain = SimpleNamespace(
        terrain_generator=_FakeTerrainGenerator(sub_cfg),
        terrain_origins=torch.tensor([[[0.0, 0.0, 0.0]]]),
        terrain_levels=torch.tensor([0]),
        terrain_types=torch.tensor([0]),
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[-1.0, 0.0, 0.9]]),
            root_link_lin_vel_b=torch.tensor([[0.5, 0.0, 0.0]]),
        )
    )
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            penetration_offset=torch.full((1, 2, 3), float(penetration)),
        )
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        scene=_FakeScene(
            robot=robot,
            terrain=terrain,
            sensors={"leg_volume_points": sensor, "shank_volume_points": sensor},
        ),
    )
    return env


def test_hurdle_crossing_reward_fires_once_per_hurdle() -> None:
    env = _make_env()
    term = hurdle_crossing_reward(
        RewardTermCfg(
            func=hurdle_crossing_reward,
            weight=1.0,
            params={"success_margin": 0.10, "min_forward_vel": 0.25},
        ),
        env,
    )

    assert term(env).item() == 0.0
    env.scene["robot"].data.root_link_pos_w[:, 0] = -0.4
    assert term(env).item() == 1.0
    assert term(env).item() == 0.0


def test_hurdle_crossing_reward_requires_clean_crossing() -> None:
    env = _make_env(penetration=0.01)
    term = hurdle_crossing_reward(
        RewardTermCfg(
            func=hurdle_crossing_reward,
            weight=1.0,
            params={"success_margin": 0.10, "min_forward_vel": 0.25},
        ),
        env,
    )

    assert term(env).item() == 0.0
    env.scene["robot"].data.root_link_pos_w[:, 0] = -0.4
    assert term(env).item() == 0.0
    env.scene.sensors["leg_volume_points"].data.penetration_offset.zero_()
    env.scene.sensors["shank_volume_points"].data.penetration_offset.zero_()
    assert term(env).item() == 0.0
