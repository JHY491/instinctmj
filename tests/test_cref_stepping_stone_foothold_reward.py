from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from mjlab.managers import RewardTermCfg, SceneEntityCfg

from instinct_mj.terrains.height_field.hf_terrains import plum_blossom_stake_centers_local
from instinct_mj.terrains.height_field.hf_terrains_cfg import PerlinPlumBlossomStakesTerrainCfg

_rewards_path = Path(__file__).resolve().parents[1] / "src/instinct_mj/tasks/parkour/mdp/rewards.py"
_rewards_spec = importlib.util.spec_from_file_location("parkour_rewards_test", _rewards_path)
assert _rewards_spec is not None and _rewards_spec.loader is not None
_rewards_mod = importlib.util.module_from_spec(_rewards_spec)
_rewards_spec.loader.exec_module(_rewards_mod)
cref_stepping_stone_foothold_reward = _rewards_mod.cref_stepping_stone_foothold_reward


def _stakes_sub_cfg(*, difficulty: float = 0.5) -> SimpleNamespace:
    return SimpleNamespace(
        stake_width_range=(0.4, 0.4),
        stake_distance_range=(0.2, 0.2),
        stake_height=0.0,
        holes_depth=-0.3,
        platform_width=1.5,
        quincunx=True,
        border_width=1.0,
        size=(8.0, 8.0),
        difficulty=difficulty,
    )


class _FakeScene:
    def __init__(
        self,
        robot: SimpleNamespace,
        terrain: SimpleNamespace,
        sensors: dict[str, SimpleNamespace],
    ) -> None:
        self.terrain = terrain
        self.sensors = sensors
        self._entities = {"robot": robot}

    def __getitem__(self, name: str) -> SimpleNamespace:
        if name in self._entities:
            return self._entities[name]
        return self.sensors[name]


class _FakeTerrainGenerator:
    def __init__(self, sub_cfg: SimpleNamespace) -> None:
        self._sub_cfg = sub_cfg

    def get_subterrain_cfg(self, row_ids: int, col_ids: int) -> SimpleNamespace:
        del row_ids, col_ids
        return self._sub_cfg


class _FakeTerrainGeneratorCfg:
    def __init__(self, *, num_cols: int = 1, sub_cfg: SimpleNamespace | None = None) -> None:
        self.num_cols = num_cols
        stakes_cfg = sub_cfg if sub_cfg is not None else SimpleNamespace(proportion=1.0)
        if not hasattr(stakes_cfg, "proportion"):
            stakes_cfg = copy.copy(stakes_cfg)
            stakes_cfg.proportion = 1.0
        self.sub_terrains: dict[str, SimpleNamespace]
        self.sub_terrains = {
            "plum_blossom_stakes": stakes_cfg,
        }


def _make_stakes_env(
    *,
    in_contact: bool,
    foot_x: float,
    foot_y: float,
    num_feet: int = 2,
    terrain_name: str = "plum_blossom_stakes",
    with_runtime_generator: bool = True,
    terrain_origin_x: float = 0.0,
    terrain_origin_y: float = 0.0,
) -> SimpleNamespace:
    sub_cfg = _stakes_sub_cfg()
    if terrain_name == "plum_blossom_stakes":
        stakes_entry = copy.copy(sub_cfg)
        stakes_entry.proportion = 1.0
        gen_cfg = _FakeTerrainGeneratorCfg(num_cols=1, sub_cfg=stakes_entry)
    else:
        gen_cfg = _FakeTerrainGeneratorCfg(num_cols=1)
        gen_cfg.sub_terrains = {terrain_name: SimpleNamespace(proportion=1.0)}
    terrain = SimpleNamespace(
        terrain_generator=_FakeTerrainGenerator(sub_cfg)
        if with_runtime_generator and terrain_name == "plum_blossom_stakes"
        else None,
        terrain_generator_cfg=gen_cfg,
        terrain_origins=torch.tensor([[[terrain_origin_x, terrain_origin_y, 0.0]]]),
        terrain_levels=torch.tensor([0]),
        terrain_types=torch.tensor([0]),
        cfg=SimpleNamespace(terrain_generator=gen_cfg),
    )
    contact_val = 5.0 if in_contact else 0.0
    force_history = torch.full((1, num_feet, 3, 3), contact_val)
    sensor = SimpleNamespace(data=SimpleNamespace(force_history=force_history))
    robot = SimpleNamespace(
        num_bodies=2,
        data=SimpleNamespace(
            body_link_pos_w=torch.tensor([[[foot_x, foot_y, 0.9], [foot_x, foot_y, 0.9]]]),
        ),
    )
    return SimpleNamespace(
        num_envs=1,
        device="cpu",
        scene=_FakeScene(
            robot=robot,
            terrain=terrain,
            sensors={"contact_forces": sensor},
        ),
    )


def _make_term(env: SimpleNamespace) -> cref_stepping_stone_foothold_reward:
    return cref_stepping_stone_foothold_reward(
        RewardTermCfg(
            func=cref_stepping_stone_foothold_reward,
            weight=1.0,
            params={
                "sensor_name": "contact_forces",
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
                ),
                "contact_force_threshold": 1.0,
                "s_xy": 0.15,
            },
        ),
        env,
    )


def _local_xy_to_pixel(
    x_local: float,
    y_local: float,
    size_x: float,
    size_y: float,
    horizontal_scale: float,
) -> tuple[int, int]:
    px = int(round((x_local + 0.5 * size_x) / horizontal_scale))
    py = int(round((y_local + 0.5 * size_y) / horizontal_scale))
    return px, py


def test_plum_blossom_stake_centers_geometry() -> None:
    sub_cfg = _stakes_sub_cfg()
    centers = plum_blossom_stake_centers_local(sub_cfg, 0.5, apply_border=False)
    assert len(centers) > 0
    xs = [x for x, _y, _z in centers]
    ys = [y for _x, y, _z in centers]
    assert min(xs) < 0.0
    assert max(xs) > 0.0
    assert min(ys) < 0.0 or max(ys) > 0.0
    inner_half_x = 0.5 * float(sub_cfg.size[0])
    assert all(-inner_half_x + sub_cfg.platform_width <= x <= inner_half_x - sub_cfg.platform_width for x in xs)


def test_plum_blossom_heightfield_stakes_and_holes() -> None:
    cfg = PerlinPlumBlossomStakesTerrainCfg(
        size=(8.0, 8.0),
        horizontal_scale=0.05,
        vertical_scale=0.005,
        stake_width_range=(0.4, 0.4),
        stake_distance_range=(0.2, 0.2),
        stake_height=0.0,
        holes_depth=-0.3,
        platform_width=1.5,
        quincunx=True,
        perlin_cfg=None,
    )
    height_field = cfg._generate_height_field(difficulty=0.5, cfg_for_gen=cfg)
    stake_height_i = round(float(cfg.stake_height) / cfg.vertical_scale)
    holes_depth_i = round(float(cfg.holes_depth) / cfg.vertical_scale)

    centers = plum_blossom_stake_centers_local(cfg, 0.5, apply_border=False)
    assert len(centers) > 0
    for x_local, y_local, _z_local in centers:
        px, py = _local_xy_to_pixel(x_local, y_local, float(cfg.size[0]), float(cfg.size[1]), cfg.horizontal_scale)
        assert int(height_field[py, px]) == stake_height_i

    mid_x = 0.5 * (centers[0][0] + centers[1][0]) if len(centers) > 1 else centers[0][0]
    mid_y = 0.5 * (centers[0][1] + centers[1][1]) if len(centers) > 1 else centers[0][1] + 0.5
    px_mid, py_mid = _local_xy_to_pixel(mid_x, mid_y, float(cfg.size[0]), float(cfg.size[1]), cfg.horizontal_scale)
    assert int(height_field[py_mid, px_mid]) == holes_depth_i


def test_cref_stepping_stone_foothold_reward_shape() -> None:
    env = _make_stakes_env(in_contact=False, foot_x=0.0, foot_y=0.0)
    term = _make_term(env)
    reward = term(env)
    assert reward.shape == (1,)
    assert reward.dtype == torch.float32


def test_no_contact_or_touchdown_without_liftoff_is_zero() -> None:
    env = _make_stakes_env(in_contact=False, foot_x=0.0, foot_y=0.0)
    term = _make_term(env)
    assert term(env).item() == 0.0

    env.scene.sensors["contact_forces"].data.force_history.fill_(5.0)
    assert term(env).item() == 0.0


def test_liftoff_freeze_then_touchdown_near_candidate_rewards() -> None:
    centers = plum_blossom_stake_centers_local(_stakes_sub_cfg(), 0.5, apply_border=False)
    start_x, start_y, _ = centers[0]
    target_x, target_y, _ = centers[1]
    env = _make_stakes_env(in_contact=True, foot_x=start_x, foot_y=start_y)
    term = _make_term(env)
    term.reset()

    env.scene.sensors["contact_forces"].data.force_history.zero_()
    assert term(env).item() == 0.0

    env.scene.sensors["contact_forces"].data.force_history.fill_(5.0)
    env.scene["robot"].data.body_link_pos_w[:, :, 0] = target_x
    env.scene["robot"].data.body_link_pos_w[:, :, 1] = target_y
    reward = term(env).item()
    assert reward > 0.9


def test_touchdown_far_from_candidate_rewards_less() -> None:
    centers = plum_blossom_stake_centers_local(_stakes_sub_cfg(), 0.5, apply_border=False)
    start_x, start_y, _ = centers[0]
    target_x, target_y, _ = centers[1]
    env_near = _make_stakes_env(in_contact=True, foot_x=start_x, foot_y=start_y)
    env_far = _make_stakes_env(in_contact=True, foot_x=start_x, foot_y=start_y)
    term_near = _make_term(env_near)
    term_far = _make_term(env_far)
    term_near.reset()
    term_far.reset()

    env_near.scene.sensors["contact_forces"].data.force_history.zero_()
    env_far.scene.sensors["contact_forces"].data.force_history.zero_()
    term_near(env_near)
    term_far(env_far)

    env_near.scene.sensors["contact_forces"].data.force_history.fill_(5.0)
    env_far.scene.sensors["contact_forces"].data.force_history.fill_(5.0)
    env_near.scene["robot"].data.body_link_pos_w[:, :, 0] = target_x
    env_near.scene["robot"].data.body_link_pos_w[:, :, 1] = target_y
    env_far.scene["robot"].data.body_link_pos_w[:, :, 0] = target_x + 1.5
    env_far.scene["robot"].data.body_link_pos_w[:, :, 1] = target_y + 1.0

    near_reward = term_near(env_near).item()
    far_reward = term_far(env_far).item()
    assert near_reward > far_reward
    assert far_reward < 0.5


def test_non_stakes_terrain_returns_zero() -> None:
    env = _make_stakes_env(
        in_contact=True,
        foot_x=0.0,
        foot_y=0.0,
        terrain_name="perlin_rough",
    )
    term = _make_term(env)
    assert term(env).item() == 0.0


def test_liftoff_forward_filter_excludes_rear_stakes() -> None:
    centers = plum_blossom_stake_centers_local(_stakes_sub_cfg(), 0.5, apply_border=False)
    liftoff_x = -1.0
    env = _make_stakes_env(in_contact=True, foot_x=liftoff_x, foot_y=centers[0][1])
    term = _make_term(env)
    term.reset()

    env.scene.sensors["contact_forces"].data.force_history.zero_()
    term(env)

    frozen_x = term.frozen_candidates_xy[0, 0, : term.frozen_candidate_counts[0, 0], 0]
    assert frozen_x.numel() > 0
    assert torch.all(frozen_x > liftoff_x + 0.02)
    rear_x = torch.tensor([x for x, _y, _z in centers if x <= liftoff_x])
    assert rear_x.numel() > 0
    for rear in rear_x.tolist():
        assert not torch.any(torch.isclose(frozen_x, torch.tensor(rear), atol=1e-4))


def test_xy_distance_matches_exp_reward() -> None:
    centers = plum_blossom_stake_centers_local(_stakes_sub_cfg(), 0.5, apply_border=False)
    start_x, start_y, _ = centers[0]
    target_x, target_y, _ = centers[1]
    env = _make_stakes_env(in_contact=True, foot_x=start_x, foot_y=start_y)
    term = _make_term(env)
    term.reset()

    env.scene.sensors["contact_forces"].data.force_history.zero_()
    term(env)

    env.scene.sensors["contact_forces"].data.force_history.fill_(5.0)
    offset = 0.05
    env.scene["robot"].data.body_link_pos_w[:, :, 0] = target_x + offset
    env.scene["robot"].data.body_link_pos_w[:, :, 1] = target_y
    reward = term(env).item()
    expected = 2.0 * math.exp(-offset / 0.15)
    assert abs(reward - expected) < 0.05


def test_touchdown_clears_armed_and_does_not_repeat_reward() -> None:
    centers = plum_blossom_stake_centers_local(_stakes_sub_cfg(), 0.5, apply_border=False)
    start_x, start_y, _ = centers[0]
    target_x, target_y, _ = centers[1]
    env = _make_stakes_env(in_contact=True, foot_x=start_x, foot_y=start_y)
    term = _make_term(env)
    term.reset()

    env.scene.sensors["contact_forces"].data.force_history.zero_()
    term(env)

    env.scene.sensors["contact_forces"].data.force_history.fill_(5.0)
    env.scene["robot"].data.body_link_pos_w[:, :, 0] = target_x
    env.scene["robot"].data.body_link_pos_w[:, :, 1] = target_y
    first_reward = term(env).item()
    assert first_reward > 0.9
    assert not term.foot_armed.any()
    assert term.frozen_candidate_counts.sum().item() == 0
    assert term(env).item() == 0.0


def test_reset_avoids_initial_touchdown_reward() -> None:
    centers = plum_blossom_stake_centers_local(_stakes_sub_cfg(), 0.5, apply_border=False)
    target_x, target_y, _ = centers[0]
    env = _make_stakes_env(in_contact=True, foot_x=target_x, foot_y=target_y)
    term = _make_term(env)
    term.reset()
    assert term(env).item() == 0.0
