from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
from mjlab.managers import RewardTermCfg, SceneEntityCfg

_rewards_path = Path(__file__).resolve().parents[1] / "src/instinct_mj/tasks/parkour/mdp/rewards.py"
_rewards_spec = importlib.util.spec_from_file_location("parkour_rewards_test", _rewards_path)
assert _rewards_spec is not None and _rewards_spec.loader is not None
_rewards_mod = importlib.util.module_from_spec(_rewards_spec)
_rewards_spec.loader.exec_module(_rewards_mod)
_tread_candidates_xz_local_neg_x_updown = _rewards_mod._tread_candidates_xz_local_neg_x_updown
cref_stair_foothold_reward = _rewards_mod.cref_stair_foothold_reward

FOOT_HEIGHT_OFFSET = 0.065


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
        stairs_cfg = sub_cfg if sub_cfg is not None else SimpleNamespace(proportion=1.0)
        if not hasattr(stairs_cfg, "proportion"):
            stairs_cfg = copy.copy(stairs_cfg)
            stairs_cfg.proportion = 1.0
        self.sub_terrains: dict[str, SimpleNamespace]
        self.sub_terrains = {
            "one_sided_stairs": stairs_cfg,
        }


def _stairs_sub_cfg(*, difficulty: float = 0.5) -> SimpleNamespace:
    return SimpleNamespace(
        stair_face="neg_x",
        up_down=True,
        step_width=0.3,
        platform_width=1.5,
        border_width=1.0,
        size=(8.0, 8.0),
        step_height_range=(0.10, 0.10),
        inverted=False,
        difficulty=difficulty,
    )


def _ankle_z_for_tread(tread_z: float) -> float:
    return tread_z + FOOT_HEIGHT_OFFSET


def _make_stairs_env(
    *,
    in_contact: bool,
    foot_x: float,
    foot_z: float,
    num_feet: int = 2,
    terrain_name: str = "one_sided_stairs",
    with_runtime_generator: bool = True,
    terrain_origin_z: float = 0.0,
) -> SimpleNamespace:
    sub_cfg = _stairs_sub_cfg()
    if terrain_name == "one_sided_stairs":
        stairs_entry = copy.copy(sub_cfg)
        stairs_entry.proportion = 1.0
        gen_cfg = _FakeTerrainGeneratorCfg(num_cols=1, sub_cfg=stairs_entry)
    else:
        gen_cfg = _FakeTerrainGeneratorCfg(num_cols=1)
        gen_cfg.sub_terrains = {terrain_name: SimpleNamespace(proportion=1.0)}
    terrain = SimpleNamespace(
        terrain_generator=_FakeTerrainGenerator(sub_cfg) if with_runtime_generator and terrain_name == "one_sided_stairs" else None,
        terrain_generator_cfg=gen_cfg,
        terrain_origins=torch.tensor([[[0.0, 0.0, terrain_origin_z]]]),
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
            body_link_pos_w=torch.tensor([[[foot_x, 0.0, foot_z], [foot_x, 0.0, foot_z]]]),
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


def _make_term(env: SimpleNamespace) -> cref_stair_foothold_reward:
    return cref_stair_foothold_reward(
        RewardTermCfg(
            func=cref_stair_foothold_reward,
            weight=1.0,
            params={
                "sensor_name": "contact_forces",
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
                ),
                "contact_force_threshold": 1.0,
                "s_xz": 0.15,
                "foot_height_offset": FOOT_HEIGHT_OFFSET,
            },
        ),
        env,
    )


def test_tread_candidates_cover_neg_x_updown_geometry() -> None:
    treads = _tread_candidates_xz_local_neg_x_updown(_stairs_sub_cfg())
    assert len(treads) > 0
    xs = [x for x, _ in treads]
    zs = [z for _, z in treads]
    assert min(xs) < 0.0
    assert max(xs) > 0.0
    assert max(zs) > 0.0

    first_x, first_z = treads[0]
    assert abs(first_x - (-2.85)) < 1e-4
    assert abs(first_z - 0.0) < 1e-4

    inner_half = 0.5 * (8.0 - 2.0 * 1.0)
    assert all(-inner_half <= x <= inner_half for x in xs)


def test_tread_candidates_accept_platform_width_range() -> None:
    cfg = _stairs_sub_cfg()
    cfg.platform_width = (1.0, 1.5)

    treads = _tread_candidates_xz_local_neg_x_updown(cfg)

    assert len(treads) > 0


def test_cref_stair_foothold_reward_shape() -> None:
    env = _make_stairs_env(in_contact=False, foot_x=0.0, foot_z=0.0)
    term = _make_term(env)
    reward = term(env)
    assert reward.shape == (1,)
    assert reward.dtype == torch.float32


def test_no_contact_or_touchdown_without_liftoff_is_zero() -> None:
    env = _make_stairs_env(in_contact=False, foot_x=0.0, foot_z=0.0)
    term = _make_term(env)
    assert term(env).item() == 0.0

    env.scene.sensors["contact_forces"].data.force_history.fill_(5.0)
    assert term(env).item() == 0.0


def test_liftoff_freeze_then_touchdown_near_candidate_rewards() -> None:
    treads = _tread_candidates_xz_local_neg_x_updown(_stairs_sub_cfg())
    start_x, start_z = treads[0]
    target_x, target_z = treads[1]
    env = _make_stairs_env(
        in_contact=True,
        foot_x=start_x,
        foot_z=_ankle_z_for_tread(start_z),
    )
    term = _make_term(env)
    term.reset()

    env.scene.sensors["contact_forces"].data.force_history.zero_()
    assert term(env).item() == 0.0

    env.scene.sensors["contact_forces"].data.force_history.fill_(5.0)
    env.scene["robot"].data.body_link_pos_w[:, :, 0] = target_x
    env.scene["robot"].data.body_link_pos_w[:, :, 2] = _ankle_z_for_tread(target_z)
    reward = term(env).item()
    assert reward > 0.9


def test_touchdown_far_from_candidate_rewards_less() -> None:
    treads = _tread_candidates_xz_local_neg_x_updown(_stairs_sub_cfg())
    start_x, start_z = treads[0]
    target_x, target_z = treads[1]
    env_near = _make_stairs_env(
        in_contact=True,
        foot_x=start_x,
        foot_z=_ankle_z_for_tread(start_z),
    )
    env_far = _make_stairs_env(
        in_contact=True,
        foot_x=start_x,
        foot_z=_ankle_z_for_tread(start_z),
    )
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
    env_near.scene["robot"].data.body_link_pos_w[:, :, 2] = _ankle_z_for_tread(target_z)
    env_far.scene["robot"].data.body_link_pos_w[:, :, 0] = target_x + 1.5
    env_far.scene["robot"].data.body_link_pos_w[:, :, 2] = _ankle_z_for_tread(target_z + 1.0)

    near_reward = term_near(env_near).item()
    far_reward = term_far(env_far).item()
    assert near_reward > far_reward
    assert far_reward < 0.5


def test_non_stairs_terrain_returns_zero() -> None:
    env = _make_stairs_env(
        in_contact=True,
        foot_x=0.0,
        foot_z=0.0,
        terrain_name="perlin_rough",
    )
    term = _make_term(env)
    assert term(env).item() == 0.0


def test_cfg_only_terrain_generator_enables_reward() -> None:
    treads = _tread_candidates_xz_local_neg_x_updown(_stairs_sub_cfg())
    start_x, start_z = treads[0]
    target_x, target_z = treads[1]
    env = _make_stairs_env(
        in_contact=True,
        foot_x=start_x,
        foot_z=_ankle_z_for_tread(start_z),
        with_runtime_generator=False,
    )
    term = _make_term(env)
    assert term._enabled
    term.reset()
    env.scene.sensors["contact_forces"].data.force_history.zero_()
    term(env)
    env.scene.sensors["contact_forces"].data.force_history.fill_(5.0)
    env.scene["robot"].data.body_link_pos_w[:, :, 0] = target_x
    env.scene["robot"].data.body_link_pos_w[:, :, 2] = _ankle_z_for_tread(target_z)
    assert term(env).item() > 0.9


def test_missing_terrain_origins_is_safe_zero() -> None:
    env = _make_stairs_env(in_contact=True, foot_x=0.0, foot_z=0.0, with_runtime_generator=False)
    env.scene.terrain.terrain_origins = None
    term = _make_term(env)
    assert not term._enabled
    assert term(env).item() == 0.0


def test_liftoff_forward_filter_excludes_rear_treads() -> None:
    treads = _tread_candidates_xz_local_neg_x_updown(_stairs_sub_cfg())
    liftoff_x = -1.0
    env = _make_stairs_env(in_contact=True, foot_x=liftoff_x, foot_z=_ankle_z_for_tread(treads[0][1]))
    term = _make_term(env)
    term.reset()

    env.scene.sensors["contact_forces"].data.force_history.zero_()
    term(env)

    frozen_x = term.frozen_candidates_xz[0, 0, : term.frozen_candidate_counts[0, 0], 0]
    assert frozen_x.numel() > 0
    assert torch.all(frozen_x > liftoff_x + 0.02)
    rear_x = torch.tensor([x for x, _ in treads if x <= liftoff_x])
    assert rear_x.numel() > 0
    for rear in rear_x.tolist():
        assert not torch.any(torch.isclose(frozen_x, torch.tensor(rear), atol=1e-4))


def test_touchdown_on_rear_tread_lower_than_forward() -> None:
    treads = _tread_candidates_xz_local_neg_x_updown(_stairs_sub_cfg())
    liftoff_x = -1.0
    rear_x, rear_z = treads[0]
    forward_candidates = [(x, z) for x, z in treads if x > liftoff_x + 0.02]
    assert len(forward_candidates) >= 1
    forward_x, forward_z = forward_candidates[0]
    assert rear_x < liftoff_x

    env_rear = _make_stairs_env(in_contact=True, foot_x=liftoff_x, foot_z=_ankle_z_for_tread(rear_z))
    env_forward = _make_stairs_env(in_contact=True, foot_x=liftoff_x, foot_z=_ankle_z_for_tread(forward_z))
    term_rear = _make_term(env_rear)
    term_forward = _make_term(env_forward)
    term_rear.reset()
    term_forward.reset()

    for env, term in ((env_rear, term_rear), (env_forward, term_forward)):
        env.scene.sensors["contact_forces"].data.force_history.zero_()
        term(env)

    env_rear.scene.sensors["contact_forces"].data.force_history.fill_(5.0)
    env_forward.scene.sensors["contact_forces"].data.force_history.fill_(5.0)
    env_rear.scene["robot"].data.body_link_pos_w[:, :, 0] = rear_x
    env_rear.scene["robot"].data.body_link_pos_w[:, :, 2] = _ankle_z_for_tread(rear_z)
    env_forward.scene["robot"].data.body_link_pos_w[:, :, 0] = forward_x
    env_forward.scene["robot"].data.body_link_pos_w[:, :, 2] = _ankle_z_for_tread(forward_z)

    rear_reward = term_rear(env_rear).item()
    forward_reward = term_forward(env_forward).item()
    assert forward_reward > rear_reward


def test_reset_avoids_initial_touchdown_reward() -> None:
    treads = _tread_candidates_xz_local_neg_x_updown(_stairs_sub_cfg())
    target_x, target_z = treads[0]
    env = _make_stairs_env(
        in_contact=True,
        foot_x=target_x,
        foot_z=_ankle_z_for_tread(target_z),
    )
    term = _make_term(env)
    term.reset()
    assert term(env).item() == 0.0


def test_nonzero_terrain_origin_z_touchdown_rewards() -> None:
    treads = _tread_candidates_xz_local_neg_x_updown(_stairs_sub_cfg())
    start_x, start_z = treads[0]
    target_x, target_z = treads[1]
    origin_z = 1.5
    env = _make_stairs_env(
        in_contact=True,
        foot_x=start_x,
        foot_z=_ankle_z_for_tread(start_z),
        terrain_origin_z=origin_z,
    )
    term = _make_term(env)
    term.reset()

    env.scene.sensors["contact_forces"].data.force_history.zero_()
    term(env)

    env.scene.sensors["contact_forces"].data.force_history.fill_(5.0)
    env.scene["robot"].data.body_link_pos_w[:, :, 0] = target_x
    env.scene["robot"].data.body_link_pos_w[:, :, 2] = _ankle_z_for_tread(target_z)
    assert term(env).item() > 0.9


def test_touchdown_clears_armed_and_does_not_repeat_reward() -> None:
    treads = _tread_candidates_xz_local_neg_x_updown(_stairs_sub_cfg())
    start_x, start_z = treads[0]
    target_x, target_z = treads[1]
    env = _make_stairs_env(
        in_contact=True,
        foot_x=start_x,
        foot_z=_ankle_z_for_tread(start_z),
    )
    term = _make_term(env)
    term.reset()

    env.scene.sensors["contact_forces"].data.force_history.zero_()
    term(env)

    env.scene.sensors["contact_forces"].data.force_history.fill_(5.0)
    env.scene["robot"].data.body_link_pos_w[:, :, 0] = target_x
    env.scene["robot"].data.body_link_pos_w[:, :, 2] = _ankle_z_for_tread(target_z)
    first_reward = term(env).item()
    assert first_reward > 0.9
    assert not term.foot_armed.any()
    assert term.frozen_candidate_counts.sum().item() == 0
    assert term(env).item() == 0.0
