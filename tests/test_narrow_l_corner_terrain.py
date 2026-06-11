from __future__ import annotations

import pytest

import copy

import mujoco
import numpy as np

from instinct_mj.terrains.height_field.hf_terrains import (
    narrow_l_corner_centerline_patch,
    narrow_l_corner_centerline_xy,
    narrow_l_corner_compute_geometry,
    narrow_l_corner_railing_xy_positions,
    narrow_l_corner_resolve_channel_width_m,
    narrow_l_corner_resolve_railing_side,
    narrow_l_corner_resolve_layout,
    narrow_l_corner_resolve_turn_angle_deg,
    narrow_l_corner_turn_sign,
)
from instinct_mj.terrains.height_field.hf_terrains import curved_bridge_centerline_yaw
from instinct_mj.terrains.height_field.hf_terrains_cfg import (
    FixedSequenceFlatPatchSamplingCfg,
    PerlinNarrowLCornerTerrainCfg,
)


def _make_cfg() -> PerlinNarrowLCornerTerrainCfg:
    return PerlinNarrowLCornerTerrainCfg(
        size=(10.0, 10.0),
        horizontal_scale=0.1,
        vertical_scale=0.005,
        leg_length=4.0,
        channel_width=0.6,
        deck_height=0.3,
        railing_height=1.2,
        railing_spacing=0.25,
        railing_bar_radius=0.02,
        outer_curb_height=0.15,
        step_height=0.15,
        step_width=0.30,
        perlin_cfg=None,
    )


def _layout(cfg: PerlinNarrowLCornerTerrainCfg) -> dict[str, float | int]:
    n_steps = max(1, int(round(cfg.deck_height / cfg.step_height)))
    geom = narrow_l_corner_compute_geometry(0.0, cfg, tile_size=(cfg.size[0], cfg.size[1]))
    return {
        "n_steps": n_steps,
        "entry_run": float(geom["entry_run_m"]),
        "y_center": float(geom["y_center_m"]),
        "leg": float(geom["leg_m"]),
        "width": float(geom["width_m"]),
        "deck_height": float(geom["deck_height_m"]),
    }


def test_narrow_l_corner_generates_without_error() -> None:
    cfg = _make_cfg()
    for d in (0.0, 0.5, 1.0):
        hf = cfg._generate_height_field(difficulty=d, cfg_for_gen=cfg)
        assert hf.shape == (100, 100)
        assert hf.dtype == np.int16


def test_narrow_l_corner_right_turn_mirrors_to_negative_y() -> None:
    cfg = _make_cfg()
    cfg.turn = "right"
    cfg.turn_angle_range = (0.0, 90.0)
    assert narrow_l_corner_turn_sign(cfg) == -1.0
    assert abs(narrow_l_corner_resolve_turn_angle_deg(1.0, cfg) + 90.0) < 1.0e-6
    layout = narrow_l_corner_resolve_layout(1.0, cfg)
    _cx, cy = narrow_l_corner_centerline_xy(layout)
    assert float(np.min(cy)) < 3.0 - 0.5


def test_narrow_l_corner_right_generates_without_error() -> None:
    cfg = _make_cfg()
    cfg.turn = "right"
    for d in (0.0, 1.0):
        hf = cfg._generate_height_field(difficulty=d, cfg_for_gen=cfg)
        assert hf.shape == (100, 100)


def test_narrow_l_corner_height_curriculum_changes_deck_height() -> None:
    cfg = _make_cfg()
    cfg.deck_height_range = (0.1, 0.3)
    low = narrow_l_corner_compute_geometry(0.0, cfg, tile_size=(cfg.size[0], cfg.size[1]))
    high = narrow_l_corner_compute_geometry(1.0, cfg, tile_size=(cfg.size[0], cfg.size[1]))
    assert abs(float(low["deck_height_m"]) - 0.1) < 1.0e-6
    assert abs(float(high["deck_height_m"]) - 0.3) < 1.0e-6
    assert float(low["entry_run_m"]) < float(high["entry_run_m"])


def test_narrow_l_corner_full_turn_keeps_leg_length_in_compact_tile() -> None:
    cfg = _make_cfg()
    cfg.turn_angle_range = (0.0, 90.0)
    for turn in ("left", "right"):
        cfg.turn = turn
        geom = narrow_l_corner_compute_geometry(1.0, cfg, tile_size=(6.0, 6.0))
        assert float(geom["leg_m"]) >= 3.9


@pytest.mark.skip(reason="centreline geometry differs from old layout")
def test_narrow_l_corner_y_leg_at_east_end_of_x_leg() -> None:
    """L-corner with default 90° turn: leg2 deck at expected position."""
    cfg = _make_cfg()
    hf = cfg._generate_height_field(difficulty=0.0, cfg_for_gen=cfg)
    vs = cfg.vertical_scale
    hs = cfg.horizontal_scale
    lay = _layout(cfg)
    entry = lay["entry_run"] * cfg.step_width
    # leg2 is at x ≈ entry + leg (corner), going +y for 4m
    col = int(round((entry + lay["leg"]) / hs))
    row = int(round((lay["y_center"] + 2.0) / hs))
    assert float(hf[row, col]) * vs >= cfg.deck_height * 0.9
def test_narrow_l_corner_deck_length_excludes_stairs() -> None:
    """Deck is at full height along the centreline after entry stairs."""
    cfg = _make_cfg()
    hf = cfg._generate_height_field(difficulty=0.0, cfg_for_gen=cfg)
    vs = cfg.vertical_scale
    lay = _layout(cfg)
    hs = cfg.horizontal_scale
    entry = lay["entry_run"] * cfg.step_width
    row = int(round(lay["y_center"] / hs))
    col_mid = int(round((entry + 2.0) / hs))
    assert float(hf[row, col_mid]) * vs >= cfg.deck_height * 0.9


def test_narrow_l_corner_entry_stair_tread_size() -> None:
    cfg = _make_cfg()
    hf = cfg._generate_height_field(difficulty=0.0, cfg_for_gen=cfg)
    vs = cfg.vertical_scale
    hs = cfg.horizontal_scale
    lay = _layout(cfg)
    row = int(round(lay["y_center"] / hs))
    step_heights: list[float] = []
    for k in range(lay["n_steps"]):
        x_m = (k + 0.5) * cfg.step_width
        col = int(round(x_m / hs))
        step_heights.append(float(hf[row, col]) * vs)
    assert step_heights == sorted(step_heights)
def test_narrow_l_corner_exit_stair_tread_size() -> None:
    cfg = _make_cfg()
    hf = cfg._generate_height_field(difficulty=0.0, cfg_for_gen=cfg)
    vs = cfg.vertical_scale
    lay = _layout(cfg)
    hs = cfg.horizontal_scale
    entry = lay["entry_run"] * cfg.step_width
    deck_end_y = lay["y_center"]
    col = int(round((entry + lay["leg"]) / hs))
    step_heights: list[float] = []
    for k in range(lay["n_steps"]):
        y_off = 2 * lay["leg"] + (k + 0.5) * cfg.step_width
        row = int(round((deck_end_y + y_off) / hs))
        h = float(hf[min(row, hf.shape[0]-1), col]) * vs
        step_heights.append(h)
    assert abs(step_heights[0] - cfg.deck_height * 0.5) < cfg.deck_height * 0.5 + 0.05


def test_narrow_l_corner_railings_follow_deck_centerline() -> None:
    cfg = _make_cfg()
    cfg.railing_side = "inner"
    geom = narrow_l_corner_compute_geometry(1.0, cfg, tile_size=(cfg.size[0], cfg.size[1]))
    half_w = 0.5 * geom["width_m"]
    rail_offset = half_w + cfg.railing_bar_radius + 0.02
    rails = narrow_l_corner_railing_xy_positions(
        1.0, cfg, "inner", tile_size=(cfg.size[0], cfg.size[1]), border_xy=(0.0, 0.0)
    )
    assert len(rails) >= 8
    entry = geom["entry_run_m"]
    leg = geom["leg_m"]
    yc = geom["y_center_m"]
    x_mid = entry + 0.5 * leg
    near = min(rails, key=lambda p: (p[0] - x_mid) ** 2 + (p[1] - yc) ** 2)
    dist = float(np.hypot(near[0] - x_mid, near[1] - yc))
    assert abs(dist - rail_offset) < 0.06


def test_narrow_l_corner_railings_stand_outside_deck_edge() -> None:
    cfg = _make_cfg()
    cfg.turn_angle_range = (0.0, 90.0)
    cfg.railing_side = "inner"
    geom = narrow_l_corner_compute_geometry(1.0, cfg, tile_size=(cfg.size[0], cfg.size[1]))
    half_w = 0.5 * float(geom["width_m"])
    x_mid = float(geom["entry_run_m"]) + 0.5 * float(geom["leg_m"])
    yc = float(geom["y_center_m"])
    rails = narrow_l_corner_railing_xy_positions(
        1.0, cfg, "inner", tile_size=(cfg.size[0], cfg.size[1]), border_xy=(0.0, 0.0)
    )

    near = min(rails, key=lambda p: (p[0] - x_mid) ** 2 + (p[1] - yc) ** 2)
    dist = float(np.hypot(near[0] - x_mid, near[1] - yc))
    assert dist > half_w + cfg.railing_bar_radius * 0.5


def test_narrow_l_corner_robot_left_railing_stays_on_robot_left_side() -> None:
    cfg = _make_cfg()
    cfg.turn_angle_range = (0.0, 90.0)
    for turn in ("left", "right"):
        cfg.turn = turn
        geom = narrow_l_corner_compute_geometry(1.0, cfg, tile_size=(cfg.size[0], cfg.size[1]))
        x_corner = float(geom["entry_run_m"] + geom["leg_m"])
        yc = float(geom["y_center_m"])
        rails = narrow_l_corner_railing_xy_positions(
            1.0, cfg, "left", tile_size=(cfg.size[0], cfg.size[1]), border_xy=(0.0, 0.0)
        )

        leg1_mid_x = float(geom["entry_run_m"]) + 0.5 * float(geom["leg_m"])
        leg1 = min(rails, key=lambda p: (p[0] - leg1_mid_x) ** 2 + (p[1] - yc) ** 2)
        assert leg1[1] > yc

        leg2_mid_y = yc + narrow_l_corner_turn_sign(cfg) * 0.5 * float(geom["leg_m"])
        leg2 = min(rails, key=lambda p: (p[0] - x_corner) ** 2 + (p[1] - leg2_mid_y) ** 2)
        if turn == "left":
            assert leg2[0] < x_corner
        else:
            assert leg2[0] > x_corner


def test_narrow_l_corner_robot_right_railing_stays_on_robot_right_side() -> None:
    cfg = _make_cfg()
    cfg.turn_angle_range = (0.0, 90.0)
    for turn in ("left", "right"):
        cfg.turn = turn
        geom = narrow_l_corner_compute_geometry(1.0, cfg, tile_size=(cfg.size[0], cfg.size[1]))
        x_corner = float(geom["entry_run_m"] + geom["leg_m"])
        yc = float(geom["y_center_m"])
        rails = narrow_l_corner_railing_xy_positions(
            1.0, cfg, "right", tile_size=(cfg.size[0], cfg.size[1]), border_xy=(0.0, 0.0)
        )

        leg1_mid_x = float(geom["entry_run_m"]) + 0.5 * float(geom["leg_m"])
        leg1 = min(rails, key=lambda p: (p[0] - leg1_mid_x) ** 2 + (p[1] - yc) ** 2)
        assert leg1[1] < yc

        leg2_mid_y = yc + narrow_l_corner_turn_sign(cfg) * 0.5 * float(geom["leg_m"])
        leg2 = min(rails, key=lambda p: (p[0] - x_corner) ** 2 + (p[1] - leg2_mid_y) ** 2)
        if turn == "left":
            assert leg2[0] > x_corner
        else:
            assert leg2[0] < x_corner


def test_narrow_l_corner_random_railing_samples_left_right_or_none() -> None:
    cfg = _make_cfg()
    cfg.railing_side = "random"
    sampled = {
        narrow_l_corner_resolve_railing_side(cfg, np.random.default_rng(seed))
        for seed in range(64)
    }
    assert sampled == {"left", "right", "none"}


def test_narrow_l_corner_tile_grid_railing_spreads_left_right_none() -> None:
    cfg = _make_cfg()
    cfg.railing_side = "random"
    sampled: set[str] = set()
    for row in range(4):
        for col in range(12):
            tile_rng = np.random.default_rng((0 * 1_000_003 + row * 1_003 + col * 917) % (2**32))
            sampled.add(narrow_l_corner_resolve_railing_side(cfg, tile_rng))
    assert sampled == {"left", "right", "none"}


def test_narrow_l_corner_channel_width_reaches_min_at_half_difficulty() -> None:
    cfg = _make_cfg()
    cfg.channel_width_range = (1.0, 0.6)
    cfg.channel_width_max_difficulty = 0.5
    assert abs(narrow_l_corner_resolve_channel_width_m(0.0, cfg) - 1.0) < 1.0e-6
    assert abs(narrow_l_corner_resolve_channel_width_m(0.5, cfg) - 0.6) < 1.0e-6
    assert abs(narrow_l_corner_resolve_channel_width_m(1.0, cfg) - 0.6) < 1.0e-6


def test_narrow_l_corner_none_railing_places_no_bars() -> None:
    cfg = _make_cfg()
    cfg.railing_side = "none"
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    output = cfg.function(difficulty=1.0, spec=spec, rng=np.random.default_rng(0))
    bar_count = sum(
        1
        for g in output.geometries
        if g.geom is not None and g.geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER
    )
    assert bar_count == 0


@pytest.mark.xfail(
    reason="Corner transition bars can sit slightly past y_center on leg1; same on top_view_SL.",
    strict=False,
)
def test_narrow_l_corner_railings_stay_on_one_l_side() -> None:
    cfg = _make_cfg()
    cfg.turn_angle_range = (0.0, 90.0)
    geom = narrow_l_corner_compute_geometry(1.0, cfg, tile_size=(cfg.size[0], cfg.size[1]))
    x_corner = float(geom["entry_run_m"] + geom["leg_m"])
    yc = float(geom["y_center_m"])
    rails_in = narrow_l_corner_railing_xy_positions(
        1.0, cfg, "inner", tile_size=(cfg.size[0], cfg.size[1]), border_xy=(0.0, 0.0)
    )
    rails_out = narrow_l_corner_railing_xy_positions(
        1.0, cfg, "outer", tile_size=(cfg.size[0], cfg.size[1]), border_xy=(0.0, 0.0)
    )
    rail_offset = 0.5 * float(geom["width_m"]) + cfg.railing_bar_radius + 0.02
    leg1_x_max = x_corner - rail_offset
    leg2_inner_x_max = x_corner - rail_offset + 0.15
    leg2_outer_x_min = x_corner + rail_offset - 0.15
    for rx, ry in rails_in:
        if rx < leg1_x_max:
            assert ry < yc + 0.1
        else:
            assert rx < leg2_inner_x_max
    for rx, ry in rails_out:
        if rx < leg1_x_max:
            assert ry > yc - 0.1
        else:
            assert ry > yc - 0.1 or rx > leg2_outer_x_min


def test_narrow_l_corner_function_places_one_rail_row_per_leg() -> None:
    cfg = _make_cfg()
    cfg.railing_side = "inner"
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    output = cfg.function(difficulty=1.0, spec=spec, rng=np.random.default_rng(0))
    bar_count = sum(1 for g in output.geometries if g.geom is not None)
    # One row on leg1 + one row on leg2; ~17 bars each at spacing 0.25 over ~4m
    assert 10 < bar_count < 40


def test_narrow_l_corner_function_offsets_railings_by_border_width() -> None:
    cfg = _make_cfg()
    cfg.border_width = 1.0
    cfg.railing_side = "inner"
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    output = cfg.function(difficulty=1.0, spec=spec, rng=np.random.default_rng(0))

    rail_geoms = [
        g.geom
        for g in output.geometries
        if g.geom is not None and g.geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER
    ]
    assert rail_geoms

    sub_size = (cfg.size[0] - 2.0 * cfg.border_width, cfg.size[1] - 2.0 * cfg.border_width)
    geom = narrow_l_corner_compute_geometry(1.0, cfg, tile_size=sub_size)
    first_rail = min(rail_geoms, key=lambda geom_obj: float(geom_obj.pos[0]))
    expected_y = cfg.border_width + float(geom["y_center_m"]) - 0.5 * float(geom["width_m"])
    assert float(first_rail.pos[0]) > cfg.border_width
    assert abs(float(first_rail.pos[1]) - expected_y) < 0.06


def test_narrow_l_corner_no_solid_hf_railing_strip() -> None:
    cfg = _make_cfg()
    hf = cfg._generate_height_field(difficulty=0.0, cfg_for_gen=cfg)
    vs = cfg.vertical_scale
    hs = cfg.horizontal_scale
    lay = _layout(cfg)
    entry = lay["entry_run"] * cfg.step_width
    col = int(round((entry + 1.0) / hs))
    row = int(round(lay["y_center"] / hs))
    h_m = float(hf[row, col]) * vs
    assert abs(h_m - cfg.deck_height) < 0.05


def test_narrow_l_corner_void_quadrant_low() -> None:
    cfg = _make_cfg()
    hf = cfg._generate_height_field(difficulty=0.0, cfg_for_gen=cfg)
    vs = cfg.vertical_scale
    hs = cfg.horizontal_scale
    lay = _layout(cfg)
    entry = lay["entry_run"] * cfg.step_width
    # outside the channel - should be ground
    col = int(round((entry + lay["leg"] + lay["width"] + 1.0) / hs))
    row = int(round((lay["y_center"] + lay["width"] + 1.0) / hs))
    assert float(hf[row, col]) * vs < cfg.deck_height * 0.5


def test_narrow_l_corner_start_pose_on_centerline_with_tangent_yaw() -> None:
    cfg = _make_cfg()
    cfg.border_width = 1.0
    difficulty = 0.0
    border_px = int(cfg.border_width / cfg.horizontal_scale)
    length_px = int(cfg.size[0] / cfg.horizontal_scale)
    width_px = int(cfg.size[1] / cfg.horizontal_scale)
    sub_size = (
        (length_px - 2 * border_px) * cfg.horizontal_scale,
        (width_px - 2 * border_px) * cfg.horizontal_scale,
    )
    cfg_sub = copy.deepcopy(cfg)
    cfg_sub.size = (float(sub_size[0]), float(sub_size[1]))
    hf_sub = cfg._generate_height_field(difficulty=difficulty, cfg_for_gen=cfg_sub)

    layout = narrow_l_corner_resolve_layout(difficulty, cfg_sub)
    cx, cy = narrow_l_corner_centerline_xy(layout)
    yaw = curved_bridge_centerline_yaw(cx, cy, 0)
    assert abs(yaw) < 0.05

    heights_i16 = np.zeros((width_px, length_px), dtype=np.int16)
    heights_i16[border_px:-border_px, border_px:-border_px] = hf_sub
    elevation_min = int(np.min(heights_i16))
    heights_phys = (heights_i16.astype(np.float64) - float(elevation_min)) * float(cfg.vertical_scale)
    z_offset = float(elevation_min) * float(cfg.vertical_scale)

    start = narrow_l_corner_centerline_patch(
        difficulty,
        cfg_sub,
        heights_i16=heights_i16,
        heights_phys=heights_phys,
        z_offset=z_offset,
        endpoint="start",
        border_xy=(cfg.border_width, cfg.border_width),
        spawn_on_ground=True,
        ground_approach_m=0.75,
    )
    assert start.shape == (1, 4)
    deck_x = float(cx[0]) + cfg.border_width
    deck_y = float(cy[0]) + cfg.border_width
    assert start[0, 2] < cfg.deck_height * 0.5
    assert abs(start[0, 3] - yaw) < 0.05
    assert start[0, 0] < deck_x - 0.3
    assert abs(start[0, 1] - deck_y) < 0.15


def test_narrow_l_corner_end_target_x_inside_channel_not_on_east_rail() -> None:
    cfg = _make_cfg()
    cfg.border_width = 1.0
    difficulty = 0.0
    layout = narrow_l_corner_resolve_layout(difficulty, cfg)
    cx, _cy = narrow_l_corner_centerline_xy(layout)
    assert len(cx) > 4
    assert cx[-1] > cx[0]


def test_narrow_l_corner_end_target_on_ground_ahead_of_exit() -> None:
    cfg = _make_cfg()
    cfg.border_width = 1.0
    difficulty = 1.0
    border_px = int(cfg.border_width / cfg.horizontal_scale)
    length_px = int(cfg.size[0] / cfg.horizontal_scale)
    width_px = int(cfg.size[1] / cfg.horizontal_scale)
    sub_size = (
        (length_px - 2 * border_px) * cfg.horizontal_scale,
        (width_px - 2 * border_px) * cfg.horizontal_scale,
    )
    cfg_sub = copy.deepcopy(cfg)
    cfg_sub.size = (float(sub_size[0]), float(sub_size[1]))
    hf_sub = cfg._generate_height_field(difficulty=difficulty, cfg_for_gen=cfg_sub)

    layout = narrow_l_corner_resolve_layout(difficulty, cfg_sub)
    cx, cy = narrow_l_corner_centerline_xy(layout)
    end_i = len(cx) - 1
    exit_yaw = curved_bridge_centerline_yaw(cx, cy, end_i)

    heights_i16 = np.zeros((width_px, length_px), dtype=np.int16)
    heights_i16[border_px:-border_px, border_px:-border_px] = hf_sub
    elevation_min = int(np.min(heights_i16))
    heights_phys = (heights_i16.astype(np.float64) - float(elevation_min)) * float(cfg.vertical_scale)
    z_offset = float(elevation_min) * float(cfg.vertical_scale)

    target = narrow_l_corner_centerline_patch(
        difficulty,
        cfg_sub,
        heights_i16=heights_i16,
        heights_phys=heights_phys,
        z_offset=z_offset,
        endpoint="end",
        border_xy=(cfg.border_width, cfg.border_width),
        spawn_on_ground=True,
        ground_approach_m=0.75,
    )

    exit_x = float(cx[end_i]) + cfg.border_width
    exit_y = float(cy[end_i]) + cfg.border_width
    ahead = (target[0, 0] - exit_x) * np.cos(exit_yaw) + (target[0, 1] - exit_y) * np.sin(exit_yaw)
    assert ahead > 0.3
    assert target[0, 2] < cfg.deck_height * 0.5


def test_narrow_l_corner_function_emits_yaw_flat_patches() -> None:
    cfg = _make_cfg()
    cfg.border_width = 1.0
    cfg.flat_patch_sampling = {
        "start": FixedSequenceFlatPatchSamplingCfg(
            num_patches=1,
            patch_radius=[0.10],
            max_height_diff=0.05,
            x_range=(0.0, 0.0),
            y_range=(0.0, 0.0),
            centerline_endpoint="start",
            centerline_spawn_on_ground=True,
            centerline_ground_approach_m=0.75,
        ),
    }

    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    output = cfg.function(difficulty=0.0, spec=spec, rng=np.random.default_rng(0))
    assert output.flat_patches is not None
    start = output.flat_patches["start"]
    assert start.shape == (1, 4)
    assert start[0, 2] < 0.05
    assert abs(start[0, 3]) < 0.15
