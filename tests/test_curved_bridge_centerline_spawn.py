from __future__ import annotations

import copy
import math

import mujoco
import numpy as np

from instinct_mj.tasks.parkour.config.terrains.presets import BRIDGE_L_CORNER_TERRAINS_CFG
from instinct_mj.terrains.height_field.hf_terrains import (
    curved_bridge_build_centerline_xy,
    curved_bridge_centerline_patch,
    curved_bridge_centerline_yaw,
    curved_bridge_curriculum_t,
    curved_bridge_resolve_radius,
    curved_bridge_use_straight_centerline,
)
from instinct_mj.terrains.height_field.hf_terrains_cfg import (
    FixedSequenceFlatPatchSamplingCfg,
    PerlinCurvedBridgeTerrainCfg,
)


def _make_cfg() -> PerlinCurvedBridgeTerrainCfg:
    return PerlinCurvedBridgeTerrainCfg(
        size=(8.0, 8.0),
        horizontal_scale=0.05,
        vertical_scale=0.005,
        bridge_length=10.0,
        bridge_width=0.5,
        bridge_height=0.1,
        curve_radius_range=(2.0, 1.0),
        perlin_cfg=None,
    )


def test_curved_bridge_start_pose_on_centerline_with_tangent_yaw() -> None:
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
    assert hf_sub.shape == (width_px - 2 * border_px, length_px - 2 * border_px)

    cx, cy = curved_bridge_build_centerline_xy(difficulty, cfg_sub, tile_size=cfg_sub.size)
    assert float(np.std(cy)) < 1.0e-6
    deck_i = int(np.argmax(cx >= 1.0 - 1.0e-6))
    assert abs(cy[deck_i] - 2.0) < 1.0e-6
    yaw = curved_bridge_centerline_yaw(cx, cy, deck_i)
    assert abs(yaw) < 0.05

    heights_i16 = np.zeros((width_px, length_px), dtype=np.int16)
    heights_i16[border_px:-border_px, border_px:-border_px] = hf_sub
    elevation_min = int(np.min(heights_i16))
    heights_phys = (heights_i16.astype(np.float64) - float(elevation_min)) * float(cfg.vertical_scale)
    z_offset = float(elevation_min) * float(cfg.vertical_scale)

    start = curved_bridge_centerline_patch(
        difficulty,
        cfg_sub,
        heights_i16=heights_i16,
        heights_phys=heights_phys,
        z_offset=z_offset,
        endpoint="start",
        terrain_size=cfg_sub.size,
        border_xy=(cfg.border_width, cfg.border_width),
        spawn_on_ground=True,
        ground_approach_m=0.75,
    )
    assert start.shape == (1, 4)
    deck_i = int(np.argmax(cx >= 1.0 - 1.0e-6))
    deck_x = float(cx[deck_i]) + cfg.border_width
    deck_y = float(cy[deck_i]) + cfg.border_width
    assert start[0, 2] < cfg.bridge_height * 0.5
    assert abs(start[0, 3] - yaw) < 0.05
    assert start[0, 0] < deck_x - 0.3


def test_curved_bridge_end_target_at_centerline_then_tangent_offset() -> None:
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
    cx, cy = curved_bridge_build_centerline_xy(difficulty, cfg_sub, tile_size=cfg_sub.size)
    end_i = len(cx) - 1
    center_x = float(cx[end_i]) + cfg.border_width
    exit_yaw = curved_bridge_centerline_yaw(cx, cy, end_i)
    tangent_m = 0.5

    hf_sub = cfg._generate_height_field(difficulty=difficulty, cfg_for_gen=cfg_sub)
    heights_i16 = np.zeros((width_px, length_px), dtype=np.int16)
    heights_i16[border_px:-border_px, border_px:-border_px] = hf_sub
    elevation_min = int(np.min(heights_i16))
    z_offset = float(elevation_min) * float(cfg.vertical_scale)
    target = curved_bridge_centerline_patch(
        difficulty,
        cfg_sub,
        heights_i16=heights_i16,
        heights_phys=np.zeros(1),
        z_offset=z_offset,
        endpoint="end",
        border_xy=(cfg.border_width, cfg.border_width),
        spawn_on_ground=False,
        tangent_offset_m=tangent_m,
    )
    assert abs(target[0, 0] - (center_x + tangent_m * math.cos(exit_yaw))) < 0.05
    assert abs(target[0, 1] - (float(cy[end_i]) + cfg.border_width + tangent_m * math.sin(exit_yaw))) < 0.05


def test_curved_bridge_end_target_on_ground_ahead_of_exit() -> None:
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

    cx, cy = curved_bridge_build_centerline_xy(difficulty, cfg_sub, tile_size=cfg_sub.size)
    end_i = len(cx) - 1
    exit_yaw = curved_bridge_centerline_yaw(cx, cy, end_i)

    heights_i16 = np.zeros((width_px, length_px), dtype=np.int16)
    heights_i16[border_px:-border_px, border_px:-border_px] = hf_sub
    elevation_min = int(np.min(heights_i16))
    heights_phys = (heights_i16.astype(np.float64) - float(elevation_min)) * float(cfg.vertical_scale)
    z_offset = float(elevation_min) * float(cfg.vertical_scale)

    tangent_m = 0.5
    target = curved_bridge_centerline_patch(
        difficulty,
        cfg_sub,
        heights_i16=heights_i16,
        heights_phys=heights_phys,
        z_offset=z_offset,
        endpoint="end",
        terrain_size=cfg_sub.size,
        border_xy=(cfg.border_width, cfg.border_width),
        spawn_on_ground=True,
        tangent_offset_m=tangent_m,
        ground_approach_m=0.75,
    )
    center_x = float(cx[end_i]) + cfg.border_width
    deck_y = float(cy[end_i]) + cfg.border_width
    assert target.shape == (1, 4)
    assert target[0, 2] < cfg.bridge_height * 0.5
    assert target[0, 0] > center_x + tangent_m * math.cos(exit_yaw) - 0.05
    assert abs(target[0, 3] - exit_yaw) < 0.05


def test_curved_bridge_function_emits_yaw_flat_patches() -> None:
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


def test_curved_bridge_low_curriculum_difficulty_is_straight() -> None:
    cfg = _make_cfg()
    cfg.straight_at_zero_difficulty = True
    cfg.straight_max_difficulty = 0.1
    assert curved_bridge_use_straight_centerline(0.05, cfg)
    cx, cy = curved_bridge_build_centerline_xy(0.05, cfg, tile_size=(6.0, 6.0))
    assert float(np.std(cy)) < 1.0e-6


def test_curved_bridge_curriculum_remaps_to_gentle_s_at_first_s_row() -> None:
    cfg = _make_cfg()
    cfg.straight_at_zero_difficulty = True
    cfg.straight_max_difficulty = 0.1
    assert curved_bridge_curriculum_t(0.05, cfg) == 0.0
    assert abs(curved_bridge_curriculum_t(0.1, cfg)) < 1.0e-9
    assert abs(curved_bridge_curriculum_t(1.0, cfg) - 1.0) < 1.0e-9
    assert abs(curved_bridge_resolve_radius(0.1, cfg, tile_size=(12.0, 12.0)) - 2.0) < 1.0e-6
    assert abs(curved_bridge_resolve_radius(1.0, cfg, tile_size=(6.0, 6.0)) - 1.0) < 1.0e-6


def test_curved_bridge_first_s_row_enters_with_forward_tangent() -> None:
    cfg = _make_cfg()
    cfg.straight_at_zero_difficulty = True
    cfg.straight_max_difficulty = 0.1
    cx, cy = curved_bridge_build_centerline_xy(0.1, cfg, tile_size=(6.0, 6.0))
    yaw = curved_bridge_centerline_yaw(cx, cy, 0)
    assert abs(yaw) < 0.35
    assert float(np.std(cy)) > 0.05


def test_curved_bridge_hard_row_uses_available_exit_length() -> None:
    cfg = _make_cfg()
    cfg.straight_at_zero_difficulty = True
    cfg.straight_max_difficulty = 0.1
    cx, cy = curved_bridge_build_centerline_xy(1.0, cfg, tile_size=(8.0, 8.0))
    arc_len = float(np.sum(np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2)))
    assert float(cx[-1]) > 7.0
    assert arc_len > 7.0


def test_precision_curved_bridge_preset_is_long_enough() -> None:
    cfg = BRIDGE_L_CORNER_TERRAINS_CFG.sub_terrains["curved_bridge"]
    assert isinstance(cfg, PerlinCurvedBridgeTerrainCfg)
    sub_size = (
        float(BRIDGE_L_CORNER_TERRAINS_CFG.size[0]) - 2.0 * float(cfg.border_width),
        float(BRIDGE_L_CORNER_TERRAINS_CFG.size[1]) - 2.0 * float(cfg.border_width),
    )
    cx, cy = curved_bridge_build_centerline_xy(1.0, cfg, tile_size=sub_size)
    arc_len = float(np.sum(np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2)))
    assert float(cx[-1]) > 9.0
    assert arc_len > 10.5


def test_precision_curved_bridge_preset_has_complete_s_shape() -> None:
    cfg = BRIDGE_L_CORNER_TERRAINS_CFG.sub_terrains["curved_bridge"]
    assert isinstance(cfg, PerlinCurvedBridgeTerrainCfg)
    sub_size = (
        float(BRIDGE_L_CORNER_TERRAINS_CFG.size[0]) - 2.0 * float(cfg.border_width),
        float(BRIDGE_L_CORNER_TERRAINS_CFG.size[1]) - 2.0 * float(cfg.border_width),
    )
    cx, cy = curved_bridge_build_centerline_xy(1.0, cfg, tile_size=sub_size)
    center_y = float(cy[0])
    yaw0 = curved_bridge_centerline_yaw(cx, cy, 0)
    yaw_end = curved_bridge_centerline_yaw(cx, cy, len(cx) - 1)
    assert float(np.max(cy)) > center_y + 1.0
    assert float(np.min(cy)) < center_y - 1.0
    assert abs(float(cy[-1]) - center_y) < 0.05
    assert abs(yaw0) < 0.1
    assert abs(yaw_end) < 0.1


def test_curved_bridge_s_curve_skips_deck_lead_straights() -> None:
    cfg = _make_cfg()
    cfg.straight_lead_in_m = 0.5
    cfg.straight_lead_out_m = 0.5
    cx, _cy = curved_bridge_build_centerline_xy(1.0, cfg, tile_size=(6.0, 6.0))
    assert float(cx[0]) >= 1.0 - 0.05


def test_curved_bridge_difficulty_one_is_s_curve() -> None:
    cfg = _make_cfg()
    cfg.border_width = 1.0
    cfg_sub = copy.deepcopy(cfg)
    cfg_sub.size = (6.0, 6.0)
    assert not curved_bridge_use_straight_centerline(1.0, cfg_sub)
    cx, cy = curved_bridge_build_centerline_xy(1.0, cfg_sub, tile_size=cfg_sub.size)
    assert float(np.std(cy)) > 0.2
    entrance_yaw = curved_bridge_centerline_yaw(cx, cy, 0)
    yaws = np.array([curved_bridge_centerline_yaw(cx, cy, i) for i in range(len(cx))])
    assert abs(entrance_yaw) < 0.35
    assert float(np.max(yaws)) > 0.6
    assert float(np.min(yaws)) < -0.6
