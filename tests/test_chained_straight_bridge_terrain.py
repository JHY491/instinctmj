"""Tests for chained straight bridge (modular collinear decks with gaps)."""

from __future__ import annotations

import copy

import numpy as np

from instinct_mj.tasks.parkour.config.terrains.presets import CHAINED_STRAIGHT_BRIDGE_TERRAINS_CFG
from instinct_mj.terrains.height_field.hf_terrains import (
    _chained_straight_bridge_point_in_deck_box,
    chained_straight_bridge_build_path_xy,
    chained_straight_bridge_compute_segments,
    chained_straight_bridge_resolve_num_segments,
    curved_bridge_centerline_yaw,
)
from instinct_mj.terrains.height_field.hf_terrains_cfg import PerlinChainedStraightBridgeTerrainCfg


def _make_cfg() -> PerlinChainedStraightBridgeTerrainCfg:
    cfg = CHAINED_STRAIGHT_BRIDGE_TERRAINS_CFG.sub_terrains["chained_straight_bridge_4"]
    assert isinstance(cfg, PerlinChainedStraightBridgeTerrainCfg)
    return copy.deepcopy(cfg)


def test_chained_straight_bridge_generates_without_error() -> None:
    cfg = _make_cfg()
    cfg._generate_height_field(0.5, cfg)


def test_chained_straight_bridge_num_segments_curriculum() -> None:
    cfg = _make_cfg()
    assert chained_straight_bridge_resolve_num_segments(0.0, cfg) == 4
    assert chained_straight_bridge_resolve_num_segments(1.0, cfg) == 4


def test_chained_straight_bridge_segments_do_not_overlap_collinear() -> None:
    cfg = _make_cfg()
    cfg.num_segments = 2
    cfg.segment_turns_deg = None
    cfg.turn_angle_range = (0.0, 0.0)
    sub_size = (10.0, 8.0)
    segments = chained_straight_bridge_compute_segments(0.0, cfg, tile_size=sub_size)
    assert len(segments) == 2
    gap = float(cfg.segment_gap_m)
    for i in range(len(segments) - 1):
        sx0, sy0, h0, len0 = segments[i]
        sx1, sy1, _h1, _len1 = segments[i + 1]
        end_x = sx0 + len0 * np.cos(h0)
        end_y = sy0 + len0 * np.sin(h0)
        gap_end_x = end_x + gap * np.cos(h0)
        gap_end_y = end_y + gap * np.sin(h0)
        assert abs(sx1 - gap_end_x) < 1.0e-6
        assert abs(sy1 - gap_end_y) < 1.0e-6
        assert abs(sy0 - sy1) < 1.0e-6


def test_chained_straight_bridge_four_module_preset_has_zigzag() -> None:
    cfg = _make_cfg()
    assert cfg.segment_turns_deg == (22.0, -22.0, 22.0)
    segments = chained_straight_bridge_compute_segments(0.0, cfg, tile_size=(10.0, 8.0))
    assert len(segments) == 4
    assert abs(segments[1][2] - np.radians(22.0)) < 1.0e-6
    assert abs(segments[2][2]) < 1.0e-6
    assert abs(segments[3][2] - np.radians(22.0)) < 1.0e-6


def test_chained_straight_bridge_path_is_longer_than_single_module() -> None:
    cfg = _make_cfg()
    sub_size = (10.0, 8.0)
    cx, cy = chained_straight_bridge_build_path_xy(1.0, cfg, tile_size=sub_size)
    arc_len = float(np.sum(np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2)))
    assert arc_len > 6.0


def test_chained_straight_bridge_entrance_yaw_is_forward() -> None:
    cfg = _make_cfg()
    sub_size = (10.0, 8.0)
    cx, cy = chained_straight_bridge_build_path_xy(0.5, cfg, tile_size=sub_size)
    yaw0 = curved_bridge_centerline_yaw(cx, cy, 0)
    assert abs(yaw0) < 0.1


def test_chained_straight_bridge_has_no_entrance_ramp_geometry() -> None:
    cfg = _make_cfg()
    cfg.size = (10.0, 8.0)
    assert float(cfg.entrance_ramp_m) == 0.0
    hf = np.asarray(cfg._generate_height_field(0.0, cfg), dtype=np.float64)
    vs = float(cfg.vertical_scale)
    deck_i = int(round(float(cfg.bridge_height) / vs))
    ramp_heights = hf[(hf > 0.0) & (hf < float(deck_i))]
    assert ramp_heights.size == 0


def test_chained_straight_bridge_deck_is_uniform_box_height() -> None:
    cfg = _make_cfg()
    cfg.size = (10.0, 8.0)
    vs = float(cfg.vertical_scale)
    hf = np.asarray(cfg._generate_height_field(0.0, cfg), dtype=np.float64)
    target_i = int(round(0.1 / vs))
    segments = chained_straight_bridge_compute_segments(0.0, cfg, tile_size=(10.0, 8.0))
    sx, sy, heading, length_m = segments[0]
    cos_h, sin_h = float(np.cos(heading)), float(np.sin(heading))
    mid_x = float(sx) + 0.5 * float(length_m) * cos_h
    mid_y = float(sy) + 0.5 * float(length_m) * sin_h
    hs = float(cfg.horizontal_scale)
    col = int(np.clip(np.round(mid_x / hs), 0, hf.shape[1] - 1))
    row = int(np.clip(np.round(mid_y / hs), 0, hf.shape[0] - 1))
    assert int(hf[row, col]) == target_i


def test_chained_straight_bridge_modules_do_not_overlap() -> None:
    cfg = _make_cfg()
    cfg.size = (10.0, 8.0)
    segments = chained_straight_bridge_compute_segments(0.0, cfg, tile_size=(10.0, 8.0))
    half_w = 0.5 * float(cfg.bridge_width)
    for seg_i, (sx, sy, heading, length_m) in enumerate(segments):
        cos_h, sin_h = float(np.cos(heading)), float(np.sin(heading))
        sample_x = float(sx) + 0.5 * float(length_m) * cos_h
        sample_y = float(sy) + 0.5 * float(length_m) * sin_h
        for other_i, (ox, oy, oh, olen) in enumerate(segments):
            if other_i == seg_i:
                continue
            ocos, osin = float(np.cos(oh)), float(np.sin(oh))
            if _chained_straight_bridge_point_in_deck_box(
                sample_x,
                sample_y,
                start_x=ox,
                start_y=oy,
                cos_h=ocos,
                sin_h=osin,
                length_m=olen,
                half_width_m=half_w,
            ):
                raise AssertionError(f"segment {seg_i} interior lies inside segment {other_i}")


def test_bridge_mix_preset_rotates_sub_terrains() -> None:
    props = {
        k: v.proportion
        for k, v in CHAINED_STRAIGHT_BRIDGE_TERRAINS_CFG.sub_terrains.items()
        if v.proportion > 0.0
    }
    assert props["chained_straight_bridge_4"] > 0.0
    assert props["curved_bridge"] > 0.0
    assert props["narrow_l_corner"] > 0.0
    assert sum(props.values()) == 1.0
