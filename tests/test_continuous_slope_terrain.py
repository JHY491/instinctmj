"""Tests for PerlinContinuousSlopeTerrainCfg (double-hump continuous slope terrain)."""

from __future__ import annotations

import math

import numpy as np

from instinct_mj.terrains.height_field.hf_terrains import _resolve_lateral_width
from instinct_mj.terrains.height_field.hf_terrains_cfg import PerlinContinuousSlopeTerrainCfg


def _make_cfg() -> PerlinContinuousSlopeTerrainCfg:
    return PerlinContinuousSlopeTerrainCfg(
        size=(10.0, 8.0),
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_angle_range=(15.0, 15.0),
        total_length=7.4,
        terrain_width=2.4,
        max_height=0.39,
        perlin_cfg=None,
    )


def _terrain_width_m(cfg: PerlinContinuousSlopeTerrainCfg) -> float:
    w = _resolve_lateral_width(cfg.terrain_width)
    if w is not None:
        return w
    if isinstance(cfg.terrain_width, tuple):
        return float(cfg.terrain_width[0])
    return float(cfg.terrain_width)


def _feature_bounds(cfg: PerlinContinuousSlopeTerrainCfg, hf: np.ndarray) -> dict[str, int]:
    hs = cfg.horizontal_scale
    ny = hf.shape[0]
    nx = hf.shape[1]
    width_px = int(round(_terrain_width_m(cfg) / hs))
    length_px = int(round(cfg.total_length / hs))
    y0 = (ny - width_px) // 2
    y1 = y0 + width_px
    x0 = (nx - length_px) // 2
    x1 = x0 + length_px
    return {"x0": x0, "x1": x1, "y0": y0, "y1": y1}


def _to_meters(hf_val, vs: float) -> float:
    return float(hf_val) * vs


def test_continuous_slope_generates_without_error() -> None:
    cfg = _make_cfg()
    for d in (0.0, 0.5, 1.0):
        hf = cfg._generate_height_field(difficulty=d, cfg_for_gen=cfg)
        assert hf.shape == (80, 100)
        assert hf.dtype == np.int16


def test_continuous_slope_has_flat_platforms() -> None:
    cfg = _make_cfg()
    hf = cfg._generate_height_field(difficulty=0.0, cfg_for_gen=cfg)
    b = _feature_bounds(cfg, hf)
    vs = cfg.vertical_scale

    mid_y = (b["y0"] + b["y1"]) // 2
    profile = hf[mid_y, b["x0"] : b["x1"]].astype(np.float64)

    max_val = profile.max()
    max_indices = np.where(profile == max_val)[0]
    # Should find two flat platform regions
    platform_groups = np.split(max_indices, np.where(np.diff(max_indices) > 1)[0] + 1)
    assert len(platform_groups) == 2
    # Each platform should be at least 3 pixels wide
    for g in platform_groups:
        assert len(g) >= 3


def test_continuous_slope_max_height_within_expected() -> None:
    cfg = _make_cfg()
    hf = cfg._generate_height_field(difficulty=0.0, cfg_for_gen=cfg)
    b = _feature_bounds(cfg, hf)
    vs = cfg.vertical_scale

    feature = hf[b["y0"] : b["y1"], b["x0"] : b["x1"]]
    h_max_m = _to_meters(feature.max(), vs)
    # max_height 0.39m, allow small rounding tolerance
    assert abs(h_max_m - cfg.max_height) < 0.02


def test_continuous_slope_starts_and_ends_at_ground() -> None:
    cfg = _make_cfg()
    hf = cfg._generate_height_field(difficulty=0.0, cfg_for_gen=cfg)
    b = _feature_bounds(cfg, hf)
    mid_y = (b["y0"] + b["y1"]) // 2

    h_start = int(hf[mid_y, b["x0"]])
    h_end = int(hf[mid_y, b["x1"] - 1])

    assert h_start == 0
    assert h_end == 0


def test_continuous_slope_steeper_angle_is_shorter_platform() -> None:
    """Higher slope angle → shorter slope horizontal → longer flat platform."""
    cfg_shallow = PerlinContinuousSlopeTerrainCfg(
        size=(20.0, 10.0),
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_angle_range=(5.0, 5.0),
        total_length=7.4,
        terrain_width=2.4,
        max_height=0.39,
        perlin_cfg=None,
    )
    cfg_steep = PerlinContinuousSlopeTerrainCfg(
        size=(20.0, 10.0),
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_angle_range=(30.0, 30.0),
        total_length=7.4,
        terrain_width=2.4,
        max_height=0.39,
        perlin_cfg=None,
    )

    hf_shallow = cfg_shallow._generate_height_field(difficulty=0.0, cfg_for_gen=cfg_shallow)
    hf_steep = cfg_steep._generate_height_field(difficulty=0.0, cfg_for_gen=cfg_steep)
    b_s = _feature_bounds(cfg_shallow, hf_shallow)
    b_t = _feature_bounds(cfg_steep, hf_steep)

    mid_y_s = (b_s["y0"] + b_s["y1"]) // 2
    mid_y_t = (b_t["y0"] + b_t["y1"]) // 2

    profile_s = hf_shallow[mid_y_s, b_s["x0"] : b_s["x1"]]
    profile_t = hf_steep[mid_y_t, b_t["x0"] : b_t["x1"]]

    max_s = profile_s.max()
    max_t = profile_t.max()
    platform_px_s = int((profile_s == max_s).sum())
    platform_px_t = int((profile_t == max_t).sum())
    assert platform_px_t > platform_px_s


def test_continuous_slope_random_width_in_range() -> None:
    cfg = PerlinContinuousSlopeTerrainCfg(
        size=(10.0, 8.0),
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_angle_range=(15.0, 15.0),
        total_length=7.4,
        terrain_width=(2.0, 2.8),
        max_height=0.39,
        perlin_cfg=None,
    )
    hs = cfg.horizontal_scale
    widths_px: set[int] = set()
    for _ in range(32):
        hf = cfg._generate_height_field(difficulty=0.0, cfg_for_gen=cfg)
        active_rows = np.where(hf.max(axis=1) > 0)[0]
        widths_px.add(int(active_rows[-1] - active_rows[0] + 1))
    assert len(widths_px) > 1
    assert min(widths_px) >= int(round(2.0 / hs)) - 1
    assert max(widths_px) <= int(round(2.8 / hs)) + 1


def test_continuous_slope_four_slope_segments() -> None:
    """Verify four non-flat slope segments: up1, down1, up2, down2."""
    cfg = _make_cfg()
    hf = cfg._generate_height_field(difficulty=0.0, cfg_for_gen=cfg)
    b = _feature_bounds(cfg, hf)
    mid_y = (b["y0"] + b["y1"]) // 2
    profile = hf[mid_y, b["x0"] : b["x1"]].astype(np.float64)

    diffs = np.diff(profile)
    sign_changes = np.where(np.diff(np.sign(diffs)) != 0)[0]
    # 4 slope-ramp pairs → 7 sign changes (up→flat, flat→down, down→flat, flat→up, up→flat, flat→down, down→flat)
    assert 5 <= len(sign_changes) <= 9
