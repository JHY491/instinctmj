"""Tests for PerlinSymmetricalSlopeTerrainCfg (V-groove symmetrical slope terrain)."""

from __future__ import annotations

import math

import numpy as np

from instinct_mj.terrains.height_field.hf_terrains_cfg import PerlinSymmetricalSlopeTerrainCfg


def _make_cfg(slope_angle: float = 20.0) -> PerlinSymmetricalSlopeTerrainCfg:
    return PerlinSymmetricalSlopeTerrainCfg(
        size=(10.0, 8.0),
        horizontal_scale=0.1,
        vertical_scale=0.005,
        total_length=3.0,
        terrain_width=3.5,
        slope_angle=slope_angle,
        flat_bottom=0.1,
        perlin_cfg=None,
    )


def _feature_bounds(cfg: PerlinSymmetricalSlopeTerrainCfg, hf: np.ndarray) -> dict[str, int]:
    hs = cfg.horizontal_scale
    ny = hf.shape[0]
    nx = hf.shape[1]
    width_px = int(round(cfg.terrain_width / hs))
    length_px = int(round(cfg.total_length / hs))
    y0 = (ny - width_px) // 2
    y1 = y0 + width_px
    x0 = (nx - length_px) // 2
    x1 = x0 + length_px
    return {"x0": x0, "x1": x1, "y0": y0, "y1": y1}


def _to_meters(hf_val, vs: float) -> float:
    return float(hf_val) * vs


def test_symmetrical_slope_generates_without_error() -> None:
    cfg = _make_cfg()
    for d in (0.0, 0.5, 1.0):
        hf = cfg._generate_height_field(difficulty=d, cfg_for_gen=cfg)
        assert hf.shape == (80, 100)
        assert hf.dtype == np.int16


def test_symmetrical_slope_height_range_within_expected() -> None:
    cfg = _make_cfg(slope_angle=20.0)
    hf = cfg._generate_height_field(difficulty=0.0, cfg_for_gen=cfg)
    b = _feature_bounds(cfg, hf)
    vs = cfg.vertical_scale

    feature = hf[b["y0"] : b["y1"], b["x0"] : b["x1"]]
    h_min_m = _to_meters(feature.min(), vs)
    h_max_m = _to_meters(feature.max(), vs)

    half_width = 0.5 * (cfg.terrain_width - cfg.flat_bottom)
    expected_max_m = half_width * math.tan(math.radians(cfg.slope_angle))
    assert h_min_m >= -0.005
    assert abs(h_max_m - expected_max_m) < 0.05


def test_symmetrical_slope_flat_bottom_at_center() -> None:
    cfg = _make_cfg(slope_angle=20.0)
    hf = cfg._generate_height_field(difficulty=0.0, cfg_for_gen=cfg)
    b = _feature_bounds(cfg, hf)
    vs = cfg.vertical_scale
    hs = cfg.horizontal_scale

    flat_px = max(1, int(round(cfg.flat_bottom / hs)))
    width_px = b["y1"] - b["y0"]
    y_center_lo = b["y0"] + (width_px - flat_px) // 2
    y_center_hi = y_center_lo + flat_px
    x_mid = (b["x0"] + b["x1"]) // 2

    center_heights = hf[y_center_lo:y_center_hi, x_mid]
    assert np.all(center_heights == 0)


def test_symmetrical_slope_symmetric_lateral_profile() -> None:
    cfg = _make_cfg(slope_angle=20.0)
    hf = cfg._generate_height_field(difficulty=0.0, cfg_for_gen=cfg)
    b = _feature_bounds(cfg, hf)
    x_mid = (b["x0"] + b["x1"]) // 2
    profile = hf[b["y0"] : b["y1"], x_mid].astype(np.float64)
    n = len(profile)
    assert np.allclose(profile, profile[::-1], atol=1.0)


def test_symmetrical_slope_angle_range_curriculum() -> None:
    cfg_steep = _make_cfg(slope_angle=30.0)
    hf_steep = cfg_steep._generate_height_field(difficulty=0.0, cfg_for_gen=cfg_steep)
    cfg_flat = _make_cfg(slope_angle=5.0)
    hf_flat = cfg_flat._generate_height_field(difficulty=0.0, cfg_for_gen=cfg_flat)

    assert hf_steep.max() > hf_flat.max()


def test_symmetrical_slope_constant_along_travel_direction() -> None:
    cfg = _make_cfg(slope_angle=20.0)
    hf = cfg._generate_height_field(difficulty=0.0, cfg_for_gen=cfg)
    b = _feature_bounds(cfg, hf)

    profile_x0 = hf[b["y0"] : b["y1"], b["x0"]].astype(np.float64)
    profile_x1 = hf[b["y0"] : b["y1"], b["x1"] - 1].astype(np.float64)
    assert np.allclose(profile_x0, profile_x1, atol=1.0)
