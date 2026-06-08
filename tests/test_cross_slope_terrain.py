from __future__ import annotations

import math

import numpy as np

from instinct_mj.terrains.height_field.hf_terrains_cfg import PerlinCrossSlopeTerrainCfg


def _make_cfg() -> PerlinCrossSlopeTerrainCfg:
    return PerlinCrossSlopeTerrainCfg(
        size=(10.0, 8.0),
        horizontal_scale=0.1,
        vertical_scale=0.005,
        total_length=6.0,
        terrain_width=3.0,
        segment_length=1.5,
        num_segments=4,
        slope_angle=15.0,
        perlin_cfg=None,
    )


def _lateral_height_diff_m(cfg: PerlinCrossSlopeTerrainCfg) -> float:
    return float(cfg.terrain_width) * math.tan(math.radians(float(cfg.slope_angle)))


def _feature_bounds(cfg: PerlinCrossSlopeTerrainCfg, hf: np.ndarray) -> dict[str, int]:
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


def _to_meters(hf_val: int, vs: float) -> float:
    return float(hf_val) * vs


def test_cross_slope_generates_without_error() -> None:
    cfg = _make_cfg()
    for d in (0.0, 0.5, 1.0):
        hf = cfg._generate_height_field(difficulty=d, cfg_for_gen=cfg)
        assert hf.shape == (80, 100)
        assert hf.dtype == np.int16


def test_cross_slope_centerline_continuous_at_junctions() -> None:
    cfg = _make_cfg()
    hf = cfg._generate_height_field(difficulty=0.0, cfg_for_gen=cfg)
    b = _feature_bounds(cfg, hf)
    vs = cfg.vertical_scale
    width_px = b["y1"] - b["y0"]
    k_lo = (width_px - 1) // 2
    k_hi = k_lo + 1
    y_lo = b["y0"] + k_lo
    y_hi = b["y0"] + k_hi
    seg_px = int(round(cfg.segment_length / cfg.horizontal_scale))
    x_junctions = [b["x0"] + seg_px, b["x0"] + 2 * seg_px, b["x0"] + 3 * seg_px]
    half_span_m = 0.5 * _lateral_height_diff_m(cfg)
    h_center = 0.5 * (_to_meters(int(hf[y_lo, b["x0"]]), vs) + _to_meters(int(hf[y_hi, b["x0"]]), vs))
    assert abs(h_center - half_span_m) < 0.03
    for xj in x_junctions:
        h_before = 0.5 * (int(hf[y_lo, xj - 1]) + int(hf[y_hi, xj - 1]))
        h_after = 0.5 * (int(hf[y_lo, xj]) + int(hf[y_hi, xj]))
        assert abs(h_before - h_after) <= 1


def test_cross_slope_left_edge_discontinuity_at_junctions() -> None:
    cfg = _make_cfg()
    hf = cfg._generate_height_field(difficulty=0.0, cfg_for_gen=cfg)
    b = _feature_bounds(cfg, hf)
    vs = cfg.vertical_scale
    y_left = b["y0"]
    seg_px = int(round(cfg.segment_length / cfg.horizontal_scale))
    xj = b["x0"] + seg_px
    jump_m = abs(_to_meters(int(hf[y_left, xj - 1]), vs) - _to_meters(int(hf[y_left, xj]), vs))
    assert abs(jump_m - _lateral_height_diff_m(cfg)) < 0.05


def test_cross_slope_segment_length_along_travel() -> None:
    cfg = _make_cfg()
    hf = cfg._generate_height_field(difficulty=0.0, cfg_for_gen=cfg)
    b = _feature_bounds(cfg, hf)
    seg_px = int(round(cfg.segment_length / cfg.horizontal_scale))
    assert b["x1"] - b["x0"] == 4 * seg_px


def test_cross_slope_lateral_profile_seg1_and_seg2() -> None:
    cfg = _make_cfg()
    hf = cfg._generate_height_field(difficulty=0.0, cfg_for_gen=cfg)
    b = _feature_bounds(cfg, hf)
    vs = cfg.vertical_scale
    h_px = int(round(_lateral_height_diff_m(cfg) / vs))
    x_mid_seg1 = b["x0"] + int(round(0.75 / cfg.horizontal_scale))
    x_mid_seg2 = b["x0"] + int(round(2.25 / cfg.horizontal_scale))
    prof1 = hf[b["y0"] : b["y1"], x_mid_seg1].astype(np.float64)
    prof2 = hf[b["y0"] : b["y1"], x_mid_seg2].astype(np.float64)
    n = b["y1"] - b["y0"]
    y_norm = (np.arange(n, dtype=np.float64) + 0.5) / n
    expected1 = y_norm * h_px
    expected2 = (1.0 - y_norm) * h_px
    assert np.max(np.abs(prof1 - expected1)) <= 1.0
    assert np.max(np.abs(prof2 - expected2)) <= 1.0
