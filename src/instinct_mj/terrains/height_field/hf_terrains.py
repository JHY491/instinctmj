from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import scipy.interpolate as interpolate

from instinct_mj.utils.perlin import generate_fractal_noise_2d

if TYPE_CHECKING:
    from . import hf_terrains_cfg

from .utils import generate_wall, height_field_to_mesh


def _spiral_theta_in_sweep(theta: float, start_angle_rad: float, sweep_rad: float) -> bool:
    return (theta - start_angle_rad) % (2.0 * math.pi) < sweep_rad


def _max_outer_radius_for_annular_sector_in_box(
    pivot_x_m: float,
    pivot_y_m: float,
    x0_m: float,
    x1_m: float,
    y0_m: float,
    y1_m: float,
    start_angle_rad: float,
    sweep_rad: float,
    margin_m: float,
) -> float:
    """Largest outer radius so the annular sector stays inside the obstacle rectangle."""
    theta_end = start_angle_rad + sweep_rad
    thetas = [start_angle_rad, theta_end]
    for cardinal in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi):
        if _spiral_theta_in_sweep(cardinal, start_angle_rad, sweep_rad):
            thetas.append(cardinal)
    limit_m = float("inf")
    for theta in thetas:
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        if cos_t < -1e-9:
            limit_m = min(limit_m, (pivot_x_m - x0_m - margin_m) / -cos_t)
        elif cos_t > 1e-9:
            limit_m = min(limit_m, (x1_m - margin_m - pivot_x_m) / cos_t)
        if sin_t < -1e-9:
            limit_m = min(limit_m, (pivot_y_m - y0_m - margin_m) / -sin_t)
        elif sin_t > 1e-9:
            limit_m = min(limit_m, (y1_m - margin_m - pivot_y_m) / sin_t)
    return max(0.1, limit_m)


def generate_perlin_noise(difficulty: float, cfg: hf_terrains_cfg.PerlinPlaneTerrainCfg) -> np.ndarray:
    noise_scale = (
        cfg.noise_scale
        if not isinstance(cfg.noise_scale, (list, tuple))
        else interpolate.interp1d([0, 1], cfg.noise_scale, kind="linear")(difficulty)
    )
    return (
        generate_fractal_noise_2d(
            xSize=cfg.size[1],
            ySize=cfg.size[0],
            xSamples=int(cfg.size[1] / cfg.horizontal_scale),
            ySamples=int(cfg.size[0] / cfg.horizontal_scale),
            frequency=cfg.noise_frequency,
            fractalOctaves=cfg.fractal_octaves,
            fractalLacunarity=cfg.fractal_lacunarity,
            fractalGain=cfg.fractal_gain,
            zScale=noise_scale,
            centering=cfg.centering,
        )
        / cfg.vertical_scale
    ).astype(np.int16)


@generate_wall
@height_field_to_mesh
def perlin_plane_terrain(difficulty: float, cfg: hf_terrains_cfg.PerlinPlaneTerrainCfg) -> np.ndarray:
    return generate_perlin_noise(
        difficulty=difficulty,
        cfg=cfg,
    )  # type: ignore[return-value, arg-type]


@generate_wall
@height_field_to_mesh
def perlin_pyramid_sloped_terrain(difficulty: float, cfg: hf_terrains_cfg.PerlinPyramidSlopedTerrainCfg) -> np.ndarray:
    """Generate a terrain with a truncated pyramid structure.

    The terrain is a pyramid-shaped sloped surface with a slope of :obj:`slope` that trims into a flat platform
    at the center. The slope is defined as the ratio of the height change along the x axis to the width along the
    x axis. For example, a slope of 1.0 means that the height changes by 1 unit for every 1 unit of width.

    If the :obj:`cfg.inverted` flag is set to :obj:`True`, the terrain is inverted such that
    the platform is at the bottom.

    .. image:: ../../_static/terrains/height_field/pyramid_sloped_terrain.jpg
       :width: 40%

    .. image:: ../../_static/terrains/height_field/inverted_pyramid_sloped_terrain.jpg
       :width: 40%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (rows_y, cols_x), where rows_y and cols_x are the number of points
        along the y and x axis, respectively.
    """
    # resolve terrain configuration
    if cfg.inverted:
        slope = -cfg.slope_range[0] - difficulty * (cfg.slope_range[1] - cfg.slope_range[0])
    else:
        slope = cfg.slope_range[0] + difficulty * (cfg.slope_range[1] - cfg.slope_range[0])

    # switch parameters to discrete units
    # -- horizontal scale
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- height
    # we want the height to be 1/2 of the width since the terrain is a pyramid
    height_max = int(slope * cfg.size[0] / 2 / cfg.vertical_scale)
    # -- center of the terrain
    center_col = int(width_pixels / 2)
    center_row = int(length_pixels / 2)

    # create a meshgrid of the terrain
    cols = np.arange(0, width_pixels)
    rows = np.arange(0, length_pixels)
    col_grid, row_grid = np.meshgrid(cols, rows, sparse=True)
    # offset the meshgrid to the center of the terrain
    col_grid = (center_col - np.abs(center_col - col_grid)) / center_col
    row_grid = (center_row - np.abs(center_row - row_grid)) / center_row
    # reshape the meshgrid to be 2D
    col_grid = col_grid.reshape(1, width_pixels)
    row_grid = row_grid.reshape(length_pixels, 1)
    # create a sloped surface
    hf_raw = np.zeros((length_pixels, width_pixels))
    hf_raw = height_max * row_grid * col_grid

    # create a flat platform at the center of the terrain
    platform_width = int(cfg.platform_width / cfg.horizontal_scale / 2)
    # get the height of the platform at the corner of the platform
    col_pf = width_pixels // 2 - platform_width
    row_pf = length_pixels // 2 - platform_width
    z_pf = hf_raw[row_pf, col_pf]
    hf_raw = np.clip(hf_raw, min(0, z_pf), max(0, z_pf))
    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(
            difficulty,
            perlin_cfg,  # type: ignore[arg-type]
        )
        # add perlin noise to the terrain
        hf_raw += perlin_noise

    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


@generate_wall
@height_field_to_mesh
def perlin_pyramid_stairs_terrain(difficulty: float, cfg: hf_terrains_cfg.PerlinPyramidStairsTerrainCfg) -> np.ndarray:
    """Generate a terrain with a pyramid stair pattern.

    The terrain is a pyramid stair pattern which trims to a flat platform at the center of the terrain.

    If the :obj:`cfg.inverted` flag is set to :obj:`True`, the terrain is inverted such that
    the platform is at the bottom.

    .. image:: ../../_static/terrains/height_field/pyramid_stairs_terrain.jpg
       :width: 40%

    .. image:: ../../_static/terrains/height_field/inverted_pyramid_stairs_terrain.jpg
       :width: 40%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (rows_y, cols_x), where rows_y and cols_x are the number of points
        along the y and x axis, respectively.
    """
    # resolve terrain configuration
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    if cfg.inverted:
        step_height *= -1
    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- stairs
    step_width = round(cfg.step_width / cfg.horizontal_scale)
    step_height = round(step_height / cfg.vertical_scale)
    # -- platform
    platform_width = round(cfg.platform_width / cfg.horizontal_scale)

    # create a terrain with a flat platform at the center
    hf_raw = np.zeros((length_pixels, width_pixels))
    # add the steps
    current_step_height = 0
    start_col, start_row = 0, 0
    stop_col, stop_row = width_pixels, length_pixels
    while (stop_col - start_col) > platform_width and (stop_row - start_row) > platform_width:
        # increment position
        # -- x
        start_col += step_width
        stop_col -= step_width
        # -- y
        start_row += step_width
        stop_row -= step_width
        # increment height
        current_step_height += step_height
        # add the step
        hf_raw[start_row:stop_row, start_col:stop_col] = current_step_height

    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(
            difficulty,
            perlin_cfg,  # type: ignore[arg-type]
        )
        # add perlin noise to the terrain
        hf_raw += perlin_noise

    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


def _one_sided_stairs_heightfield_neg_x(
    width_pixels: int,
    length_pixels: int,
    step_width_px: int,
    step_height_i: int,
    platform_width_px: int,
) -> np.ndarray:
    """Stairs climb along +x from x=0 (low) toward a flat platform at high x."""
    hf_raw = np.zeros((length_pixels, width_pixels), dtype=np.float64)
    if platform_width_px <= 0 or platform_width_px >= width_pixels:
        return np.rint(hf_raw).astype(np.int16)
    if step_width_px <= 0:
        step_width_px = 1

    current_h = 0.0
    col = 0
    while col < width_pixels - platform_width_px:
        next_col = min(col + step_width_px, width_pixels - platform_width_px)
        hf_raw[:, col:next_col] = current_h
        current_h += float(step_height_i)
        col = next_col
    hf_raw[:, width_pixels - platform_width_px :] = current_h
    return np.rint(hf_raw).astype(np.int16)


def _one_sided_stairs_heightfield_neg_y(
    width_pixels: int,
    length_pixels: int,
    step_width_px: int,
    step_height_i: int,
    platform_width_px: int,
) -> np.ndarray:
    """Stairs climb along +y from y=0 (low) toward a flat platform at high y."""
    hf_raw = np.zeros((length_pixels, width_pixels), dtype=np.float64)
    if platform_width_px <= 0 or platform_width_px >= length_pixels:
        return np.rint(hf_raw).astype(np.int16)
    if step_width_px <= 0:
        step_width_px = 1

    current_h = 0.0
    row = 0
    while row < length_pixels - platform_width_px:
        next_row = min(row + step_width_px, length_pixels - platform_width_px)
        hf_raw[row:next_row, :] = current_h
        current_h += float(step_height_i)
        row = next_row
    hf_raw[length_pixels - platform_width_px :, :] = current_h
    return np.rint(hf_raw).astype(np.int16)


def _one_sided_up_down_stairs_heightfield_x(
    width_pixels: int,
    length_pixels: int,
    step_width_px: int,
    step_height_i: int,
    platform_width_px: int,
) -> np.ndarray:
    """Low at both x edges; ascend to a center platform, then descend (symmetric along x)."""
    hf_raw = np.zeros((length_pixels, width_pixels), dtype=np.float64)
    if step_width_px <= 0:
        step_width_px = 1
    if width_pixels <= 1:
        return np.rint(hf_raw).astype(np.int16)
    if platform_width_px < 1:
        platform_width_px = 1
    if platform_width_px > width_pixels:
        platform_width_px = width_pixels
    rise_span_px = max(0, (width_pixels - platform_width_px) // 2)
    max_steps = rise_span_px // step_width_px
    for col in range(width_pixels):
        steps_left = col // step_width_px
        steps_right = (width_pixels - 1 - col) // step_width_px
        si = min(steps_left, steps_right, max_steps)
        hf_raw[:, col] = float(si * step_height_i)
    return np.rint(hf_raw).astype(np.int16)


def _one_sided_up_down_stairs_heightfield_y(
    width_pixels: int,
    length_pixels: int,
    step_width_px: int,
    step_height_i: int,
    platform_width_px: int,
) -> np.ndarray:
    """Low at both y edges; ascend to a center platform, then descend (symmetric along y)."""
    hf_raw = np.zeros((length_pixels, width_pixels), dtype=np.float64)
    if step_width_px <= 0:
        step_width_px = 1
    if length_pixels <= 1:
        return np.rint(hf_raw).astype(np.int16)
    if platform_width_px < 1:
        platform_width_px = 1
    if platform_width_px > length_pixels:
        platform_width_px = length_pixels
    rise_span_px = max(0, (length_pixels - platform_width_px) // 2)
    max_steps = rise_span_px // step_width_px
    for row in range(length_pixels):
        steps_left = row // step_width_px
        steps_right = (length_pixels - 1 - row) // step_width_px
        si = min(steps_left, steps_right, max_steps)
        hf_raw[row, :] = float(si * step_height_i)
    return np.rint(hf_raw).astype(np.int16)


def _apply_lateral_width_strip(
    *,
    hf_raw: np.ndarray,
    stair_face: str,
    lateral_width_m: float | None,
    horizontal_scale: float,
) -> np.ndarray:
    """Keep only a centered lateral strip for one-sided stairs."""
    if lateral_width_m is None:
        return hf_raw
    lateral_width_m = float(lateral_width_m)
    if lateral_width_m <= 0.0:
        return hf_raw

    length_pixels, width_pixels = hf_raw.shape
    if stair_face in ("neg_y", "pos_y"):
        lateral_axis_size = width_pixels
        axis = 1
    else:
        lateral_axis_size = length_pixels
        axis = 0

    keep_pixels = int(round(lateral_width_m / float(horizontal_scale)))
    keep_pixels = max(1, min(keep_pixels, lateral_axis_size))
    start = max(0, (lateral_axis_size - keep_pixels) // 2)
    end = min(lateral_axis_size, start + keep_pixels)

    filtered = np.zeros_like(hf_raw, dtype=np.float64)
    if axis == 0:
        filtered[start:end, :] = hf_raw[start:end, :]
    else:
        filtered[:, start:end] = hf_raw[:, start:end]
    return filtered


def _resolve_lateral_width(lateral_width: float | tuple[float, float] | None) -> float | None:
    """Resolve lateral width value, optionally sampling from a range."""
    if lateral_width is None:
        return None
    if isinstance(lateral_width, tuple):
        lo = float(lateral_width[0])
        hi = float(lateral_width[1])
        if lo > hi:
            lo, hi = hi, lo
        return float(np.random.uniform(lo, hi))
    return float(lateral_width)


@generate_wall
@height_field_to_mesh
def perlin_one_sided_stairs_terrain(
    difficulty: float, cfg: hf_terrains_cfg.PerlinOneSidedStairsTerrainCfg
) -> np.ndarray:
    """Single-run stairs: steps along one axis only; perpendicular direction is flat.

    Unlike :func:`perlin_pyramid_stairs_terrain`, only one edge is the low side; stairs
    advance toward a flat platform at the opposite side. Height is uniform along the
    axis perpendicular to the stair run.

    For x: ``neg_x`` is low at x=0 and climbs with +x (upstairs); ``pos_x`` flips the
    field so +x goes downhill (downstairs). For y: ``neg_y`` / ``pos_y`` behave analogously.

    If ``cfg.up_down`` is True, one tile has **connected** stairs up then down along the
    stair axis (``neg_x``/``pos_x`` → x, ``neg_y``/``pos_y`` → y); edges are low and the
    center is highest.
    """
    step_height_phys = cfg.step_height_range[0] + difficulty * (
        cfg.step_height_range[1] - cfg.step_height_range[0]
    )
    if cfg.inverted:
        step_height_phys *= -1.0

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    step_width_px = max(1, round(cfg.step_width / cfg.horizontal_scale))
    step_height_i = round(step_height_phys / cfg.vertical_scale)
    platform_width_px = max(1, round(cfg.platform_width / cfg.horizontal_scale))

    if cfg.up_down:
        face = cfg.stair_face
        if face in ("neg_x", "pos_x"):
            hf_raw = _one_sided_up_down_stairs_heightfield_x(
                width_pixels, length_pixels, step_width_px, step_height_i, platform_width_px
            ).astype(np.float64)
            if face == "pos_x":
                hf_raw = np.flip(hf_raw, axis=1)
        elif face in ("neg_y", "pos_y"):
            hf_raw = _one_sided_up_down_stairs_heightfield_y(
                width_pixels, length_pixels, step_width_px, step_height_i, platform_width_px
            ).astype(np.float64)
            if face == "pos_y":
                hf_raw = np.flip(hf_raw, axis=0)
        else:
            raise ValueError(f"Unknown stair_face: {face!r}")

        if cfg.perlin_cfg is not None:
            perlin_cfg = cfg.perlin_cfg
            perlin_cfg.size = cfg.size
            perlin_cfg.horizontal_scale = cfg.horizontal_scale
            perlin_cfg.vertical_scale = cfg.vertical_scale
            perlin_cfg.slope_threshold = cfg.slope_threshold
            perlin_noise = generate_perlin_noise(
                difficulty,
                perlin_cfg,  # type: ignore[arg-type]
            )
            hf_raw = hf_raw.astype(np.float64) + perlin_noise.astype(np.float64)

        hf_raw = _apply_lateral_width_strip(
            hf_raw=hf_raw,
            stair_face=face,
            lateral_width_m=_resolve_lateral_width(cfg.lateral_width),
            horizontal_scale=cfg.horizontal_scale,
        )

        return np.rint(hf_raw).astype(np.int16)

    face = cfg.stair_face
    if face == "neg_x":
        hf_raw = _one_sided_stairs_heightfield_neg_x(
            width_pixels, length_pixels, step_width_px, step_height_i, platform_width_px
        ).astype(np.float64)
    elif face == "pos_x":
        hf_raw = _one_sided_stairs_heightfield_neg_x(
            width_pixels, length_pixels, step_width_px, step_height_i, platform_width_px
        ).astype(np.float64)
        hf_raw = np.flip(hf_raw, axis=1)
    elif face == "neg_y":
        hf_raw = _one_sided_stairs_heightfield_neg_y(
            width_pixels, length_pixels, step_width_px, step_height_i, platform_width_px
        ).astype(np.float64)
    elif face == "pos_y":
        hf_raw = _one_sided_stairs_heightfield_neg_y(
            width_pixels, length_pixels, step_width_px, step_height_i, platform_width_px
        ).astype(np.float64)
        hf_raw = np.flip(hf_raw, axis=0)
    else:
        raise ValueError(f"Unknown stair_face: {face!r}")

    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(
            difficulty,
            perlin_cfg,  # type: ignore[arg-type]
        )
        hf_raw = hf_raw.astype(np.float64) + perlin_noise.astype(np.float64)

    hf_raw = _apply_lateral_width_strip(
        hf_raw=hf_raw,
        stair_face=face,
        lateral_width_m=_resolve_lateral_width(cfg.lateral_width),
        horizontal_scale=cfg.horizontal_scale,
    )

    return np.rint(hf_raw).astype(np.int16)


@generate_wall
@height_field_to_mesh
def perlin_spiral_stairs_terrain(difficulty: float, cfg: hf_terrains_cfg.PerlinSpiralStairsTerrainCfg) -> np.ndarray:
    """Generate a switchback stair block matching the referenced spiral-stair obstacle.

    The long side follows the Y axis, matching the other parkour stair terrains.
    The shape is a high rectangular block with two narrow stair flights cut into
    it and a large top platform between them.
    """
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    obstacle_length_px = max(2, int(round(float(cfg.obstacle_length) / float(cfg.horizontal_scale))))
    obstacle_width_px = max(2, int(round(float(cfg.obstacle_width) / float(cfg.horizontal_scale))))
    obstacle_width_px = min(obstacle_width_px, width_pixels)
    obstacle_length_px = min(obstacle_length_px, length_pixels)
    num_steps = max(1, int(cfg.num_steps))
    stair_width_px = max(1, int(round(float(cfg.stair_width) / float(cfg.horizontal_scale))))
    platform_length_px = max(1, int(round(float(cfg.platform_length) / float(cfg.horizontal_scale))))
    half_width_px = max(1, obstacle_width_px // 2)
    stair_width_px = min(stair_width_px, half_width_px)
    max_platform_length_px = max(1, obstacle_length_px - 2 * num_steps)
    platform_length_px = min(platform_length_px, max_platform_length_px)

    if isinstance(cfg.step_height, (list, tuple)):
        step_height = float(cfg.step_height[0]) + float(difficulty) * (
            float(cfg.step_height[1]) - float(cfg.step_height[0])
        )
    else:
        step_height = float(cfg.step_height)
    step_height_i = int(round(step_height / float(cfg.vertical_scale)))
    max_height = num_steps * step_height_i
    if isinstance(cfg.flight_angle_range, (list, tuple)):
        flight_angle = float(cfg.flight_angle_range[0]) + float(difficulty) * (
            float(cfg.flight_angle_range[1]) - float(cfg.flight_angle_range[0])
        )
    else:
        flight_angle = float(difficulty) * float(cfg.flight_angle_range)

    x0 = max(0, (width_pixels - obstacle_width_px) // 2)
    x1 = x0 + obstacle_width_px
    y0 = max(0, (length_pixels - obstacle_length_px) // 2)
    y1 = y0 + obstacle_length_px
    x_mid = x0 + obstacle_width_px // 2
    platform_start_y = y0 + (obstacle_length_px - platform_length_px) // 2
    platform_end_y = platform_start_y + platform_length_px
    entrance_margin = max(0, (x_mid - x0 - stair_width_px) // 2)
    entrance_lane_x0 = x0 + entrance_margin
    entrance_lane_x1 = min(x_mid, entrance_lane_x0 + stair_width_px)
    exit_margin = max(0, (x1 - x_mid - stair_width_px) // 2)
    exit_lane_x0 = x_mid + exit_margin
    exit_lane_x1 = min(x1, exit_lane_x0 + stair_width_px)
    entrance_step_length_px = max(1, (platform_start_y - y0) // num_steps)
    exit_step_length_px = max(1, (y1 - platform_end_y) // num_steps)
    max_angle_offset_px = min(
        max(0, entrance_step_length_px - 1),
        max(0, exit_step_length_px - 1),
        int(round(np.tan(np.deg2rad(abs(flight_angle))) * max(0, stair_width_px - 1))),
    )

    entrance_offsets = np.rint(
        np.linspace(max_angle_offset_px, 0, max(1, entrance_lane_x1 - entrance_lane_x0))
    ).astype(np.int64)
    exit_offsets = np.rint(
        np.linspace(max_angle_offset_px, 0, max(1, exit_lane_x1 - exit_lane_x0))
    ).astype(np.int64)

    hf_raw = np.zeros((width_pixels, length_pixels), dtype=np.float64)
    hf_raw[x0:x1, platform_start_y:platform_end_y] = float(max_height)

    entrance_wall_offsets = np.rint(np.linspace(max_angle_offset_px, 0, max(1, x_mid - x0))).astype(np.int64)
    for row_idx, x in enumerate(range(x0, x_mid)):
        offset = int(entrance_wall_offsets[row_idx])
        wall_start_y = min(platform_start_y, y0 + offset)
        hf_raw[x, wall_start_y:platform_start_y] = float(max_height)

    exit_wall_offsets = np.rint(np.linspace(max_angle_offset_px, 0, max(1, x1 - x_mid))).astype(np.int64)
    for row_idx, x in enumerate(range(x_mid, x1)):
        offset = int(exit_wall_offsets[row_idx])
        wall_start_y = min(y1, platform_end_y + offset)
        hf_raw[x, wall_start_y:y1] = float(max_height)

    for step_idx in range(num_steps):
        start_y = y0 + step_idx * entrance_step_length_px
        stop_y = y0 + (step_idx + 1) * entrance_step_length_px
        if step_idx == num_steps - 1:
            stop_y = platform_start_y
        entrance_height = (step_idx + 1) * step_height_i
        for row_idx, x in enumerate(range(entrance_lane_x0, entrance_lane_x1)):
            offset = int(entrance_offsets[row_idx])
            if step_idx == 0:
                row_start_y = y0
            else:
                row_start_y = min(platform_start_y, start_y + offset)
            row_stop_y = min(platform_start_y, stop_y + offset)
            hf_raw[x, row_start_y:row_stop_y] = float(entrance_height)

        exit_start_y = platform_end_y + step_idx * exit_step_length_px
        exit_stop_y = platform_end_y + (step_idx + 1) * exit_step_length_px
        if step_idx == num_steps - 1:
            exit_stop_y = y1
        exit_height = (num_steps - step_idx - 1) * step_height_i
        for row_idx, x in enumerate(range(exit_lane_x0, exit_lane_x1)):
            offset = int(exit_offsets[row_idx])
            row_start_y = min(y1, exit_start_y + offset)
            if step_idx == num_steps - 1:
                row_stop_y = y1
            else:
                row_stop_y = min(y1, exit_stop_y + offset)
            hf_raw[x, row_start_y:row_stop_y] = float(exit_height)

    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(
            difficulty,
            perlin_cfg,  # type: ignore[arg-type]
        )
        hf_raw += perlin_noise

    return np.rint(hf_raw).astype(np.int16)


@generate_wall
@height_field_to_mesh
def perlin_discrete_obstacles_terrain(
    difficulty: float, cfg: hf_terrains_cfg.PerlinDiscreteObstaclesTerrainCfg
) -> np.ndarray:
    """Generate a terrain with randomly generated obstacles as pillars with positive and negative heights.

    The terrain is a flat platform at the center of the terrain with randomly generated obstacles as pillars
    with positive and negative height. The obstacles are randomly generated cuboids with a random width and
    height. They are placed randomly on the terrain with a minimum distance of :obj:`cfg.platform_width`
    from the center of the terrain.

    .. image:: ../../_static/terrains/height_field/discrete_obstacles_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (rows_y, cols_x), where rows_y and cols_x are the number of points
        along the y and x axis, respectively.
    """
    # resolve terrain configuration
    obs_height = cfg.obstacle_height_range[0] + difficulty * (
        cfg.obstacle_height_range[1] - cfg.obstacle_height_range[0]
    )

    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- obstacles
    obs_height = round(obs_height / cfg.vertical_scale)
    obs_width_min = round(cfg.obstacle_width_range[0] / cfg.horizontal_scale)
    obs_width_max = round(cfg.obstacle_width_range[1] / cfg.horizontal_scale)
    # -- center of the terrain
    platform_width = round(cfg.platform_width / cfg.horizontal_scale)

    # create discrete ranges for the obstacles
    # -- shape
    obs_width_range = np.arange(obs_width_min, obs_width_max, 4)
    obs_length_range = np.arange(obs_width_min, obs_width_max, 4)
    # -- position
    obs_x_range = np.arange(1, width_pixels, 4)
    obs_y_range = np.arange(1, length_pixels, 4)

    # create a terrain with a flat platform at the center
    hf_raw = np.zeros((length_pixels, width_pixels))
    # generate the obstacles
    for _ in range(cfg.num_obstacles):
        # sample size
        if cfg.obstacle_height_mode == "choice":
            height = np.random.choice([-obs_height, -obs_height // 2, obs_height // 2, obs_height])
        elif cfg.obstacle_height_mode == "fixed":
            height = obs_height
        else:
            raise ValueError(f"Unknown obstacle height mode '{cfg.obstacle_height_mode}'. Must be 'choice' or 'fixed'.")
        width = round(np.random.choice(obs_width_range))
        length = round(np.random.choice(obs_length_range))
        # sample position
        x_start = int(np.random.choice(obs_x_range))
        y_start = int(np.random.choice(obs_y_range))
        # clip start position to the terrain
        x_start = max(0, x_start)
        if x_start + width > width_pixels:
            width = max(1, width_pixels - x_start)
        y_start = max(0, y_start)
        if y_start + length > length_pixels:
            length = max(1, length_pixels - y_start)
        # add to terrain
        hf_raw[y_start : y_start + length, x_start : x_start + width] = height
    # clip the terrain to the platform
    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[y1:y2, x1:x2] = 0
    # round off the heights to the nearest vertical step

    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(
            difficulty,
            perlin_cfg,  # type: ignore[arg-type]
        )
        # add perlin noise to the terrain
        hf_raw += perlin_noise

    return np.rint(hf_raw).astype(np.int16)


@generate_wall
@height_field_to_mesh
def perlin_wave_terrain(difficulty: float, cfg: hf_terrains_cfg.PerlinWaveTerrainCfg) -> np.ndarray:
    r"""Generate a terrain with a wave pattern.

    The terrain is a flat platform at the center of the terrain with a wave pattern. The wave pattern
    is generated by adding sinusoidal waves based on the number of waves and the amplitude of the waves.

    The height of the terrain at a point :math:`(x, y)` is given by:

    .. math::

        h(x, y) =  A \left(\sin\left(\frac{2 \pi x}{\lambda}\right) + \cos\left(\frac{2 \pi y}{\lambda}\right) \right)

    where :math:`A` is the amplitude of the waves, :math:`\lambda` is the wavelength of the waves.

    .. image:: ../../_static/terrains/height_field/wave_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (rows_y, cols_x), where rows_y and cols_x are the number of points
        along the y and x axis, respectively.

    Raises:
        ValueError: When the number of waves is non-positive.
    """
    # check number of waves
    if cfg.num_waves <= 0:
        raise ValueError(f"Number of waves must be a positive integer. Got: {cfg.num_waves}.")

    # resolve terrain configuration
    amplitude = cfg.amplitude_range[0] + difficulty * (cfg.amplitude_range[1] - cfg.amplitude_range[0])
    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    amplitude_pixels = int(0.5 * amplitude / cfg.vertical_scale)

    # compute the wave number: nu = 2 * pi / lambda
    wave_length = length_pixels / cfg.num_waves
    wave_number = 2 * np.pi / wave_length
    # create meshgrid for the terrain
    cols = np.arange(0, width_pixels)
    rows = np.arange(0, length_pixels)
    col_grid, row_grid = np.meshgrid(cols, rows, sparse=True)
    col_grid = col_grid.reshape(1, width_pixels)
    row_grid = row_grid.reshape(length_pixels, 1)

    # create a terrain with a flat platform at the center
    hf_raw = np.zeros((length_pixels, width_pixels))
    # add the waves
    hf_raw += amplitude_pixels * (np.cos(row_grid * wave_number) + np.sin(col_grid * wave_number))
    # round off the heights to the nearest vertical step

    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(
            difficulty,
            perlin_cfg,  # type: ignore[arg-type]
        )
        # add perlin noise to the terrain
        hf_raw += perlin_noise

    return np.rint(hf_raw).astype(np.int16)


@generate_wall
@height_field_to_mesh
def perlin_stepping_stones_terrain(
    difficulty: float, cfg: hf_terrains_cfg.PerlinSteppingStonesTerrainCfg
) -> np.ndarray:
    """Generate a terrain with a stepping stones pattern.

    The terrain is a stepping stones pattern which trims to a flat platform at the center of the terrain.

    .. image:: ../../_static/terrains/height_field/stepping_stones_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (rows_y, cols_x), where rows_y and cols_x are the number of points
        along the y and x axis, respectively.
    """
    # resolve terrain configuration
    stone_width = cfg.stone_width_range[1] - difficulty * (cfg.stone_width_range[1] - cfg.stone_width_range[0])
    stone_distance = cfg.stone_distance_range[0] + difficulty * (
        cfg.stone_distance_range[1] - cfg.stone_distance_range[0]
    )

    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- stones
    stone_distance = int(stone_distance / cfg.horizontal_scale)
    stone_width = int(stone_width / cfg.horizontal_scale)
    stone_height_max = int(cfg.stone_height_max / cfg.vertical_scale)
    # -- holes
    holes_depth = int(cfg.holes_depth / cfg.vertical_scale)
    # -- platform
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)
    # create range of heights
    stone_height_range = np.arange(-stone_height_max - 1, stone_height_max, step=1)

    # create a terrain with a flat platform at the center
    hf_raw = np.full((length_pixels, width_pixels), holes_depth)
    # add the stones
    start_x, start_y = 0, 0
    # -- if the terrain is longer than it is wide then fill the terrain column by column
    if length_pixels >= width_pixels:
        while start_y < length_pixels:
            # ensure that stone stops along y-axis
            stop_y = min(length_pixels, start_y + stone_width)
            # randomly sample x-position
            start_x = np.random.randint(0, stone_width)
            stop_x = max(0, start_x - stone_distance)
            # fill first stone
            hf_raw[start_y:stop_y, 0:stop_x] = np.random.choice(stone_height_range)
            # fill row with stones
            while start_x < width_pixels:
                stop_x = min(width_pixels, start_x + stone_width)
                hf_raw[start_y:stop_y, start_x:stop_x] = np.random.choice(stone_height_range)
                start_x += stone_width + stone_distance
            # update y-position
            start_y += stone_width + stone_distance
    elif width_pixels > length_pixels:
        while start_x < width_pixels:
            # ensure that stone stops along x-axis
            stop_x = min(width_pixels, start_x + stone_width)
            # randomly sample y-position
            start_y = np.random.randint(0, stone_width)
            stop_y = max(0, start_y - stone_distance)
            # fill first stone
            hf_raw[0:stop_y, start_x:stop_x] = np.random.choice(stone_height_range)
            # fill column with stones
            while start_y < length_pixels:
                stop_y = min(length_pixels, start_y + stone_width)
                hf_raw[start_y:stop_y, start_x:stop_x] = np.random.choice(stone_height_range)
                start_y += stone_width + stone_distance
            # update x-position
            start_x += stone_width + stone_distance
    # add the platform in the center
    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[y1:y2, x1:x2] = 0
    # round off the heights to the nearest vertical step

    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(
            difficulty,
            perlin_cfg,  # type: ignore[arg-type]
        )
        # add perlin noise to the terrain
        hf_raw += perlin_noise

    return np.rint(hf_raw).astype(np.int16)


# -- Newly added terrains for parkour -- #


@generate_wall
@height_field_to_mesh
def perlin_parapet_terrain(difficulty: float, cfg: hf_terrains_cfg.PerlinParapetTerrainCfg) -> np.ndarray:
    """Generate a terrain with a parapet pattern.

    The terrain is a parapets pattern standing out from the ground.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (rows_y, cols_x), where rows_y and cols_x are the number of points
        along the y and x axis, respectively.
    """

    # resolve terrain configuration

    if isinstance(cfg.parapet_height, (list, tuple)):
        parapet_height = cfg.parapet_height[0] + difficulty * (cfg.parapet_height[1] - cfg.parapet_height[0])
    else:
        parapet_height = cfg.parapet_height

    if isinstance(cfg.parapet_length, (list, tuple)):
        parapet_length = cfg.parapet_length[0] + difficulty * (cfg.parapet_length[1] - cfg.parapet_length[0])
    else:
        parapet_length = cfg.parapet_length

    if cfg.parapet_width is None:
        parapet_width = cfg.size[0]
    else:
        parapet_width = cfg.parapet_width

    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- parapets
    parapet_height = int(parapet_height / cfg.vertical_scale)
    parapet_width = int(parapet_width / cfg.horizontal_scale)
    parapet_length = int(parapet_length / cfg.horizontal_scale)

    # create a terrain with a flat platform at the center
    hf_raw = np.zeros((length_pixels, width_pixels))
    start_x = width_pixels // 2 - parapet_width // 2
    start_y = length_pixels // 2 - parapet_length // 2

    end_x = start_x + parapet_width
    end_y = start_y + parapet_length

    hf_raw[start_y:end_y, start_x:end_x] = parapet_height

    if cfg.curved_top_rate is not None and cfg.curved_top_rate > 0.0 and cfg.curved_top_rate > np.random.uniform():
        curve = 1 - np.square(np.linspace(-1, 1, parapet_length)) * 0.5
        hf_raw[start_y:end_y, start_x:end_x] = parapet_height * curve.reshape(-1, 1)

    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(
            difficulty,
            perlin_cfg,  # type: ignore[arg-type]
        )
        hf_raw += perlin_noise

    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


@generate_wall
@height_field_to_mesh
def perlin_gutter_terrain(difficulty: float, cfg: hf_terrains_cfg.PerlinGutterTerrainCfg):
    """Generate a terrain with a gutter pattern.

    The terrain is a gutter pattern sinking down from the ground.

    """
    # resolve terrain configuration
    if isinstance(cfg.gutter_length, (list, tuple)):
        gutter_length = cfg.gutter_length[0] + difficulty * (cfg.gutter_length[1] - cfg.gutter_length[0])
    else:
        gutter_length = cfg.gutter_length

    if isinstance(cfg.gutter_depth, (list, tuple)):
        gutter_depth = np.random.uniform(cfg.gutter_depth[0], cfg.gutter_depth[1])
    else:
        gutter_depth = cfg.gutter_depth

    if cfg.gutter_width is None:
        gutter_width = cfg.size[0]
    else:
        gutter_width = cfg.gutter_width

    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- gutters
    gutter_depth = int(gutter_depth / cfg.vertical_scale)
    gutter_width = int(gutter_width / cfg.horizontal_scale)
    gutter_length = int(gutter_length / cfg.horizontal_scale)

    # create a terrain with a flat platform at the center
    hf_raw = np.zeros((length_pixels, width_pixels))

    start_x = width_pixels // 2 - gutter_width // 2
    start_y = length_pixels // 2 - gutter_length // 2

    end_x = start_x + gutter_width
    end_y = start_y + gutter_length

    hf_raw[start_y:end_y, start_x:end_x] = -gutter_depth

    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(
            difficulty,
            perlin_cfg,  # type: ignore[arg-type]
        )
        hf_raw += perlin_noise

    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


@generate_wall
@height_field_to_mesh
def perlin_stairs_up_down_terrain(difficulty: float, cfg: hf_terrains_cfg.PerlinStairsUpDownTerrainCfg):
    """Generate a terrain with stairs going up and down."""
    # resolve terrain configuration
    if isinstance(cfg.per_step_height, (list, tuple)):
        per_step_height = cfg.per_step_height[0] + difficulty * (cfg.per_step_height[1] - cfg.per_step_height[0])
    else:
        per_step_height = cfg.per_step_height
    if isinstance(cfg.per_step_length, (list, tuple)):
        per_step_length = cfg.per_step_length[0] + difficulty * (cfg.per_step_length[1] - cfg.per_step_length[0])
    else:
        per_step_length = cfg.per_step_length
    if isinstance(cfg.num_steps, (list, tuple)):
        num_steps = cfg.num_steps[0] + difficulty * (cfg.num_steps[1] - cfg.num_steps[0])
    else:
        num_steps = cfg.num_steps
    platform_length = cfg.platform_length
    if cfg.per_step_width is None:
        per_step_width = cfg.size[0]
    else:
        per_step_width = cfg.per_step_width

    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- steps
    per_step_height = int(per_step_height / cfg.vertical_scale)
    per_step_width = int(per_step_width / cfg.horizontal_scale)
    per_step_length = int(per_step_length / cfg.horizontal_scale)
    platform_length = int(platform_length / cfg.horizontal_scale)
    num_steps = int(num_steps)
    num_steps = min(num_steps, (length_pixels - platform_length) // (2 * per_step_length))

    # create a terrain with a flat platform at the center
    hf_raw = np.zeros((length_pixels, width_pixels))
    middle_x = width_pixels // 2
    middle_y = length_pixels // 2
    start_x = middle_x - per_step_width // 2
    end_x = start_x + per_step_width
    start_y_up = middle_y - platform_length // 2
    start_y_down = start_y_up + platform_length
    for i in range(num_steps):
        # going up
        start_y = start_y_up - i * per_step_length
        end_y = start_y + per_step_length
        hf_raw[start_y:end_y, start_x:end_x] = (num_steps - i) * per_step_height

        # going down
        start_y = start_y_down + i * per_step_length
        end_y = start_y + per_step_length
        hf_raw[start_y:end_y, start_x:end_x] = (num_steps - i) * per_step_height

    # add the platform in the center
    hf_raw[start_y_up:start_y_down, start_x:end_x] = num_steps * per_step_height

    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(
            difficulty,
            perlin_cfg,  # type: ignore[arg-type]
        )
        # add perlin noise to the terrain
        hf_raw += perlin_noise

    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


@generate_wall
@height_field_to_mesh
def perlin_stairs_down_up_terrain(difficulty: float, cfg: hf_terrains_cfg.PerlinStairsDownUpTerrainCfg):
    """Generate a terrain with stairs going down and up."""
    # resolve terrain configuration
    if isinstance(cfg.per_step_height, (list, tuple)):
        per_step_height = cfg.per_step_height[0] + difficulty * (cfg.per_step_height[1] - cfg.per_step_height[0])
    else:
        per_step_height = cfg.per_step_height
    if isinstance(cfg.per_step_length, (list, tuple)):
        per_step_length = cfg.per_step_length[0] + difficulty * (cfg.per_step_length[1] - cfg.per_step_length[0])
    else:
        per_step_length = cfg.per_step_length
    if isinstance(cfg.num_steps, (list, tuple)):
        num_steps = cfg.num_steps[0] + difficulty * (cfg.num_steps[1] - cfg.num_steps[0])
    else:
        num_steps = cfg.num_steps

    platform_length = cfg.platform_length
    if cfg.per_step_width is None:
        per_step_width = cfg.size[0]
    else:
        per_step_width = cfg.per_step_width

    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- steps
    per_step_height = int(per_step_height / cfg.vertical_scale)
    per_step_width = int(per_step_width / cfg.horizontal_scale)
    per_step_length = int(per_step_length / cfg.horizontal_scale)
    platform_length = int(platform_length / cfg.horizontal_scale)
    num_steps = int(num_steps)
    num_steps = min(num_steps, (length_pixels - platform_length) // (2 * per_step_length))

    # create a terrain with a flat platform at the center
    hf_raw = np.zeros((length_pixels, width_pixels))
    middle_x = width_pixels // 2
    middle_y = length_pixels // 2
    start_x = middle_x - per_step_width // 2
    end_x = start_x + per_step_width
    start_y_up = middle_y - platform_length // 2
    start_y_down = start_y_up + platform_length
    for i in range(num_steps):
        # going up
        start_y = start_y_up - i * per_step_length
        end_y = start_y + per_step_length
        hf_raw[start_y:end_y, start_x:end_x] = -(num_steps - i) * per_step_height

        # going down
        start_y = start_y_down + i * per_step_length
        end_y = start_y + per_step_length
        hf_raw[start_y:end_y, start_x:end_x] = -(num_steps - i) * per_step_height

    # add the platform in the center
    hf_raw[start_y_up:start_y_down, start_x:end_x] = -num_steps * per_step_height

    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(
            difficulty,
            perlin_cfg,  # type: ignore[arg-type]
        )
        # add perlin noise to the terrain
        hf_raw += perlin_noise

    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


@generate_wall
@height_field_to_mesh
def perlin_tilt_terrain(difficulty: float, cfg: hf_terrains_cfg.PerlinTiltTerrainCfg):
    """Generate a terrain with a tilt pattern.

    The terrain is a tilted plane with a slope that increases with difficulty.
    """

    # resolve terrain configuration
    if isinstance(cfg.wall_length, (list, tuple)):
        wall_length = np.random.uniform(*cfg.wall_length)
    else:
        wall_length = cfg.wall_length

    if isinstance(cfg.wall_opening_width, (list, tuple)):
        wall_opening_width = cfg.wall_opening_width[1] - difficulty * (
            cfg.wall_opening_width[1] - cfg.wall_opening_width[0]
        )
    else:
        wall_opening_width = cfg.wall_opening_width

    if isinstance(cfg.wall_opening_angle, (list, tuple)):
        wall_opening_angle = np.random.uniform(*cfg.wall_opening_angle)
    else:
        wall_opening_angle = cfg.wall_opening_angle

    if isinstance(cfg.wall_height, (list, tuple)):
        wall_height = np.random.uniform(*cfg.wall_height)
    else:
        wall_height = cfg.wall_height

    if cfg.wall_width is None:
        wall_width = cfg.size[0]
    else:
        wall_width = cfg.wall_width

    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    # -- walls
    wall_length = int(wall_length / cfg.horizontal_scale)
    wall_opening_width = int(wall_opening_width / cfg.horizontal_scale)
    wall_height = int(wall_height / cfg.vertical_scale)
    wall_width = int(wall_width / cfg.horizontal_scale)

    # create a terrain with a flat platform at the center
    hf_raw = np.zeros((length_pixels, width_pixels))
    start_x = width_pixels // 2 - wall_width // 2
    end_x = start_x + wall_width
    start_y = length_pixels // 2 - wall_length // 2
    end_y = start_y + wall_length
    middle_x = width_pixels // 2

    # generate the wall
    hf_raw[start_y:end_y, start_x:end_x] = wall_height

    # generate the wall opening
    if wall_opening_angle > 0:
        for length_px in range(wall_length + 1):
            current_y = end_y - length_px
            open_width = int(length_px * np.tan(np.deg2rad(wall_opening_angle)))
            open_width_start = middle_x - wall_opening_width // 2 - open_width
            open_width_end = open_width_start + wall_opening_width + 2 * open_width

            open_width_start = max(open_width_start, 0)
            open_width_end = min(open_width_end, width_pixels)
            current_y = max(current_y, 0)

            hf_raw[current_y, open_width_start:open_width_end] = 0
    else:
        open_width_start = middle_x - wall_opening_width // 2
        open_width_end = open_width_start + wall_opening_width
        open_width_start = max(open_width_start, 0)
        open_width_end = min(open_width_end, width_pixels)
        hf_raw[start_y:end_y, open_width_start:open_width_end] = 0

    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(
            difficulty,
            perlin_cfg,  # type: ignore[arg-type]
        )
        # add perlin noise to the terrain
        hf_raw += perlin_noise

    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


@generate_wall
@height_field_to_mesh
def perlin_tilted_ramp_terrain(difficulty: float, cfg: hf_terrains_cfg.PerlinTiltedRampTerrainCfg):
    """Generate a terrain with a tilted ramp pattern."""

    # resolve terrain configuration
    if isinstance(cfg.tilt_angle, (list, tuple)):
        tilt_angle = (1 - difficulty) * cfg.tilt_angle[0] + difficulty * cfg.tilt_angle[1]
    else:
        tilt_angle = cfg.tilt_angle

    if isinstance(cfg.tilt_height, (list, tuple)):
        tilt_height = np.random.uniform(cfg.tilt_height[0], cfg.tilt_height[1])
    else:
        tilt_height = cfg.tilt_height

    if isinstance(cfg.tilt_width, (list, tuple)):
        tilt_width = np.random.uniform(cfg.tilt_width[0], cfg.tilt_width[1])
    else:
        tilt_width = cfg.tilt_width

    if isinstance(cfg.tilt_length, (list, tuple)):
        tilt_length = (1 - difficulty) * cfg.tilt_length[0] + difficulty * cfg.tilt_length[1]
    else:
        tilt_length = cfg.tilt_length

    if isinstance(cfg.switch_spacing, (list, tuple)):
        if cfg.spacing_curriculum:
            switch_spacing = (1 - difficulty) * cfg.switch_spacing[0] + difficulty * cfg.switch_spacing[1]
        else:
            switch_spacing = np.random.uniform(*cfg.switch_spacing)
    else:
        switch_spacing = cfg.switch_spacing

    # switch parameters to discrete units
    # --terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    # --ramp
    tilt_height = int(tilt_height / cfg.vertical_scale)
    tilt_width = int(tilt_width / cfg.horizontal_scale)
    tilt_length = int(tilt_length / cfg.horizontal_scale)
    width_empty_space = int((width_pixels - tilt_width) / 2)

    hf_raw = np.zeros((length_pixels, width_pixels))
    tilt_angle = np.deg2rad(tilt_angle)

    if switch_spacing > 0.0:
        switch_spacing_px = int(switch_spacing / cfg.horizontal_scale)
        assert cfg.overlap_size is not None, "No overlap_size specified."
        overlap_size_px = int(cfg.overlap_size / cfg.horizontal_scale)
        start_left_px = int(width_pixels / 2 + overlap_size_px / 2)
        start_right_px = int(width_pixels / 2 - overlap_size_px / 2)
        tilt_left = True if np.random.uniform() < 0.5 else False
        start_y_px = 1
        end_y_px = start_y_px + switch_spacing_px
        while end_y_px < length_pixels and end_y_px <= tilt_length:
            if tilt_left:
                tilt_left_px = start_left_px - width_empty_space
                tilt_profile = (
                    np.linspace(tilt_left_px, 0, tilt_left_px)
                    * np.tan(tilt_angle)
                    * cfg.horizontal_scale
                    / cfg.vertical_scale
                    + tilt_height
                )
                y_size = end_y_px - start_y_px
                tilt_profile_2d = np.tile(tilt_profile[np.newaxis, :], (y_size, 1))
                hf_raw[start_y_px:end_y_px, width_empty_space:start_left_px] += tilt_profile_2d
            else:
                tilt_right_px = width_pixels - start_right_px - width_empty_space
                tilt_profile = (
                    np.linspace(0, tilt_right_px, tilt_right_px)
                    * np.tan(tilt_angle)
                    * cfg.horizontal_scale
                    / cfg.vertical_scale
                    + tilt_height
                )
                y_size = end_y_px - start_y_px
                tilt_profile_2d = np.tile(tilt_profile[np.newaxis, :], (y_size, 1))
                hf_raw[start_y_px:end_y_px, start_right_px:-width_empty_space] += tilt_profile_2d
            start_y_px = end_y_px
            end_y_px = start_y_px + switch_spacing_px
            tilt_left = not tilt_left
    else:
        # two-side tilted ramp
        start_px = int(width_pixels / 2)
        tilt_left_px = start_px - width_empty_space
        tilt_right_px = width_pixels - start_px - width_empty_space
        start_y_px = int((length_pixels - tilt_length) / 2)
        end_y_px = start_y_px + tilt_length
        tilt_profile = (
            np.linspace(tilt_left_px, 0, tilt_left_px) * np.tan(tilt_angle) * cfg.horizontal_scale / cfg.vertical_scale
            + tilt_height
        )
        tilt_Profile_2d = np.tile(tilt_profile[np.newaxis, :], (end_y_px - start_y_px, 1))
        hf_raw[start_y_px:end_y_px, width_empty_space:start_px] += tilt_Profile_2d
        right_tilt_profile = (
            np.linspace(0, tilt_right_px, tilt_right_px)
            * np.tan(tilt_angle)
            * cfg.horizontal_scale
            / cfg.vertical_scale
            + tilt_height
        )
        right_tilt_profile_2d = np.tile(right_tilt_profile[np.newaxis, :], (end_y_px - start_y_px, 1))
        hf_raw[start_y_px:end_y_px, start_px : width_pixels - width_empty_space] += right_tilt_profile_2d

    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(
            difficulty,
            perlin_cfg,  # type: ignore[arg-type]
        )
        # add perlin noise to the terrain
        hf_raw += perlin_noise

    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


@generate_wall
@height_field_to_mesh
def perlin_slope_terrain(difficulty: float, cfg: hf_terrains_cfg.PerlinSlopeTerrainCfg) -> np.ndarray:
    """Generate a terrain with a slope going up and down."""
    # resolve terrain configuration
    if isinstance(cfg.slope_angle, (list, tuple)):
        slope_angle = cfg.slope_angle[0] + difficulty * (cfg.slope_angle[1] - cfg.slope_angle[0])
    else:
        slope_angle = cfg.slope_angle
    if isinstance(cfg.per_slope_length, (list, tuple)):
        slope_length = cfg.per_slope_length[0] + difficulty * (cfg.per_slope_length[1] - cfg.per_slope_length[0])
    else:
        slope_length = cfg.per_slope_length
    platform_length = cfg.platform_length
    if cfg.slope_width is None:
        slope_width = cfg.size[0]
    else:
        slope_width = cfg.slope_width

    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- slopes
    slope_angle = np.deg2rad(slope_angle)
    slope_width = int(slope_width / cfg.horizontal_scale)
    slope_length = int(slope_length / cfg.horizontal_scale)
    platform_length = int(platform_length / cfg.horizontal_scale)

    slope_height = int(slope_length * np.tan(slope_angle) * cfg.horizontal_scale / cfg.vertical_scale)

    if cfg.up_down == False:
        slope_height = -slope_height
    # create a terrain with a flat platform at the center
    hf_raw = np.zeros((length_pixels, width_pixels))
    middle_x = width_pixels // 2
    middle_y = length_pixels // 2
    start_x = middle_x - slope_width // 2
    end_x = start_x + slope_width
    end_y_up = middle_y - platform_length // 2
    start_y_up = end_y_up - slope_length
    start_y_down = end_y_up + platform_length
    end_y_down = start_y_down + slope_length

    up_slope_profile = np.linspace(0, slope_height, slope_length)
    down_slope_profile = np.linspace(slope_height, 0, slope_length)

    # going up
    hf_raw[start_y_up:end_y_up, start_x:end_x] = np.tile(up_slope_profile[:, np.newaxis], (1, slope_width))
    # going down
    hf_raw[start_y_down:end_y_down, start_x:end_x] = np.tile(down_slope_profile[:, np.newaxis], (1, slope_width))
    # add the platform in the center
    hf_raw[end_y_up:start_y_down, start_x:end_x] = slope_height

    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(
            difficulty,
            perlin_cfg,  # type: ignore[arg-type]
        )
        # add perlin noise to the terrain
        hf_raw += perlin_noise

    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


@generate_wall
@height_field_to_mesh
def perlin_cross_stone_terrain(difficulty: float, cfg: hf_terrains_cfg.PerlinCrossStoneTerrainCfg) -> np.ndarray:
    """Generate a terrain with a cross stone pattern."""
    # resolve terrain configuration
    stone_size = cfg.stone_size
    if isinstance(cfg.stone_height, (list, tuple)):
        stone_height = cfg.stone_height[0] + difficulty * (cfg.stone_height[1] - cfg.stone_height[0])
    else:
        stone_height = cfg.stone_height
    if isinstance(cfg.stone_spacing, (list, tuple)):
        stone_spacing = cfg.stone_spacing[0] + difficulty * (cfg.stone_spacing[1] - cfg.stone_spacing[0])
    else:
        stone_spacing = cfg.stone_spacing
    ground_depth = cfg.ground_depth
    platform_width = cfg.platform_width
    xy_random_ratio = cfg.xy_random_ratio

    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    stone_width = int(stone_size[0] / cfg.horizontal_scale)
    stone_length = int(stone_size[1] / cfg.horizontal_scale)
    stone_height = int(stone_height / cfg.vertical_scale)
    stone_spacing = int(stone_spacing / cfg.horizontal_scale)
    ground_depth = int(ground_depth / cfg.vertical_scale)
    platform_width = int(platform_width / cfg.horizontal_scale)

    hf_raw = np.zeros((length_pixels, width_pixels))
    hf_raw += ground_depth

    # create platform (square in the center)
    platform_start_x = (width_pixels - platform_width) // 2
    platform_end_x = platform_start_x + platform_width
    platform_start_y = (length_pixels - platform_width) // 2
    platform_end_y = platform_start_y + platform_width
    hf_raw[platform_start_y:platform_end_y, platform_start_x:platform_end_x] = 0

    # generate stone positions
    # Get platform center coordinates
    platform_center_x = width_pixels // 2
    platform_center_y = length_pixels // 2

    # Create cross pattern from platform center extending in four directions

    # From platform extending left (moving away from center)
    x = platform_start_x - stone_spacing - stone_width
    while x >= 0:
        if xy_random_ratio > 0.0:
            rand_x = int((np.random.uniform() - 0.5) * stone_spacing * xy_random_ratio)
            rand_y = int((np.random.uniform() - 0.5) * stone_spacing * xy_random_ratio)
        else:
            rand_x, rand_y = 0, 0
        stone_x1 = np.clip(x + rand_x, 0, width_pixels - stone_width)
        stone_x2 = np.clip(stone_x1 + stone_width, 0, width_pixels)
        stone_y1 = np.clip(platform_center_y - stone_length // 2 + rand_y, 0, length_pixels - stone_length)
        stone_y2 = np.clip(stone_y1 + stone_length, 0, length_pixels)
        if stone_x2 > stone_x1 and stone_y2 > stone_y1:
            hf_raw[stone_y1:stone_y2, stone_x1:stone_x2] = stone_height
        x -= stone_width + stone_spacing

    # Add one more stone in the remaining space if there's room
    if x + stone_width + stone_spacing > 0:
        if xy_random_ratio > 0.0:
            rand_x = int((np.random.uniform() - 0.5) * stone_spacing * xy_random_ratio)
            rand_y = int((np.random.uniform() - 0.5) * stone_spacing * xy_random_ratio)
        else:
            rand_x, rand_y = 0, 0
        stone_x1 = np.clip(0 + rand_x, 0, width_pixels - stone_width)
        stone_x2 = np.clip(stone_x1 + stone_width, 0, width_pixels)
        stone_y1 = np.clip(platform_center_y - stone_length // 2 + rand_y, 0, length_pixels - stone_length)
        stone_y2 = np.clip(stone_y1 + stone_length, 0, length_pixels)
        if stone_x2 > stone_x1 and stone_y2 > stone_y1:
            hf_raw[stone_y1:stone_y2, stone_x1:stone_x2] = stone_height

    # From platform extending right (moving away from center)
    x = platform_end_x + stone_spacing
    while x + stone_width <= width_pixels:
        if xy_random_ratio > 0.0:
            rand_x = int((np.random.uniform() - 0.5) * stone_spacing * xy_random_ratio)
            rand_y = int((np.random.uniform() - 0.5) * stone_spacing * xy_random_ratio)
        else:
            rand_x, rand_y = 0, 0
        stone_x1 = np.clip(x + rand_x, 0, width_pixels - stone_width)
        stone_x2 = np.clip(stone_x1 + stone_width, 0, width_pixels)
        stone_y1 = np.clip(platform_center_y - stone_length // 2 + rand_y, 0, length_pixels - stone_length)
        stone_y2 = np.clip(stone_y1 + stone_length, 0, length_pixels)
        if stone_x2 > stone_x1 and stone_y2 > stone_y1:
            hf_raw[stone_y1:stone_y2, stone_x1:stone_x2] = stone_height
        x += stone_width + stone_spacing

    # Add one more stone in the remaining space if there's room
    if x - stone_width - stone_spacing < width_pixels:
        if xy_random_ratio > 0.0:
            rand_x = int((np.random.uniform() - 0.5) * stone_spacing * xy_random_ratio)
            rand_y = int((np.random.uniform() - 0.5) * stone_spacing * xy_random_ratio)
        else:
            rand_x, rand_y = 0, 0
        stone_x1 = np.clip(width_pixels - stone_width + rand_x, 0, width_pixels - stone_width)
        stone_x2 = np.clip(stone_x1 + stone_width, 0, width_pixels)
        stone_y1 = np.clip(platform_center_y - stone_length // 2 + rand_y, 0, length_pixels - stone_length)
        stone_y2 = np.clip(stone_y1 + stone_length, 0, length_pixels)
        if stone_x2 > stone_x1 and stone_y2 > stone_y1:
            hf_raw[stone_y1:stone_y2, stone_x1:stone_x2] = stone_height

    # From platform extending up (moving away from center)
    y = platform_start_y - stone_spacing - stone_length
    while y >= 0:
        if xy_random_ratio > 0.0:
            rand_x = int((np.random.uniform() - 0.5) * stone_spacing * xy_random_ratio)
            rand_y = int((np.random.uniform() - 0.5) * stone_spacing * xy_random_ratio)
        else:
            rand_x, rand_y = 0, 0
        stone_x1 = np.clip(platform_center_x - stone_width // 2 + rand_x, 0, width_pixels - stone_width)
        stone_x2 = np.clip(stone_x1 + stone_width, 0, width_pixels)
        stone_y1 = np.clip(y + rand_y, 0, length_pixels - stone_length)
        stone_y2 = np.clip(stone_y1 + stone_length, 0, length_pixels)
        if stone_x2 > stone_x1 and stone_y2 > stone_y1:
            hf_raw[stone_y1:stone_y2, stone_x1:stone_x2] = stone_height
        y -= stone_length + stone_spacing

    # Add one more stone in the remaining space if there's room
    if y + stone_length + stone_spacing > 0:
        if xy_random_ratio > 0.0:
            rand_x = int((np.random.uniform() - 0.5) * stone_spacing * xy_random_ratio)
            rand_y = int((np.random.uniform() - 0.5) * stone_spacing * xy_random_ratio)
        else:
            rand_x, rand_y = 0, 0
        stone_x1 = np.clip(platform_center_x - stone_width // 2 + rand_x, 0, width_pixels - stone_width)
        stone_x2 = np.clip(stone_x1 + stone_width, 0, width_pixels)
        stone_y1 = np.clip(0 + rand_y, 0, length_pixels - stone_length)
        stone_y2 = np.clip(stone_y1 + stone_length, 0, length_pixels)
        if stone_x2 > stone_x1 and stone_y2 > stone_y1:
            hf_raw[stone_y1:stone_y2, stone_x1:stone_x2] = stone_height

    # From platform extending down (moving away from center)
    y = platform_end_y + stone_spacing
    while y + stone_length <= length_pixels:
        if xy_random_ratio > 0.0:
            rand_x = int((np.random.uniform() - 0.5) * stone_spacing * xy_random_ratio)
            rand_y = int((np.random.uniform() - 0.5) * stone_spacing * xy_random_ratio)
        else:
            rand_x, rand_y = 0, 0
        stone_x1 = np.clip(platform_center_x - stone_width // 2 + rand_x, 0, width_pixels - stone_width)
        stone_x2 = np.clip(stone_x1 + stone_width, 0, width_pixels)
        stone_y1 = np.clip(y + rand_y, 0, length_pixels - stone_length)
        stone_y2 = np.clip(stone_y1 + stone_length, 0, length_pixels)
        if stone_x2 > stone_x1 and stone_y2 > stone_y1:
            hf_raw[stone_y1:stone_y2, stone_x1:stone_x2] = stone_height
        y += stone_length + stone_spacing

    # Add one more stone in the remaining space if there's room
    if y - stone_length - stone_spacing < length_pixels:
        if xy_random_ratio > 0.0:
            rand_x = int((np.random.uniform() - 0.5) * stone_spacing * xy_random_ratio)
            rand_y = int((np.random.uniform() - 0.5) * stone_spacing * xy_random_ratio)
        else:
            rand_x, rand_y = 0, 0
        stone_x1 = np.clip(platform_center_x - stone_width // 2 + rand_x, 0, width_pixels - stone_width)
        stone_x2 = np.clip(stone_x1 + stone_width, 0, width_pixels)
        stone_y1 = np.clip(length_pixels - stone_length + rand_y, 0, length_pixels - stone_length)
        stone_y2 = np.clip(stone_y1 + stone_length, 0, length_pixels)
        if stone_x2 > stone_x1 and stone_y2 > stone_y1:
            hf_raw[stone_y1:stone_y2, stone_x1:stone_x2] = stone_height

    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(
            difficulty,
            perlin_cfg,  # type: ignore[arg-type]
        )
        # add perlin noise to the terrain
        hf_raw += perlin_noise

    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


@generate_wall
@height_field_to_mesh
def perlin_square_gap_terrain(difficulty: float, cfg: hf_terrains_cfg.PerlinSquareGapTerrainCfg) -> np.ndarray:
    """Generate a terrain with a square gap pattern."""
    # resolve terrain configuration
    gap_distance = cfg.gap_distance_range[0] + difficulty * (cfg.gap_distance_range[1] - cfg.gap_distance_range[0])

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- platform
    platform_width = round(cfg.platform_width / cfg.horizontal_scale)

    # create a terrain with a flat platform at the center
    hf_raw = np.zeros((length_pixels, width_pixels))
    # add the steps
    # compute discrete gap size
    gap_distance_px = round(gap_distance / cfg.horizontal_scale)

    # platform coordinates (centered square)
    platform_start_x = (width_pixels - platform_width) // 2
    platform_end_x = platform_start_x + platform_width
    platform_start_y = (length_pixels - platform_width) // 2
    platform_end_y = platform_start_y + platform_width

    # ensure platform inside bounds
    platform_start_x = max(0, platform_start_x)
    platform_start_y = max(0, platform_start_y)
    platform_end_x = min(width_pixels, platform_end_x)
    platform_end_y = min(length_pixels, platform_end_y)

    # create central platform
    hf_raw[platform_start_y:platform_end_y, platform_start_x:platform_end_x] = 0

    # groove depth in pixels (use cfg.gap_depth if available, else default to one vertical unit)
    gap_depth = np.random.uniform(cfg.gap_depth[0], cfg.gap_depth[1])
    gap_value = -round(gap_depth / cfg.vertical_scale)

    # draw square "spiral" grooves (concentric square borders) outward from the platform
    # Only draw one ring of gap around the platform
    sx = platform_start_x - gap_distance_px
    ex = platform_end_x + gap_distance_px
    sy = platform_start_y - gap_distance_px
    ey = platform_end_y + gap_distance_px

    # clamp to terrain bounds
    sx_clamped = max(0, sx)
    ex_clamped = min(width_pixels, ex)
    sy_clamped = max(0, sy)
    ey_clamped = min(length_pixels, ey)

    t = gap_distance_px  # thickness of the groove

    # Draw the groove border (top, bottom, left, right)
    # Top edge
    hf_raw[sy_clamped : sy_clamped + t, sx_clamped:ex_clamped] = gap_value
    # Bottom edge
    hf_raw[max(sy_clamped, ey_clamped - t) : ey_clamped, sx_clamped:ex_clamped] = gap_value
    # Left edge
    hf_raw[sy_clamped:ey_clamped, sx_clamped : sx_clamped + t] = gap_value
    # Right edge
    hf_raw[sy_clamped:ey_clamped, max(sx_clamped, ex_clamped - t) : ex_clamped] = gap_value
    ey_clamped = min(length_pixels, ey)

    t = gap_distance_px  # thickness of the groove

    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(
            difficulty,
            perlin_cfg,  # type: ignore[arg-type]
        )
        # add perlin noise to the terrain
        hf_raw += perlin_noise

    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


@generate_wall
@height_field_to_mesh
def perlin_continuous_slope_terrain(difficulty: float, cfg: hf_terrains_cfg.PerlinContinuousSlopeTerrainCfg) -> np.ndarray:
    """Generate a terrain with two consecutive slope humps (double hump).

    Travel direction is along the x-axis. Profile along x:
    up-slope → platform → down-slope → up-slope → platform → down-slope

    Total length and max height are fixed. Slope angle varies with difficulty;
    platform length along x adapts automatically. Lateral width is fixed or
    sampled uniformly when ``terrain_width`` is a ``(min, max)`` tuple.
    """
    # resolve slope angle from difficulty
    angle_lo, angle_hi = cfg.slope_angle_range
    slope_angle_deg = angle_lo + difficulty * (angle_hi - angle_lo)
    slope_angle = np.deg2rad(slope_angle_deg)

    # hf_raw shape: (ny, nx) = (size[1]/hs, size[0]/hs)
    ny = int(cfg.size[1] / cfg.horizontal_scale)
    nx = int(cfg.size[0] / cfg.horizontal_scale)
    hs = cfg.horizontal_scale
    vs = cfg.vertical_scale

    max_height_i = int(round(cfg.max_height / vs))
    terrain_width_m = _resolve_lateral_width(cfg.terrain_width)
    if terrain_width_m is None:
        terrain_width_m = 2.4
    width_px = int(round(terrain_width_m / hs))       # along y (rows)
    length_px = int(round(cfg.total_length / hs))        # along x (cols)

    # constrain to tile bounds
    width_px = min(width_px, ny)
    length_px = min(length_px, nx)

    # compute slope horizontal projection from max height and angle
    slope_horiz_px = int(round(max_height_i * vs / np.tan(slope_angle) / hs))
    # clamp so 4 slopes fit within total length
    slope_horiz_px = min(slope_horiz_px, length_px // 4)

    # compute platform width (must be >= 0)
    remaining = length_px - 4 * slope_horiz_px
    platform_width_px = max(0, remaining // 2)

    # center the feature in the tile
    y_start = (ny - width_px) // 2
    y_end = y_start + width_px
    x_start = (nx - length_px) // 2
    x_end = x_start + length_px

    hf_raw = np.zeros((ny, nx), dtype=np.float64)

    # define segment boundaries along x (columns)
    seg1_up_end = x_start + slope_horiz_px
    seg1_plat_end = seg1_up_end + platform_width_px
    seg1_down_end = seg1_plat_end + slope_horiz_px
    seg2_up_end = seg1_down_end + slope_horiz_px
    seg2_plat_end = seg2_up_end + platform_width_px

    # up slope 1: linear from 0 to max_height (along x)
    up1 = np.linspace(0, max_height_i, slope_horiz_px, dtype=np.float64)
    hf_raw[y_start:y_end, x_start:seg1_up_end] = np.tile(up1[np.newaxis, :], (width_px, 1))

    # platform 1: constant at max_height
    hf_raw[y_start:y_end, seg1_up_end:seg1_plat_end] = max_height_i

    # down slope 1: linear from max_height to 0 (along x)
    down1 = np.linspace(max_height_i, 0, slope_horiz_px, dtype=np.float64)
    hf_raw[y_start:y_end, seg1_plat_end:seg1_down_end] = np.tile(down1[np.newaxis, :], (width_px, 1))

    # up slope 2: linear from 0 to max_height (along x)
    up2 = np.linspace(0, max_height_i, slope_horiz_px, dtype=np.float64)
    hf_raw[y_start:y_end, seg1_down_end:seg2_up_end] = np.tile(up2[np.newaxis, :], (width_px, 1))

    # platform 2: constant at max_height
    hf_raw[y_start:y_end, seg2_up_end:seg2_plat_end] = max_height_i

    # down slope 2: linear from max_height to 0 (along x)
    down2 = np.linspace(max_height_i, 0, slope_horiz_px, dtype=np.float64)
    hf_raw[y_start:y_end, seg2_plat_end:x_end] = np.tile(down2[np.newaxis, :], (width_px, 1))

    # optional perlin noise overlay
    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(difficulty, perlin_cfg)
        hf_raw += perlin_noise

    return np.rint(hf_raw).astype(np.int16)


@generate_wall
@height_field_to_mesh
def perlin_cross_slope_terrain(difficulty: float, cfg: hf_terrains_cfg.PerlinCrossSlopeTerrainCfg) -> np.ndarray:
    """Generate four-segment lateral cross-slope (交叉斜坡) terrain.

    Travel along x; lateral profile along y alternates every segment_length.
    Centerline height is continuous (~height_diff/2); lateral edges jump at junctions.
    """
    if cfg.slope_angle_range is not None:
        angle_lo, angle_hi = cfg.slope_angle_range
        slope_angle_deg = float(angle_lo) + float(difficulty) * (float(angle_hi) - float(angle_lo))
    else:
        slope_angle_deg = float(cfg.slope_angle)
    height_diff_m = float(cfg.terrain_width) * math.tan(math.radians(slope_angle_deg))

    ny = int(cfg.size[1] / cfg.horizontal_scale)
    nx = int(cfg.size[0] / cfg.horizontal_scale)
    hs = cfg.horizontal_scale
    vs = cfg.vertical_scale

    height_diff_i = int(round(height_diff_m / vs))
    width_px = int(round(cfg.terrain_width / hs))
    length_px = int(round(cfg.total_length / hs))
    seg_px = int(round(cfg.segment_length / hs))

    width_px = min(width_px, ny)
    length_px = min(length_px, nx)
    seg_px = max(1, seg_px)
    assert cfg.num_segments * seg_px <= length_px, "segments exceed total_length"

    y_start = (ny - width_px) // 2
    y_end = y_start + width_px
    x_start = (nx - length_px) // 2
    x_end = x_start + length_px

    hf_raw = np.zeros((ny, nx), dtype=np.float64)

    y_norm = (np.arange(width_px, dtype=np.float64) + 0.5) / width_px
    rise_profile = y_norm * height_diff_i
    fall_profile = (1.0 - y_norm) * height_diff_i

    for i in range(cfg.num_segments):
        x0 = x_start + i * seg_px
        x1 = min(x_start + (i + 1) * seg_px, x_end)
        if x1 <= x0:
            continue
        profile = rise_profile if i % 2 == 0 else fall_profile
        hf_raw[y_start:y_end, x0:x1] = profile[:, np.newaxis]

    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(difficulty, perlin_cfg)
        hf_raw += perlin_noise

    return np.rint(hf_raw).astype(np.int16)


@generate_wall
@height_field_to_mesh
def perlin_symmetrical_slope_terrain(difficulty: float, cfg: hf_terrains_cfg.PerlinSymmetricalSlopeTerrainCfg) -> np.ndarray:
    """Generate a V-groove symmetrical slope terrain.

    Cross-section along y is a V-groove: flat bottom with symmetric ramps
    on both sides. Constant cross-section along travel direction (x).
    """
    if cfg.slope_angle_range is not None:
        lo, hi = cfg.slope_angle_range
        slope_angle_deg = float(lo) + float(difficulty) * (float(hi) - float(lo))
    else:
        slope_angle_deg = float(cfg.slope_angle)

    ny = int(cfg.size[1] / cfg.horizontal_scale)
    nx = int(cfg.size[0] / cfg.horizontal_scale)
    hs = cfg.horizontal_scale
    vs = cfg.vertical_scale

    width_px = int(round(cfg.terrain_width / hs))   # along y
    length_px = int(round(cfg.total_length / hs))    # along x

    width_px = min(width_px, ny)
    length_px = min(length_px, nx)

    flat_px = max(1, int(round(cfg.flat_bottom / hs)))
    ramp_px = max(1, (width_px - flat_px) // 2)
    width_px = 2 * ramp_px + flat_px  # enforce symmetry

    slope_angle = np.deg2rad(slope_angle_deg)
    max_height_i = int(round(ramp_px * hs * np.tan(slope_angle) / vs))

    y_start = (ny - width_px) // 2
    y_end = y_start + width_px
    x_start = (nx - length_px) // 2
    x_end = x_start + length_px

    hf_raw = np.zeros((ny, nx), dtype=np.float64)

    # V-groove profile along y: left ramp -> flat bottom -> right ramp
    left = np.linspace(max_height_i, 0, ramp_px, dtype=np.float64)
    bottom = np.zeros(flat_px, dtype=np.float64)
    right = np.linspace(0, max_height_i, ramp_px, dtype=np.float64)
    profile = np.concatenate([left, bottom, right])  # shape: (width_px,)

    hf_raw[y_start:y_end, x_start:x_end] = profile[:, np.newaxis]

    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(difficulty, perlin_cfg)
        hf_raw += perlin_noise

    return np.rint(hf_raw).astype(np.int16)


def curved_bridge_curriculum_t(
    difficulty: float,
    cfg: hf_terrains_cfg.PerlinCurvedBridgeTerrainCfg,
) -> float:
    """Map terrain difficulty to [0, 1] for S-curve radius/width (after the straight tier)."""
    d = float(difficulty)
    if curved_bridge_use_straight_centerline(d, cfg):
        return 0.0
    max_straight = float(getattr(cfg, "straight_max_difficulty", 0.1))
    d = min(max(d, max_straight), 1.0)
    span = max(1.0 - max_straight, 1.0e-6)
    return (d - max_straight) / span


def curved_bridge_resolve_bridge_width_m(
    difficulty: float,
    cfg: hf_terrains_cfg.PerlinCurvedBridgeTerrainCfg,
) -> float:
    """Walkable deck width (m) for the current difficulty."""
    if cfg.bridge_width_range is not None:
        lo, hi = cfg.bridge_width_range
        if curved_bridge_use_straight_centerline(difficulty, cfg):
            return float(lo)
        t = curved_bridge_curriculum_t(difficulty, cfg)
        return float(lo + t * (hi - lo))
    return float(cfg.bridge_width)


def curved_bridge_resolve_radius(
    difficulty: float,
    cfg: hf_terrains_cfg.PerlinCurvedBridgeTerrainCfg,
    *,
    tile_size: tuple[float, float] | None = None,
) -> float:
    """Resolve semicircle radius R (meters) for the current difficulty."""
    t = curved_bridge_curriculum_t(difficulty, cfg)
    if cfg.curve_radius is not None:
        radius = float(cfg.curve_radius)
    elif cfg.curve_radius_range is not None:
        lo, hi = cfg.curve_radius_range
        radius = float(lo + t * (hi - lo))
    else:
        radius = float(cfg.bridge_length / (2.0 * np.pi))
    tile_x = float(tile_size[0] if tile_size is not None else cfg.size[0])
    max_radius = (tile_x - 1.0) / 4.0
    return min(radius, max_radius)


def curved_bridge_use_straight_centerline(
    difficulty: float,
    cfg: hf_terrains_cfg.PerlinCurvedBridgeTerrainCfg,
) -> bool:
    """True when this difficulty should use a straight +x bridge (no S-bends)."""
    if not bool(getattr(cfg, "straight_at_zero_difficulty", False)):
        return False
    max_d = float(getattr(cfg, "straight_max_difficulty", 0.1))
    return float(difficulty) < max_d


def curved_bridge_straight_centerline_xy(
    cfg: hf_terrains_cfg.PerlinCurvedBridgeTerrainCfg,
    *,
    tile_size: tuple[float, float] | None = None,
    num_samples: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Straight bridge along +x; same entrance anchor as the S-curve (x=1, y=2 corner frame)."""
    tile_x = float(tile_size[0] if tile_size is not None else cfg.size[0])
    x_offset = 1.0
    y_center = 2.0
    max_len = max(0.5, tile_x - x_offset - 0.5)
    length_m = min(float(cfg.bridge_length), max_len)
    n = max(2, num_samples)
    cx = np.linspace(x_offset, x_offset + length_m, n, dtype=np.float64)
    cy = np.full(n, y_center, dtype=np.float64)

    lead_in = float(getattr(cfg, "straight_lead_in_m", 0.0))
    lead_out = float(getattr(cfg, "straight_lead_out_m", 0.0))
    if lead_in > 0.0 or lead_out > 0.0:
        ds = length_m / float(max(n - 1, 1))
        if lead_in > 0.0:
            n_in = max(1, int(round(lead_in / ds)))
            lead_x = np.linspace(x_offset - lead_in, x_offset, n_in, dtype=np.float64)
            lead_y = np.full(n_in, y_center, dtype=np.float64)
            cx = np.concatenate([lead_x, cx])
            cy = np.concatenate([lead_y, cy])
        if lead_out > 0.0:
            n_out = max(1, int(round(lead_out / ds)))
            x_end = float(cx[-1])
            trail_x = np.linspace(x_end, x_end + lead_out, n_out, dtype=np.float64)
            trail_y = np.full(n_out, y_center, dtype=np.float64)
            cx = np.concatenate([cx, trail_x])
            cy = np.concatenate([cy, trail_y])
    return cx, cy


def curved_bridge_build_centerline_xy(
    difficulty: float,
    cfg: hf_terrains_cfg.PerlinCurvedBridgeTerrainCfg,
    *,
    tile_size: tuple[float, float] | None = None,
    num_samples: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """S-curve centerline, or a straight +x segment in the easiest curriculum band."""
    if curved_bridge_use_straight_centerline(difficulty, cfg):
        return curved_bridge_straight_centerline_xy(cfg, tile_size=tile_size, num_samples=num_samples)
    radius = curved_bridge_resolve_radius(difficulty, cfg, tile_size=tile_size)
    tile_x = float(tile_size[0] if tile_size is not None else cfg.size[0])
    tile_y = float(tile_size[1] if tile_size is not None else cfg.size[1])
    t = curved_bridge_curriculum_t(difficulty, cfg)
    x_span_m = min(float(cfg.bridge_length), max(0.5, tile_x - 1.5))
    lateral_amplitude_m = min(0.22 * tile_y, 0.8 + 0.9 * t)
    return curved_bridge_centerline_xy(
        radius,
        y_offset=0.5 * tile_y,
        x_span_m=x_span_m,
        lateral_amplitude_m=lateral_amplitude_m,
        straight_lead_in_m=0.0,
        straight_lead_out_m=0.0,
        num_samples=num_samples,
    )


def _centerline_index_at_arc_offset(
    cx: np.ndarray,
    cy: np.ndarray,
    arc_offset_m: float,
    *,
    from_start: bool,
) -> int:
    """Index on the polyline at ``arc_offset_m`` from start or end."""
    if arc_offset_m <= 0.0:
        return 0 if from_start else len(cx) - 1
    seg_len = np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    if from_start:
        target = float(arc_offset_m)
        index = int(np.searchsorted(cum, target, side="left"))
        return int(np.clip(index, 0, len(cx) - 1))
    target = float(cum[-1] - arc_offset_m)
    index = int(np.searchsorted(cum, target, side="left"))
    return int(np.clip(index, 0, len(cx) - 1))


def curved_bridge_centerline_xy(
    radius: float,
    *,
    x_offset: float = 1.0,
    y_offset: float | None = None,
    x_span_m: float | None = None,
    lateral_amplitude_m: float | None = None,
    num_samples: int = 500,
    straight_lead_in_m: float = 0.0,
    straight_lead_out_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return S-bridge centerline (x, y) in corner-frame meters (progress along +x).

    The path keeps a forward tangent at the entrance/exit so body-frame +X
    commands remain meaningful while curvature increases with difficulty.
    """
    radius = float(radius)
    if y_offset is None:
        y_offset = 2.0
    n = max(4, int(num_samples))
    s = np.linspace(0.0, 1.0, n, dtype=np.float64)
    max_heading = float(np.clip(1.2 / max(radius, 1.0e-6), 0.25, 1.05))
    x_span = max(0.5, 4.0 * radius if x_span_m is None else float(x_span_m))
    amplitude = (
        x_span * np.tan(max_heading) / (2.0 * np.pi)
        if lateral_amplitude_m is None
        else float(lateral_amplitude_m)
    )
    phase = s * s * (3.0 - 2.0 * s)
    cx = x_span * s
    cy = amplitude * np.sin(2.0 * np.pi * phase)
    ds = x_span / float(max(n - 1, 1))
    cx += float(x_offset)
    cy += float(y_offset)

    lead_in = float(straight_lead_in_m)
    if lead_in > 0.0:
        n = max(1, int(round(lead_in / ds)))
        entrance_y = float(cy[0])
        lead_x = np.linspace(float(cx[0]) - lead_in, float(cx[0]), n)
        lead_y = np.full(n, entrance_y)
        cx = np.concatenate([lead_x, cx])
        cy = np.concatenate([lead_y, cy])

    lead_out = float(straight_lead_out_m)
    if lead_out > 0.0:
        n = max(1, int(round(lead_out / ds)))
        exit_y = float(cy[-1])
        trail_x = np.linspace(float(cx[-1]), float(cx[-1]) + lead_out, n)
        trail_y = np.full(n, exit_y)
        cx = np.concatenate([cx, trail_x])
        cy = np.concatenate([cy, trail_y])

    return cx, cy


def curved_bridge_centerline_yaw(cx: np.ndarray, cy: np.ndarray, index: int) -> float:
    """Yaw (rad) from centerline tangent; positive about +z (MuJoCo world)."""
    index = int(np.clip(index, 0, len(cx) - 1))
    if index < len(cx) - 1:
        dx = float(cx[index + 1] - cx[index])
        dy = float(cy[index + 1] - cy[index])
    else:
        dx = float(cx[index] - cx[index - 1])
        dy = float(cy[index] - cy[index - 1])
    return float(np.arctan2(dy, dx))


def _height_at_corner_xy(
    heights_i16: np.ndarray,
    *,
    horizontal_scale: float,
    vertical_scale: float,
    elevation_min: int,
    x_m: float,
    y_m: float,
) -> float:
    """Bilinear sample of walkable height at corner-frame (x, y) on the full-tile hfield."""
    hs = float(horizontal_scale)
    vs = float(vertical_scale)
    z_offset = float(elevation_min) * vs
    col_f = float(x_m) / hs
    row_f = float(y_m) / hs
    col0 = int(np.floor(col_f))
    row0 = int(np.floor(row_f))
    col1 = min(col0 + 1, heights_i16.shape[1] - 1)
    row1 = min(row0 + 1, heights_i16.shape[0] - 1)
    col0 = int(np.clip(col0, 0, heights_i16.shape[1] - 1))
    row0 = int(np.clip(row0, 0, heights_i16.shape[0] - 1))
    tx = col_f - col0
    ty = row_f - row0

    def _z_at(row: int, col: int) -> float:
        return (float(heights_i16[row, col]) - float(elevation_min)) * vs + z_offset

    z00 = _z_at(row0, col0)
    z01 = _z_at(row0, col1)
    z10 = _z_at(row1, col0)
    z11 = _z_at(row1, col1)
    return float((1.0 - ty) * ((1.0 - tx) * z00 + tx * z01) + ty * (z10 * (1.0 - tx) + tx * z11))


def _is_bridge_height_at(
    heights_i16: np.ndarray,
    *,
    horizontal_scale: float,
    vertical_scale: float,
    bridge_height_m: float,
    x_m: float,
    y_m: float,
) -> bool:
    hs = float(horizontal_scale)
    vs = float(vertical_scale)
    col = int(np.clip(np.round(x_m / hs), 0, heights_i16.shape[1] - 1))
    row = int(np.clip(np.round(y_m / hs), 0, heights_i16.shape[0] - 1))
    return float(heights_i16[row, col]) * vs >= 0.5 * float(bridge_height_m)


def _curved_bridge_ground_approach_xy(
    deck_x: float,
    deck_y: float,
    entrance_yaw: float,
    heights_i16: np.ndarray,
    *,
    horizontal_scale: float,
    vertical_scale: float,
    bridge_height_m: float,
    ground_approach_m: float,
    ground_offset_m: tuple[float, float],
) -> tuple[float, float]:
    """Spawn on ground behind the entrance, offset opposite to the centerline tangent."""
    yaw = float(entrance_yaw)
    tangent_x = float(np.cos(yaw))
    tangent_y = float(np.sin(yaw))
    extra_dx, extra_dy = float(ground_offset_m[0]), float(ground_offset_m[1])
    for extra_back in (0.0, 0.12, 0.24, 0.36, 0.48, 0.6):
        back = float(ground_approach_m) + extra_back
        x_m = deck_x - back * tangent_x + extra_dx
        y_m = deck_y - back * tangent_y + extra_dy
        if not _is_bridge_height_at(
            heights_i16,
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            bridge_height_m=bridge_height_m,
            x_m=x_m,
            y_m=y_m,
        ):
            return x_m, y_m
    back = float(ground_approach_m) + 0.6
    return deck_x - back * tangent_x + extra_dx, deck_y - back * tangent_y + extra_dy


def _curved_bridge_along_tangent_xy(
    x_m: float,
    y_m: float,
    yaw: float,
    offset_m: float,
    ground_offset_m: tuple[float, float],
) -> tuple[float, float]:
    """Translate (x, y) by ``offset_m`` along the centerline tangent (+forward at exit)."""
    yaw = float(yaw)
    return (
        x_m + float(offset_m) * float(np.cos(yaw)) + float(ground_offset_m[0]),
        y_m + float(offset_m) * float(np.sin(yaw)) + float(ground_offset_m[1]),
    )


def _curved_bridge_ground_exit_xy(
    deck_x: float,
    deck_y: float,
    exit_yaw: float,
    heights_i16: np.ndarray,
    *,
    horizontal_scale: float,
    vertical_scale: float,
    bridge_height_m: float,
    ground_approach_m: float,
    ground_offset_m: tuple[float, float],
) -> tuple[float, float]:
    """From exit centerline point: step along +tangent; continue forward until off the bridge."""
    yaw = float(exit_yaw)
    tangent_x = float(np.cos(yaw))
    tangent_y = float(np.sin(yaw))
    extra_dx, extra_dy = float(ground_offset_m[0]), float(ground_offset_m[1])
    for extra_forward in (0.0, 0.12, 0.24, 0.36, 0.48, 0.6):
        forward = float(ground_approach_m) + extra_forward
        x_m = deck_x + forward * tangent_x + extra_dx
        y_m = deck_y + forward * tangent_y + extra_dy
        if not _is_bridge_height_at(
            heights_i16,
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            bridge_height_m=bridge_height_m,
            x_m=x_m,
            y_m=y_m,
        ):
            return x_m, y_m
    forward = float(ground_approach_m) + 0.6
    return deck_x + forward * tangent_x + extra_dx, deck_y + forward * tangent_y + extra_dy


def curved_bridge_centerline_patch(
    difficulty: float,
    cfg: hf_terrains_cfg.PerlinCurvedBridgeTerrainCfg,
    *,
    heights_i16: np.ndarray,
    heights_phys: np.ndarray,
    z_offset: float,
    endpoint: str,
    terrain_size: tuple[float, float] | None = None,
    border_xy: tuple[float, float] = (0.0, 0.0),
    arc_offset_m: float = 0.0,
    spawn_on_ground: bool = False,
    ground_approach_m: float = 0.75,
    ground_offset_m: tuple[float, float] = (0.0, 0.0),
    tangent_offset_m: float | None = None,
) -> np.ndarray:
    """Flat patch row(s) as (N, 4): x, y, z, yaw in full-tile corner-frame meters."""
    del heights_phys, z_offset
    sub_size = terrain_size if terrain_size is not None else (float(cfg.size[0]), float(cfg.size[1]))
    cx, cy = curved_bridge_build_centerline_xy(difficulty, cfg, tile_size=sub_size)
    from_start = endpoint == "start"
    deck_index = _centerline_index_at_arc_offset(cx, cy, float(arc_offset_m), from_start=from_start)
    bx, by = float(border_xy[0]), float(border_xy[1])
    deck_x = float(cx[deck_index]) + bx
    deck_y = float(cy[deck_index]) + by

    yaw = curved_bridge_centerline_yaw(cx, cy, deck_index)
    if spawn_on_ground and endpoint == "start":
        x_m, y_m = _curved_bridge_ground_approach_xy(
            deck_x,
            deck_y,
            yaw,
            heights_i16,
            horizontal_scale=float(cfg.horizontal_scale),
            vertical_scale=float(cfg.vertical_scale),
            bridge_height_m=float(cfg.bridge_height),
            ground_approach_m=float(ground_approach_m),
            ground_offset_m=ground_offset_m,
        )
    elif endpoint == "end":
        forward_m = (
            float(ground_approach_m)
            if tangent_offset_m is None
            else float(tangent_offset_m)
        )
        if spawn_on_ground:
            x_m, y_m = _curved_bridge_ground_exit_xy(
                deck_x,
                deck_y,
                yaw,
                heights_i16,
                horizontal_scale=float(cfg.horizontal_scale),
                vertical_scale=float(cfg.vertical_scale),
                bridge_height_m=float(cfg.bridge_height),
                ground_approach_m=forward_m,
                ground_offset_m=ground_offset_m,
            )
        else:
            x_m, y_m = _curved_bridge_along_tangent_xy(
                deck_x, deck_y, yaw, forward_m, ground_offset_m
            )
    else:
        x_m, y_m = deck_x, deck_y

    elevation_min = int(np.min(heights_i16))
    z_m = _height_at_corner_xy(
        heights_i16,
        horizontal_scale=float(cfg.horizontal_scale),
        vertical_scale=float(cfg.vertical_scale),
        elevation_min=elevation_min,
        x_m=x_m,
        y_m=y_m,
    )
    return np.array([[x_m, y_m, z_m, yaw]], dtype=np.float64)


@generate_wall
@height_field_to_mesh
def perlin_curved_bridge_terrain(difficulty: float, cfg: hf_terrains_cfg.PerlinCurvedBridgeTerrainCfg) -> np.ndarray:
    """Generate an S-curved bridge terrain.

    Two opposing semicircles form an S-shape. Entrance at the west end; path
    progresses along +x with lateral S deflection in y. Bridge is raised above flat ground.
    """
    ny = int(cfg.size[1] / cfg.horizontal_scale)
    nx = int(cfg.size[0] / cfg.horizontal_scale)
    hs = cfg.horizontal_scale
    vs = cfg.vertical_scale

    bridge_width_m = curved_bridge_resolve_bridge_width_m(difficulty, cfg)

    bridge_height_i = int(round(cfg.bridge_height / vs))
    half_width_px = max(1, int(round(bridge_width_m * 0.5 / hs)))

    cx, cy = curved_bridge_build_centerline_xy(difficulty, cfg, tile_size=(float(cfg.size[0]), float(cfg.size[1])))

    hf_raw = np.zeros((ny, nx), dtype=np.float64)
    half_width_m = half_width_px * hs

    # Row-by-row distance computation (memory efficient)
    for r in range(ny):
        row_y = r * hs
        dx = cx[:, np.newaxis] - (np.arange(nx) * hs)[np.newaxis, :]
        dy = cy[:, np.newaxis] - row_y
        dist2 = dx * dx + dy * dy
        min_dist = np.sqrt(np.min(dist2, axis=0))
        mask = min_dist <= half_width_m
        hf_raw[r, mask] = bridge_height_i

    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(difficulty, perlin_cfg)
        hf_raw += perlin_noise

    return np.rint(hf_raw).astype(np.int16)


def _narrow_l_corner_discrete_stair_height_i(
    arc_pos: np.ndarray,
    *,
    deck_start_m: float,
    deck_end_m: float,
    entry_run_m: float,
    exit_run_m: float,
    num_entry_steps: int,
    num_exit_steps: int,
    step_h_m: float,
    step_w_m: float,
    deck_height_m: float,
    deck_i: int,
    vs: float,
) -> np.ndarray:
    """Discrete entry/exit tread heights along centreline arc length (quantized)."""
    arc = np.asarray(arc_pos, dtype=np.float64)
    heights = np.zeros(arc.shape, dtype=np.float64)
    step_w = max(float(step_w_m), 1.0e-6)
    entry_run = max(float(entry_run_m), 1.0e-6)
    exit_run = max(float(exit_run_m), 1.0e-6)

    on_entry = arc < float(deck_start_m) - 1.0e-9
    if np.any(on_entry):
        rel = arc[on_entry]
        step_idx = np.floor(rel / step_w).astype(np.int64)
        np.clip(step_idx, 0, max(num_entry_steps, 1) - 1, out=step_idx)
        rise_m = (step_idx + 1).astype(np.float64) * float(step_h_m)
        heights[on_entry] = np.minimum(np.rint(rise_m / vs), float(deck_i))

    on_deck = (arc >= float(deck_start_m) - 1.0e-9) & (arc <= float(deck_end_m) + 1.0e-9)
    heights[on_deck] = float(deck_i)

    on_exit = arc > float(deck_end_m) + 1.0e-9
    if np.any(on_exit):
        rel = arc[on_exit] - float(deck_end_m)
        step_idx = np.floor(rel / step_w).astype(np.int64)
        np.clip(step_idx, 0, max(num_exit_steps, 1) - 1, out=step_idx)
        rise_m = (max(num_exit_steps, 1) - step_idx - 1).astype(np.float64) * float(step_h_m)
        rise_m = np.minimum(rise_m, float(deck_height_m))
        heights[on_exit] = np.maximum(1.0, np.rint(rise_m / vs))

    return heights


def narrow_l_corner_turn_sign(cfg: hf_terrains_cfg.PerlinNarrowLCornerTerrainCfg) -> float:
    """+1 for left (+y leg2), -1 for right (-y leg2)."""
    if cfg.turn == "left":
        return 1.0
    if cfg.turn == "right":
        return -1.0
    raise ValueError(f"Unknown narrow L-corner turn {cfg.turn!r}; expected 'left' or 'right'.")


def narrow_l_corner_resolve_railing_side(
    cfg: hf_terrains_cfg.PerlinNarrowLCornerTerrainCfg,
    rng: np.random.Generator,
) -> Literal["inner", "outer", "left", "right"]:
    """Pick guard-rail side for this sub-terrain tile."""
    side = str(getattr(cfg, "railing_side", "random"))
    if side == "random":
        return "left" if float(rng.random()) < 0.5 else "right"
    if side in ("inner", "outer", "left", "right"):
        return cast(Literal["inner", "outer", "left", "right"], side)
    raise ValueError(
        "railing_side must be 'inner', 'outer', 'left', 'right', or 'random'. "
        f"Got: {side!r}"
    )


def narrow_l_corner_resolve_turn_angle_deg(
    difficulty: float,
    cfg: hf_terrains_cfg.PerlinNarrowLCornerTerrainCfg,
) -> float:
    """Signed turn angle (deg): 0 = straight +x, +90 = left L, -90 = right L."""
    sign = narrow_l_corner_turn_sign(cfg)
    if cfg.turn_angle_range is not None:
        ta_lo, ta_hi = cfg.turn_angle_range
        magnitude = float(ta_lo + difficulty * (ta_hi - ta_lo))
    else:
        magnitude = 90.0
    return sign * magnitude


def narrow_l_corner_compute_geometry(
    difficulty: float,
    cfg: hf_terrains_cfg.PerlinNarrowLCornerTerrainCfg,
    *,
    tile_size: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Layout in sub-terrain corner frame (matches height-field centreline, no border)."""
    tile_x_m = float(tile_size[0] if tile_size is not None else cfg.size[0])
    tile_y_m = float(tile_size[1] if tile_size is not None else cfg.size[1])
    if cfg.channel_width_range is not None:
        lo, hi = cfg.channel_width_range
        width_m = lo + difficulty * (hi - lo)
    else:
        width_m = float(cfg.channel_width)
    if cfg.deck_height_range is not None:
        lo_h, hi_h = cfg.deck_height_range
        deck_height_m = lo_h + difficulty * (hi_h - lo_h)
    else:
        deck_height_m = float(cfg.deck_height)
    turn_angle_deg = narrow_l_corner_resolve_turn_angle_deg(difficulty, cfg)
    step_h_m = float(cfg.step_height)
    step_w_m = float(cfg.step_width)
    num_entry_steps = max(1, int(round(deck_height_m / step_h_m)))
    entry_run_m = num_entry_steps * step_w_m
    exit_run_m = num_entry_steps * step_w_m
    rail_clearance_m = max(float(getattr(cfg, "railing_bar_radius", 0.0)), 0.0) + 0.05
    edge_clearance_m = 0.5 * width_m + rail_clearance_m
    desired_leg_m = float(cfg.leg_length)
    if narrow_l_corner_turn_sign(cfg) >= 0.0:
        y_center_m = tile_y_m - edge_clearance_m - exit_run_m - desired_leg_m
    else:
        y_center_m = edge_clearance_m + exit_run_m + desired_leg_m
    y_center_m = float(np.clip(y_center_m, edge_clearance_m, tile_y_m - edge_clearance_m))
    turn_rad = np.deg2rad(turn_angle_deg)
    cos_a, sin_a = np.cos(turn_rad), np.sin(turn_rad)
    max_x_span = max(0.1, tile_x_m - entry_run_m)
    if sin_a >= 0.0:
        y_span = max(0.1, tile_y_m - edge_clearance_m - y_center_m - exit_run_m)
    else:
        y_span = max(0.1, y_center_m - edge_clearance_m - exit_run_m)
    leg_from_x = max(0.5, max_x_span / (1.0 + abs(cos_a)))
    leg_from_y = max(0.5, y_span / (abs(sin_a) + 1e-6)) if abs(sin_a) > 1e-6 else 1e9
    leg_m = min(float(cfg.leg_length), leg_from_x, leg_from_y)
    return {
        "y_center_m": y_center_m,
        "entry_run_m": entry_run_m,
        "exit_run_m": exit_run_m,
        "leg_m": leg_m,
        "width_m": width_m,
        "deck_height_m": deck_height_m,
        "turn_angle_deg": turn_angle_deg,
        "turn_sign": narrow_l_corner_turn_sign(cfg),
        "deck_start_m": entry_run_m,
        "deck_end_m": entry_run_m + 2.0 * leg_m,
    }


def narrow_l_corner_resolve_layout(
    difficulty: float,
    cfg: hf_terrains_cfg.PerlinNarrowLCornerTerrainCfg,
    *,
    tile_size: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Sub-terrain layout (corner frame, before border) matching centreline-based generation."""
    return narrow_l_corner_compute_geometry(difficulty, cfg, tile_size=tile_size)


def narrow_l_corner_build_centerline_polyline(
    geom: dict[str, float],
    *,
    horizontal_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centreline polyline and cumulative arc length (same sampling as the height field)."""
    entry_run_m = float(geom["entry_run_m"])
    exit_run_m = float(geom["exit_run_m"])
    leg_m = float(geom["leg_m"])
    y_center_m = float(geom["y_center_m"])
    turn_angle_deg = float(geom["turn_angle_deg"])
    hs = float(horizontal_scale)
    turn_rad = np.deg2rad(turn_angle_deg)
    cos_a, sin_a = np.cos(turn_rad), np.sin(turn_rad)

    cl_x: list[float] = [0.0]
    cl_y: list[float] = [y_center_m]
    n_pts = max(2, int(round(entry_run_m / hs)))
    for i in range(1, n_pts + 1):
        cl_x.append((i / n_pts) * entry_run_m)
        cl_y.append(y_center_m)
    n_pts = max(2, int(round(leg_m / hs)))
    for i in range(1, n_pts + 1):
        cl_x.append(entry_run_m + (i / n_pts) * leg_m)
        cl_y.append(y_center_m)
    x_corner = entry_run_m + leg_m
    y_corner = y_center_m
    for i in range(1, n_pts + 1):
        frac = i / n_pts
        cl_x.append(x_corner + frac * leg_m * cos_a)
        cl_y.append(y_corner + frac * leg_m * sin_a)
    x_leg2_end = x_corner + leg_m * cos_a
    y_leg2_end = y_corner + leg_m * sin_a
    n_pts = max(2, int(round(exit_run_m / hs)))
    for i in range(1, n_pts + 1):
        frac = i / n_pts
        cl_x.append(x_leg2_end + frac * exit_run_m * cos_a)
        cl_y.append(y_leg2_end + frac * exit_run_m * sin_a)

    cx = np.array(cl_x, dtype=np.float64)
    cy = np.array(cl_y, dtype=np.float64)
    seg_len = np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2)
    arc_cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    return cx, cy, arc_cum


def _narrow_l_corner_channel_edge_offset(
    cfg: hf_terrains_cfg.PerlinNarrowLCornerTerrainCfg,
    geom: dict[str, float],
    arc_m: float,
    *,
    half_w: float,
    rail_side: Literal["inner", "outer", "left", "right"],
    tangent: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Offset to one channel edge.

    ``left``/``right`` are robot-body sides along the current centerline tangent.
    ``inner``/``outer`` keep the old geometric concave/convex side semantics.
    """
    if rail_side in ("left", "right"):
        if tangent is None:
            raise ValueError("Robot-side railings require a centerline tangent.")
        tx, ty = float(tangent[0]), float(tangent[1])
        norm = float(np.hypot(tx, ty))
        if norm < 1.0e-9:
            return 0.0, 0.0
        tx /= norm
        ty /= norm
        if rail_side == "left":
            return -ty * half_w, tx * half_w
        return ty * half_w, -tx * half_w

    inner = rail_side == "inner"
    entry = float(geom["entry_run_m"])
    leg = float(geom["leg_m"])
    turn_rad = np.deg2rad(float(geom["turn_angle_deg"]))
    cos_a, sin_a = float(np.cos(turn_rad)), float(np.sin(turn_rad))
    n_inner = (-sin_a * half_w, cos_a * half_w)
    n_outer = (sin_a * half_w, -cos_a * half_w)

    if abs(sin_a) < 1.0e-3:
        if cfg.turn == "left":
            y_sign = -1.0 if inner else 1.0
        else:
            y_sign = 1.0 if inner else -1.0
        return 0.0, y_sign * half_w

    if float(arc_m) <= entry + leg + 1.0e-6:
        if cfg.turn == "left":
            y_sign = -1.0 if inner else 1.0
        else:
            y_sign = 1.0 if inner else -1.0
        return 0.0, y_sign * half_w

    return n_inner if inner else n_outer


def _narrow_l_corner_xy_at_arc(
    cx: np.ndarray,
    cy: np.ndarray,
    arc_cum: np.ndarray,
    arc_m: float,
) -> tuple[float, float, tuple[float, float]]:
    """Interpolate (x, y, tangent) on the centreline at arc length ``arc_m``."""
    arc_m = float(np.clip(arc_m, 0.0, float(arc_cum[-1])))
    seg_i = int(np.searchsorted(arc_cum, arc_m, side="right") - 1)
    seg_i = int(np.clip(seg_i, 0, len(cx) - 2))
    seg_len = float(arc_cum[seg_i + 1] - arc_cum[seg_i])
    if seg_len < 1.0e-9:
        t = 0.0
    else:
        t = (arc_m - float(arc_cum[seg_i])) / seg_len
    x = float(cx[seg_i] + t * (cx[seg_i + 1] - cx[seg_i]))
    y = float(cy[seg_i] + t * (cy[seg_i + 1] - cy[seg_i]))
    tx = float(cx[seg_i + 1] - cx[seg_i])
    ty = float(cy[seg_i + 1] - cy[seg_i])
    return x, y, (tx, ty)


def narrow_l_corner_railing_xy_positions(
    difficulty: float,
    cfg: hf_terrains_cfg.PerlinNarrowLCornerTerrainCfg,
    rail_side: Literal["inner", "outer", "left", "right"],
    *,
    tile_size: tuple[float, float] | None = None,
    border_xy: tuple[float, float] = (0.0, 0.0),
) -> list[tuple[float, float]]:
    """Guard-bar (x, y) along one continuous channel side."""
    geom = narrow_l_corner_compute_geometry(difficulty, cfg, tile_size=tile_size)
    cx, cy, arc_cum = narrow_l_corner_build_centerline_polyline(
        geom, horizontal_scale=float(cfg.horizontal_scale)
    )
    half_w = (
        0.5 * float(geom["width_m"])
        + max(float(getattr(cfg, "railing_bar_radius", 0.0)), 0.0)
        + 0.001
    )
    spacing = float(cfg.railing_spacing)
    path_len = float(arc_cum[-1])
    bx, by = float(border_xy[0]), float(border_xy[1])

    positions: list[tuple[float, float]] = []
    arc = 0.5 * spacing
    while arc <= path_len - 0.5 * spacing + 1.0e-6:
        x, y, tangent = _narrow_l_corner_xy_at_arc(cx, cy, arc_cum, arc)
        off_x, off_y = _narrow_l_corner_channel_edge_offset(
            cfg, geom, arc, half_w=half_w, rail_side=rail_side, tangent=tangent
        )
        rx, ry = x + off_x, y + off_y

        # At the L-corner the perpendicular offset from one leg can place the
        # railing inside the deck of the other leg.  Iteratively push radially
        # outward from the nearest centreline point until the minimum distance
        # to *any* centreline point >= half_w.
        for _ in range(16):
            dx = cx - rx
            dy = cy - ry
            dist_sq = dx * dx + dy * dy
            min_idx = int(np.argmin(dist_sq))
            min_dist = float(np.sqrt(dist_sq[min_idx]))
            if min_dist >= half_w - 1.0e-9:
                break
            nx = float(cx[min_idx])
            ny = float(cy[min_idx])
            dir_x = rx - nx
            dir_y = ry - ny
            norm = float(np.hypot(dir_x, dir_y))
            if norm > 1.0e-12:
                rx = nx + dir_x / norm * half_w
                ry = ny + dir_y / norm * half_w
            else:
                break

        positions.append((rx + bx, ry + by))
        arc += spacing
    return positions


def narrow_l_corner_centerline_xy(
    layout: dict[str, float],
    *,
    num_samples_per_leg: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """L-channel centerline: entry->leg1 (+x)->leg2 (at turn_angle)->exit."""
    geom = dict(layout)
    if "y_center_m" not in geom:
        geom["y_center_m"] = 3.0
    hs = max(
        float(geom.get("horizontal_scale", 0.05)),
        float(geom["leg_m"]) / float(max(num_samples_per_leg, 2)),
    )
    cx, cy, _arc = narrow_l_corner_build_centerline_polyline(geom, horizontal_scale=hs)
    return cx, cy


def narrow_l_corner_centerline_patch(
    difficulty: float,
    cfg: hf_terrains_cfg.PerlinNarrowLCornerTerrainCfg,
    *,
    heights_i16: np.ndarray,
    heights_phys: np.ndarray,
    z_offset: float,
    endpoint: str,
    terrain_size: tuple[float, float] | None = None,
    border_xy: tuple[float, float] = (0.0, 0.0),
    arc_offset_m: float = 0.0,
    spawn_on_ground: bool = False,
    ground_approach_m: float = 0.75,
    ground_offset_m: tuple[float, float] = (0.0, 0.0),
    tangent_offset_m: float | None = None,
) -> np.ndarray:
    """Flat patch as (N, 4): x, y, z, yaw in full-tile corner-frame meters."""
    del heights_phys, z_offset, terrain_size
    layout = narrow_l_corner_resolve_layout(difficulty, cfg)
    cx, cy = narrow_l_corner_centerline_xy(layout)
    from_start = endpoint == "start"
    deck_index = _centerline_index_at_arc_offset(cx, cy, float(arc_offset_m), from_start=from_start)
    bx, by = float(border_xy[0]), float(border_xy[1])
    deck_x = float(cx[deck_index]) + bx
    deck_y = float(cy[deck_index]) + by
    yaw = curved_bridge_centerline_yaw(cx, cy, deck_index)
    deck_height_m = float(layout["deck_height_m"])
    if spawn_on_ground and endpoint == "start":
        x_m, y_m = _curved_bridge_ground_approach_xy(
            deck_x,
            deck_y,
            yaw,
            heights_i16,
            horizontal_scale=float(cfg.horizontal_scale),
            vertical_scale=float(cfg.vertical_scale),
            bridge_height_m=deck_height_m,
            ground_approach_m=float(ground_approach_m),
            ground_offset_m=ground_offset_m,
        )
    elif endpoint == "end":
        forward_m = (
            float(ground_approach_m)
            if tangent_offset_m is None
            else float(tangent_offset_m)
        )
        if spawn_on_ground:
            x_m, y_m = _curved_bridge_ground_exit_xy(
                deck_x,
                deck_y,
                yaw,
                heights_i16,
                horizontal_scale=float(cfg.horizontal_scale),
                vertical_scale=float(cfg.vertical_scale),
                bridge_height_m=deck_height_m,
                ground_approach_m=forward_m,
                ground_offset_m=ground_offset_m,
            )
        else:
            x_m, y_m = _curved_bridge_along_tangent_xy(
                deck_x, deck_y, yaw, forward_m, ground_offset_m
            )
    else:
        x_m, y_m = deck_x, deck_y
    elevation_min = int(np.min(heights_i16))
    z_m = _height_at_corner_xy(
        heights_i16,
        horizontal_scale=float(cfg.horizontal_scale),
        vertical_scale=float(cfg.vertical_scale),
        elevation_min=elevation_min,
        x_m=x_m,
        y_m=y_m,
    )
    return np.array([[x_m, y_m, z_m, yaw]], dtype=np.float64)


@generate_wall
@height_field_to_mesh
def perlin_continuous_hurdle_terrain(difficulty: float, cfg: hf_terrains_cfg.PerlinContinuousHurdleTerrainCfg) -> np.ndarray:
    """Generate flat ground for continuous hurdle terrain.

    Flat ground only; BOX geoms are added by the config's overridden
    function() method directly to the MuJoCo spec.
    """
    sub_width = int(cfg.size[1] / cfg.horizontal_scale)
    sub_length = int(cfg.size[0] / cfg.horizontal_scale)
    hf_raw = np.zeros((sub_width, sub_length), dtype=np.float64)

    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        hf_raw += generate_perlin_noise(difficulty, perlin_cfg)

    return np.rint(hf_raw).astype(np.int16)


@generate_wall
@height_field_to_mesh
def perlin_narrow_l_corner_terrain(
    difficulty: float, cfg: hf_terrains_cfg.PerlinNarrowLCornerTerrainCfg
) -> np.ndarray:
    """Generate an L-corner channel. Turn angle is curriculum-driven.

    At turn_angle=0 deg the bridge is straight (+x only). At +90 deg it is a left
    L (+x then +y); at -90 deg a right L (+x then -y). Set ``cfg.turn`` to
    ``'left'`` or ``'right'``.
    """
    ny = int(cfg.size[1] / cfg.horizontal_scale)
    nx = int(cfg.size[0] / cfg.horizontal_scale)
    hs = cfg.horizontal_scale
    vs = cfg.vertical_scale

    geom = narrow_l_corner_compute_geometry(
        difficulty,
        cfg,
        tile_size=(float(cfg.size[0]), float(cfg.size[1])),
    )
    width_m = float(geom["width_m"])
    deck_height_m = float(geom["deck_height_m"])
    entry_run_m = float(geom["entry_run_m"])
    exit_run_m = float(geom["exit_run_m"])
    leg_m = float(geom["leg_m"])
    step_h_m = float(cfg.step_height)
    step_w_m = float(cfg.step_width)
    num_entry_steps = max(1, int(round(deck_height_m / step_h_m)))
    num_exit_steps = num_entry_steps

    cx, cy, arc_cum = narrow_l_corner_build_centerline_polyline(geom, horizontal_scale=hs)
    deck_start = float(geom["deck_start_m"])
    deck_end = float(geom["deck_end_m"])

    deck_i = int(round(deck_height_m / vs))
    ground_i = int(round(cfg.ground_height / vs))
    half_width_px = max(1, int(round(width_m * 0.5 / hs)))

    hf_raw = np.full((ny, nx), ground_i, dtype=np.float64)
    half_width_m = half_width_px * hs

    for r in range(ny):
        row_y = r * hs
        col_xs = np.arange(nx, dtype=np.float64) * hs
        dx = cx[:, np.newaxis] - col_xs[np.newaxis, :]
        dy = cy[:, np.newaxis] - row_y
        dist2 = dx * dx + dy * dy
        min_dist = np.sqrt(np.min(dist2, axis=0))
        nearest = np.argmin(dist2, axis=0)
        on_deck = min_dist <= half_width_m
        if not np.any(on_deck):
            continue
        arc_pos = arc_cum[nearest[on_deck]]
        heights = _narrow_l_corner_discrete_stair_height_i(
            arc_pos,
            deck_start_m=deck_start,
            deck_end_m=deck_end,
            entry_run_m=entry_run_m,
            exit_run_m=exit_run_m,
            num_entry_steps=num_entry_steps,
            num_exit_steps=num_exit_steps,
            step_h_m=step_h_m,
            step_w_m=step_w_m,
            deck_height_m=deck_height_m,
            deck_i=deck_i,
            vs=vs,
        )
        hf_raw[r, on_deck] = heights

    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        hf_raw += generate_perlin_noise(difficulty, perlin_cfg)

    return np.rint(hf_raw).astype(np.int16)


def _slalom_pole_deck_heightfield(cfg: hf_terrains_cfg.PerlinSlalomPoleTerrainCfg) -> np.ndarray:
    """Ground at z=0 with a flat raised deck (default 10 cm); no entry stairs."""
    hs = cfg.horizontal_scale
    vs = cfg.vertical_scale
    length_pixels = int(cfg.size[1] / hs)
    width_pixels = int(cfg.size[0] / hs)

    ground_i = int(round(cfg.ground_height / vs))
    deck_i = int(round(cfg.deck_height / vs))
    hf_raw = np.full((length_pixels, width_pixels), float(ground_i), dtype=np.float64)

    total_span = (cfg.num_poles - 1) * cfg.pole_spacing
    x_start_pole = (cfg.size[0] - total_span) * 0.5
    x_end_pole = x_start_pole + total_span
    y_center = cfg.size[1] * 0.5

    deck_x0_m = max(0.0, x_start_pole - cfg.deck_margin_x)
    deck_x1_m = min(float(cfg.size[0]), x_end_pole + cfg.deck_margin_x)
    deck_y0_m = y_center - cfg.deck_width * 0.5
    deck_y1_m = y_center + cfg.deck_width * 0.5

    x0_px = max(0, int(round(deck_x0_m / hs)))
    x1_px = min(width_pixels, int(round(deck_x1_m / hs)))
    y0_px = max(0, int(round(deck_y0_m / hs)))
    y1_px = min(length_pixels, int(round(deck_y1_m / hs)))
    if x1_px <= x0_px or y1_px <= y0_px:
        return np.rint(hf_raw).astype(np.int16)

    hf_raw[y0_px:y1_px, x0_px:x1_px] = float(deck_i)
    return np.rint(hf_raw).astype(np.int16)


@generate_wall
@height_field_to_mesh
def perlin_slalom_pole_terrain(difficulty: float, cfg: hf_terrains_cfg.PerlinSlalomPoleTerrainCfg) -> np.ndarray:
    """Flat raised deck (default 10 cm); poles sit on the deck."""
    hf_raw = _slalom_pole_deck_heightfield(cfg).astype(np.float64)
    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        hf_raw += generate_perlin_noise(difficulty, perlin_cfg)
    return np.rint(hf_raw).astype(np.int16)
