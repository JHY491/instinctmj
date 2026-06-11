import copy
import uuid
from dataclasses import MISSING, dataclass, field
from typing import List, Literal, cast

import mujoco
import numpy as np
import trimesh
from mjlab.terrains.terrain_generator import FlatPatchSamplingCfg, SubTerrainCfg, TerrainGeometry, TerrainOutput
from scipy import ndimage

from .utils import convert_height_field_to_mesh


@dataclass(kw_only=True)
class FixedSequenceFlatPatchSamplingCfg(FlatPatchSamplingCfg):
    """Explicit (x, y) patch centers in the same convention as ``x_range`` / ``y_range``.

    Each component uses center-relative coordinates when it lies in ``[-size/2, size/2]`` on that
    axis (then ``+ size/2`` to corner frame), identical to ``_convert_range_to_grid`` in
    ``_find_flat_patches_from_heightfield_nearest``. Values outside that band are treated as
    already corner-frame meters on that axis.

    ``num_patches`` must equal ``len(fixed_xy_points)``. Z comes from the heightfield at the
    snapped grid cell; if that cell fails the flatness mask, the nearest flat cell is used.

    ``reset_command_patch_index``: flat-patch index used on episode reset when the command
    uses fixed-sequence advance. ``0`` is the first point; ``-1`` is the last valid patch
    (e.g. stair target on the far low ground across the run).
    """

    fixed_xy_points: tuple[tuple[float, float], ...] = ()
    reset_command_patch_index: int = 0
    centerline_endpoint: Literal["start", "end"] | None = None
    """When set on bridge terrain cfgs, patch (x, y, z, yaw) is taken from the path centerline
    tangent instead of ``fixed_xy_points``."""
    centerline_arc_offset_m: float = 0.0
    """Shift spawn along the centerline by this arc length (m) from start/end."""
    centerline_spawn_on_ground: bool = False
    """On bridge terrains: ``start`` -> ground before the entrance; ``end`` -> ground ahead of the
    exit (off the deck)."""
    centerline_ground_approach_m: float = 0.75
    """Meters along the centerline tangent from the deck point (backward at start, forward at end)."""
    centerline_tangent_offset_m: float | None = None
    """For curved-bridge ``end``: shift from the centerline exit (channel midline) along +tangent.
    Defaults to ``centerline_ground_approach_m`` when None."""
    centerline_ground_offset_m: tuple[float, float] = (0.0, 0.0)
    """Extra (dx, dy) added after the tangent-based ground approach offset."""


def terrain_column_target_uses_fixed_sequence_flat_patch(terrain_generator_cfg: object) -> list[bool]:
    """Return one flag per terrain column (``terrain_types`` index).

    ``True`` when that column's sub-terrain uses ``FixedSequenceFlatPatchSamplingCfg`` for the
    ``"target"`` flat-patch entry.
    """
    if terrain_generator_cfg is None:
        return []
    num_cols = int(getattr(terrain_generator_cfg, "num_cols", 0) or 0)
    sub_terrains = getattr(terrain_generator_cfg, "sub_terrains", None)
    if sub_terrains is None or num_cols <= 0:
        return []
    proportions = np.array([sub_cfg.proportion for sub_cfg in sub_terrains.values()], dtype=np.float64)
    denom = float(np.sum(proportions))
    if denom <= 0.0:
        return [False] * num_cols
    proportions = proportions / denom
    sub_terrains_names = list(sub_terrains.keys())
    sub_indices: list[int] = []
    for index in range(num_cols):
        sub_index = int(np.min(np.where(index / num_cols + 0.001 < np.cumsum(proportions))[0]))
        sub_indices.append(sub_index)
    out: list[bool] = []
    for col in range(num_cols):
        name = sub_terrains_names[sub_indices[col]]
        sub_cfg = sub_terrains[name]
        fps = getattr(sub_cfg, "flat_patch_sampling", None)
        if not isinstance(fps, dict):
            out.append(False)
            continue
        tgt = fps.get("target")
        out.append(isinstance(tgt, FixedSequenceFlatPatchSamplingCfg))
    return out


def terrain_column_fixed_sequence_reset_patch_index(terrain_generator_cfg: object) -> list[int]:
    """Per terrain column: flat-patch index for command target on episode reset."""
    if terrain_generator_cfg is None:
        return []
    num_cols = int(getattr(terrain_generator_cfg, "num_cols", 0) or 0)
    sub_terrains = getattr(terrain_generator_cfg, "sub_terrains", None)
    if sub_terrains is None or num_cols <= 0:
        return []
    proportions = np.array([sub_cfg.proportion for sub_cfg in sub_terrains.values()], dtype=np.float64)
    denom = float(np.sum(proportions))
    if denom <= 0.0:
        return [0] * num_cols
    proportions = proportions / denom
    sub_terrains_names = list(sub_terrains.keys())
    sub_indices: list[int] = []
    for index in range(num_cols):
        sub_index = int(np.min(np.where(index / num_cols + 0.001 < np.cumsum(proportions))[0]))
        sub_indices.append(sub_index)
    out: list[int] = []
    for col in range(num_cols):
        name = sub_terrains_names[sub_indices[col]]
        sub_cfg = sub_terrains[name]
        fps = getattr(sub_cfg, "flat_patch_sampling", None)
        if not isinstance(fps, dict):
            out.append(0)
            continue
        tgt = fps.get("target")
        if not isinstance(tgt, FixedSequenceFlatPatchSamplingCfg):
            out.append(0)
            continue
        idx = int(tgt.reset_command_patch_index)
        n = len(tgt.fixed_xy_points)
        if idx < 0:
            idx = max(0, n + idx)
        if n > 0:
            idx = min(idx, n - 1)
        out.append(idx)
    return out


_CENTER_SPAWN_SUBTERRAIN_NAMES = frozenset({"perlin_rough", "perlin_rough_stand"})


def _subterrain_reset_root_xy_offset(
    subterrain_name: str,
    sub_cfg: object,
    *,
    edge_offset: tuple[float, float] = (-4.0, 0.0),
    center_offset: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float]:
    """Return env-local spawn (x, y) offset for episode reset on this sub-terrain."""
    explicit = getattr(sub_cfg, "reset_root_xy_offset", None)
    if explicit is not None:
        return (float(explicit[0]), float(explicit[1]))
    if subterrain_name in _CENTER_SPAWN_SUBTERRAIN_NAMES:
        return center_offset
    fps = getattr(sub_cfg, "flat_patch_sampling", None)
    if isinstance(fps, dict):
        tgt = fps.get("target")
        if isinstance(tgt, FixedSequenceFlatPatchSamplingCfg):
            return edge_offset
        if tgt is not None:
            x_range = getattr(tgt, "x_range", None)
            if x_range is not None and len(x_range) >= 2:
                x0 = float(x_range[0])
                x1 = float(x_range[1])
                if abs(x0 - x1) < 1.0e-6 and x0 >= 3.0:
                    return edge_offset
    return edge_offset


def terrain_column_reset_root_xy_offset(
    terrain_generator_cfg: object,
    *,
    edge_offset: tuple[float, float] = (-4.0, 0.0),
    center_offset: tuple[float, float] = (0.0, 0.0),
) -> list[tuple[float, float]]:
    """Per terrain column: robot spawn (x, y) offset in the sub-terrain cell frame."""
    if terrain_generator_cfg is None:
        return []
    num_cols = int(getattr(terrain_generator_cfg, "num_cols", 0) or 0)
    sub_terrains = getattr(terrain_generator_cfg, "sub_terrains", None)
    if sub_terrains is None or num_cols <= 0:
        return []
    proportions = np.array([sub_cfg.proportion for sub_cfg in sub_terrains.values()], dtype=np.float64)
    denom = float(np.sum(proportions))
    if denom <= 0.0:
        return [edge_offset] * num_cols
    proportions = proportions / denom
    sub_terrains_names = list(sub_terrains.keys())
    sub_indices: list[int] = []
    for index in range(num_cols):
        sub_index = int(np.min(np.where(index / num_cols + 0.001 < np.cumsum(proportions))[0]))
        sub_indices.append(sub_index)
    out: list[tuple[float, float]] = []
    for col in range(num_cols):
        name = sub_terrains_names[sub_indices[col]]
        sub_cfg = sub_terrains[name]
        out.append(
            _subterrain_reset_root_xy_offset(
                name,
                sub_cfg,
                edge_offset=edge_offset,
                center_offset=center_offset,
            )
        )
    return out


def _flat_patch_xy_center_relative_to_corner_frame(
    fx: float,
    fy: float,
    terrain_size: tuple[float, float],
) -> tuple[float, float]:
    """Apply the same per-axis shift as ``_convert_range_to_grid`` for a single (x, y) pair."""
    half_x = 0.5 * float(terrain_size[0])
    half_y = 0.5 * float(terrain_size[1])
    ax = float(fx)
    ay = float(fy)
    if ax >= -half_x - 1.0e-6 and ax <= half_x + 1.0e-6:
        ax += half_x
    if ay >= -half_y - 1.0e-6 and ay <= half_y + 1.0e-6:
        ay += half_y
    return (ax, ay)


def _unwrap_height_field_function(function_obj: object) -> object:
    """Unwrap decorator layers and return the raw height-field callable."""
    raw_function = function_obj
    wrapped_function = getattr(raw_function, "__wrapped__", None)
    while wrapped_function is not None:
        raw_function = wrapped_function
        wrapped_function = getattr(raw_function, "__wrapped__", None)
    return raw_function


def _add_wall_geometries(
    *,
    body: mujoco.MjsBody,
    cfg: "HfTerrainBaseCfg",
    rng: np.random.Generator,
) -> list[TerrainGeometry]:
    """Add optional side walls, matching legacy `generate_wall` semantics."""
    wall_prob = cfg.wall_prob
    if wall_prob is None:
        return []

    wall_height = float(cfg.wall_height)
    wall_thickness = float(cfg.wall_thickness)
    if wall_height <= 0.0 or wall_thickness <= 0.0:
        return []

    min_x, max_x = 0.0, float(cfg.size[0])
    min_y, max_y = 0.0, float(cfg.size[1])
    geoms: list[TerrainGeometry] = []

    # Left wall
    if rng.uniform() < wall_prob[0]:
        left_wall = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(wall_thickness * 0.5, (max_y - min_y) * 0.5, wall_height * 0.5),
            pos=(min_x - wall_thickness * 0.5, (min_y + max_y) * 0.5, wall_height * 0.5),
        )
        geoms.append(TerrainGeometry(geom=left_wall))

    # Right wall
    if rng.uniform() < wall_prob[1]:
        right_wall = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(wall_thickness * 0.5, (max_y - min_y) * 0.5, wall_height * 0.5),
            pos=(max_x + wall_thickness * 0.5, (min_y + max_y) * 0.5, wall_height * 0.5),
        )
        geoms.append(TerrainGeometry(geom=right_wall))

    # Front wall
    if rng.uniform() < wall_prob[2]:
        front_wall = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=((max_x - min_x) * 0.5, wall_thickness * 0.5, wall_height * 0.5),
            pos=((min_x + max_x) * 0.5, min_y - wall_thickness * 0.5, wall_height * 0.5),
        )
        geoms.append(TerrainGeometry(geom=front_wall))

    # Back wall
    if rng.uniform() < wall_prob[3]:
        back_wall = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=((max_x - min_x) * 0.5, wall_thickness * 0.5, wall_height * 0.5),
            pos=((min_x + max_x) * 0.5, max_y + wall_thickness * 0.5, wall_height * 0.5),
        )
        geoms.append(TerrainGeometry(geom=back_wall))

    return geoms


def _find_flat_patches_from_heightfield_nearest(
    *,
    heights: np.ndarray,
    terrain_size: tuple[float, float],
    horizontal_scale: float,
    z_offset: float,
    cfg: FlatPatchSamplingCfg,
    rng: np.random.Generator,
) -> np.ndarray:
    """Find flat patches on hfield; if range is empty, choose nearest valid patch."""
    heights_work = np.asarray(heights, dtype=np.float64)
    grid_scale = float(horizontal_scale)
    if cfg.grid_resolution is not None and float(cfg.grid_resolution) < grid_scale:
        zoom_factor = grid_scale / float(cfg.grid_resolution)
        heights_work = np.asarray(ndimage.zoom(heights_work, zoom_factor, order=1))
        grid_scale = float(cfg.grid_resolution)

    num_rows, num_cols = heights_work.shape

    if isinstance(cfg.patch_radius, (list, tuple)):
        patch_radius = max(float(radius) for radius in cfg.patch_radius)
    else:
        patch_radius = float(cfg.patch_radius)
    radius_pixels = int(np.ceil(patch_radius / grid_scale))

    y_grid, x_grid = np.ogrid[
        -radius_pixels : radius_pixels + 1,
        -radius_pixels : radius_pixels + 1,
    ]
    footprint = (x_grid**2 + y_grid**2) <= radius_pixels**2

    max_h = ndimage.maximum_filter(
        heights_work,
        footprint=footprint,
        mode="constant",
        cval=-np.inf,
    )
    min_h = ndimage.minimum_filter(
        heights_work,
        footprint=footprint,
        mode="constant",
        cval=np.inf,
    )
    valid_base_mask = (max_h - min_h) <= float(cfg.max_height_diff)
    valid_base_mask[:radius_pixels, :] = False
    valid_base_mask[-radius_pixels:, :] = False
    valid_base_mask[:, :radius_pixels] = False
    valid_base_mask[:, -radius_pixels:] = False

    z_values = heights_work + float(z_offset)
    z_valid = (z_values >= float(cfg.z_range[0])) & (z_values <= float(cfg.z_range[1]))
    valid_base_mask &= z_valid

    fixed_xy = getattr(cfg, "fixed_xy_points", None)
    if fixed_xy is not None and len(fixed_xy) > 0:
        if int(cfg.num_patches) != len(fixed_xy):
            raise ValueError(
                "fixed_xy_points length must match num_patches for fixed-sequence flat targets. "
                f"Got len(fixed_xy_points)={len(fixed_xy)}, num_patches={cfg.num_patches}."
            )
        flat_pool = np.argwhere(valid_base_mask)
        if flat_pool.size == 0:
            raise RuntimeError("No flat heightfield cells available for fixed_xy_points targets.")
        out = np.zeros((len(fixed_xy), 3), dtype=np.float64)
        for i, (fx, fy) in enumerate(fixed_xy):
            ax, ay = _flat_patch_xy_center_relative_to_corner_frame(
                float(fx), float(fy), (float(terrain_size[0]), float(terrain_size[1]))
            )
            col = int(np.clip(np.round(ax / grid_scale), 0, num_cols - 1))
            row = int(np.clip(np.round(ay / grid_scale), 0, num_rows - 1))
            if bool(valid_base_mask[row, col]):
                pick_r, pick_c = row, col
            else:
                dist2 = (flat_pool[:, 0] - row) ** 2 + (flat_pool[:, 1] - col) ** 2
                pick = flat_pool[int(np.argmin(dist2))]
                pick_r, pick_c = int(pick[0]), int(pick[1])
            out[i, 0] = float(pick_c) * grid_scale
            out[i, 1] = float(pick_r) * grid_scale
            out[i, 2] = float(heights_work[pick_r, pick_c]) + float(z_offset)
        return out

    x_coords = np.arange(num_cols) * grid_scale
    y_coords = np.arange(num_rows) * grid_scale

    def _convert_range_to_grid(
        value_range: tuple[float, float],
        axis_size: float,
        axis_coords: np.ndarray,
    ) -> tuple[tuple[float, float], float]:
        lo = float(value_range[0])
        hi = float(value_range[1])
        if lo > hi:
            lo, hi = hi, lo

        half_axis = 0.5 * float(axis_size)
        # InstinctLab ranges are center-relative; keep absolute ranges unchanged.
        if lo >= -half_axis - 1.0e-6 and hi <= half_axis + 1.0e-6:
            lo += half_axis
            hi += half_axis

        min_coord = float(axis_coords[0]) if axis_coords.size > 0 else 0.0
        max_coord = float(axis_coords[-1]) if axis_coords.size > 0 else 0.0
        lo = float(np.clip(lo, min_coord, max_coord))
        hi = float(np.clip(hi, min_coord, max_coord))
        if lo > hi:
            lo, hi = hi, lo

        if (hi - lo) < grid_scale:
            target = 0.5 * (lo + hi)
            snapped = float(np.clip(np.round(target / grid_scale) * grid_scale, min_coord, max_coord))
            return (snapped, snapped), snapped

        lo = float(np.clip(np.floor(lo / grid_scale) * grid_scale, min_coord, max_coord))
        hi = float(np.clip(np.ceil(hi / grid_scale) * grid_scale, min_coord, max_coord))
        if lo > hi:
            hi = lo
        return (lo, hi), 0.5 * (lo + hi)

    x_range, target_x = _convert_range_to_grid(cfg.x_range, float(terrain_size[0]), x_coords)
    y_range, target_y = _convert_range_to_grid(cfg.y_range, float(terrain_size[1]), y_coords)

    x_valid = (x_coords >= float(x_range[0])) & (x_coords <= float(x_range[1]))
    y_valid = (y_coords >= float(y_range[0])) & (y_coords <= float(y_range[1]))
    valid_mask = valid_base_mask & y_valid[:, None] & x_valid[None, :]
    valid_indices = np.argwhere(valid_mask)

    if len(valid_indices) == 0:
        target_col = int(np.clip(np.round(target_x / grid_scale), 0, num_cols - 1))
        target_row = int(np.clip(np.round(target_y / grid_scale), 0, num_rows - 1))
        fallback_indices = np.argwhere(valid_base_mask)
        if len(fallback_indices) == 0:
            selected = np.tile(np.array([target_row, target_col], dtype=np.int64), (cfg.num_patches, 1))
        else:
            dist2 = (fallback_indices[:, 0] - target_row) ** 2 + (fallback_indices[:, 1] - target_col) ** 2
            sorted_idx = np.argsort(dist2)
            nearest = fallback_indices[sorted_idx]
            pool_size = max(1, min(len(nearest), cfg.num_patches * 4))
            pool = nearest[:pool_size]
            replace = pool_size < cfg.num_patches
            chosen = rng.choice(pool_size, size=cfg.num_patches, replace=replace)
            selected = pool[chosen]
    else:
        replace = len(valid_indices) < cfg.num_patches
        chosen = rng.choice(len(valid_indices), size=cfg.num_patches, replace=replace)
        selected = valid_indices[chosen]

    x = selected[:, 1] * grid_scale
    y = selected[:, 0] * grid_scale
    z = heights_work[selected[:, 0], selected[:, 1]] + float(z_offset)
    return np.stack([x, y, z], axis=-1)


def _height_field_to_output(
    *,
    heights: np.ndarray,
    cfg: "HfTerrainBaseCfg",
    spec: mujoco.MjSpec,
    rng: np.random.Generator,
) -> TerrainOutput:
    """Convert integer height field to MuJoCo hfield + TerrainOutput."""
    body = spec.body("terrain")
    heights_i16 = np.asarray(np.rint(heights), dtype=np.int16)

    elevation_min = int(np.min(heights_i16))
    elevation_max = int(np.max(heights_i16))
    elevation_range_i16 = elevation_max - elevation_min
    elevation_range_i16 = elevation_range_i16 if elevation_range_i16 > 0 else 1

    normalized_elevation = (heights_i16 - elevation_min) / elevation_range_i16
    max_physical_height = float(elevation_range_i16) * float(cfg.vertical_scale)
    base_thickness = max_physical_height * float(cfg.base_thickness_ratio)
    # MuJoCo hfield top height uses `geom_z + normalized * elevation_range`.
    # `base_thickness` only controls underside thickness and must NOT be
    # subtracted from geom z, otherwise each tile gets an extra depth-dependent
    # downward shift and seam steps appear between neighboring terrains.
    geom_z = float(elevation_min) * float(cfg.vertical_scale)

    unique_id = uuid.uuid4().hex
    hfield = spec.add_hfield(
        name=f"hfield_{unique_id}",
        size=[
            float(cfg.size[0]) / 2.0,
            float(cfg.size[1]) / 2.0,
            max_physical_height,
            base_thickness,
        ],
        nrow=heights_i16.shape[0],
        ncol=heights_i16.shape[1],
        userdata=normalized_elevation.astype(np.float32).flatten().tolist(),
    )

    hfield_geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_HFIELD,
        hfieldname=hfield.name,
        pos=(float(cfg.size[0]) * 0.5, float(cfg.size[1]) * 0.5, geom_z),
    )

    col1 = int((float(cfg.size[0]) * 0.5 - 1.0) / float(cfg.horizontal_scale))
    col2 = int((float(cfg.size[0]) * 0.5 + 1.0) / float(cfg.horizontal_scale))
    row1 = int((float(cfg.size[1]) * 0.5 - 1.0) / float(cfg.horizontal_scale))
    row2 = int((float(cfg.size[1]) * 0.5 + 1.0) / float(cfg.horizontal_scale))
    row1 = max(0, min(row1, heights_i16.shape[0] - 1))
    row2 = max(row1 + 1, min(row2, heights_i16.shape[0]))
    col1 = max(0, min(col1, heights_i16.shape[1] - 1))
    col2 = max(col1 + 1, min(col2, heights_i16.shape[1]))
    origin_z = float(np.max(heights_i16[row1:row2, col1:col2])) * float(cfg.vertical_scale)

    flat_patches: dict[str, np.ndarray] | None = None
    if cfg.flat_patch_sampling is not None:
        heights_phys = (heights_i16.astype(np.float64) - float(elevation_min)) * float(cfg.vertical_scale)
        z_offset = float(elevation_min) * float(cfg.vertical_scale)
        flat_patches = {}
        for patch_name, patch_cfg in cfg.flat_patch_sampling.items():
            if getattr(patch_cfg, "centerline_endpoint", None) in ("start", "end"):
                continue
            flat_patches[patch_name] = _find_flat_patches_from_heightfield_nearest(
                heights=heights_phys,
                terrain_size=cfg.size,
                horizontal_scale=float(cfg.horizontal_scale),
                z_offset=z_offset,
                cfg=patch_cfg,
                rng=rng,
            )
    origin = np.array([float(cfg.size[0]) * 0.5, float(cfg.size[1]) * 0.5, origin_z], dtype=np.float64)

    geometries = [TerrainGeometry(geom=hfield_geom, hfield=hfield)]
    geometries.extend(_add_wall_geometries(body=body, cfg=cfg, rng=rng))
    return TerrainOutput(origin=origin, geometries=geometries, flat_patches=flat_patches)


def _height_field_to_hfield_surface_mesh(
    heights: np.ndarray,
    cfg: "HfTerrainBaseCfg",
) -> trimesh.Trimesh:
    """Convert heights into a source-style surface mesh for virtual obstacles."""
    heights_i16 = np.asarray(heights, dtype=np.int16)
    nrow, ncol = heights_i16.shape
    if nrow <= 1 or ncol <= 1:
        return trimesh.Trimesh(vertices=np.empty((0, 3), dtype=np.float32), faces=np.empty((0, 3), dtype=np.int32))

    horizontal_step_y = float(cfg.size[1]) / float(nrow - 1)
    vertices, faces = convert_height_field_to_mesh(
        heights_i16,
        horizontal_scale=horizontal_step_y,
        vertical_scale=float(cfg.vertical_scale),
        slope_threshold=cfg.slope_threshold,
    )
    vertices = np.asarray(vertices, dtype=np.float32)
    if horizontal_step_y > 0.0:
        horizontal_step_x = float(cfg.size[0]) / float(ncol - 1)
        vertices[:, 0] *= float(horizontal_step_x / horizontal_step_y)
    return trimesh.Trimesh(vertices=vertices, faces=faces.astype(np.int32, copy=False), process=False)


@dataclass(kw_only=True)
class HfTerrainBaseCfg(SubTerrainCfg):
    border_width: float = 0.0
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    slope_threshold: float | None = None
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None
    reset_root_xy_offset: tuple[float, float] | None = None
    """Env-local spawn (x, y) offset for ``reset_root_state_on_target_patch``; z still from flat patch."""
    base_thickness_ratio: float = 1.0
    wall_prob: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    wall_height: float = 5.0
    wall_thickness: float = 0.05
    virtual_obstacle_keys: tuple[str, ...] = ()
    """Keys in ``TerrainImporterCfg.virtual_obstacles`` that extract geometry from this tile only."""

    def _generate_height_field(self, difficulty: float, cfg_for_gen: "HfTerrainBaseCfg") -> np.ndarray:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _generate_height_field(difficulty, cfg_for_gen)."
        )

    def function(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
    ) -> TerrainOutput:
        if self.border_width > 0.0 and self.border_width < self.horizontal_scale:
            raise ValueError(
                f"The border width ({self.border_width}) must be greater than or equal to the"
                f" horizontal scale ({self.horizontal_scale})."
            )

        # NOTE:
        # MuJoCo hfield sampling in mjlab assumes grid counts use
        # `int(size / horizontal_scale)` (no `+1`), and border uses
        # `int(border_width / horizontal_scale)`.
        # Keeping legacy `+1` here introduces seam artifacts between
        # neighboring tiles (boundary samples fall into mismatched cells),
        # which amplifies with deeper terrains.
        width_pixels = int(self.size[1] / self.horizontal_scale)
        length_pixels = int(self.size[0] / self.horizontal_scale)
        border_pixels = int(self.border_width / self.horizontal_scale)
        heights = np.zeros((width_pixels, length_pixels), dtype=np.int16)

        sub_terrain_size = [length_pixels - 2 * border_pixels, width_pixels - 2 * border_pixels]
        sub_terrain_size = [dim * self.horizontal_scale for dim in sub_terrain_size]

        cfg_for_gen = copy.deepcopy(self)
        cfg_for_gen.size = tuple(sub_terrain_size)
        generated = self._generate_height_field(difficulty, cfg_for_gen)
        generated_i16 = np.asarray(np.rint(generated), dtype=np.int16)
        if border_pixels > 0:
            heights[border_pixels:-border_pixels, border_pixels:-border_pixels] = generated_i16
        else:
            heights[:, :] = generated_i16

        output = _height_field_to_output(heights=heights, cfg=self, spec=spec, rng=rng)
        # Keep a mesh representation for virtual-obstacle generation, sampled on
        # the same XY lattice as MuJoCo hfield to avoid edge position drift.
        output.instinct_surface_mesh = _height_field_to_hfield_surface_mesh(heights, self)
        return output


@dataclass(kw_only=True)
class HfPyramidSlopedTerrainCfg(HfTerrainBaseCfg):
    slope_range: tuple[float, float] = MISSING
    platform_width: float = 1.0
    inverted: bool = False


@dataclass(kw_only=True)
class HfInvertedPyramidSlopedTerrainCfg(HfPyramidSlopedTerrainCfg):
    inverted: bool = True


@dataclass(kw_only=True)
class HfPyramidStairsTerrainCfg(HfTerrainBaseCfg):
    step_height_range: tuple[float, float] = MISSING
    step_width: float = MISSING
    platform_width: float = 1.0
    inverted: bool = False


@dataclass(kw_only=True)
class HfInvertedPyramidStairsTerrainCfg(HfPyramidStairsTerrainCfg):
    inverted: bool = True


@dataclass(kw_only=True)
class HfOneSidedStairsTerrainCfg(HfTerrainBaseCfg):
    """Single-run stairs: steps along one world axis; constant height along the other axis.

    If ``up_down`` is False, ``stair_face`` picks the low edge: ``neg_x`` climbs toward +x;
    ``pos_x`` is the x-mirror (down along +x); ``neg_y`` / ``pos_y`` along y.

    If ``up_down`` is True, one tile has **connected** stairs up then down along the
    chosen axis (``neg_x``/``pos_x`` → x, ``neg_y``/``pos_y`` → y); edges are low
    and the center is highest. ``platform_width`` is unused in this mode.
    """

    step_height_range: tuple[float, float] = MISSING
    step_width: float = MISSING
    platform_width: float = 1.0
    # Optional lateral usable width (meters) orthogonal to stair run direction.
    # - None: use full terrain width.
    # - float: fixed lateral width.
    # - tuple(min, max): sample uniformly per generated sub-terrain.
    lateral_width: float | tuple[float, float] | None = None
    stair_face: Literal["neg_x", "pos_x", "neg_y", "pos_y"] = "neg_x"
    inverted: bool = False
    up_down: bool = False


@dataclass(kw_only=True)
class HfSpiralStairsTerrainCfg(HfTerrainBaseCfg):
    """Annular-sector spiral stairs inside the obstacle block.

    Set ``spiral_pivot_xy`` (meters from the obstacle -X/-Y corner) to place the
    pivot yourself; otherwise the pivot is inferred from the west-edge entrance
    and inner radius. ``spiral_pivot_backoff_x`` shifts the pivot west (toward the
    entrance); large values can push the west-side tread past the obstacle -X
    edge when the sweep starts at 180 deg. ``spiral_center_offset`` adds on top
    in either mode.
    """

    obstacle_length: float = 5.04
    obstacle_width: float = 3.0
    step_height: tuple[float, float] | float = 0.15
    step_depth: float = 0.3
    num_steps: int = 6
    platform_length: float = 1.44
    stair_width: float = 1.2
    flight_angle_range: tuple[float, float] | float = (0.0, 30.0)
    edge_resolution: float = 0.02
    spiral_radius: float = 1.0
    spiral_step_arc_length: float | None = 0.3
    spiral_pivot_xy: tuple[float, float] | None = None
    spiral_entrance_along: float = 0.0
    spiral_entrance_inset_x: float = 0.0
    spiral_pivot_backoff_x: float = 0.0
    spiral_center_offset: tuple[float, float] = (0.0, 0.0)
    spiral_up_start_angle_deg: float = 270.0
    spiral_total_angle_deg: float = 90.0
    add_down_spiral: bool = False


@dataclass(kw_only=True)
class HfDiscreteObstaclesTerrainCfg(HfTerrainBaseCfg):
    obstacle_height_mode: str = "choice"
    obstacle_width_range: tuple[float, float] = MISSING
    obstacle_height_range: tuple[float, float] = MISSING
    num_obstacles: int = MISSING
    platform_width: float = 1.0


@dataclass(kw_only=True)
class HfWaveTerrainCfg(HfTerrainBaseCfg):
    amplitude_range: tuple[float, float] = MISSING
    num_waves: int = 1


@dataclass(kw_only=True)
class HfSteppingStonesTerrainCfg(HfTerrainBaseCfg):
    stone_height_max: float = MISSING
    stone_width_range: tuple[float, float] = MISSING
    stone_distance_range: tuple[float, float] = MISSING
    holes_depth: float = -10.0
    platform_width: float = 1.0


@dataclass
class WallTerrainCfgMixin:
    border_width: float = 0.0
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    slope_threshold: float | None = None
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None
    wall_prob: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0]
    )  # Probability of generating walls on [left, right, front, back] sides
    wall_height: float = 5.0  # Height of the walls
    wall_thickness: float = 0.05  # Thickness of the walls


@dataclass(kw_only=True)
class PerlinPlaneTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    noise_scale: float | List[float] = 0.05
    noise_frequency: int = 20

    fractal_octaves: int = 2

    fractal_lacunarity: float = 2.0

    fractal_gain: float = 0.25

    centering: bool = False  # If True, the noise will be centered around 0

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_PLANE_TERRAIN_FN(difficulty, cfg_for_gen)


@dataclass(kw_only=True)
class PerlinPyramidSlopedTerrainCfg(HfPyramidSlopedTerrainCfg, WallTerrainCfgMixin):
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    slope_range: tuple[float, float] = MISSING
    platform_width: float = 1.0
    inverted: bool = False
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_PYRAMID_SLOPED_TERRAIN_FN(difficulty, cfg_for_gen)


@dataclass(kw_only=True)
class PerlinInvertedPyramidSlopedTerrainCfg(HfInvertedPyramidSlopedTerrainCfg, WallTerrainCfgMixin):
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    slope_range: tuple[float, float] = MISSING
    platform_width: float = 1.0
    inverted: bool = True
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_PYRAMID_SLOPED_TERRAIN_FN(difficulty, cfg_for_gen)


@dataclass(kw_only=True)
class PerlinPyramidStairsTerrainCfg(HfPyramidStairsTerrainCfg, WallTerrainCfgMixin):
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    step_height_range: tuple[float, float] = MISSING
    step_width: float = MISSING
    platform_width: float = 1.0
    inverted: bool = False
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_PYRAMID_STAIRS_TERRAIN_FN(difficulty, cfg_for_gen)


@dataclass(kw_only=True)
class PerlinInvertedPyramidStairsTerrainCfg(HfInvertedPyramidStairsTerrainCfg, WallTerrainCfgMixin):
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    step_height_range: tuple[float, float] = MISSING
    step_width: float = MISSING
    platform_width: float = 1.0
    inverted: bool = True
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_PYRAMID_STAIRS_TERRAIN_FN(difficulty, cfg_for_gen)


@dataclass(kw_only=True)
class PerlinOneSidedStairsTerrainCfg(HfOneSidedStairsTerrainCfg, WallTerrainCfgMixin):
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    step_height_range: tuple[float, float] = MISSING
    step_width: float = MISSING
    platform_width: float = 1.0
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_ONE_SIDED_STAIRS_TERRAIN_FN(difficulty, cfg_for_gen)


@dataclass(kw_only=True)
class PerlinSpiralStairsTerrainCfg(HfSpiralStairsTerrainCfg, WallTerrainCfgMixin):
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    obstacle_length: float = 5.04
    obstacle_width: float = 3.0
    step_height: tuple[float, float] | float = 0.15
    step_depth: float = 0.3
    num_steps: int = 6
    platform_length: float = 1.44
    stair_width: float = 1.2
    flight_angle_range: tuple[float, float] | float = (0.0, 30.0)
    edge_resolution: float = 0.02
    spiral_radius: float = 1.0
    spiral_step_arc_length: float | None = 0.3
    spiral_entrance_along: float = 0.0
    spiral_entrance_inset_x: float = 0.0
    spiral_pivot_backoff_x: float = 0.0
    spiral_center_offset: tuple[float, float] = (0.0, 0.0)
    spiral_up_start_angle_deg: float = 270.0
    spiral_total_angle_deg: float = 90.0
    add_down_spiral: bool = False
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_SPIRAL_STAIRS_TERRAIN_FN(difficulty, cfg_for_gen)

    def function(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
    ) -> TerrainOutput:
        if self.border_width > 0.0 and self.border_width < self.horizontal_scale:
            raise ValueError(
                f"The border width ({self.border_width}) must be greater than or equal to the"
                f" horizontal scale ({self.horizontal_scale})."
            )

        cfg_for_output = copy.deepcopy(self)
        cfg_for_output.horizontal_scale = min(float(self.horizontal_scale), float(self.edge_resolution))

        width_pixels = int(cfg_for_output.size[0] / cfg_for_output.horizontal_scale)
        length_pixels = int(cfg_for_output.size[1] / cfg_for_output.horizontal_scale)
        border_pixels = int(cfg_for_output.border_width / cfg_for_output.horizontal_scale)
        heights = np.zeros((width_pixels, length_pixels), dtype=np.int16)

        sub_terrain_size = [width_pixels - 2 * border_pixels, length_pixels - 2 * border_pixels]
        sub_terrain_size = [dim * cfg_for_output.horizontal_scale for dim in sub_terrain_size]

        cfg_for_gen = copy.deepcopy(cfg_for_output)
        cfg_for_gen.size = tuple(sub_terrain_size)
        generated = self._generate_height_field(difficulty, cfg_for_gen)
        generated_i16 = np.asarray(np.rint(generated), dtype=np.int16)
        if border_pixels > 0:
            heights[border_pixels:-border_pixels, border_pixels:-border_pixels] = generated_i16
        else:
            heights[:, :] = generated_i16

        output = _height_field_to_output(heights=heights, cfg=cfg_for_output, spec=spec, rng=rng)
        output.instinct_surface_mesh = _height_field_to_hfield_surface_mesh(heights, cfg_for_output)
        return output


@dataclass(kw_only=True)
class PerlinDiscreteObstaclesTerrainCfg(HfDiscreteObstaclesTerrainCfg, WallTerrainCfgMixin):
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    obstacle_height_mode: str = "choice"
    obstacle_width_range: tuple[float, float] = MISSING
    obstacle_height_range: tuple[float, float] = MISSING
    num_obstacles: int = MISSING
    platform_width: float = 1.0
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_DISCRETE_OBSTACLES_TERRAIN_FN(difficulty, cfg_for_gen)


@dataclass(kw_only=True)
class PerlinWaveTerrainCfg(HfWaveTerrainCfg, WallTerrainCfgMixin):
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    amplitude_range: tuple[float, float] = MISSING
    num_waves: int = 1
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_WAVE_TERRAIN_FN(difficulty, cfg_for_gen)


@dataclass(kw_only=True)
class PerlinSteppingStonesTerrainCfg(HfSteppingStonesTerrainCfg, WallTerrainCfgMixin):
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    stone_height_max: float = MISSING
    stone_width_range: tuple[float, float] = MISSING
    stone_distance_range: tuple[float, float] = MISSING
    holes_depth: float = -10.0
    platform_width: float = 1.0
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_STEPPING_STONES_TERRAIN_FN(difficulty, cfg_for_gen)


# -- Newly added terrain configurations for parkour terrains-- #
@dataclass(kw_only=True)
class PerlinParapetTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a parapet terrain, can be used for jump and hurdle tasks."""

    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    parapet_height: tuple[float, float] | float = (0.1, 0.3)
    parapet_length: tuple[float, float] | float = (0.1, 0.3)
    parapet_width: float | None = None
    curved_top_rate: float | None = None
    """The rate to generate curved top. If None, the top will be flat."""
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_PARAPET_TERRAIN_FN(difficulty, cfg_for_gen)


@dataclass(kw_only=True)
class PerlinGutterTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a gutter parkour terrain."""

    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    gutter_length: tuple[float, float] | float = (0.5, 1.5)  # the distance between gutters
    gutter_depth: tuple[float, float] | float = (0.1, 0.3)  # the depth of the gutter
    gutter_width: float | None = None  # the length of the gutter
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_GUTTER_TERRAIN_FN(difficulty, cfg_for_gen)


@dataclass(kw_only=True)
class PerlinStairsUpDownTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a stairs up and down parkour terrain."""

    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    per_step_height: tuple[float, float] | float = MISSING
    """The height of each step. Could be a fixed value or a range (min, max)."""
    per_step_width: float | None = None
    """The width of each step. If None, it will be equal to the width of the terrain."""
    per_step_length: tuple[float, float] | float = MISSING
    """The length of each step along the y-axis."""
    num_steps: tuple[int, int] | int = MISSING
    """The number of steps. Could be a fixed value or a range (min, max)."""

    platform_length: float = 1.0
    """The length of the platform at the bottom of the stairs."""

    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_STAIRS_UP_DOWN_TERRAIN_FN(difficulty, cfg_for_gen)


@dataclass(kw_only=True)
class PerlinStairsDownUpTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a stairs down and up parkour terrain."""

    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    per_step_height: tuple[float, float] | float = MISSING
    """The height of each step. Could be a fixed value or a range (min, max)."""
    per_step_width: float | None = None
    """The width of each step. If None, it will be equal to the width of the terrain."""
    per_step_length: tuple[float, float] | float = MISSING
    """The length of each step along the y-axis."""
    num_steps: tuple[int, int] | int = MISSING
    """The number of steps. Could be a fixed value or a range (min, max)."""

    platform_length: float = 1.0
    """The length of the platform at the bottom of the stairs."""

    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_STAIRS_DOWN_UP_TERRAIN_FN(difficulty, cfg_for_gen)


@dataclass(kw_only=True)
class PerlinTiltTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a tilt terrain."""

    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    wall_height: tuple[float, float] | float = MISSING
    wall_width: float | None = None
    wall_length: tuple[float, float] | float = MISSING
    wall_opening_angle: tuple[float, float] | float = MISSING  # in degrees
    wall_opening_width: tuple[float, float] | float = MISSING
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_TILT_TERRAIN_FN(difficulty, cfg_for_gen)


@dataclass(kw_only=True)
class PerlinTiltedRampTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a tilted ramp terrain."""

    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    tilt_angle: tuple[float, float] | float = MISSING  # in degrees
    tilt_height: tuple[float, float] | float = MISSING
    tilt_width: tuple[float, float] | float = MISSING
    tilt_length: tuple[float, float] | float = MISSING
    switch_spacing: tuple[float, float] | float = MISSING
    spacing_curriculum: bool | None = None
    overlap_size: float | None = None
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_TILTED_RAMP_TERRAIN_FN(difficulty, cfg_for_gen)


@dataclass(kw_only=True)
class PerlinSlopeTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a slope up and down terrain with a flat ground in the middle."""

    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    slope_angle: tuple[float, float] | float = MISSING  # in degrees
    per_slope_length: tuple[float, float] | float = MISSING
    platform_length: float = 1.0
    slope_width: float | None = None
    up_down: bool | None = None  # If True or None, the slope will be up and down, otherwise it will be down and up.
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_SLOPE_TERRAIN_FN(difficulty, cfg_for_gen)


@dataclass(kw_only=True)
class PerlinCrossStoneTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a cross stone terrain."""

    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    stone_size: tuple[float, float] = MISSING
    stone_height: tuple[float, float] | float = MISSING
    stone_spacing: tuple[float, float] | float = MISSING
    ground_depth: float = -0.5
    platform_width: float = 1.5
    xy_random_ratio: float = 0.2
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_CROSS_STONE_TERRAIN_FN(difficulty, cfg_for_gen)


@dataclass(kw_only=True)
class PerlinSquareGapTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    gap_distance_range: tuple[float, float] = (0.1, 0.5)
    gap_depth: tuple[float, float] = (0.2, 0.5)
    platform_width: float = 1.5
    border_width: float = 0.0

    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_SQUARE_GAP_TERRAIN_FN(difficulty, cfg_for_gen)


@dataclass(kw_only=True)
class PerlinContinuousSlopeTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a continuous double-hump slope terrain.

    Two consecutive convex humps along the x-axis (travel direction).
    Slope angle varies with difficulty; total length and max height are fixed.
    ``terrain_width`` may be a ``(min, max)`` tuple for uniform random width
    per sub-terrain tile. Platform length along x adapts to the slope angle.
    """

    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    slope_angle_range: tuple[float, float] = (12.0, 20.0)
    """Slope angle range in degrees, interpolated by difficulty (low=12,
    high=20). Higher angle = steeper slope = harder."""

    total_length: float = 7.4
    """Fixed total terrain length along travel direction (meters)."""

    terrain_width: float | tuple[float, float] = 2.4
    """Terrain width in meters (lateral, centered in tile). Use ``(lo, hi)`` for uniform random width per tile."""

    max_height: float = 0.39
    """Fixed max height of each hump (meters)."""

    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_CONTINUOUS_SLOPE_TERRAIN_FN(difficulty, cfg_for_gen)


@dataclass(kw_only=True)
class PerlinCrossSlopeTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Four-segment lateral cross-slope (交叉斜坡) terrain.

    Travel along x; height varies along y. Adjacent segments flip tilt direction.
    Lateral span is ``terrain_width * tan(slope_angle)``; use ``slope_angle_range`` for curriculum.
    """

    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    total_length: float = 6.0
    """Terrain extent along travel direction (meters)."""

    terrain_width: float = 3.0
    """Terrain extent along lateral direction (meters)."""

    segment_length: float = 1.5
    """Length of each cross-slope segment along travel (meters)."""

    num_segments: int = 4
    """Number of alternating cross-slope segments."""

    slope_angle: float = 15.0
    """Cross-slope angle in degrees when ``slope_angle_range`` is None."""

    slope_angle_range: tuple[float, float] | None = None
    """Optional (low, high) angle in degrees, interpolated by difficulty."""

    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_CROSS_SLOPE_TERRAIN_FN(difficulty, cfg_for_gen)


@dataclass(kw_only=True)
class PerlinSymmetricalSlopeTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """V-groove symmetrical slope (对称斜坡) terrain.

    Cross-section along y is a V-groove: flat bottom with symmetric ramps on both sides.
    Use ``slope_angle_range`` for curriculum; otherwise ``slope_angle`` is fixed.
    """

    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    total_length: float = 3.0
    """Terrain extent along travel direction (x, meters)."""

    terrain_width: float = 3.5
    """Terrain extent along lateral direction (y, meters)."""

    slope_angle: float = 20.0
    """Fixed V-groove ramp angle (degrees) when ``slope_angle_range`` is None."""

    slope_angle_range: tuple[float, float] | None = None
    """Optional (low, high) ramp angle (degrees), interpolated by difficulty."""

    flat_bottom: float = 0.1
    """Flat bottom width at the V-groove center (meters)."""

    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_SYMMETRICAL_SLOPE_TERRAIN_FN(difficulty, cfg_for_gen)


@dataclass(kw_only=True)
class PerlinCurvedBridgeTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """S-curved bridge (S形弯桥) terrain.

    Narrow bridge in an S-shape formed by two opposing semicircles, raised above flat
    ground. Centerline runs along +x with lateral bends in y (total length ~bridge_length).
    """

    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    bridge_length: float = 10.0
    """Total path length along the S-curve centerline (meters)."""

    bridge_width: float = 0.5
    """Road width (meters) when ``bridge_width_range`` is None."""

    bridge_width_range: tuple[float, float] | None = None
    """Optional (easy, hard) road width (meters), interpolated by difficulty."""

    bridge_height: float = 0.1
    """Bridge surface height above ground (meters)."""

    straight_lead_in_m: float = 0.0
    """Extra straight deck before the S-curve (meters, along +x)."""

    straight_lead_out_m: float = 0.0
    """Extra straight deck after the S-curve (meters, along +x)."""

    curve_radius: float | None = None
    """Radius of each semicircle. Auto-derived from bridge_length if None and ``curve_radius_range`` is None."""

    curve_radius_range: tuple[float, float] | None = None
    """Optional (easy, hard) curve radius (meters), interpolated by difficulty when not straight."""

    straight_at_zero_difficulty: bool = True
    """When True, the easiest curriculum band uses a straight +x bridge instead of an S-curve."""

    straight_max_difficulty: float = 0.1
    """Upper difficulty (exclusive) for the straight bridge."""

    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_CURVED_BRIDGE_TERRAIN_FN(difficulty, cfg_for_gen)

    def function(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
    ) -> TerrainOutput:
        from . import hf_terrains as hf_terrains_mod

        output = super().function(difficulty, spec, rng)
        if self.flat_patch_sampling is None or output.flat_patches is None:
            return output

        width_pixels = int(self.size[1] / self.horizontal_scale)
        length_pixels = int(self.size[0] / self.horizontal_scale)
        border_pixels = int(self.border_width / self.horizontal_scale)
        sub_terrain_size = [
            (length_pixels - 2 * border_pixels) * self.horizontal_scale,
            (width_pixels - 2 * border_pixels) * self.horizontal_scale,
        ]
        cfg_sub = copy.deepcopy(self)
        cfg_sub.size = (float(sub_terrain_size[0]), float(sub_terrain_size[1]))
        generated_i16 = np.asarray(
            self._generate_height_field(difficulty, cfg_sub),
            dtype=np.int16,
        )
        heights_i16 = np.zeros((width_pixels, length_pixels), dtype=np.int16)
        if border_pixels > 0:
            heights_i16[
                border_pixels:-border_pixels,
                border_pixels:-border_pixels,
            ] = generated_i16
        else:
            heights_i16[:, :] = generated_i16

        elevation_min = int(np.min(heights_i16))
        z_offset = float(elevation_min) * float(self.vertical_scale)
        heights_phys = (heights_i16.astype(np.float64) - float(elevation_min)) * float(self.vertical_scale)

        patched: dict[str, np.ndarray] = dict(output.flat_patches or {})
        for patch_name, patch_cfg in self.flat_patch_sampling.items():
            endpoint = getattr(patch_cfg, "centerline_endpoint", None)
            if endpoint not in ("start", "end"):
                continue
            patched[patch_name] = hf_terrains_mod.curved_bridge_centerline_patch(
                difficulty=float(difficulty),
                cfg=cfg_sub,
                heights_i16=heights_i16,
                heights_phys=heights_phys,
                z_offset=z_offset,
                endpoint=cast(Literal["start", "end"], endpoint),
                terrain_size=(float(sub_terrain_size[0]), float(sub_terrain_size[1])),
                border_xy=(
                    float(border_pixels) * float(self.horizontal_scale),
                    float(border_pixels) * float(self.horizontal_scale),
                ),
                arc_offset_m=float(getattr(patch_cfg, "centerline_arc_offset_m", 0.0)),
                spawn_on_ground=bool(getattr(patch_cfg, "centerline_spawn_on_ground", False)),
                ground_approach_m=float(getattr(patch_cfg, "centerline_ground_approach_m", 0.75)),
                ground_offset_m=tuple(getattr(patch_cfg, "centerline_ground_offset_m", (0.0, 0.0))),
                tangent_offset_m=getattr(patch_cfg, "centerline_tangent_offset_m", None),
            )
        output.flat_patches = patched
        return output


@dataclass(kw_only=True)
class PerlinChainedStraightBridgeTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Chained straight bridge modules (串联直桥).

    Each module is a uniform ``bridge_height`` (default 0.1 m) oriented rectangular
    deck. Ground gaps and joint turns keep boxes disjoint—no overlapping geometry.
    """

    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    bridge_width: float = 0.5
    bridge_width_range: tuple[float, float] | None = None
    bridge_height: float = 0.1

    num_segments_range: tuple[int, int] = (2, 4)
    """Inclusive (easy, hard) module count, interpolated by difficulty."""

    segment_length_m: float = 2.5
    """Deck length per module when ``segment_length_range`` is None."""

    segment_length_range: tuple[float, float] | None = None

    segment_gap_m: float = 0.2
    """Ground gap (m) between consecutive modules along the exit tangent."""

    turn_angle_range: tuple[float, float] = (0.0, 0.0)
    """Heading change (deg) after each module when ``segment_turns_deg`` is None."""

    segment_turns_deg: tuple[float, ...] | None = None
    """Explicit per-joint turn angles (deg) after each module; length = num_segments - 1.

    Positive turns left (CCW). Use alternating signs (e.g. ``(25, -25)``) for zigzag swings."""

    entrance_approach_m: float = 0.0
    """Flat ground strip (m) before the first deck; 0 disables extra entrance geometry."""

    entrance_ramp_m: float = 0.0
    """Ramp length (m) before the first deck; 0 means no entrance ramp."""

    exit_approach_m: float = 0.0
    """Flat ground strip (m) after the last deck; 0 disables extra exit geometry."""

    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_CHAINED_STRAIGHT_BRIDGE_TERRAIN_FN(difficulty, cfg_for_gen)

    def function(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
    ) -> TerrainOutput:
        from . import hf_terrains as hf_terrains_mod

        output = super().function(difficulty, spec, rng)
        if self.flat_patch_sampling is None or output.flat_patches is None:
            return output

        width_pixels = int(self.size[1] / self.horizontal_scale)
        length_pixels = int(self.size[0] / self.horizontal_scale)
        border_pixels = int(self.border_width / self.horizontal_scale)
        sub_terrain_size = [
            (length_pixels - 2 * border_pixels) * self.horizontal_scale,
            (width_pixels - 2 * border_pixels) * self.horizontal_scale,
        ]
        cfg_sub = copy.deepcopy(self)
        cfg_sub.size = (float(sub_terrain_size[0]), float(sub_terrain_size[1]))
        generated_i16 = np.asarray(
            self._generate_height_field(difficulty, cfg_sub),
            dtype=np.int16,
        )
        heights_i16 = np.zeros((width_pixels, length_pixels), dtype=np.int16)
        if border_pixels > 0:
            heights_i16[
                border_pixels:-border_pixels,
                border_pixels:-border_pixels,
            ] = generated_i16
        else:
            heights_i16[:, :] = generated_i16

        elevation_min = int(np.min(heights_i16))
        z_offset = float(elevation_min) * float(self.vertical_scale)
        heights_phys = (heights_i16.astype(np.float64) - float(elevation_min)) * float(self.vertical_scale)

        patched: dict[str, np.ndarray] = dict(output.flat_patches or {})
        for patch_name, patch_cfg in self.flat_patch_sampling.items():
            endpoint = getattr(patch_cfg, "centerline_endpoint", None)
            if endpoint not in ("start", "end"):
                continue
            patched[patch_name] = hf_terrains_mod.chained_straight_bridge_centerline_patch(
                difficulty=float(difficulty),
                cfg=cfg_sub,
                heights_i16=heights_i16,
                heights_phys=heights_phys,
                z_offset=z_offset,
                endpoint=cast(Literal["start", "end"], endpoint),
                terrain_size=(float(sub_terrain_size[0]), float(sub_terrain_size[1])),
                border_xy=(
                    float(border_pixels) * float(self.horizontal_scale),
                    float(border_pixels) * float(self.horizontal_scale),
                ),
                arc_offset_m=float(getattr(patch_cfg, "centerline_arc_offset_m", 0.0)),
                spawn_on_ground=bool(getattr(patch_cfg, "centerline_spawn_on_ground", False)),
                ground_approach_m=float(getattr(patch_cfg, "centerline_ground_approach_m", 0.75)),
                ground_offset_m=tuple(getattr(patch_cfg, "centerline_ground_offset_m", (0.0, 0.0))),
                tangent_offset_m=getattr(patch_cfg, "centerline_tangent_offset_m", None),
            )
        output.flat_patches = patched
        return output


@dataclass(kw_only=True)
class PerlinContinuousHurdleTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Continuous hurdle (连续跨栏) terrain.

    Multiple thin rectangular bars arranged linearly along the travel direction.
    Each bar spans 2.4m laterally, stands 0.3m tall, and is 0.05m thick.
    Hurdles are added as BOX geoms (too thin for hfield); ground is flat.
    """

    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    num_hurdles: int = 5
    """Number of hurdles."""

    hurdle_width: float = 2.4
    """Lateral span along y (meters)."""

    hurdle_height: float = 0.3
    """Hurdle height (meters). Used when ``hurdle_height_range`` is None."""

    hurdle_height_range: tuple[float, float] | None = None
    """Optional (min, max) hurdle height in meters, interpolated by difficulty for curriculum."""

    hurdle_thickness: float = 0.05
    """Hurdle thickness along travel direction x (meters)."""

    hurdle_spacing: float = 1.1
    """Clear spacing between hurdles, trailing edge to leading edge (meters).
    Used when ``hurdle_spacing_range`` is None."""

    hurdle_spacing_range: tuple[float, float] | None = None
    """Optional (min, max) hurdle spacing in meters, interpolated by difficulty for curriculum."""

    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_CONTINUOUS_HURDLE_TERRAIN_FN(difficulty, cast(PerlinContinuousHurdleTerrainCfg, cfg_for_gen))

    def function(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
    ) -> TerrainOutput:
        # Generate ground hfield (optional Perlin overlay) via parent
        output = super().function(difficulty, spec, rng)

        # Add hurdle box geoms to the terrain body and output geometries
        # so _create_terrain_geom applies the world_position offset.
        body = spec.body("terrain")
        hurdle_height = self.hurdle_height
        if self.hurdle_height_range is not None:
            lo, hi = self.hurdle_height_range
            hurdle_height = lo + difficulty * (hi - lo)
        hurdle_spacing = self.hurdle_spacing
        if self.hurdle_spacing_range is not None:
            lo, hi = self.hurdle_spacing_range
            hurdle_spacing = lo + difficulty * (hi - lo)

        half_thick = self.hurdle_thickness * 0.5
        half_width = self.hurdle_width * 0.5
        half_height = hurdle_height * 0.5

        total_thick = self.num_hurdles * self.hurdle_thickness
        total_space = (self.num_hurdles - 1) * hurdle_spacing
        total_length = total_thick + total_space
        x_start = (self.size[0] - total_length) * 0.5
        y_center = self.size[1] * 0.5

        hurdle_meshes: list[trimesh.Trimesh] = []
        for i in range(self.num_hurdles):
            cx = x_start + i * (self.hurdle_thickness + hurdle_spacing) + half_thick
            geom = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[half_thick, half_width, half_height],
                pos=[cx, y_center, half_height],
            )
            output.geometries.append(TerrainGeometry(geom=geom))
            size = np.asarray(geom.size, dtype=np.float64).reshape(-1)
            extents = 2.0 * size[:3]
            transform = np.eye(4, dtype=np.float64)
            quat = np.asarray(geom.quat, dtype=np.float64).reshape(-1)
            if quat.size >= 4:
                transform[:3, :3] = trimesh.transformations.quaternion_matrix(quat[:4])[:3, :3]
            transform[:3, 3] = np.asarray(geom.pos, dtype=np.float64).reshape(3)
            hurdle_meshes.append(trimesh.creation.box(extents=extents, transform=transform))

        # Hfield-only surface mesh omits hurdle BOX geoms; merge them for virtual_obstacle_keys extraction.
        mesh_parts: list[trimesh.Trimesh] = []
        base_surface = getattr(output, "instinct_surface_mesh", None)
        if isinstance(base_surface, trimesh.Trimesh):
            mesh_parts.append(base_surface)
        mesh_parts.extend(hurdle_meshes)
        if len(mesh_parts) == 1:
            output.instinct_surface_mesh = mesh_parts[0]
        elif len(mesh_parts) > 1:
            output.instinct_surface_mesh = trimesh.util.concatenate(mesh_parts)

        return output


@dataclass(kw_only=True)
class PerlinNarrowLCornerTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Narrow L-corner elevated channel (限宽 L 型弯).

    Travel +x on leg1, turn at the east end onto leg2 along +y.
    Discrete entry/exit stairs; one row of guard bars on inner or outer channel edge (MuJoCo geoms).
    v1: fixed geometry (difficulty ignored).
    """

    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    leg_length: float = 4.0
    """Each straight leg length (meters)."""

    channel_width: float = 0.6
    """Walkable deck width (meters) when ``channel_width_range`` is None."""

    channel_width_range: tuple[float, float] | None = None
    """Optional (easy, hard) deck width (meters), interpolated by difficulty.
    E.g. (1.0, 0.5): 1.0m at difficulty=0, 0.5m when difficulty reaches ``channel_width_max_difficulty``."""

    channel_width_max_difficulty: float = 1.0
    """Difficulty at which ``channel_width_range`` reaches its hard (hi) value; values above clamp."""

    deck_height: float = 0.3
    """Elevated channel floor height (meters) when ``deck_height_range`` is None."""

    deck_height_range: tuple[float, float] | None = None
    """Optional (easy, hard) deck height (meters), interpolated by difficulty."""

    railing_height: float = 1.2
    """Inner guard bar height above deck (meters)."""

    railing_spacing: float = 0.25
    """Center-to-center spacing of vertical guard bars (meters)."""

    railing_bar_radius: float = 0.02
    """Radius of each vertical guard bar (meters)."""

    railing_side: Literal["inner", "outer", "left", "right", "none", "random"] = "random"
    """Guard bar side. ``random`` picks left, right, or none. ``none`` omits guard bars."""

    outer_curb_height: float = 0.15
    """Low lip height on outer deck edges (meters)."""

    ground_height: float = 0.0
    """Surrounding ground height (meters)."""

    step_height: float = 0.15
    """Rise per stair tread (meters)."""

    step_width: float = 0.30
    """Tread run per step along travel direction (meters); excluded from leg_length."""

    turn: Literal["left", "right"] = "left"
    """Turn direction: ``left`` (+x then +y), ``right`` mirrors to +x then -y."""

    turn_angle_range: tuple[float, float] | None = None
    """Optional (easy, hard) turn angle in degrees, interpolated by difficulty.
    0 deg = straight bridge (leg2 continues +x), 90 deg = full L-turn (+y)."""

    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_NARROW_L_CORNER_TERRAIN_FN(difficulty, cfg_for_gen)

    def function(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
    ) -> TerrainOutput:
        output = super().function(difficulty, spec, rng)

        from . import hf_terrains as hf_terrains_mod

        bw = float(self.border_width)
        sub_size = (
            float(self.size[0]) - 2.0 * bw,
            float(self.size[1]) - 2.0 * bw,
        )
        geom = hf_terrains_mod.narrow_l_corner_compute_geometry(
            difficulty, self, tile_size=sub_size
        )
        deck_h = float(geom["deck_height_m"])
        bar_r = max(self.railing_bar_radius, 0.01)
        bar_half_h = self.railing_height * 0.5
        bar_z = deck_h + bar_half_h
        rail_side = hf_terrains_mod.narrow_l_corner_resolve_railing_side(self, rng)
        rail_xy: list[tuple[float, float]] = []
        if rail_side != "none":
            rail_xy = hf_terrains_mod.narrow_l_corner_railing_xy_positions(
                difficulty,
                self,
                rail_side,
                tile_size=sub_size,
                border_xy=(bw, bw),
            )

        body = spec.body("terrain")
        bar_rgba = {
            "left": (0.92, 0.35, 0.08, 0.95),
            "right": (0.12, 0.45, 0.95, 0.95),
            "inner": (0.92, 0.35, 0.08, 0.95),
            "outer": (0.12, 0.45, 0.95, 0.95),
        }.get(rail_side, (0.75, 0.75, 0.75, 0.95))
        bar_geoms: list[TerrainGeometry] = []
        for x_m, y_m in rail_xy:
            geom = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[bar_r, bar_half_h],
                pos=[x_m, y_m, bar_z],
            )
            geom.contype = 1
            geom.conaffinity = 1
            geom.rgba[:] = bar_rgba
            bar_geoms.append(TerrainGeometry(geom=geom))
        output.geometries.extend(bar_geoms)

        if self.flat_patch_sampling is not None:
            width_pixels = int(self.size[1] / self.horizontal_scale)
            length_pixels = int(self.size[0] / self.horizontal_scale)
            border_pixels = int(self.border_width / self.horizontal_scale)
            sub_terrain_size = [
                (length_pixels - 2 * border_pixels) * self.horizontal_scale,
                (width_pixels - 2 * border_pixels) * self.horizontal_scale,
            ]
            cfg_sub = copy.deepcopy(self)
            cfg_sub.size = (float(sub_terrain_size[0]), float(sub_terrain_size[1]))
            generated_i16 = np.asarray(
                self._generate_height_field(difficulty, cfg_sub),
                dtype=np.int16,
            )
            heights_i16 = np.zeros((width_pixels, length_pixels), dtype=np.int16)
            if border_pixels > 0:
                heights_i16[
                    border_pixels:-border_pixels,
                    border_pixels:-border_pixels,
                ] = generated_i16
            else:
                heights_i16[:, :] = generated_i16
            elevation_min = int(np.min(heights_i16))
            z_offset = float(elevation_min) * float(self.vertical_scale)
            heights_phys = (heights_i16.astype(np.float64) - float(elevation_min)) * float(self.vertical_scale)

            patched: dict[str, np.ndarray] = dict(output.flat_patches or {})
            for patch_name, patch_cfg in self.flat_patch_sampling.items():
                endpoint = getattr(patch_cfg, "centerline_endpoint", None)
                if endpoint not in ("start", "end"):
                    continue
                patched[patch_name] = hf_terrains_mod.narrow_l_corner_centerline_patch(
                    difficulty=float(difficulty),
                    cfg=cfg_sub,
                    heights_i16=heights_i16,
                    heights_phys=heights_phys,
                    z_offset=z_offset,
                    endpoint=cast(Literal["start", "end"], endpoint),
                    border_xy=(
                        float(border_pixels) * float(self.horizontal_scale),
                        float(border_pixels) * float(self.horizontal_scale),
                    ),
                    arc_offset_m=float(getattr(patch_cfg, "centerline_arc_offset_m", 0.0)),
                    spawn_on_ground=bool(getattr(patch_cfg, "centerline_spawn_on_ground", False)),
                    ground_approach_m=float(getattr(patch_cfg, "centerline_ground_approach_m", 0.75)),
                    ground_offset_m=tuple(getattr(patch_cfg, "centerline_ground_offset_m", (0.0, 0.0))),
                    tangent_offset_m=getattr(patch_cfg, "centerline_tangent_offset_m", None),
                )
            output.flat_patches = patched
        return output


@dataclass(kw_only=True)
class PerlinSlalomPoleTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Slalom pole (绕桩) terrain.

    Surrounding ground with a flat raised deck (default 10 cm, no entry stairs).
    Thin tall cylinders on the deck at 1.1 m spacing; robot approaches on the
    ground, steps onto the deck, then weaves between poles.
    """

    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    num_poles: int = 5
    """Number of poles arranged in a line."""

    pole_height: float = 1.5
    """Pole height (meters)."""

    pole_radius: float = 0.015
    """Pole radius (meters); 0.03m diameter."""

    pole_spacing: float = 1.1
    """Center-to-center spacing between poles (meters)."""

    deck_height: float = 0.10
    """Elevated deck height (meters) under the pole line."""

    deck_width: float = 3.0
    """Deck width along y, centered on the tile (meters)."""

    deck_margin_x: float = 0.85
    """Deck extension beyond the first/last pole along x (meters)."""

    ground_height: float = 0.0
    """Height of ground outside the deck (meters)."""

    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    def _generate_height_field(self, difficulty: float, cfg_for_gen: HfTerrainBaseCfg) -> np.ndarray:
        return _RAW_PERLIN_SLALOM_POLE_TERRAIN_FN(difficulty, cast(PerlinSlalomPoleTerrainCfg, cfg_for_gen))

    def function(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
    ) -> TerrainOutput:
        output = super().function(difficulty, spec, rng)

        body = spec.body("terrain")
        total_span = (self.num_poles - 1) * self.pole_spacing
        x_start = (self.size[0] - total_span) * 0.5
        y_center = self.size[1] * 0.5
        pole_half_height = self.pole_height * 0.5
        pole_base_z = self.deck_height + pole_half_height
        min_radius = max(self.pole_radius, 0.01)

        for i in range(self.num_poles):
            x = x_start + i * self.pole_spacing
            geom = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[min_radius, pole_half_height],
                pos=[x, y_center, pole_base_z],
            )
            output.geometries.append(TerrainGeometry(geom=geom))

        return output


# ---------------------------------------------------------------------------
# Deferred import: hf_terrains depends on this module's base classes, so we
# import it AFTER all Cfg classes are defined to break the circular-import
# cycle (the _RAW_* names are only looked up at method-call time).
# ---------------------------------------------------------------------------
from . import hf_terrains

_RAW_PERLIN_PLANE_TERRAIN_FN = _unwrap_height_field_function(hf_terrains.perlin_plane_terrain)
_RAW_PERLIN_PYRAMID_SLOPED_TERRAIN_FN = _unwrap_height_field_function(hf_terrains.perlin_pyramid_sloped_terrain)
_RAW_PERLIN_PYRAMID_STAIRS_TERRAIN_FN = _unwrap_height_field_function(hf_terrains.perlin_pyramid_stairs_terrain)
_RAW_PERLIN_ONE_SIDED_STAIRS_TERRAIN_FN = _unwrap_height_field_function(hf_terrains.perlin_one_sided_stairs_terrain)
_RAW_PERLIN_SPIRAL_STAIRS_TERRAIN_FN = _unwrap_height_field_function(hf_terrains.perlin_spiral_stairs_terrain)
_RAW_PERLIN_DISCRETE_OBSTACLES_TERRAIN_FN = _unwrap_height_field_function(hf_terrains.perlin_discrete_obstacles_terrain)
_RAW_PERLIN_WAVE_TERRAIN_FN = _unwrap_height_field_function(hf_terrains.perlin_wave_terrain)
_RAW_PERLIN_STEPPING_STONES_TERRAIN_FN = _unwrap_height_field_function(hf_terrains.perlin_stepping_stones_terrain)
_RAW_PERLIN_PARAPET_TERRAIN_FN = _unwrap_height_field_function(hf_terrains.perlin_parapet_terrain)
_RAW_PERLIN_GUTTER_TERRAIN_FN = _unwrap_height_field_function(hf_terrains.perlin_gutter_terrain)
_RAW_PERLIN_STAIRS_UP_DOWN_TERRAIN_FN = _unwrap_height_field_function(hf_terrains.perlin_stairs_up_down_terrain)
_RAW_PERLIN_STAIRS_DOWN_UP_TERRAIN_FN = _unwrap_height_field_function(hf_terrains.perlin_stairs_down_up_terrain)
_RAW_PERLIN_TILT_TERRAIN_FN = _unwrap_height_field_function(hf_terrains.perlin_tilt_terrain)
_RAW_PERLIN_TILTED_RAMP_TERRAIN_FN = _unwrap_height_field_function(hf_terrains.perlin_tilted_ramp_terrain)
_RAW_PERLIN_SLOPE_TERRAIN_FN = _unwrap_height_field_function(hf_terrains.perlin_slope_terrain)
_RAW_PERLIN_CROSS_STONE_TERRAIN_FN = _unwrap_height_field_function(hf_terrains.perlin_cross_stone_terrain)
_RAW_PERLIN_SQUARE_GAP_TERRAIN_FN = _unwrap_height_field_function(hf_terrains.perlin_square_gap_terrain)
_RAW_PERLIN_CONTINUOUS_SLOPE_TERRAIN_FN = _unwrap_height_field_function(hf_terrains.perlin_continuous_slope_terrain)
_RAW_PERLIN_CROSS_SLOPE_TERRAIN_FN = _unwrap_height_field_function(hf_terrains.perlin_cross_slope_terrain)
_RAW_PERLIN_SYMMETRICAL_SLOPE_TERRAIN_FN = _unwrap_height_field_function(hf_terrains.perlin_symmetrical_slope_terrain)
_RAW_PERLIN_CURVED_BRIDGE_TERRAIN_FN = _unwrap_height_field_function(hf_terrains.perlin_curved_bridge_terrain)
_RAW_PERLIN_CHAINED_STRAIGHT_BRIDGE_TERRAIN_FN = _unwrap_height_field_function(
    hf_terrains.perlin_chained_straight_bridge_terrain
)
_RAW_PERLIN_CONTINUOUS_HURDLE_TERRAIN_FN = _unwrap_height_field_function(hf_terrains.perlin_continuous_hurdle_terrain)
_RAW_PERLIN_NARROW_L_CORNER_TERRAIN_FN = _unwrap_height_field_function(hf_terrains.perlin_narrow_l_corner_terrain)
_RAW_PERLIN_SLALOM_POLE_TERRAIN_FN = _unwrap_height_field_function(hf_terrains.perlin_slalom_pole_terrain)
