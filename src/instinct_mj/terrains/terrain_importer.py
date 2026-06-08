from __future__ import annotations

import copy
import gc
import math
import time
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch
import trimesh
from mjlab.terrains import SubTerrainCfg as SubTerrainBaseCfg
from mjlab.terrains import TerrainEntity as TerrainEntityBase
from mjlab.terrains import TerrainGenerator

from .height_field.utils import convert_height_field_to_mesh


class Timer:
    """Simple timer context manager."""

    def __init__(self, message: str):
        self.message = message
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed = time.perf_counter() - self._start
        print(f"{self.message}: {elapsed:.4f}s")
        return False


if TYPE_CHECKING:
    from .terrain_importer_cfg import TerrainImporterCfg
    from .virtual_obstacle import VirtualObstacleBase


class TerrainImporter(TerrainEntityBase):
    def __init__(self, cfg: TerrainImporterCfg, device: str):
        runtime_terrain_type = cfg.terrain_type
        if runtime_terrain_type == "hacked_generator":
            self._hacked_terrain_type = "hacked_generator"
            runtime_terrain_type = "plane"

        self._debug_vis_enabled = False
        self._collision_debug_vis = bool(cfg.collision_debug_vis)
        self._collision_debug_rgba = tuple(cfg.collision_debug_rgba)
        self._virtual_obstacle_source = str(cfg.virtual_obstacle_source).lower()
        if self._virtual_obstacle_source not in ("mesh", "heightfield"):
            raise ValueError(
                f"virtual_obstacle_source must be 'mesh' or 'heightfield'. Got: {self._virtual_obstacle_source!r}"
            )
        self._virtual_obstacle_hfield_method = str(cfg.virtual_obstacle_hfield_method).lower()
        if self._virtual_obstacle_hfield_method != "mesh_like":
            raise ValueError(
                f"virtual_obstacle_hfield_method must be 'mesh_like'. Got: {self._virtual_obstacle_hfield_method!r}"
            )
        self._virtual_obstacles = {}
        for name, virtual_obstacle_cfg in cfg.virtual_obstacles.items():
            if virtual_obstacle_cfg is None:
                continue
            virtual_obstacle = virtual_obstacle_cfg.class_type(virtual_obstacle_cfg)
            self._virtual_obstacles[name] = virtual_obstacle

        # Build a runtime cfg and map hacked_generator -> generator.
        cfg_runtime = copy.deepcopy(cfg)
        cfg_runtime.terrain_type = runtime_terrain_type

        # -----------------------------------------------------------------------------
        # mjlab-native terrain importer flow (mirrors mjlab.terrains.TerrainImporter)
        # -----------------------------------------------------------------------------
        self.cfg = cfg_runtime
        self.device = device
        self._device = device
        self._spec = mujoco.MjSpec()
        self.env_origins = None
        self.terrain_origins = None
        self.terrain_generator = None
        self._flat_patches = {}
        self._flat_patch_radii = {}
        self._flat_patch_num_valid: dict[str, torch.Tensor] = {}
        self._flat_patch_headings: dict[str, torch.Tensor] = {}
        self._height_query_mesh: trimesh.Trimesh | None = None

        if self.cfg.terrain_type == "generator":
            if self.cfg.terrain_generator is None:
                raise ValueError("Input terrain type is 'generator' but no value provided for 'terrain_generator'.")
            terrain_generator_cls = getattr(
                self.cfg.terrain_generator,
                "class_type",
                TerrainGenerator,
            )
            if terrain_generator_cls is None:
                terrain_generator_cls = TerrainGenerator
            self.terrain_generator = terrain_generator_cls(
                self.cfg.terrain_generator,
                device=self.device,
            )
            self.terrain_generator.compile(self._spec)
            self.configure_env_origins(self.terrain_generator.terrain_origins)
            self._flat_patches = {
                name: torch.from_numpy(arr).to(device=self.device, dtype=torch.float)
                for name, arr in self.terrain_generator.flat_patches.items()
            }
            self._flat_patch_radii = dict(self.terrain_generator.flat_patch_radii)
            fnv = getattr(self.terrain_generator, "flat_patch_num_valid", None)
            if fnv is not None:
                self._flat_patch_num_valid = {
                    k: torch.from_numpy(v).to(device=self.device, dtype=torch.long)
                    for k, v in fnv.items()
                }
            headings = getattr(self.terrain_generator, "flat_patch_headings", None)
            if headings is not None:
                self._flat_patch_headings = {
                    k: torch.from_numpy(v).to(device=self.device, dtype=torch.float)
                    for k, v in headings.items()
                }
        elif self.cfg.terrain_type == "plane":
            self._import_ground_plane("terrain")
            if self.env_origins is None:
                self.configure_env_origins()
        else:
            raise ValueError(f"Unknown terrain type: {self.cfg.terrain_type}")

        self._add_env_origin_sites()
        self._add_terrain_origin_sites()
        self._add_flat_patch_sites()

        if len(self._virtual_obstacles) > 0:
            self._validate_subterrain_virtual_obstacle_keys()
            if self._uses_selective_subterrain_virtual_obstacles():
                self._generate_virtual_obstacles_by_subterrain()
            else:
                generated_from_hfield = False
                if self._virtual_obstacle_source == "heightfield":
                    generated_from_hfield = self._generate_virtual_obstacles_from_heightfield()

                if not generated_from_hfield:
                    terrain_mesh = self._get_terrain_mesh_for_virtual_obstacles()
                    if terrain_mesh is None:
                        raise RuntimeError(
                            "virtual obstacles are configured but no terrain mesh is available from terrain generator."
                        )
                    self._generate_virtual_obstacles(terrain_mesh)
            self._release_terrain_mesh_cache()

        if self._collision_debug_vis:
            self._apply_collision_debug_visual_style()

        # Keep Entity internals aligned with mjlab's initialization flow after
        # building terrain spec manually in this terrain importer.
        self._actuators = []
        self._identify_joints()
        self._apply_spec_editors()
        self._add_actuators()
        self._add_initial_state_keyframe()

    @property
    def virtual_obstacles(self) -> dict[str, VirtualObstacleBase]:
        """Get the virtual obstacles representing the edges.
        TODO: Make the returned value more general.
        """
        # still pointing the same VirtualObstacleBase objects but the dict is a copy.
        return self._virtual_obstacles.copy()

    @property
    def subterrain_specific_cfgs(self) -> list[SubTerrainBaseCfg] | None:
        """Get the specific configurations for all subterrains."""
        # This is a placeholder. The actual implementation should return the specific configurations.
        return (
            self.terrain_generator.subterrain_specific_cfgs
            if hasattr(self, "terrain_generator") and hasattr(self.terrain_generator, "subterrain_specific_cfgs")
            else None
        )

    @property
    def flat_patch_num_valid(self) -> dict[str, torch.Tensor]:
        """Per (terrain_row, terrain_col) count of real flat-patch slots (excludes spawn padding)."""
        return self._flat_patch_num_valid

    @property
    def flat_patch_headings(self) -> dict[str, torch.Tensor]:
        """Per-patch spawn yaw (rad) from terrain centerline, when available."""
        return self._flat_patch_headings

    """
    Operations - Import.
    """

    def _import_ground_plane(self, name: str):
        """Import a flat MuJoCo plane terrain."""
        if getattr(self, "_hacked_terrain_type", None) == "hacked_generator":
            # check config is provided
            if self.cfg.terrain_generator is None:
                raise ValueError("Input terrain type is 'generator' but no value provided for 'terrain_generator'.")
            # generate the terrain
            self.terrain_generator = getattr(
                self.cfg.terrain_generator,
                "class_type",
                TerrainGenerator,
            )(cfg=self.cfg.terrain_generator, device=self.device)
            self.terrain_generator.compile(self._spec)
            # configure the terrain origins based on the terrain generator
            self.configure_env_origins(self.terrain_generator.terrain_origins)
            # refer to the flat patches
            self._flat_patches = {
                name: torch.from_numpy(arr).to(device=self.device, dtype=torch.float)
                for name, arr in self.terrain_generator.flat_patches.items()
            }
            self._flat_patch_radii = dict(self.terrain_generator.flat_patch_radii)
            fnv = getattr(self.terrain_generator, "flat_patch_num_valid", None)
            if fnv is not None:
                self._flat_patch_num_valid = {
                    k: torch.from_numpy(v).to(device=self.device, dtype=torch.long)
                    for k, v in fnv.items()
                }
            headings = getattr(self.terrain_generator, "flat_patch_headings", None)
            if headings is not None:
                self._flat_patch_headings = {
                    k: torch.from_numpy(v).to(device=self.device, dtype=torch.float)
                    for k, v in headings.items()
                }
            return

        self._spec.worldbody.add_body(name=name).add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=(0, 0, 0.01),
        )

    def _get_terrain_mesh_for_virtual_obstacles(self) -> trimesh.Trimesh | None:
        if self.terrain_generator is None:
            return None
        # When low-step repair is requested, rebuild the virtual-obstacle mesh
        # from hfield specs so short stair risers/lips can use the same selective
        # height-threshold repair as the mesh_like heightfield path.
        if self._get_hfield_mesh_like_height_threshold() is not None:
            repaired_mesh = self._collect_hfield_surface_mesh(use_cached_terrain_mesh=False)
            if repaired_mesh is not None:
                return repaired_mesh
        terrain_mesh = self.terrain_generator.terrain_mesh
        return terrain_mesh

    @staticmethod
    def _subterrain_virtual_obstacle_keys(subterrain_cfg: SubTerrainBaseCfg | None) -> tuple[str, ...]:
        if subterrain_cfg is None:
            return ()
        keys = getattr(subterrain_cfg, "virtual_obstacle_keys", ())
        if keys is None:
            return ()
        if isinstance(keys, str):
            raise TypeError("virtual_obstacle_keys must be a sequence of obstacle names, not a single string.")
        return tuple(keys)

    def _uses_selective_subterrain_virtual_obstacles(self) -> bool:
        gen_cfg = self.cfg.terrain_generator
        if gen_cfg is None or not hasattr(gen_cfg, "sub_terrains"):
            return False
        return any(
            len(self._subterrain_virtual_obstacle_keys(sub_cfg)) > 0 for sub_cfg in gen_cfg.sub_terrains.values()
        )

    def _validate_subterrain_virtual_obstacle_keys(self) -> None:
        if not self._uses_selective_subterrain_virtual_obstacles():
            return
        known_keys = set(self._virtual_obstacles)
        gen_cfg = self.cfg.terrain_generator
        if gen_cfg is None or not hasattr(gen_cfg, "sub_terrains"):
            return
        for sub_name, sub_cfg in gen_cfg.sub_terrains.items():
            for key in self._subterrain_virtual_obstacle_keys(sub_cfg):
                if key not in known_keys:
                    raise ValueError(
                        f"sub_terrain {sub_name!r} lists virtual_obstacle_keys {key!r}, "
                        f"but terrain importer virtual_obstacles defines {sorted(known_keys)}."
                    )

    def _get_per_tile_meshes_for_virtual_obstacles(self) -> list[trimesh.Trimesh]:
        if self.terrain_generator is None:
            return []
        terrain_meshes = getattr(self.terrain_generator, "_terrain_meshes", None)
        if isinstance(terrain_meshes, list) and len(terrain_meshes) > 0:
            return list(terrain_meshes)
        terrain_mesh = getattr(self.terrain_generator, "terrain_mesh", None)
        if terrain_mesh is not None:
            return [terrain_mesh]
        return []

    def _repair_subterrain_mesh_for_virtual_obstacles(
        self,
        mesh: trimesh.Trimesh,
        geom,
        subterrain_cfg: SubTerrainBaseCfg | None,
    ) -> trimesh.Trimesh:
        if not self._should_apply_mesh_like_height_repair(subterrain_cfg):
            return mesh
        height_threshold = self._get_hfield_mesh_like_height_threshold()
        if height_threshold is None or geom is None:
            return mesh
        hfield_name = geom.hfieldname
        if not isinstance(hfield_name, str) or hfield_name == "":
            return mesh
        hfield_spec = self._spec.hfield(hfield_name)
        if hfield_spec is None:
            return mesh
        geom_pos = np.asarray(geom.pos, dtype=np.float64).reshape(3)
        repaired_mesh = self._hfield_spec_to_world_mesh(
            hfield_spec,
            geom_pos,
            slope_threshold=self._get_hfield_mesh_like_slope_threshold(),
            height_threshold=height_threshold,
        )
        if repaired_mesh is None:
            return mesh
        return repaired_mesh

    def _collect_meshes_by_virtual_obstacle_key(self) -> dict[str, list[trimesh.Trimesh]]:
        meshes_by_key: dict[str, list[trimesh.Trimesh]] = {name: [] for name in self._virtual_obstacles}
        tile_meshes = self._get_per_tile_meshes_for_virtual_obstacles()
        subterrain_cfgs = self.subterrain_specific_cfgs or []
        if len(tile_meshes) == 0:
            raise RuntimeError(
                "selective virtual_obstacle_keys require per-tile terrain meshes from the terrain generator."
            )
        if len(subterrain_cfgs) not in (0, len(tile_meshes)):
            raise RuntimeError(
                "subterrain_specific_cfgs length "
                f"({len(subterrain_cfgs)}) must match per-tile mesh count ({len(tile_meshes)})."
            )
        hfield_pairs = list(self._iter_hfield_geoms_with_subterrain_cfgs())
        for tile_index, mesh in enumerate(tile_meshes):
            subterrain_cfg = subterrain_cfgs[tile_index] if tile_index < len(subterrain_cfgs) else None
            obstacle_keys = self._subterrain_virtual_obstacle_keys(subterrain_cfg)
            if len(obstacle_keys) == 0:
                continue
            tile_mesh = mesh
            if tile_index < len(hfield_pairs):
                geom, _ = hfield_pairs[tile_index]
                tile_mesh = self._repair_subterrain_mesh_for_virtual_obstacles(mesh, geom, subterrain_cfg)
            for key in obstacle_keys:
                meshes_by_key[key].append(tile_mesh.copy())
        return meshes_by_key

    @staticmethod
    def _prepare_mesh_for_virtual_obstacle(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        prepared = mesh.copy()
        prepared.merge_vertices()
        prepared.update_faces(prepared.unique_faces())
        prepared.remove_unreferenced_vertices()
        return prepared

    @staticmethod
    def _empty_mesh_for_virtual_obstacle() -> trimesh.Trimesh:
        return trimesh.Trimesh(
            vertices=np.zeros((0, 3), dtype=np.float64),
            faces=np.zeros((0, 3), dtype=np.int32),
            process=False,
        )

    def _generate_virtual_obstacle_from_mesh_parts(
        self,
        name: str,
        virtual_obstacle: VirtualObstacleBase,
        mesh_parts: list[trimesh.Trimesh],
    ) -> None:
        if len(mesh_parts) == 0:
            mesh = self._empty_mesh_for_virtual_obstacle()
        elif len(mesh_parts) == 1:
            mesh = mesh_parts[0]
        else:
            mesh = trimesh.util.concatenate(mesh_parts)
        with Timer(f"Generate virtual obstacle {name}"):
            virtual_obstacle.generate(self._prepare_mesh_for_virtual_obstacle(mesh), device=self.device)

    def _generate_virtual_obstacles_by_subterrain(self) -> None:
        meshes_by_key = self._collect_meshes_by_virtual_obstacle_key()
        for name, virtual_obstacle in self._virtual_obstacles.items():
            self._generate_virtual_obstacle_from_mesh_parts(name, virtual_obstacle, meshes_by_key[name])

    def _generate_virtual_obstacles(self, mesh: trimesh.Trimesh):
        """Generate virtual obstacles from a terrain mesh."""
        prepared_mesh = self._prepare_mesh_for_virtual_obstacle(mesh)
        # NOTE: generate virtual obstacle first because it might modify the mesh.
        for name, virtual_obstacle in self._virtual_obstacles.items():
            with Timer(f"Generate virtual obstacle {name}"):
                virtual_obstacle.generate(prepared_mesh, device=self.device)

    @staticmethod
    def _simplify_polyline_collinear_indices(
        points_xyz: np.ndarray,
        *,
        distance_epsilon: float,
        angle_threshold_deg: float,
    ) -> np.ndarray:
        """Simplify polyline by removing only near-collinear interior points."""
        num_points = points_xyz.shape[0]
        if num_points <= 2:
            return np.arange(num_points, dtype=np.int64)

        dist_eps = max(float(distance_epsilon), 0.0)
        sin_threshold = math.sin(math.radians(max(float(angle_threshold_deg), 0.0)))

        keep: list[int] = [0]
        prev_kept = 0
        for idx in range(1, num_points - 1):
            prev_point = points_xyz[prev_kept, :2]
            current_point = points_xyz[idx, :2]
            next_point = points_xyz[idx + 1, :2]

            vec_a = current_point - prev_point
            vec_b = next_point - current_point
            norm_a = float(np.linalg.norm(vec_a))
            norm_b = float(np.linalg.norm(vec_b))
            if norm_a <= 1.0e-12 or norm_b <= 1.0e-12:
                keep.append(idx)
                prev_kept = idx
                continue

            cross_mag = abs(vec_a[0] * vec_b[1] - vec_a[1] * vec_b[0])
            sin_angle = cross_mag / max(norm_a * norm_b, 1.0e-12)
            if sin_angle > sin_threshold:
                keep.append(idx)
                prev_kept = idx
                continue

            line_vec = next_point - prev_point
            line_len = float(np.linalg.norm(line_vec))
            if line_len <= 1.0e-12:
                keep.append(idx)
                prev_kept = idx
                continue
            rel = current_point - prev_point
            line_dist = abs(line_vec[0] * rel[1] - line_vec[1] * rel[0]) / line_len
            if line_dist > dist_eps:
                keep.append(idx)
                prev_kept = idx

        keep.append(num_points - 1)
        return np.asarray(keep, dtype=np.int64)

    @staticmethod
    def _trace_segments_as_graph_polylines(
        edge_segments: np.ndarray,
        *,
        simplify_epsilon: float,
        min_segment_length: float,
        snap_xy: float | None,
        snap_z: float | None,
        collinear_angle_threshold: float,
    ) -> np.ndarray:
        """Trace primitive edge segments into continuous graph polylines."""
        if edge_segments.size == 0:
            return np.empty((0, 6), dtype=np.float32)

        segments = edge_segments.reshape(-1, 6).astype(np.float64, copy=False)
        if segments.shape[0] == 0:
            return np.empty((0, 6), dtype=np.float32)

        if snap_xy is None or float(snap_xy) <= 0.0:
            primitive_len = np.linalg.norm(segments[:, 3:5] - segments[:, 0:2], axis=1)
            positive = primitive_len[primitive_len > 1.0e-9]
            if positive.size == 0:
                snap_xy_local = 1.0e-4
            else:
                snap_xy_local = max(float(np.median(positive)) * 0.25, 1.0e-4)
        else:
            snap_xy_local = float(snap_xy)

        if snap_z is None or float(snap_z) <= 0.0:
            endpoint_z = np.concatenate([segments[:, 2], segments[:, 5]], axis=0)
            unique_z = np.unique(np.round(endpoint_z, decimals=4))
            if unique_z.size <= 1:
                snap_z_local = 1.0e-3
            else:
                z_deltas = np.diff(np.sort(unique_z))
                positive_deltas = z_deltas[z_deltas > 1.0e-6]
                if positive_deltas.size == 0:
                    snap_z_local = 1.0e-3
                else:
                    snap_z_local = max(float(np.median(positive_deltas)) * 0.5, 1.0e-3)
        else:
            snap_z_local = float(snap_z)

        node_map: dict[tuple[int, int, int], int] = {}
        node_xyz_sum: list[np.ndarray] = []
        node_count: list[int] = []

        def node_id_from_point(point_xyz: np.ndarray) -> int:
            key = (
                int(np.round(float(point_xyz[0]) / snap_xy_local)),
                int(np.round(float(point_xyz[1]) / snap_xy_local)),
                int(np.round(float(point_xyz[2]) / snap_z_local)),
            )
            node_id = node_map.get(key)
            if node_id is None:
                node_id = len(node_xyz_sum)
                node_map[key] = node_id
                node_xyz_sum.append(np.asarray(point_xyz, dtype=np.float64).copy())
                node_count.append(1)
                return node_id
            node_xyz_sum[node_id] = node_xyz_sum[node_id] + np.asarray(point_xyz, dtype=np.float64)
            node_count[node_id] += 1
            return node_id

        edges: list[tuple[int, int]] = []
        edge_key_set: set[tuple[int, int]] = set()
        for segment in segments:
            start_id = node_id_from_point(segment[:3])
            end_id = node_id_from_point(segment[3:6])
            if start_id == end_id:
                continue
            edge_key = (start_id, end_id) if start_id < end_id else (end_id, start_id)
            if edge_key in edge_key_set:
                continue
            edge_key_set.add(edge_key)
            edges.append(edge_key)

        if len(edges) == 0:
            return np.empty((0, 6), dtype=np.float32)

        adjacency: list[list[int]] = [[] for _ in range(len(node_xyz_sum))]
        for edge_id, (u, v) in enumerate(edges):
            adjacency[u].append(edge_id)
            adjacency[v].append(edge_id)

        visited = np.zeros(len(edges), dtype=bool)

        def edge_other_node(edge_id: int, node_id: int) -> int:
            u, v = edges[edge_id]
            return v if u == node_id else u

        def trace_from(node_start: int, edge_start: int) -> list[int]:
            path = [node_start]
            current_node = node_start
            current_edge = edge_start
            while True:
                if visited[current_edge]:
                    break
                visited[current_edge] = True
                next_node = edge_other_node(current_edge, current_node)
                path.append(next_node)
                current_node = next_node
                if len(adjacency[current_node]) != 2:
                    break
                next_edge = -1
                for edge_candidate in adjacency[current_node]:
                    if not visited[edge_candidate]:
                        next_edge = edge_candidate
                        break
                if next_edge < 0:
                    break
                current_edge = next_edge
            return path

        polylines: list[list[int]] = []
        # Start traces from branch/end nodes first.
        for node_id, node_edges in enumerate(adjacency):
            if len(node_edges) == 2:
                continue
            for edge_id in node_edges:
                if visited[edge_id]:
                    continue
                polylines.append(trace_from(node_id, edge_id))
        # Handle closed loops where every node degree is exactly 2.
        for edge_id in range(len(edges)):
            if visited[edge_id]:
                continue
            node_start = edges[edge_id][0]
            polylines.append(trace_from(node_start, edge_id))

        node_xyz = np.asarray(node_xyz_sum, dtype=np.float64)
        node_count_arr = np.asarray(node_count, dtype=np.float64).reshape(-1, 1)
        node_xyz = node_xyz / np.maximum(node_count_arr, 1.0)
        simplify_eps = max(float(simplify_epsilon), 0.0)
        min_seg_len = max(float(min_segment_length), 0.0)

        traced_segments: list[np.ndarray] = []
        for node_path in polylines:
            if len(node_path) < 2:
                continue
            polyline = node_xyz[np.asarray(node_path, dtype=np.int64)]
            if simplify_eps > 0.0 and polyline.shape[0] > 2:
                keep_indices = TerrainImporter._simplify_polyline_collinear_indices(
                    polyline,
                    distance_epsilon=simplify_eps,
                    angle_threshold_deg=collinear_angle_threshold,
                )
                polyline = polyline[keep_indices]
            if polyline.shape[0] < 2:
                continue
            for idx in range(polyline.shape[0] - 1):
                point_a = polyline[idx]
                point_b = polyline[idx + 1]
                if np.linalg.norm(point_b[:2] - point_a[:2]) < min_seg_len:
                    continue
                traced_segments.append(np.concatenate([point_a, point_b]))

        if len(traced_segments) == 0:
            return np.empty((0, 6), dtype=np.float32)
        return np.asarray(traced_segments, dtype=np.float32).reshape(-1, 6)

    @staticmethod
    def _hfield_spec_to_world_mesh(
        hfield_spec,
        geom_pos: np.ndarray,
        slope_threshold: float | None = None,
        height_threshold: float | None = None,
    ) -> trimesh.Trimesh | None:
        """Convert one MuJoCo hfield spec + geom pose into world-frame trimesh."""
        nrow = int(hfield_spec.nrow)
        ncol = int(hfield_spec.ncol)
        if nrow <= 1 or ncol <= 1:
            return None

        userdata = np.asarray(hfield_spec.userdata, dtype=np.float64)
        if userdata.size != nrow * ncol:
            return None
        normalized_heights = userdata.reshape(nrow, ncol)

        size = np.asarray(hfield_spec.size, dtype=np.float64).reshape(-1)
        if size.size < 4:
            return None
        half_x, half_y, elevation_range, _base_thickness = size[:4]

        # Reconstruct mesh with the same slope-threshold projection behavior as
        # height-field terrain conversion to reduce hfield triangulation artifacts.
        heights_m = normalized_heights * float(elevation_range)
        y_step = (2.0 * float(half_y)) / float(nrow - 1)
        x_step = (2.0 * float(half_x)) / float(ncol - 1)
        if y_step <= 0.0 or x_step <= 0.0:
            return None

        if height_threshold is not None:
            height_threshold = float(height_threshold)
            if height_threshold <= 0.0:
                height_threshold = None

        if height_threshold is not None:
            # Legacy low-step repair should only affect discontinuous hfield
            # terrains. It overrides slope-threshold projection locally so
            # short stair risers/gap lips remain connected in mesh_like mode.
            slope_threshold = height_threshold / float(y_step)
        elif slope_threshold is not None:
            slope_threshold = float(slope_threshold)
            if slope_threshold <= 0.0:
                slope_threshold = None

        vertices_local, faces = convert_height_field_to_mesh(
            height_field=heights_m,
            horizontal_scale=float(y_step),
            vertical_scale=1.0,
            slope_threshold=slope_threshold,
        )

        vertices = vertices_local.copy()
        vertices[:, 0] *= float(x_step / y_step)
        x_world = float(geom_pos[0] - half_x) + vertices[:, 0]
        y_world = float(geom_pos[1] - half_y) + vertices[:, 1]
        z_world = float(geom_pos[2]) + vertices[:, 2]
        vertices[:, 0] = x_world
        vertices[:, 1] = y_world
        vertices[:, 2] = z_world

        return trimesh.Trimesh(vertices=vertices, faces=faces.astype(np.int32, copy=False), process=False)

    def _get_hfield_mesh_like_slope_threshold(self) -> float | None:
        """Resolve slope-threshold used when reconstructing hfield surface mesh."""
        terrain_gen_cfg = self.cfg.terrain_generator
        if terrain_gen_cfg is None:
            return None
        slope_threshold = terrain_gen_cfg.slope_threshold
        if slope_threshold is None:
            return None
        slope_threshold = float(slope_threshold)
        if slope_threshold <= 0.0:
            return None
        return slope_threshold

    def _get_hfield_mesh_like_height_threshold(self) -> float | None:
        """Resolve absolute height-threshold override for discrete hfield repair."""
        height_threshold = self.cfg.virtual_obstacle_hfield_height_threshold
        if height_threshold is None:
            return None
        height_threshold = float(height_threshold)
        if height_threshold <= 0.0:
            return None
        return height_threshold

    @staticmethod
    def _should_apply_mesh_like_height_repair(subterrain_cfg: SubTerrainBaseCfg | None) -> bool:
        """Whether legacy low-step repair should apply to this subterrain."""
        if subterrain_cfg is None:
            return True
        cfg_type_name = type(subterrain_cfg).__name__
        if cfg_type_name == "PerlinPlaneTerrainCfg":
            return False
        if any(token in cfg_type_name for token in ("Slope", "Sloped", "Tilt", "Ramp", "Wave")):
            return False
        return True

    def _iter_hfield_geoms_with_subterrain_cfgs(self):
        """Iterate terrain hfield geoms together with their subterrain cfgs."""
        terrain_body = self._spec.body("terrain")
        if terrain_body is None:
            return

        subterrain_cfgs = self.subterrain_specific_cfgs or []
        hfield_geom_index = 0
        for geom in terrain_body.geoms:
            hfield_name = geom.hfieldname
            if not isinstance(hfield_name, str) or hfield_name == "":
                continue
            subterrain_cfg = subterrain_cfgs[hfield_geom_index] if hfield_geom_index < len(subterrain_cfgs) else None
            hfield_geom_index += 1
            yield geom, subterrain_cfg

    def _collect_hfield_surface_mesh(self, *, use_cached_terrain_mesh: bool = True) -> trimesh.Trimesh | None:
        """Collect and concatenate hfield surface meshes from terrain spec."""
        if use_cached_terrain_mesh:
            terrain_mesh = self.terrain_generator.terrain_mesh if self.terrain_generator is not None else None
        else:
            terrain_mesh = None
        if terrain_mesh is not None:
            return terrain_mesh.copy()

        slope_threshold = self._get_hfield_mesh_like_slope_threshold()
        height_threshold = self._get_hfield_mesh_like_height_threshold()
        meshes: list[trimesh.Trimesh] = []
        for geom, subterrain_cfg in self._iter_hfield_geoms_with_subterrain_cfgs():
            hfield_name = geom.hfieldname
            hfield_spec = self._spec.hfield(hfield_name)
            if hfield_spec is None:
                continue
            geom_pos = np.asarray(geom.pos, dtype=np.float64).reshape(3)
            geom_height_threshold = (
                height_threshold if self._should_apply_mesh_like_height_repair(subterrain_cfg) else None
            )
            world_mesh = self._hfield_spec_to_world_mesh(
                hfield_spec,
                geom_pos,
                slope_threshold=slope_threshold,
                height_threshold=geom_height_threshold,
            )
            if world_mesh is not None:
                meshes.append(world_mesh)

        if len(meshes) == 0:
            return None
        if len(meshes) == 1:
            return meshes[0]
        return trimesh.util.concatenate(meshes)

    def _collect_hfield_mesh_like_edge_segments(
        self,
        *,
        angle_threshold: float,
        drop_cell_diagonals: bool,
        trace_segments: bool,
        min_edge_length: float,
    ) -> np.ndarray:
        """Collect sharp edges from hfield-surface mesh while filtering cell diagonals."""
        slope_threshold = self._get_hfield_mesh_like_slope_threshold()
        height_threshold = self._get_hfield_mesh_like_height_threshold()
        sharp_threshold = np.deg2rad(float(angle_threshold))
        edge_segments_list: list[np.ndarray] = []
        for geom, subterrain_cfg in self._iter_hfield_geoms_with_subterrain_cfgs():
            hfield_name = geom.hfieldname
            hfield_spec = self._spec.hfield(hfield_name)
            if hfield_spec is None:
                continue
            ncol = int(hfield_spec.ncol)
            if ncol <= 1:
                continue
            geom_pos = np.asarray(geom.pos, dtype=np.float64).reshape(3)
            geom_height_threshold = (
                height_threshold if self._should_apply_mesh_like_height_repair(subterrain_cfg) else None
            )
            world_mesh = self._hfield_spec_to_world_mesh(
                hfield_spec,
                geom_pos,
                slope_threshold=slope_threshold,
                height_threshold=geom_height_threshold,
            )
            if world_mesh is None:
                continue

            angles = world_mesh.face_adjacency_angles
            if angles.size == 0:
                continue
            sharp_mask = angles > sharp_threshold
            if not np.any(sharp_mask):
                continue

            sharp_edges = world_mesh.face_adjacency_edges[sharp_mask]
            if sharp_edges.size == 0:
                continue

            if drop_cell_diagonals:
                start_idx = sharp_edges[:, 0]
                end_idx = sharp_edges[:, 1]
                start_row = start_idx // ncol
                start_col = start_idx % ncol
                end_row = end_idx // ncol
                end_col = end_idx % ncol
                row_delta = np.abs(end_row - start_row)
                col_delta = np.abs(end_col - start_col)
                # Structured hfield cell diagonal edges (delta=(1,1)) are
                # triangulation artifacts and create fragmented pseudo-obstacles.
                keep_mask = (row_delta + col_delta) == 1
                sharp_edges = sharp_edges[keep_mask]
                if sharp_edges.size == 0:
                    continue

            vertices = world_mesh.vertices
            edge_segments = np.hstack([vertices[sharp_edges[:, 0]], vertices[sharp_edges[:, 1]]]).astype(
                np.float32,
                copy=False,
            )
            if min_edge_length > 0.0:
                edge_lengths = np.linalg.norm(edge_segments[:, 3:6] - edge_segments[:, 0:3], axis=1)
                edge_segments = edge_segments[edge_lengths >= min_edge_length]
                if edge_segments.size == 0:
                    continue
            edge_segments_list.append(edge_segments)

        if len(edge_segments_list) == 0:
            return np.empty((0, 6), dtype=np.float32)
        edge_segments = np.concatenate(edge_segments_list, axis=0).astype(np.float32, copy=False)
        if not trace_segments:
            return edge_segments

        simplify_epsilon = float(self.cfg.virtual_obstacle_hfield_trace_simplify_epsilon)
        min_segment_length = float(self.cfg.virtual_obstacle_hfield_trace_min_segment_length)
        snap_xy = self.cfg.virtual_obstacle_hfield_trace_snap_xy
        if snap_xy is not None:
            snap_xy = float(snap_xy)
            if snap_xy <= 0.0:
                snap_xy = None
        snap_z = self.cfg.virtual_obstacle_hfield_trace_snap_z
        if snap_z is not None:
            snap_z = float(snap_z)
            if snap_z <= 0.0:
                snap_z = None
        collinear_angle_threshold = float(self.cfg.virtual_obstacle_hfield_trace_collinear_angle_threshold)
        return self._trace_segments_as_graph_polylines(
            edge_segments,
            simplify_epsilon=simplify_epsilon,
            min_segment_length=min_segment_length,
            snap_xy=snap_xy,
            snap_z=snap_z,
            collinear_angle_threshold=collinear_angle_threshold,
        )

    def _generate_virtual_obstacles_from_heightfield(self) -> bool:
        """Generate virtual obstacles directly from MuJoCo hfield data."""
        if len(self._virtual_obstacles) == 0:
            return True
        if self._uses_selective_subterrain_virtual_obstacles():
            self._generate_virtual_obstacles_by_subterrain()
            return True

        # Preserve original InstinctLab semantics as closely as possible:
        # reconstruct one mesh-like hfield surface first, merge shared tile
        # vertices, then let each obstacle use its native `generate(mesh)` flow.
        # This avoids discontinuities introduced by per-tile edge tracing.
        hfield_surface_mesh = self._collect_hfield_surface_mesh()
        if hfield_surface_mesh is None:
            return False
        self._generate_virtual_obstacles(hfield_surface_mesh)
        del hfield_surface_mesh
        gc.collect()
        if torch.cuda.is_available():
            device_type = torch.device(self.device).type
            if device_type == "cuda":
                torch.cuda.empty_cache()
        return True

    def _cleanup_heightfield_virtual_obstacle_cache(self, cached_segments_by_angle: dict[float, np.ndarray]) -> None:
        # Mesh-like extraction can allocate large temporary numpy/trimesh buffers.
        # Release them immediately after startup generation to reduce peak memory.
        cached_segments_by_angle.clear()
        gc.collect()
        if torch.cuda.is_available():
            device_type = torch.device(self.device).type
            if device_type == "cuda":
                torch.cuda.empty_cache()

    def _release_terrain_mesh_cache(self) -> None:
        """Release terrain mesh cache once virtual obstacles are built."""
        from .terrain_generator import FiledTerrainGenerator

        if self.terrain_generator is None:
            return
        if isinstance(self.terrain_generator, FiledTerrainGenerator):
            if self._height_query_mesh is None:
                terrain_mesh = self.terrain_generator.terrain_mesh
                if terrain_mesh is not None:
                    self._height_query_mesh = terrain_mesh.copy()
            self.terrain_generator.terrain_mesh = None
            terrain_meshes = self.terrain_generator._terrain_meshes
            if isinstance(terrain_meshes, list):
                terrain_meshes.clear()

    def _apply_collision_debug_visual_style(self) -> None:
        """Tint terrain collision geoms so they are visible in the viewer."""
        terrain_body = self._spec.body("terrain")
        if terrain_body is None:
            return
        for geom in terrain_body.geoms:
            contype = int(geom.contype)
            conaffinity = int(geom.conaffinity)
            if contype == 0 and conaffinity == 0:
                continue
            geom.rgba[:] = self._collision_debug_rgba

    def set_debug_vis(self, debug_vis: bool) -> bool:
        """Set the debug visualization flag.

        Args:
            vis: True to enable debug visualization, False to disable.
        """
        self._debug_vis_enabled = debug_vis
        results = True

        for name, virtual_obstacle in self._virtual_obstacles.items():
            if debug_vis:
                virtual_obstacle.visualize()
            else:
                virtual_obstacle.disable_visualizer()

        return results

    def debug_vis(self, visualizer) -> None:
        """Draw virtual obstacles using viewer-native debug visualizer."""
        if not self._debug_vis_enabled:
            return
        for virtual_obstacle in self._virtual_obstacles.values():
            virtual_obstacle.debug_vis(visualizer)

    def configure_env_origins(self, origins: np.ndarray | torch.Tensor | None = None):
        """Configure the environment origins.

        Args:
            origins: The origins of the environments. Shape is (num_envs, 3).
        """
        if origins is None and getattr(self, "_hacked_terrain_type", None) == "hacked_generator":
            # In case of None override, we don't need to do anything
            return
        return super().configure_env_origins(origins)
