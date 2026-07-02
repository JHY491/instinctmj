"""Sub-module containing Flat-Patch based velocity command generators."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.utils.lab_api import math as math_utils
from mjlab.utils.lab_api.math import quat_apply_inverse as quat_rotate_inverse
from mjlab.utils.lab_api.math import wrap_to_pi, yaw_quat

from instinct_mj.managers import CommandTerm
from instinct_mj.tasks.parkour.mdp.rewards import (
    _cref_all_stake_marker_centers,
    cref_foothold_reward_visualizable,
)
from instinct_mj.terrains import TerrainImporter
from instinct_mj.terrains.height_field.hf_terrains_cfg import (
    terrain_column_fixed_sequence_reset_patch_index,
    terrain_column_target_uses_fixed_sequence_flat_patch,
)

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

    from .commands_cfg import PoseVelocityCommandCfg


def _cref_all_tread_marker_centers(
    origins: torch.Tensor,
    treads_xz: torch.Tensor,
    tread_counts: torch.Tensor,
    *,
    z_offset: float = 0.15,
) -> list[tuple[str, np.ndarray]]:
    """Return marker centers for every valid CReF tread candidate in the terrain grid.

    The markers are intentionally lifted above the analytic tread height because
    one_sided_stairs can add Perlin roughness comparable to low-difficulty step
    heights, which otherwise hides markers inside the terrain surface.
    """
    origins_np = origins.detach().cpu().numpy()
    treads_np = treads_xz.detach().cpu().numpy()
    counts_np = tread_counts.detach().cpu().numpy()

    centers: list[tuple[str, np.ndarray]] = []
    num_rows, num_cols = counts_np.shape
    for row in range(num_rows):
        for col in range(num_cols):
            count = int(counts_np[row, col])
            if count <= 0:
                continue
            origin = origins_np[row, col]
            for tread_idx in range(count):
                tread_xz = treads_np[row, col, tread_idx]
                center = np.array(
                    [origin[0] + tread_xz[0], origin[1], tread_xz[1] + z_offset],
                    dtype=np.float64,
                )
                centers.append((f"cref_tread_{row}_{col}_{tread_idx}", center))
    return centers


def _cref_reward_term_from_manager(reward_manager: object | None, term_name: str) -> object | None:
    if reward_manager is None:
        return None
    get_term_func = getattr(reward_manager, "get_term_func", None)
    if callable(get_term_func):
        func = get_term_func(term_name)
        if func is not None:
            return func
    get_term_cfg = getattr(reward_manager, "get_term_cfg", None)
    if not callable(get_term_cfg):
        return None

    group_names: list[str | None] = ["rewards", None]
    group_term_names = getattr(reward_manager, "_MultiRewardManager__group_term_names", None)
    if isinstance(group_term_names, dict):
        group_names = list(dict.fromkeys([*group_term_names.keys(), "rewards", None]))

    for group_name in group_names:
        try:
            if group_name is None:
                term_cfg = get_term_cfg(term_name)
            else:
                term_cfg = get_term_cfg(term_name, group_name=group_name)
        except (AttributeError, ValueError, TypeError):
            continue
        func = getattr(term_cfg, "func", None)
        if func is not None:
            return func
    return None


class PoseVelocityCommand(CommandTerm):
    """Velocity command based on the 2D flat patch command generator."""

    cfg: PoseVelocityCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: PoseVelocityCommandCfg, env: ManagerBasedRlEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot and terrain assets
        # -- robot
        self.robot: Entity = env.scene[cfg.entity_name]

        # crete buffers to store the command
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        self.pos_command_b = torch.zeros_like(self.pos_command_w)
        self.heading_command_b = torch.zeros_like(self.heading_command_w)
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.max_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._target_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # -- metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["tracking_exp_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["tracking_exp_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        # debug ratios to localize "standing still" behavior
        self.metrics["command_nonzero_ratio"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["target_near_ratio"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["standing_env_ratio"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["random_velocity_env_ratio"] = torch.zeros(self.num_envs, device=self.device)

        # obtain the terrain asset
        self.terrain: TerrainImporter = env.scene["terrain"]

        self.lin_vel_x_range = torch.zeros(self.num_envs, 2, device=self.device)
        self.lin_vel_y_range = torch.zeros(self.num_envs, 2, device=self.device)
        self.ang_vel_z_range = torch.zeros(self.num_envs, 2, device=self.device)

        self.random_lin_vel_x_range = torch.zeros(self.num_envs, 2, device=self.device)
        self.random_lin_vel_y_range = torch.zeros(self.num_envs, 2, device=self.device)
        self.random_ang_vel_z_range = torch.zeros(self.num_envs, 2, device=self.device)
        self.random_velocity_indices = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Descent-slowdown state: which envs are eligible, and the smoothed base vz used to detect descent.
        self.descend_speed_indices = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._base_vz_ema = torch.zeros(self.num_envs, device=self.device)

        self.random_lin_vel_x = torch.zeros(self.num_envs, device=self.device)
        self.random_lin_vel_y = torch.zeros(self.num_envs, device=self.device)
        self.random_ang_vel_z = torch.zeros(self.num_envs, device=self.device)

        if self.cfg.velocity_ranges is not None:
            terrain_generator_cfg = self.terrain.cfg.terrain_generator
            proportions = np.array([sub_cfg.proportion for sub_cfg in terrain_generator_cfg.sub_terrains.values()])
            proportions /= np.sum(proportions)

            # find the sub-terrain index for each column
            # we generate the terrains based on their proportion (not randomly sampled)
            sub_indices = []
            for index in range(terrain_generator_cfg.num_cols):
                sub_index = np.min(np.where(index / terrain_generator_cfg.num_cols + 0.001 < np.cumsum(proportions))[0])
                sub_indices.append(sub_index)
            sub_indices = np.array(sub_indices, dtype=np.int32)
            sub_terrains_names = list(terrain_generator_cfg.sub_terrains.keys())
            for key, value in self.cfg.velocity_ranges.items():
                if key in sub_terrains_names:
                    terrain_type_index = sub_terrains_names.index(key)
                    type_indices = np.where(sub_indices == terrain_type_index)[0]
                    for type_indice in type_indices:
                        env_indices = torch.where(self.terrain.terrain_types == type_indice)[0]
                        self.lin_vel_x_range[env_indices, 0] = value["lin_vel_x"][0]
                        self.lin_vel_x_range[env_indices, 1] = value["lin_vel_x"][1]
                        self.lin_vel_y_range[env_indices, 0] = value["lin_vel_y"][0]
                        self.lin_vel_y_range[env_indices, 1] = value["lin_vel_y"][1]
                        self.ang_vel_z_range[env_indices, 0] = value["ang_vel_z"][0]
                        self.ang_vel_z_range[env_indices, 1] = value["ang_vel_z"][1]
                else:
                    raise RuntimeError(f"Terrain type {key} not found in the terrain generator sub-terrain names.")

            if self.cfg.random_velocity_terrain is not None:
                for key in self.cfg.random_velocity_terrain:
                    terrain_type_index = sub_terrains_names.index(key)
                    type_indices = np.where(sub_indices == terrain_type_index)[0]
                    for type_indice in type_indices:
                        env_indices = torch.where(self.terrain.terrain_types == type_indice)[0]
                        self.random_velocity_indices[env_indices] = True

        self.random_lin_vel_x_range[:, 0] = self.cfg.ranges.lin_vel_x[0]
        self.random_lin_vel_x_range[:, 1] = self.cfg.ranges.lin_vel_x[1]
        self.random_lin_vel_y_range[:, 0] = self.cfg.ranges.lin_vel_y[0]
        self.random_lin_vel_y_range[:, 1] = self.cfg.ranges.lin_vel_y[1]
        self.random_ang_vel_z_range[:, 0] = self.cfg.ranges.ang_vel_z[0]
        self.random_ang_vel_z_range[:, 1] = self.cfg.ranges.ang_vel_z[1]

        if self.cfg.random_velocity_terrain is not None and self.cfg.velocity_ranges is not None:
            terrain_generator_cfg = self.terrain.cfg.terrain_generator
            proportions = np.array([sub_cfg.proportion for sub_cfg in terrain_generator_cfg.sub_terrains.values()])
            proportions /= np.sum(proportions)
            sub_indices = []
            for index in range(terrain_generator_cfg.num_cols):
                sub_index = np.min(np.where(index / terrain_generator_cfg.num_cols + 0.001 < np.cumsum(proportions))[0])
                sub_indices.append(sub_index)
            sub_indices = np.array(sub_indices, dtype=np.int32)
            sub_terrains_names = list(terrain_generator_cfg.sub_terrains.keys())
            for key in self.cfg.random_velocity_terrain:
                if key not in self.cfg.velocity_ranges:
                    continue
                value = self.cfg.velocity_ranges[key]
                terrain_type_index = sub_terrains_names.index(key)
                type_indices = np.where(sub_indices == terrain_type_index)[0]
                for type_indice in type_indices:
                    env_indices = torch.where(self.terrain.terrain_types == type_indice)[0]
                    self.random_lin_vel_x_range[env_indices, 0] = value["lin_vel_x"][0]
                    self.random_lin_vel_x_range[env_indices, 1] = value["lin_vel_x"][1]
                    self.random_lin_vel_y_range[env_indices, 0] = value["lin_vel_y"][0]
                    self.random_lin_vel_y_range[env_indices, 1] = value["lin_vel_y"][1]
                    self.random_ang_vel_z_range[env_indices, 0] = value["ang_vel_z"][0]
                    self.random_ang_vel_z_range[env_indices, 1] = value["ang_vel_z"][1]

        if self.cfg.descend_speed_terrain is None:
            self.descend_speed_indices[:] = True
        elif self.cfg.velocity_ranges is not None:
            terrain_generator_cfg = self.terrain.cfg.terrain_generator
            proportions = np.array([sub_cfg.proportion for sub_cfg in terrain_generator_cfg.sub_terrains.values()])
            proportions /= np.sum(proportions)
            sub_indices = []
            for index in range(terrain_generator_cfg.num_cols):
                sub_index = np.min(np.where(index / terrain_generator_cfg.num_cols + 0.001 < np.cumsum(proportions))[0])
                sub_indices.append(sub_index)
            sub_indices = np.array(sub_indices, dtype=np.int32)
            sub_terrains_names = list(terrain_generator_cfg.sub_terrains.keys())
            for key in self.cfg.descend_speed_terrain:
                terrain_type_index = sub_terrains_names.index(key)
                type_indices = np.where(sub_indices == terrain_type_index)[0]
                for type_indice in type_indices:
                    env_indices = torch.where(self.terrain.terrain_types == type_indice)[0]
                    self.descend_speed_indices[env_indices] = True

        # obtain the valid targets from the terrain
        if "target" not in self.terrain.flat_patches:
            raise RuntimeError(
                "The terrain-based command generator requires a valid flat patch under 'target' in the terrain."
                f" Found: {list(self.terrain.flat_patches.keys())}"
            )
        # valid targets: (terrain_level, terrain_type, num_patches, 3) — ignore optional yaw slot
        targets = self.terrain.flat_patches["target"]
        self.valid_targets = targets[..., :3] if targets.shape[-1] > 3 else targets
        self._flat_patch_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._flat_patch_arrival_hold = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._flat_patch_seq_terminal = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        fnv = getattr(self.terrain, "flat_patch_num_valid", None)
        self._flat_patch_num_valid: torch.Tensor | None = None
        if isinstance(fnv, dict) and "target" in fnv:
            self._flat_patch_num_valid = fnv["target"]

        ncol = int(self.valid_targets.shape[1])
        tgen = getattr(self.terrain.cfg, "terrain_generator", None)
        col_flags = terrain_column_target_uses_fixed_sequence_flat_patch(tgen)
        col_reset_idx = terrain_column_fixed_sequence_reset_patch_index(tgen)
        if len(col_flags) != ncol:
            if len(col_flags) < ncol:
                col_flags = col_flags + [False] * (ncol - len(col_flags))
            else:
                col_flags = col_flags[:ncol]
        if len(col_reset_idx) != ncol:
            if len(col_reset_idx) < ncol:
                col_reset_idx = col_reset_idx + [0] * (ncol - len(col_reset_idx))
            else:
                col_reset_idx = col_reset_idx[:ncol]
        self._flat_patch_target_fixed_sequence_by_col = torch.tensor(col_flags, dtype=torch.bool, device=self.device)
        self._flat_patch_reset_index_by_col = torch.tensor(col_reset_idx, dtype=torch.long, device=self.device)

        self._guided_mode: bool = False
        self._guided_world: torch.Tensor | None = None
        self._guided_counts: torch.Tensor | None = None
        self._guided_wp_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._guided_arrival_hold = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._init_guided_waypoints()

    def __str__(self) -> str:
        msg = "PositionCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b

    @property
    def pose_command(self) -> torch.Tensor:
        """The desired base pose command in the base frame. Shape is (num_envs, 3)."""
        return torch.cat([self.pos_command_b[:, :2], self.vel_command_b[:, 0:1]], dim=1)

    """
    Implementation specific functions.
    """

    def _init_guided_waypoints(self) -> None:
        """Build world-frame waypoint tables from sub-terrain-local poses plus ``terrain_origins``."""
        guided = self.cfg.guided_waypoints_local
        if guided is None or len(guided) == 0:
            return

        terr_orig = getattr(self.terrain, "terrain_origins", None)
        if terr_orig is None:
            raise RuntimeError("guided_waypoints_local requires scene terrain with ``terrain_origins`` populated.")
        if isinstance(terr_orig, torch.Tensor):
            terr_orig_np = terr_orig.detach().cpu().numpy()
        else:
            terr_orig_np = np.asarray(terr_orig)
        if terr_orig_np.ndim != 3 or terr_orig_np.shape[2] != 3:
            raise RuntimeError(
                "guided_waypoints_local expects ``terrain_origins`` shaped (num_rows, num_cols, 3). "
                f"Got shape {terr_orig_np.shape}."
            )

        terrain_generator_cfg = self.terrain.cfg.terrain_generator
        if terrain_generator_cfg is None:
            raise RuntimeError("guided_waypoints_local requires a terrain_generator on the terrain config.")

        proportions = np.array([sub_cfg.proportion for sub_cfg in terrain_generator_cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)
        sub_indices: list[int] = []
        for index in range(terrain_generator_cfg.num_cols):
            sub_index = int(np.min(np.where(index / terrain_generator_cfg.num_cols + 0.001 < np.cumsum(proportions))[0]))
            sub_indices.append(sub_index)
        sub_terrains_names = list(terrain_generator_cfg.sub_terrains.keys())

        for key in guided:
            if key not in sub_terrains_names:
                raise RuntimeError(
                    f"guided_waypoints_local key {key!r} is not a sub_terrain name. "
                    f"Valid names: {sub_terrains_names}"
                )
            if len(guided[key]) == 0:
                raise RuntimeError(f"guided_waypoints_local[{key!r}] must be a non-empty list of (x, y, z) tuples.")

        nrow, ncol = int(terr_orig_np.shape[0]), int(terr_orig_np.shape[1])
        max_wp = max(len(pts) for pts in guided.values())
        terr_orig_t = torch.as_tensor(terr_orig_np, dtype=torch.float32, device=self.device)

        guided_world = torch.zeros(nrow, ncol, max_wp, 3, dtype=torch.float32, device=self.device)
        guided_counts = torch.zeros(nrow, ncol, dtype=torch.long, device=self.device)

        for col in range(ncol):
            name = sub_terrains_names[sub_indices[col]]
            pts = guided.get(name)
            if pts is None:
                continue
            local = torch.tensor(pts, dtype=torch.float32, device=self.device)
            n = int(local.shape[0])
            for level in range(nrow):
                origin = terr_orig_t[level, col]
                guided_world[level, col, :n] = local + origin
                guided_counts[level, col] = n

        self._guided_mode = True
        self._guided_world = guided_world
        self._guided_counts = guided_counts

    def _guided_counts_at_envs(self, env_ids: torch.Tensor) -> torch.Tensor:
        assert self._guided_counts is not None
        levels = self.terrain.terrain_levels[env_ids]
        cols = self.terrain.terrain_types[env_ids]
        return self._guided_counts[levels, cols]

    def _velocity_stop_xy_dist(self) -> torch.Tensor:
        """Per-env distance below which linear/angular commands are zeroed.

        Guided columns use ``guided_arrival_xy_dist`` so the robot is not frozen
        outside the waypoint arrival radius (``target_dis_threshold`` alone can
        be larger and block ever reaching ``guided_arrival_xy_dist``).
        """
        stop = torch.full(
            (self.num_envs,),
            self.cfg.target_dis_threshold,
            device=self.device,
            dtype=torch.float32,
        )
        if self._guided_mode and self._guided_counts is not None:
            env_all = torch.arange(self.num_envs, device=self.device)
            guided_active = self._guided_counts_at_envs(env_all) > 0
            stop = torch.where(
                guided_active,
                torch.full_like(stop, self.cfg.guided_arrival_xy_dist),
                stop,
            )
        return stop

    def _set_guided_positions(self, env_ids: Sequence[int], *, reset_index: bool) -> None:
        assert self._guided_world is not None
        ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if ids_t.numel() == 0:
            return
        if reset_index:
            self._guided_wp_idx[ids_t] = 0
        levels = self.terrain.terrain_levels[ids_t]
        cols = self.terrain.terrain_types[ids_t]
        wp_i = self._guided_wp_idx[ids_t]
        self.pos_command_w[ids_t] = self._guided_world[levels, cols, wp_i]

    def _nv_vec_for_envs(self, ids_t: torch.Tensor) -> torch.Tensor:
        max_np = int(self.valid_targets.shape[2])
        levels = self.terrain.terrain_levels[ids_t]
        tts = self.terrain.terrain_types[ids_t]
        if self._flat_patch_num_valid is not None:
            return self._flat_patch_num_valid[levels, tts].clamp(min=1, max=max_np)
        return torch.full((ids_t.shape[0],), max_np, device=self.device, dtype=torch.long)

    def _flat_patch_arrival_sequence_eligible(self) -> torch.Tensor:
        """Per env: True when ``flat_patch_advance_on_arrival`` applies (fixed-sequence target column, not guided)."""
        if not self.cfg.flat_patch_advance_on_arrival:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        env_all = torch.arange(self.num_envs, device=self.device)
        cols = self.terrain.terrain_types[env_all]
        by_col = self._flat_patch_target_fixed_sequence_by_col[cols]
        if not self._guided_mode or self._guided_counts is None:
            return by_col
        gc = self._guided_counts_at_envs(env_all)
        return by_col & (gc <= 0)

    def _sample_flat_patch_positions(self, env_ids: Sequence[int], *, force_restart: bool = False) -> None:
        ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if ids_t.numel() == 0:
            return
        max_np = int(self.valid_targets.shape[2])
        if max_np <= 0:
            return
        levels = self.terrain.terrain_levels[ids_t]
        tts = self.terrain.terrain_types[ids_t]
        nv_vec = self._nv_vec_for_envs(ids_t)
        if force_restart:
            reset_ep = torch.ones(ids_t.shape[0], dtype=torch.bool, device=self.device)
        else:
            reset_ep = self._env.episode_length_buf[ids_t] <= 1
        self._flat_patch_seq_terminal[ids_t] = torch.where(
            reset_ep, torch.zeros_like(self._flat_patch_seq_terminal[ids_t]), self._flat_patch_seq_terminal[ids_t]
        )
        self._flat_patch_arrival_hold[ids_t] = torch.where(
            reset_ep, torch.zeros_like(self._flat_patch_arrival_hold[ids_t]), self._flat_patch_arrival_hold[ids_t]
        )

        use_seq = self.cfg.flat_patch_advance_on_arrival & self._flat_patch_target_fixed_sequence_by_col[tts]
        if self.cfg.flat_patch_advance_on_arrival:
            term = self._flat_patch_seq_terminal[ids_t]
            ids_last = nv_vec - 1
            kept = self._flat_patch_idx[ids_t]
            reset_idx = self._flat_patch_reset_index_by_col[tts].clamp(min=0)
            reset_idx = torch.minimum(reset_idx, torch.clamp(nv_vec - 1, min=0))
            ids_seq_start = torch.where(use_seq, reset_idx, torch.zeros_like(nv_vec))
            # Episode start: per-column reset index (e.g. far stair low). After final patch: stay on last.
            ids_seq = torch.where(reset_ep, ids_seq_start, torch.where(term, ids_last, kept))
        else:
            ids_seq = torch.zeros(ids_t.shape[0], dtype=torch.long, device=self.device)

        if self.cfg.flat_patch_random_index:
            u = torch.rand(ids_t.shape[0], device=self.device)
            ids_rand = (u * nv_vec.to(dtype=torch.float32)).floor().to(torch.long)
            ids_rand = torch.minimum(ids_rand, nv_vec - 1)
        else:
            ids_rand = torch.zeros(ids_t.shape[0], dtype=torch.long, device=self.device)

        ids = torch.where(use_seq, ids_seq, ids_rand)
        self._flat_patch_idx[ids_t] = ids
        self.pos_command_w[ids_t] = self.valid_targets[levels, tts, ids]

    def _maybe_advance_flat_patch_by_arrival(self) -> None:
        if not self.cfg.flat_patch_advance_on_arrival:
            return
        env_all = torch.arange(self.num_envs, device=self.device)
        use_flat = self._flat_patch_arrival_sequence_eligible()
        nv_vec = self._nv_vec_for_envs(env_all)
        active = use_flat & (nv_vec > 0)
        if not bool(active.any()):
            return

        target_xy = self.pos_command_w[:, :2]
        robot_xy = self.robot.data.root_link_pos_w[:, :2]
        dist = torch.norm(robot_xy - target_xy, dim=1)
        near = active & ~self._flat_patch_seq_terminal & (dist <= self.cfg.flat_patch_arrival_xy_dist)
        self._flat_patch_arrival_hold = torch.where(
            near,
            self._flat_patch_arrival_hold + 1,
            torch.zeros_like(self._flat_patch_arrival_hold),
        )
        advance = active & (self._flat_patch_arrival_hold >= self.cfg.flat_patch_arrival_hold_steps)
        env_ids = advance.nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return

        levels = self.terrain.terrain_levels[env_ids]
        tts = self.terrain.terrain_types[env_ids]
        last_i = nv_vec[env_ids] - 1
        cur = self._flat_patch_idx[env_ids]
        not_last = cur < last_i
        self._flat_patch_idx[env_ids] = torch.where(not_last, cur + 1, cur)
        self._flat_patch_seq_terminal[env_ids] = torch.where(~not_last, True, self._flat_patch_seq_terminal[env_ids])
        new_i = self._flat_patch_idx[env_ids]
        self.pos_command_w[env_ids] = self.valid_targets[levels, tts, new_i]
        self._flat_patch_arrival_hold[env_ids] = 0
        # Latched ``_target_reached`` zeros velocity until resample; clear it so the next patch gets a command.
        self._target_reached[env_ids] = False

        if self.cfg.flat_patch_resample_velocity_on_advance:
            self._resample_velocity_internal([int(i) for i in env_ids.tolist()])

    def _resample_velocity_internal(self, env_ids: Sequence[int]) -> None:
        ids_t = torch.as_tensor(list(env_ids), device=self.device, dtype=torch.long)
        if ids_t.numel() == 0:
            return
        n = int(ids_t.shape[0])
        r = torch.empty(n, device=self.device)
        self.max_command_b[ids_t, 0] = self.lin_vel_x_range[ids_t, 0] + r.uniform_(0.0, 1.0) * (
            self.lin_vel_x_range[ids_t, 1] - self.lin_vel_x_range[ids_t, 0]
        )
        self.max_command_b[ids_t, 1] = self.lin_vel_y_range[ids_t, 0] + r.uniform_(0.0, 1.0) * (
            self.lin_vel_y_range[ids_t, 1] - self.lin_vel_y_range[ids_t, 0]
        )
        self.max_command_b[ids_t, 2] = self.ang_vel_z_range[ids_t, 0] + r.uniform_(0.0, 1.0) * (
            self.ang_vel_z_range[ids_t, 1] - self.ang_vel_z_range[ids_t, 0]
        )
        self.is_standing_env[ids_t] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

        current_batch_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        current_batch_mask[ids_t] = True
        update_mask = current_batch_mask & self.random_velocity_indices
        random_velocity_env_ids = update_mask.nonzero(as_tuple=False).flatten()

        if len(random_velocity_env_ids) > 0:
            self.random_lin_vel_x[random_velocity_env_ids] = self.random_lin_vel_x_range[
                random_velocity_env_ids, 0
            ] + torch.rand(len(random_velocity_env_ids), device=self.device) * (
                self.random_lin_vel_x_range[random_velocity_env_ids, 1]
                - self.random_lin_vel_x_range[random_velocity_env_ids, 0]
            )
            self.random_lin_vel_y[random_velocity_env_ids] = self.random_lin_vel_y_range[
                random_velocity_env_ids, 0
            ] + torch.rand(len(random_velocity_env_ids), device=self.device) * (
                self.random_lin_vel_y_range[random_velocity_env_ids, 1]
                - self.random_lin_vel_y_range[random_velocity_env_ids, 0]
            )
            self.random_ang_vel_z[random_velocity_env_ids] = self.random_ang_vel_z_range[
                random_velocity_env_ids, 0
            ] + torch.rand(len(random_velocity_env_ids), device=self.device) * (
                self.random_ang_vel_z_range[random_velocity_env_ids, 1]
                - self.random_ang_vel_z_range[random_velocity_env_ids, 0]
            )
            self.random_ang_vel_z *= torch.abs(self.random_ang_vel_z) > 0.1

    def _maybe_advance_guided_waypoints(self) -> None:
        if not self._guided_mode or self._guided_world is None or self._guided_counts is None:
            return

        env_all = torch.arange(self.num_envs, device=self.device)
        counts_e = self._guided_counts_at_envs(env_all)
        active = counts_e > 0
        if not bool(active.any()):
            return

        target_xy = self.pos_command_w[:, :2]
        robot_xy = self.robot.data.root_link_pos_w[:, :2]
        dist = torch.norm(robot_xy - target_xy, dim=1)
        near = active & (dist <= self.cfg.guided_arrival_xy_dist)
        self._guided_arrival_hold = torch.where(
            near,
            self._guided_arrival_hold + 1,
            torch.zeros_like(self._guided_arrival_hold),
        )
        advance = active & (self._guided_arrival_hold >= self.cfg.guided_arrival_hold_steps)
        env_ids = advance.nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return

        levels = self.terrain.terrain_levels[env_ids]
        cols = self.terrain.terrain_types[env_ids]
        counts = self._guided_counts[levels, cols]
        if self.cfg.guided_loop_waypoints:
            new_idx = (self._guided_wp_idx[env_ids] + 1) % torch.clamp(counts, min=1)
        else:
            new_idx = torch.minimum(self._guided_wp_idx[env_ids] + 1, counts - 1)

        self._guided_wp_idx[env_ids] = new_idx
        self.pos_command_w[env_ids] = self._guided_world[levels, cols, new_idx]
        self._guided_arrival_hold[env_ids] = 0
        self._target_reached[env_ids] = False

        if self.cfg.guided_resample_velocity_on_advance:
            self._resample_velocity_internal([int(i) for i in env_ids.tolist()])

    def _update_metrics(self):
        # logs data
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1)
            / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2]) / max_command_step
        )
        lin_vel_error = torch.sum(
            torch.square(self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2]),
            dim=1,
        )
        self.metrics["tracking_exp_vel_xy"] += (
            torch.exp(-lin_vel_error / self.cfg.lin_vel_metrics_std**2) / self._env.max_episode_length
        )
        angular_vel_error = torch.square(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2])
        self.metrics["tracking_exp_vel_yaw"] += (
            torch.exp(-angular_vel_error / self.cfg.ang_vel_metrics_std**2) / self._env.max_episode_length
        )
        target_vec = self.pos_command_w - self.robot.data.root_link_pos_w[:, :3]
        target_dist = torch.norm(target_vec[:, :2], dim=1)
        cmd_lin_norm = torch.norm(self.vel_command_b[:, :2], dim=1)
        cmd_nonzero = torch.logical_or(cmd_lin_norm > 1e-6, torch.abs(self.vel_command_b[:, 2]) > 1e-6)
        self.metrics["command_nonzero_ratio"] += cmd_nonzero.float() / self._env.max_episode_length
        self.metrics["target_near_ratio"] += (
            target_dist <= self.cfg.target_dis_threshold
        ).float() / self._env.max_episode_length
        self.metrics["standing_env_ratio"] += self.is_standing_env.float() / self._env.max_episode_length
        self.metrics["random_velocity_env_ratio"] += self.random_velocity_indices.float() / self._env.max_episode_length

    def _resample_command(self, env_ids: Sequence[int]):
        ids_t = torch.as_tensor(list(env_ids), device=self.device, dtype=torch.long)
        if ids_t.numel() == 0:
            return
        self._target_reached[ids_t] = False

        if self._guided_mode and self._guided_world is not None:
            counts_e = self._guided_counts_at_envs(ids_t)
            guided_ok = counts_e > 0
            if bool((~guided_ok).any()):
                self._sample_flat_patch_positions([int(i) for i in ids_t[~guided_ok].tolist()])
        else:
            self._sample_flat_patch_positions([int(i) for i in ids_t.tolist()])

        self._resample_velocity_internal([int(i) for i in ids_t.tolist()])

    def reset(self, env_ids: Sequence[int] | None = None) -> dict:
        """Episode reset: restart flat-patch / guided sequence (``episode_length_buf`` may not be updated yet)."""
        if env_ids is None:
            ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            ids_t = torch.as_tensor(list(env_ids), device=self.device, dtype=torch.long)
        if ids_t.numel() > 0:
            self._target_reached[ids_t] = False
            self._flat_patch_arrival_hold[ids_t] = 0
            self._flat_patch_seq_terminal[ids_t] = False
            self._flat_patch_idx[ids_t] = 0
            if self._guided_mode and self._guided_world is not None:
                self._guided_arrival_hold[ids_t] = 0
                counts_e = self._guided_counts_at_envs(ids_t)
                guided_ok = counts_e > 0
                if bool(guided_ok.any()):
                    self._guided_wp_idx[ids_t[guided_ok]] = 0
                    self._set_guided_positions([int(i) for i in ids_t[guided_ok].tolist()], reset_index=True)
                if bool((~guided_ok).any()):
                    self._sample_flat_patch_positions(
                        [int(i) for i in ids_t[~guided_ok].tolist()], force_restart=True
                    )
            else:
                self._sample_flat_patch_positions([int(i) for i in ids_t.tolist()], force_restart=True)

        base_reset = getattr(super(), "reset", None)
        if callable(base_reset):
            out = base_reset(env_ids)
            return out if isinstance(out, dict) else {}
        return {}

    def _update_command(self):
        """Re-target the position command to the current root state."""
        self._maybe_advance_guided_waypoints()
        self._maybe_advance_flat_patch_by_arrival()

        target_vec = self.pos_command_w - self.robot.data.root_link_pos_w[:, :3]
        target_dist = torch.norm(target_vec[:, :2], dim=1)
        self.pos_command_b[:] = quat_rotate_inverse(yaw_quat(self.robot.data.root_link_quat_w), target_vec)
        self.vel_command_b[:, :2] = self.pos_command_b[:, :2] * self.cfg.velocity_control_stiffness

        # set heading command to point towards target
        target_vec = self.pos_command_w - self.robot.data.root_link_pos_w
        target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])

        # compute errors to find the closest direction to the current heading
        # this is done to avoid the discontinuity at the -pi/pi boundary
        self.heading_command_w = wrap_to_pi(target_direction - self.robot.data.heading_w)

        self.vel_command_b[:, 2] = self.heading_command_w * self.cfg.heading_control_stiffness

        if self.cfg.yaw_vel_gate_threshold is not None:
            yaw_gate_mask = self.heading_command_w.abs() > self.cfg.yaw_vel_gate_threshold
            self.vel_command_b[yaw_gate_mask, :2] = 0.0

        # Descent slowdown: shrink the forward speed cap while the base is steadily sinking,
        # so stair descent takes short steps instead of long ones that skip a tread / land on an edge.
        base_vz = self.robot.data.root_link_lin_vel_w[:, 2]
        self._base_vz_ema = (
            1.0 - self.cfg.descend_vz_ema_alpha
        ) * self._base_vz_ema + self.cfg.descend_vz_ema_alpha * base_vz
        descending = self.descend_speed_indices & (self._base_vz_ema < self.cfg.descend_vz_threshold)
        forward_cap = self.max_command_b[:, 0] * torch.where(
            descending,
            torch.full_like(self._base_vz_ema, self.cfg.descend_speed_scale),
            torch.ones_like(self._base_vz_ema),
        )

        # scale linear velocity so the dominant axis hits its limit and
        # the other axis preserves its ratio
        vx = self.vel_command_b[:, 0]
        vy = self.vel_command_b[:, 1]
        min_x = -forward_cap if not self.cfg.only_positive_lin_vel_x else torch.zeros_like(forward_cap)
        min_y = -self.max_command_b[:, 1]
        max_x = forward_cap
        max_y = self.max_command_b[:, 1]
        eps = 1e-6

        abs_vx = vx.abs()
        abs_vy = vy.abs()

        if not self.cfg.only_positive_lin_vel_x:
            # clamp each axis independently
            clamped_vx = torch.clamp(abs_vx, min=min_x, max=max_x)
            clamped_vy = torch.clamp(abs_vy, min=min_y, max=max_y)

            # compute scale for whichever axis is dominant
            scale_x = clamped_vx / (abs_vx + eps)
            scale_y = clamped_vy / (abs_vy + eps)
            scale = torch.where(abs_vx >= abs_vy, scale_x, scale_y)

            # apply scale and restore sign
            self.vel_command_b[:, 0] = vx * scale
            self.vel_command_b[:, 1] = vy * scale

        else:
            self.vel_command_b[:, 0] = torch.clamp(vx, min=min_x, max=max_x)
            self.vel_command_b[:, 1] = torch.clamp(vy, min=min_y, max=max_y)

        self.vel_command_b[:, 2] = torch.clamp(
            self.vel_command_b[:, 2],
            self.ang_vel_z_range[:, 0],
            self.ang_vel_z_range[:, 1],
        )
        random_velocity_env_ids = self.random_velocity_indices.nonzero(as_tuple=False).flatten()
        self.vel_command_b[random_velocity_env_ids, 0] = self.random_lin_vel_x[random_velocity_env_ids]
        self.vel_command_b[random_velocity_env_ids, 1] = self.random_lin_vel_y[random_velocity_env_ids]
        self.vel_command_b[random_velocity_env_ids, 2] = self.random_ang_vel_z[random_velocity_env_ids]

        stop_dist = self._velocity_stop_xy_dist()
        self._target_reached |= target_dist <= stop_dist
        self.vel_command_b[:] *= (~self._target_reached).unsqueeze(-1)
        self.vel_command_b[:, :2] *= (
            (torch.norm(self.vel_command_b[:, :2], dim=1) > self.cfg.lin_vel_threshold).float().unsqueeze(-1)
        )
        self.vel_command_b[:, 2] *= (torch.abs(self.vel_command_b[:, 2]) > self.cfg.ang_vel_threshold).float()
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

    def _debug_vis_impl(self, visualizer) -> None:
        """Draw target positions and velocity arrows using MuJoCo geometry.

        This replaces the legacy marker path with native MuJoCo visualization.
        """
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return

        # Convert to numpy for visualization
        pos_commands_w = self.pos_command_w.cpu().numpy()
        vel_commands_b = self.vel_command_b.cpu().numpy()
        base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
        base_quat_ws = self.robot.data.root_link_quat_w.cpu().numpy()
        lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()
        ang_vel_bs = self.robot.data.root_link_ang_vel_b.cpu().numpy()

        # Visualization parameters
        goal_radius = self.cfg.target_dis_threshold
        goal_height = 0.1
        patch_height = 0.05
        arrow_z_offset = 0.5
        arrow_scale = 0.5
        tangential_radius = 0.35
        ang_hub_z_extra = 0.12

        cref_reward_term = None
        if self.cfg.show_cref_foothold_candidates:
            cref_reward_term = _cref_reward_term_from_manager(
                getattr(self._env, "reward_manager", None),
                self.cfg.cref_foothold_reward_name,
            )
            if cref_foothold_reward_visualizable(cref_reward_term):
                origins_t = getattr(cref_reward_term, "terrain_origins", None)
                stakes_xy_t = getattr(cref_reward_term, "stake_centers_xy_local", None)
                stake_counts_t = getattr(cref_reward_term, "stake_counts", None)
                stake_z_t = getattr(cref_reward_term, "stake_z_local", None)
                treads_t = getattr(cref_reward_term, "treads_xz_local", None)
                tread_counts_t = getattr(cref_reward_term, "tread_counts", None)
                if (
                    self.cfg.show_cref_foothold_all_treads
                    and isinstance(origins_t, torch.Tensor)
                    and isinstance(stakes_xy_t, torch.Tensor)
                    and isinstance(stake_counts_t, torch.Tensor)
                    and isinstance(stake_z_t, torch.Tensor)
                ):
                    for label, center in _cref_all_stake_marker_centers(
                        origins_t, stakes_xy_t, stake_counts_t, stake_z_t
                    ):
                        visualizer.add_sphere(
                            center=center,
                            radius=0.045,
                            color=(0.05, 1.0, 0.25, 0.85),
                            label=label,
                        )
                elif (
                    self.cfg.show_cref_foothold_all_treads
                    and isinstance(origins_t, torch.Tensor)
                    and isinstance(treads_t, torch.Tensor)
                    and isinstance(tread_counts_t, torch.Tensor)
                ):
                    for label, center in _cref_all_tread_marker_centers(origins_t, treads_t, tread_counts_t):
                        visualizer.add_sphere(
                            center=center,
                            radius=0.045,
                            color=(0.05, 1.0, 0.25, 0.85),
                            label=label,
                        )

        for batch in env_indices:
            # Skip if robot appears uninitialized (at origin)
            if np.linalg.norm(base_pos_ws[batch]) < 1e-6:
                continue

            # 1. Draw target position as red cylinder
            goal_pos = pos_commands_w[batch]
            goal_start = goal_pos.copy()
            goal_end = goal_pos.copy()
            goal_end[2] += goal_height
            visualizer.add_cylinder(
                start=goal_start,
                end=goal_end,
                radius=goal_radius,
                color=(1.0, 0.0, 0.0, 0.6),
                label=f"goal_{batch}",
            )

            # 2. Draw commanded velocity arrow (green)
            base_pos = base_pos_ws[batch]
            arrow_start = base_pos.copy()
            arrow_start[2] += arrow_z_offset

            # Convert velocity command from body frame to world frame
            vel_cmd_b = vel_commands_b[batch]
            quat_w = base_quat_ws[batch]
            # Simple rotation: only yaw matters for horizontal velocity
            yaw = np.arctan2(
                2.0 * (quat_w[0] * quat_w[3] + quat_w[1] * quat_w[2]), 1.0 - 2.0 * (quat_w[2] ** 2 + quat_w[3] ** 2)
            )
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            vel_cmd_w = np.array(
                [cos_yaw * vel_cmd_b[0] - sin_yaw * vel_cmd_b[1], sin_yaw * vel_cmd_b[0] + cos_yaw * vel_cmd_b[1], 0.0]
            )

            arrow_end = arrow_start + vel_cmd_w * arrow_scale
            visualizer.add_arrow(
                start=arrow_start,
                end=arrow_end,
                color=(0.1, 1.0, 0.1, 0.8),  # Bright green
                width=0.02,
                label=f"cmd_vel_{batch}",
            )

            # 3. Draw actual velocity arrow (blue)
            lin_vel_b = lin_vel_bs[batch]
            vel_actual_w = np.array(
                [cos_yaw * lin_vel_b[0] - sin_yaw * lin_vel_b[1], sin_yaw * lin_vel_b[0] + cos_yaw * lin_vel_b[1], 0.0]
            )

            arrow_end_actual = arrow_start + vel_actual_w * arrow_scale
            visualizer.add_arrow(
                start=arrow_start,
                end=arrow_end_actual,
                color=(0.1, 0.1, 1.0, 0.8),  # Bright blue
                width=0.02,
                label=f"actual_vel_{batch}",
            )

            # Tangential yaw-rate arrows at body +x (ω×r hub); +ω_z → body +y (CCW from above).
            ang_hub = base_pos.copy()
            ang_hub[0] += cos_yaw * tangential_radius
            ang_hub[1] += sin_yaw * tangential_radius
            ang_hub[2] += arrow_z_offset + ang_hub_z_extra
            tang_dir_w = np.array([-sin_yaw, cos_yaw, 0.0])

            def _draw_tangential_ang_vel(omega_z: float, color: tuple[float, float, float, float], label: str) -> None:
                if abs(omega_z) < 1e-5:
                    return
                arc_len = abs(omega_z) * tangential_radius * arrow_scale
                tang = tang_dir_w * (np.sign(omega_z) * arc_len)
                visualizer.add_arrow(
                    start=ang_hub,
                    end=ang_hub + tang,
                    color=color,
                    width=0.018,
                    label=label,
                )

            _draw_tangential_ang_vel(vel_cmd_b[2], (1.0, 0.95, 0.1, 0.85), f"cmd_ang_vel_{batch}")
            _draw_tangential_ang_vel(ang_vel_bs[batch][2], (1.0, 0.1, 0.75, 0.85), f"actual_ang_vel_{batch}")

            # 4. Flat-patch target candidates (terrain flat_patches["target"]) for this env's (level, type).
            if self.cfg.patch_vis or self.cfg.show_flat_patch_candidates:
                valid_targets = self.valid_targets.cpu().numpy()
                terrain_level = self.terrain.terrain_levels[batch].item()
                terrain_type = self.terrain.terrain_types[batch].item()
                patches = valid_targets[terrain_level, terrain_type]
                n_p_cell = int(patches.shape[0])
                if self._flat_patch_num_valid is not None:
                    n_p_cell = int(self._flat_patch_num_valid[terrain_level, terrain_type].item())
                    n_p_cell = max(1, min(n_p_cell, int(patches.shape[0])))
                n_cap = self.cfg.flat_patch_target_count
                if n_cap is None:
                    n_cap = n_p_cell
                else:
                    n_cap = min(int(n_cap), n_p_cell)
                patch_radius_vis = float(goal_radius) * 0.5
                cur_patch_i = int(self._flat_patch_idx[batch].item())
                for i in range(n_cap):
                    patch_pos = patches[i]
                    if not np.all(np.isfinite(patch_pos)):
                        break
                    if i > 0 and np.allclose(patch_pos, patches[i - 1], atol=2e-3, rtol=0.0):
                        break
                    patch_start = patch_pos.copy()
                    patch_end = patch_pos.copy()
                    patch_end[2] += patch_height
                    is_current = i == cur_patch_i
                    rgba = ((1.0, 0.45, 0.05, 0.75) if is_current else (0.0, 0.35, 1.0, 0.55))
                    visualizer.add_cylinder(
                        start=patch_start,
                        end=patch_end,
                        radius=patch_radius_vis,
                        color=rgba,
                        label=f"flat_target_{batch}_i{i}{'_current' if is_current else ''}",
                    )

            # 4b. CReF-style foothold candidates from the stateful reward term.
            if self.cfg.show_cref_foothold_candidates:
                reward_term = cref_reward_term
                if cref_foothold_reward_visualizable(reward_term):
                    terrain_level = int(self.terrain.terrain_levels[batch].item())
                    terrain_type = int(self.terrain.terrain_types[batch].item())
                    origins_t = getattr(reward_term, "terrain_origins", None)
                    frozen_xy_t = getattr(reward_term, "frozen_candidates_xy", None)
                    frozen_xz_t = getattr(reward_term, "frozen_candidates_xz", None)
                    frozen_counts_t = getattr(reward_term, "frozen_candidate_counts", None)
                    stake_z_t = getattr(reward_term, "stake_z_local", None)
                    if (
                        isinstance(origins_t, torch.Tensor)
                        and isinstance(frozen_counts_t, torch.Tensor)
                    ):
                        origin = origins_t[terrain_level, terrain_type].detach().cpu().numpy()
                        frozen_counts = frozen_counts_t[batch].detach().cpu().numpy()
                        foot_colors = (
                            (1.0, 0.85, 0.05, 0.9),
                            (0.9, 0.1, 1.0, 0.9),
                        )
                        if isinstance(frozen_xy_t, torch.Tensor):
                            frozen_xy = frozen_xy_t[batch].detach().cpu().numpy()
                            for foot_idx in range(frozen_xy.shape[0]):
                                n_frozen = int(frozen_counts[foot_idx])
                                if n_frozen <= 0:
                                    continue
                                color = foot_colors[min(foot_idx, len(foot_colors) - 1)]
                                for i in range(n_frozen):
                                    candidate_xy = frozen_xy[foot_idx, i]
                                    z_local = 0.0
                                    if isinstance(stake_z_t, torch.Tensor):
                                        cell_z = stake_z_t[terrain_level, terrain_type].detach().cpu().numpy()
                                        z_local = float(np.max(cell_z)) if cell_z.size > 0 else 0.0
                                    center = np.array(
                                        [
                                            origin[0] + candidate_xy[0],
                                            origin[1] + candidate_xy[1],
                                            origin[2] + z_local + 0.08,
                                        ],
                                        dtype=np.float64,
                                    )
                                    visualizer.add_sphere(
                                        center=center,
                                        radius=0.045,
                                        color=color,
                                        label=f"cref_frozen_{batch}_foot{foot_idx}_{i}",
                                    )
                        elif (
                            isinstance(frozen_xz_t, torch.Tensor)
                        ):
                            frozen_xz = frozen_xz_t[batch].detach().cpu().numpy()
                            foot_y_offsets = (-0.12, 0.12)
                            for foot_idx in range(frozen_xz.shape[0]):
                                n_frozen = int(frozen_counts[foot_idx])
                                if n_frozen <= 0:
                                    continue
                                color = foot_colors[min(foot_idx, len(foot_colors) - 1)]
                                y_offset = foot_y_offsets[min(foot_idx, len(foot_y_offsets) - 1)]
                                for i in range(n_frozen):
                                    candidate_xz = frozen_xz[foot_idx, i]
                                    center = np.array(
                                        [origin[0] + candidate_xz[0], origin[1] + y_offset, candidate_xz[1] + 0.08],
                                        dtype=np.float64,
                                    )
                                    visualizer.add_sphere(
                                        center=center,
                                        radius=0.045,
                                        color=color,
                                        label=f"cref_frozen_{batch}_foot{foot_idx}_{i}",
                                    )

            # 5. Draw full guided waypoint sequence for this env's (level, type) cell
            if self._guided_mode and self._guided_world is not None and self._guided_counts is not None:
                terrain_level = self.terrain.terrain_levels[batch].item()
                terrain_type = self.terrain.terrain_types[batch].item()
                n_wp = int(self._guided_counts[terrain_level, terrain_type].item())
                if n_wp > 0:
                    wp_xyz = self._guided_world[terrain_level, terrain_type, :n_wp].detach().cpu().numpy()
                    cur_i = int(self._guided_wp_idx[batch].item())
                    wp_radius = float(goal_radius) * 0.55
                    for i, wp in enumerate(wp_xyz):
                        if float(np.linalg.norm(wp)) < 1e-9:
                            continue
                        wp_start = wp.copy()
                        wp_end = wp.copy()
                        wp_end[2] += patch_height
                        is_active = i == cur_i
                        rgba = (
                            (1.0, 0.55, 0.05, 0.65) if is_active else (0.05, 0.85, 0.9, 0.4)
                        )
                        visualizer.add_cylinder(
                            start=wp_start,
                            end=wp_end,
                            radius=wp_radius,
                            color=rgba,
                            label=f"guided_wp_{batch}_{i}",
                        )
