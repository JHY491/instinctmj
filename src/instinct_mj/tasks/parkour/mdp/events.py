from __future__ import annotations

from typing import TYPE_CHECKING

import mjlab.envs.mdp as envs_mdp
import torch
from mjlab.managers import SceneEntityCfg
from mjlab.utils.lab_api import math as math_utils
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

from instinct_mj.terrains.height_field.hf_terrains_cfg import (
    terrain_column_reset_root_xy_offset,
    terrain_column_target_uses_fixed_sequence_flat_patch,
)

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv


def randomize_rigid_body_material(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    static_friction_range: tuple[float, float] = (0.3, 1.0),
    dynamic_friction_range: tuple[float, float] = (0.3, 1.0),
    make_consistent: bool = True,
) -> None:
    slide_friction_range = (
        min(static_friction_range[0], dynamic_friction_range[0]),
        max(static_friction_range[1], dynamic_friction_range[1]),
    )
    envs_mdp.dr.geom_friction(
        env=env,
        env_ids=env_ids,
        ranges=slide_friction_range,
        operation="abs",
        asset_cfg=asset_cfg,
        shared_random=make_consistent,
    )


def push_by_setting_velocity_without_stand(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    velocity_threshold: float = 0.15,
) -> None:
    """Push the asset by setting root velocity. No push when standing still."""
    asset: Entity = env.scene[asset_cfg.name]
    vel_w = asset.data.root_link_vel_w[env_ids]

    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)
    add_vel = sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=env.device)

    command = env.command_manager.get_command(command_name)
    if command is None:
        raise RuntimeError(f"Command {command_name!r} is required for push event.")
    cmd = command[env_ids]
    lin_vel = torch.norm(cmd[:, :2], dim=1) > velocity_threshold
    ang_vel = torch.abs(cmd[:, 2]) > velocity_threshold

    should_push = torch.logical_or(lin_vel, ang_vel).float().unsqueeze(-1)
    vel_w += add_vel * should_push
    asset.write_root_link_velocity_to_sim(vel_w, env_ids=env_ids)


def reset_root_state_on_target_patch(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    patch_name: str = "target",
    pose_range: dict[str, tuple[float, float]] | None = None,
    velocity_range: dict[str, tuple[float, float]] | None = None,
    align_z_to_patch: bool = True,
    align_z_to_spawn: bool | None = None,
    edge_spawn_xy_offset: tuple[float, float] = (-4.0, 0.0),
    center_spawn_xy_offset: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Reset root pose using per-column spawn xy and flat-patch ground height.

    Spawn x/y come from ``terrain_column_reset_root_xy_offset`` (edge for course terrains like
    one-sided stairs, center for perlin noise). Z comes from a sampled ``flat_patches[patch_name]``
    surface plus ``init_state.pos.z`` when ``align_z_to_patch`` is True. Fixed-sequence columns
    always use flat-patch index 0 for spawn height so the robot does not drop from mid-course z.
    """
    if env_ids.numel() == 0:
        return

    align_z = align_z_to_patch if align_z_to_spawn is None else align_z_to_spawn

    asset: Entity = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    if terrain is None:
        raise RuntimeError("Terrain entity is required for target-patch reset.")
    flat_patches = terrain.flat_patches
    if patch_name not in flat_patches:
        raise RuntimeError(f"Patch {patch_name!r} not found in terrain.flat_patches: {list(flat_patches.keys())}")

    targets: torch.Tensor = flat_patches[patch_name]
    default_root_state = asset.data.default_root_state[env_ids]
    env_origins = env.scene.env_origins[env_ids]
    levels = terrain.terrain_levels[env_ids]
    tts = terrain.terrain_types[env_ids]

    tgen = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    col_xy_offsets = terrain_column_reset_root_xy_offset(
        tgen,
        edge_offset=edge_spawn_xy_offset,
        center_offset=center_spawn_xy_offset,
    )
    ncol = int(targets.shape[1])
    if len(col_xy_offsets) < ncol:
        col_xy_offsets = col_xy_offsets + [edge_spawn_xy_offset] * (ncol - len(col_xy_offsets))
    else:
        col_xy_offsets = col_xy_offsets[:ncol]
    spawn_xy = torch.tensor(col_xy_offsets, dtype=torch.float32, device=env.device)[tts]

    base_pos_w = env_origins + default_root_state[:, 0:3]
    base_pos_w[:, 0] = env_origins[:, 0] + spawn_xy[:, 0]
    base_pos_w[:, 1] = env_origins[:, 1] + spawn_xy[:, 1]

    max_np = int(targets.shape[2])
    fnv_map = getattr(terrain, "flat_patch_num_valid", None)
    if fnv_map is not None and patch_name in fnv_map:
        nv = fnv_map[patch_name][levels, tts].clamp(min=1, max=max_np)
        u = torch.rand(len(env_ids), device=env.device)
        patch_ids = (u * nv.to(dtype=torch.float32)).floor().to(torch.long)
        patch_ids = torch.minimum(patch_ids, nv - 1)
    else:
        patch_ids = torch.randint(0, max_np, (len(env_ids),), device=env.device)

    fixed_seq_cols = terrain_column_target_uses_fixed_sequence_flat_patch(tgen)
    if fixed_seq_cols:
        col_flags = torch.tensor(fixed_seq_cols, dtype=torch.bool, device=env.device)
        if col_flags.shape[0] == ncol:
            on_fixed_seq = col_flags[tts]
            patch_ids = torch.where(on_fixed_seq, torch.zeros_like(patch_ids), patch_ids)

    if align_z:
        target_pos_w = targets[levels, tts, patch_ids]
        base_pos_w[:, 2] = target_pos_w[:, 2] + default_root_state[:, 2]

    base_quat_w = default_root_state[:, 3:7].clone()
    base_lin_vel_w = default_root_state[:, 7:10].clone()
    base_ang_vel_w = default_root_state[:, 10:13].clone()

    if pose_range is not None:
        pose_ranges = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        pose_ranges_t = torch.tensor(pose_ranges, device=env.device)
        pose_samples = sample_uniform(pose_ranges_t[:, 0], pose_ranges_t[:, 1], (len(env_ids), 6), device=env.device)
        base_pos_w += pose_samples[:, 0:3]
        quat_delta = math_utils.quat_from_euler_xyz(pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5])
        base_quat_w = math_utils.quat_mul(quat_delta, base_quat_w)

    if velocity_range is not None:
        vel_ranges = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        vel_ranges_t = torch.tensor(vel_ranges, device=env.device)
        vel_samples = sample_uniform(vel_ranges_t[:, 0], vel_ranges_t[:, 1], (len(env_ids), 6), device=env.device)
        base_lin_vel_w += vel_samples[:, 0:3]
        base_ang_vel_w += vel_samples[:, 3:6]

    asset.write_root_link_pose_to_sim(torch.cat([base_pos_w, base_quat_w], dim=-1), env_ids=env_ids)
    asset.write_root_link_velocity_to_sim(torch.cat([base_lin_vel_w, base_ang_vel_w], dim=-1), env_ids=env_ids)


def reset_root_state_from_flat_patches(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    patch_name: str = "spawn",
    pose_range: dict[str, tuple[float, float]] | None = None,
    velocity_range: dict[str, tuple[float, float]] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    use_patch_heading: bool = True,
) -> None:
    """Reset on a flat patch; apply stored centerline yaw when ``flat_patch_headings`` exists."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    terrain = env.scene.terrain
    if terrain is None or patch_name not in terrain.flat_patches:
        envs_mdp.reset_root_state_uniform(
            env,
            env_ids,
            pose_range=pose_range or {},
            velocity_range=velocity_range,
            asset_cfg=asset_cfg,
        )
        return

    patches = terrain.flat_patches[patch_name]
    num_patches = patches.shape[2]

    levels = terrain.terrain_levels[env_ids]
    types = terrain.terrain_types[env_ids]

    fnv_map = getattr(terrain, "flat_patch_num_valid", None)
    if fnv_map is not None and patch_name in fnv_map:
        max_np = int(patches.shape[2])
        nv = fnv_map[patch_name][levels, types].clamp(min=1, max=max_np)
        u = torch.rand(len(env_ids), device=env.device)
        patch_ids = (u * nv.to(dtype=torch.float32)).floor().to(torch.long)
        patch_ids = torch.minimum(patch_ids, nv - 1)
    else:
        patch_ids = torch.randint(0, num_patches, (len(env_ids),), device=env.device)

    positions = patches[levels, types, patch_ids]

    asset: Entity = env.scene[asset_cfg.name]
    default_root_state = asset.data.default_root_state
    assert default_root_state is not None
    root_states = default_root_state[env_ids].clone()

    if pose_range is None:
        pose_range = {}
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)
    pose_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)

    final_positions = positions.clone()
    final_positions[:, 0] += pose_samples[:, 0]
    final_positions[:, 1] += pose_samples[:, 1]
    final_positions[:, 2] += root_states[:, 2] + pose_samples[:, 2]

    headings_map = getattr(terrain, "flat_patch_headings", None)
    patch_yaw: torch.Tensor | None = None
    if use_patch_heading and headings_map is not None and patch_name in headings_map:
        patch_yaw = headings_map[patch_name][levels, types, patch_ids]

    if patch_yaw is not None:
        has_heading = ~torch.isnan(patch_yaw)
        yaw_w = patch_yaw + pose_samples[:, 5]
        orientations_heading = quat_from_euler_xyz(pose_samples[:, 3], pose_samples[:, 4], yaw_w)
        orientations_delta = quat_from_euler_xyz(pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5])
        orientations_default = quat_mul(root_states[:, 3:7], orientations_delta)
        orientations = torch.where(
            has_heading.unsqueeze(-1),
            orientations_heading,
            orientations_default,
        )
    else:
        orientations_delta = quat_from_euler_xyz(pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5])
        orientations = quat_mul(root_states[:, 3:7], orientations_delta)

    if asset.is_fixed_base:
        if not asset.is_mocap:
            raise ValueError(
                f"Cannot reset root state for fixed-base non-mocap entity '{asset_cfg.name}'."
            )
        asset.write_mocap_pose_to_sim(
            torch.cat([final_positions, orientations], dim=-1),
            env_ids=env_ids,
        )
        return

    if velocity_range is None:
        velocity_range = {}
    vel_range_list = [
        velocity_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    vel_ranges = torch.tensor(vel_range_list, device=env.device)
    vel_samples = sample_uniform(vel_ranges[:, 0], vel_ranges[:, 1], (len(env_ids), 6), device=env.device)
    velocities = root_states[:, 7:13] + vel_samples

    asset.write_root_link_pose_to_sim(
        torch.cat([final_positions, orientations], dim=-1),
        env_ids=env_ids,
    )
    asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)
