from __future__ import annotations

from typing import TYPE_CHECKING

import mjlab.envs.mdp as envs_mdp
import numpy as np
import torch
import trimesh
from mjlab.managers import SceneEntityCfg
from mjlab.utils.lab_api import math as math_utils
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

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


def _terrain_height_query_mesh(terrain: object) -> trimesh.Trimesh | None:
    """Return a world-frame terrain surface mesh for spawn-height raycasts."""
    cached = getattr(terrain, "_height_query_mesh", None)
    if isinstance(cached, trimesh.Trimesh):
        return cached
    terrain_gen = getattr(terrain, "terrain_generator", None)
    if terrain_gen is None:
        return None
    mesh = getattr(terrain_gen, "terrain_mesh", None)
    if isinstance(mesh, trimesh.Trimesh):
        return mesh
    return None


def _terrain_surface_z_at_xy_w(env: ManagerBasedRlEnv, xy_w: torch.Tensor) -> torch.Tensor | None:
    """Raycast terrain surface height (world z) at each (x, y) spawn location."""
    terrain = env.scene.terrain
    if terrain is None:
        return None
    mesh = _terrain_height_query_mesh(terrain)
    if mesh is None:
        return None

    intersector = getattr(terrain, "_spawn_height_intersector", None)
    if intersector is None:
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
        terrain._spawn_height_intersector = intersector

    xy_np = xy_w.detach().cpu().numpy()
    n = int(xy_np.shape[0])
    z_hi = float(np.max(mesh.vertices[:, 2])) + 5.0
    origins = np.column_stack([xy_np[:, 0], xy_np[:, 1], np.full(n, z_hi, dtype=np.float64)])
    directions = np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float64), (n, 1))
    locations, ray_ids, _ = intersector.intersects_location(
        ray_origins=origins,
        ray_directions=directions,
    )
    if len(ray_ids) == 0:
        return None

    surface_z_np = np.full(n, np.nan, dtype=np.float64)
    for hit_xyz, ray_i in zip(locations, ray_ids):
        ray_i = int(ray_i)
        hit_z = float(hit_xyz[2])
        if np.isnan(surface_z_np[ray_i]) or hit_z > surface_z_np[ray_i]:
            surface_z_np[ray_i] = hit_z
    if np.isnan(surface_z_np).any():
        return None
    return torch.tensor(surface_z_np, device=env.device, dtype=xy_w.dtype)


def reset_root_state_on_target_patch(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    patch_name: str = "target",
    pose_range: dict[str, tuple[float, float]] | None = None,
    velocity_range: dict[str, tuple[float, float]] | None = None,
    align_z_to_spawn: bool = True,
) -> None:
    """Reset root pose/velocity at the default spawn offset, with optional terrain z alignment.

    Spawn x/y come from ``env_origin + init_state.pos`` plus optional ``pose_range`` jitter.
    When ``align_z_to_spawn`` is True, root z is set to the terrain surface under that xy
    plus ``init_state.pos.z`` (pelvis height above ground), not a flat-patch target height.
    """
    if env_ids.numel() == 0:
        return

    asset: Entity = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    if terrain is None:
        raise RuntimeError("Terrain entity is required for target-patch reset.")
    flat_patches = terrain.flat_patches
    if patch_name not in flat_patches:
        raise RuntimeError(f"Patch {patch_name!r} not found in terrain.flat_patches: {list(flat_patches.keys())}")

    default_root_state = asset.data.default_root_state[env_ids]
    env_origins = env.scene.env_origins[env_ids]
    base_pos_w = env_origins + default_root_state[:, 0:3]

    pose_samples: torch.Tensor | None = None
    if pose_range is not None:
        pose_ranges = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        pose_ranges_t = torch.tensor(pose_ranges, device=env.device)
        pose_samples = sample_uniform(pose_ranges_t[:, 0], pose_ranges_t[:, 1], (len(env_ids), 6), device=env.device)
        base_pos_w[:, 0] += pose_samples[:, 0]
        base_pos_w[:, 1] += pose_samples[:, 1]
        if not align_z_to_spawn:
            base_pos_w[:, 2] += pose_samples[:, 2]

    if align_z_to_spawn:
        surface_z = _terrain_surface_z_at_xy_w(env, base_pos_w[:, :2])
        if surface_z is not None:
            z_pose = pose_samples[:, 2] if pose_samples is not None else 0.0
            base_pos_w[:, 2] = surface_z + default_root_state[:, 2] + z_pose

    base_quat_w = default_root_state[:, 3:7].clone()
    base_lin_vel_w = default_root_state[:, 7:10].clone()
    base_ang_vel_w = default_root_state[:, 10:13].clone()

    if pose_samples is not None:
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
