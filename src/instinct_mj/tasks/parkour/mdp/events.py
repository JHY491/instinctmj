from __future__ import annotations

from typing import TYPE_CHECKING

import mjlab.envs.mdp as envs_mdp
import torch
from mjlab.managers import SceneEntityCfg
from mjlab.utils.lab_api import math as math_utils
from mjlab.utils.lab_api.math import sample_uniform

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
) -> None:
    """Reset root pose/velocity by sampling terrain target flat patches."""
    if env_ids.numel() == 0:
        return

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
    base_pos_w = env_origins + default_root_state[:, 0:3]

    patch_ids = torch.randint(0, targets.shape[2], (len(env_ids),), device=env.device)
    target_pos_w = targets[terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids], patch_ids]
    # Keep regular spawn x/y, align z to target patch plus pelvis/root offset.
    # The offset comes from robot default root z (driven by init_state.pos.z).
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
