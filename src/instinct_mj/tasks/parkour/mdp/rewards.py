from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import torch
from mjlab.managers import SceneEntityCfg
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.sensor import ContactSensor, RayCastSensor
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from instinct_mj.terrains.height_field.hf_terrains import plum_blossom_stake_centers_local

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv


def track_lin_vel_xy_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of reference linear velocity (x/y in body frame)."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    lin_vel_error = torch.sum(
        torch.square(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of reference yaw angular velocity (body frame)."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    ang_vel_error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def ang_vel_z_overshoot(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ang_vel_tolerance: float = 0.15,
) -> torch.Tensor:
    """Penalize yaw rate magnitude that exceeds the commanded magnitude (body frame).

    One-sided: only over-spinning is penalized (|ωz| above |ωz_cmd| + tolerance);
    under-spinning and wrong-direction turns are left to the tracking reward. With a zero
    yaw command this reduces to penalizing any |ωz| above tolerance (no-command drift),
    so it subsumes the previous no-command-only behavior while also curbing the
    over-fast in-place turns. Returns a nonnegative penalty; use a negative weight.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    cmd_z_abs = torch.abs(cmd[:, 2])
    wz_abs = torch.abs(asset.data.root_link_ang_vel_b[:, 2])
    excess = torch.clamp(wz_abs - cmd_z_abs - ang_vel_tolerance, min=0.0)
    return torch.square(excess)


def ang_vel_z_random_velocity_band(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cmd_epsilon: float = 0.1,
    slow_tolerance: float = 0.2,
    fast_tolerance: float = 0.25,
    std: float = 0.25,
) -> torch.Tensor:
    """Reward keeping stand random-spin ωz within a band around the sampled command.

    Only active on ``random_velocity_indices`` envs (e.g. ``perlin_rough_stand``) with
    |ωz_cmd| > cmd_epsilon. Inside the [slow_tolerance, fast_tolerance] band the reward
    is 1.0; outside it decays as exp(-violation² / std²). Use a positive reward weight.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd_z = env.command_manager.get_command(command_name)[:, 2]
    actual_z = asset.data.root_link_ang_vel_b[:, 2]

    cmd_term = env.command_manager.get_term(command_name)
    random_mask = getattr(cmd_term, "random_velocity_indices", None)
    if random_mask is None:
        return torch.zeros(env.num_envs, device=env.device)

    active = random_mask & (torch.abs(cmd_z) > cmd_epsilon)
    mag_cmd = torch.abs(cmd_z)
    mag_act = torch.abs(actual_z)
    sign_match = cmd_z * actual_z > 0.0

    slow_excess = torch.clamp(mag_cmd - slow_tolerance - mag_act, min=0.0)
    fast_excess = torch.clamp(mag_act - mag_cmd - fast_tolerance, min=0.0)
    violation_sq = torch.square(slow_excess) + torch.square(fast_excess)
    band_reward = torch.exp(-violation_sq / std**2)
    misalign_reward = torch.exp(-torch.square(actual_z - cmd_z) / std**2)
    reward = torch.where(sign_match, band_reward, misalign_reward)

    return reward * active.float()


def heading_error(
    env: ManagerBasedRlEnv,
    command_name: str,
    skip_random_velocity: bool = True,
) -> torch.Tensor:
    """Penalize yaw velocity command magnitude (|ωz_cmd|).

    Random-velocity envs (e.g. ``perlin_rough_stand``) sample ωz directly; skip there so
    in-place spins are not penalized for holding a sampled spin rate command.
    """
    command = env.command_manager.get_command(command_name)
    penalty = torch.abs(command[:, 2])
    if skip_random_velocity:
        cmd_term = env.command_manager.get_term(command_name)
        random_mask = getattr(cmd_term, "random_velocity_indices", None)
        if random_mask is not None:
            penalty = penalty * (~random_mask).float()
    return penalty


def dont_wait(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize standing still when there is a forward velocity command."""
    asset: Entity = env.scene[asset_cfg.name]
    lin_vel_cmd_x = env.command_manager.get_command(command_name)[:, 0]
    lin_vel_x = asset.data.root_link_lin_vel_b[:, 0]

    return (lin_vel_cmd_x > 0.3) * (
        (lin_vel_x < 0.15).float() + (lin_vel_x < 0.0).float() + (lin_vel_x < -0.15).float()
    )


def stand_still(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.15,
    offset: float = 1.0,
) -> torch.Tensor:
    """Penalize moving when there is no velocity command."""
    asset: Entity = env.scene[asset_cfg.name]
    dof_error = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)

    cmd = env.command_manager.get_command(command_name)
    cmd_lin_norm = torch.norm(cmd[:, :2], dim=1)
    cmd_yaw_abs = torch.abs(cmd[:, 2])

    return (dof_error - offset) * (cmd_lin_norm < threshold) * (cmd_yaw_abs < threshold)


class hurdle_crossing_reward(ManagerTermBase):
    """Give a one-shot reward when the base cleanly crosses each hurdle centerline."""

    def __init__(self, cfg: Any, env: ManagerBasedRlEnv):
        super().__init__(env)
        self.asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.success_margin = float(cfg.params.get("success_margin", 0.10))
        self.min_forward_vel = float(cfg.params.get("min_forward_vel", 0.25))
        self.collision_sensor_names = tuple(
            cfg.params.get("collision_sensor_names", ("leg_volume_points", "shank_volume_points"))
        )
        self.collision_tolerance = float(cfg.params.get("collision_tolerance", 1.0e-4))

        terrain = env.scene.terrain
        terrain_generator = getattr(terrain, "terrain_generator", None)
        terrain_origins = getattr(terrain, "terrain_origins", None)
        if terrain_generator is None or terrain_origins is None:
            raise RuntimeError("hurdle_crossing_reward requires generated parkour terrain.")

        terrain_origins_t = torch.as_tensor(terrain_origins, dtype=torch.float32, device=env.device)
        num_rows = int(terrain_origins_t.shape[0])
        num_cols = int(terrain_origins_t.shape[1])
        max_hurdles = 0
        hurdle_centers: list[list[list[float]]] = []
        hurdle_counts: list[list[int]] = []
        for row in range(num_rows):
            centers_row: list[list[float]] = []
            counts_row: list[int] = []
            for col in range(num_cols):
                sub_cfg = terrain_generator.get_subterrain_cfg(row, col)
                count = int(getattr(sub_cfg, "num_hurdles", 0) or 0)
                centers = self._hurdle_centers_world_x(sub_cfg, terrain_origins_t[row, col, 0])
                centers_row.append(centers)
                counts_row.append(count)
                max_hurdles = max(max_hurdles, len(centers))
            hurdle_centers.append(centers_row)
            hurdle_counts.append(counts_row)

        if max_hurdles <= 0:
            raise RuntimeError("hurdle_crossing_reward did not find any hurdle sub-terrain configs.")

        centers_tensor = torch.zeros((num_rows, num_cols, max_hurdles), dtype=torch.float32, device=env.device)
        counts_tensor = torch.zeros((num_rows, num_cols), dtype=torch.long, device=env.device)
        for row in range(num_rows):
            for col in range(num_cols):
                centers = hurdle_centers[row][col]
                count = len(centers)
                counts_tensor[row, col] = count
                if count > 0:
                    centers_tensor[row, col, :count] = torch.tensor(centers, dtype=torch.float32, device=env.device)
        self.hurdle_centers_x_w = centers_tensor
        self.hurdle_counts = counts_tensor
        self.prev_root_x_w = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        self.next_hurdle_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    @staticmethod
    def _hurdle_centers_world_x(sub_cfg: object, terrain_origin_x_w: torch.Tensor) -> list[float]:
        count = int(getattr(sub_cfg, "num_hurdles", 0) or 0)
        if count <= 0:
            return []
        difficulty = float(getattr(sub_cfg, "difficulty", 0.0))
        thickness = float(getattr(sub_cfg, "hurdle_thickness"))
        spacing = float(getattr(sub_cfg, "hurdle_spacing"))
        spacing_range = getattr(sub_cfg, "hurdle_spacing_range", None)
        if spacing_range is not None:
            lo, hi = spacing_range
            spacing = float(lo) + difficulty * (float(hi) - float(lo))

        size_x = float(getattr(sub_cfg, "size")[0])
        half_thick = 0.5 * thickness
        total_length = count * thickness + (count - 1) * spacing
        x_start = 0.5 * (size_x - total_length)
        tile_corner_x_w = float(terrain_origin_x_w.item()) - 0.5 * size_x
        return [
            tile_corner_x_w + x_start + i * (thickness + spacing) + half_thick
            for i in range(count)
        ]

    def reset(self, env_ids: torch.Tensor | slice | None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        asset = self._env.scene[self.asset_cfg.name]
        root_x = asset.data.root_link_pos_w[env_ids, 0]
        self.prev_root_x_w[env_ids] = root_x
        self.next_hurdle_idx[env_ids] = self._next_hurdle_index(env_ids, root_x)
        self._initialized[env_ids] = True

    def _next_hurdle_index(self, env_ids: torch.Tensor | slice, root_x: torch.Tensor) -> torch.Tensor:
        terrain = self._env.scene.terrain
        levels = terrain.terrain_levels[env_ids]
        cols = terrain.terrain_types[env_ids]
        centers = self.hurdle_centers_x_w[levels, cols]
        counts = self.hurdle_counts[levels, cols]
        passed = torch.sum(root_x.unsqueeze(-1) >= centers + self.success_margin, dim=1)
        return torch.minimum(passed.to(torch.long), counts)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        success_margin: float = 0.10,
        min_forward_vel: float = 0.25,
        collision_sensor_names: tuple[str, ...] = ("leg_volume_points", "shank_volume_points"),
        collision_tolerance: float = 1.0e-4,
    ) -> torch.Tensor:
        del asset_cfg, success_margin, min_forward_vel, collision_sensor_names, collision_tolerance
        asset = env.scene[self.asset_cfg.name]
        root_x = asset.data.root_link_pos_w[:, 0]
        uninitialized = ~self._initialized
        reset_like = root_x + 0.5 < self.prev_root_x_w
        refresh = uninitialized | reset_like
        if bool(refresh.any()):
            env_ids = refresh.nonzero(as_tuple=False).flatten()
            self.prev_root_x_w[env_ids] = root_x[env_ids]
            self.next_hurdle_idx[env_ids] = self._next_hurdle_index(env_ids, root_x[env_ids])
            self._initialized[env_ids] = True

        terrain = env.scene.terrain
        levels = terrain.terrain_levels
        cols = terrain.terrain_types
        counts = self.hurdle_counts[levels, cols]
        active = self.next_hurdle_idx < counts
        idx = torch.clamp(self.next_hurdle_idx, max=self.hurdle_centers_x_w.shape[2] - 1)
        next_center_x = self.hurdle_centers_x_w[levels, cols, idx]
        crossed = active & (self.prev_root_x_w < next_center_x) & (root_x >= next_center_x + self.success_margin)
        moving_forward = asset.data.root_link_lin_vel_b[:, 0] >= self.min_forward_vel
        clean = self._clean_hurdle_crossing(env)
        reward = (crossed & moving_forward & clean).float()

        self.next_hurdle_idx = torch.where(crossed, torch.minimum(self.next_hurdle_idx + 1, counts), self.next_hurdle_idx)
        self.prev_root_x_w[:] = root_x
        return reward

    def _clean_hurdle_crossing(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        clean = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        for sensor_name in self.collision_sensor_names:
            sensor = env.scene.sensors[sensor_name]
            penetration = torch.linalg.vector_norm(
                sensor.data.penetration_offset.reshape(env.num_envs, -1, 3),
                dim=-1,
            )
            clean &= torch.max(penetration, dim=1)[0] <= self.collision_tolerance
        return clean


def feet_air_time(
    env: ManagerBasedRlEnv,
    command_name: str,
    vel_threshold: float,
    sensor_name: str,
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold
    and also keeping one foot at a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step),
    then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene[sensor_name]
    air_time = contact_sensor.data.current_air_time
    contact_time = contact_sensor.data.current_contact_time
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]

    # no reward for zero command
    cmd = env.command_manager.get_command(command_name)
    reward *= torch.logical_or(
        torch.norm(cmd[:, :2], dim=1) > vel_threshold,
        torch.abs(cmd[:, 2]) > vel_threshold,
    )
    return reward


def yaw_cmd_double_support(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    yaw_threshold: float = 0.15,
    grace_period: float = 0.08,
    max_penalty_time: float = 0.4,
    random_velocity_only: bool = False,
) -> torch.Tensor:
    """Penalize staying in double support when a yaw command asks the robot to turn."""
    contact_sensor: ContactSensor = env.scene[sensor_name]
    contact_time = contact_sensor.data.current_contact_time
    assert contact_time is not None

    cmd = env.command_manager.get_command(command_name)
    active = torch.abs(cmd[:, 2]) > yaw_threshold
    if random_velocity_only:
        cmd_term = env.command_manager.get_term(command_name)
        random_mask = getattr(cmd_term, "random_velocity_indices", None)
        if random_mask is None:
            return torch.zeros(env.num_envs, device=env.device)
        active = active & random_mask

    in_contact = contact_time > 0.0
    double_support = torch.sum(in_contact.int(), dim=1) == 2
    support_time = torch.min(contact_time, dim=1)[0]
    penalty = torch.clamp(support_time - grace_period, min=0.0, max=max_penalty_time)
    return penalty * active.float() * double_support.float()


def both_feet_air_time(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    grace_period: float = 0.0,       # 任何腾空时间都会产生惩罚
    max_penalty_time: float = 0.2,
) -> torch.Tensor:
    """Penalize simultaneous flight time to discourage two-foot hopping.
    
    Now penalizes ANY time both feet are off the ground, regardless of command velocity.
    """
    contact_sensor: ContactSensor = env.scene[sensor_name]
    air_time = contact_sensor.data.current_air_time
    contact_time = contact_sensor.data.current_contact_time
    
    assert air_time is not None
    assert contact_time is not None

    # 判断是否双足都未接触地面 (in_contact 为 False)
    in_contact = contact_time > 0.0
    both_feet_air = torch.sum(in_contact.int(), dim=1) == 0
    
    
    return both_feet_air.float()


    


def feet_slide(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize foot sliding speed while feet are in contact."""
    asset: Entity = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene[sensor_name]

    in_contact = torch.max(torch.linalg.vector_norm(sensor.data.force_history, dim=-1), dim=2)[0] > threshold
    foot_vel_xy = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :2]
    slip_speed = torch.norm(foot_vel_xy, dim=-1)
    return torch.sum(slip_speed * in_contact.float(), dim=1)


def ang_vel_xy_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def joint_deviation_square(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    joint_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(joint_error), dim=1)


def joint_deviation_l1(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    joint_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(joint_error), dim=1)


def link_orientation(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize non-flat link orientation using L2 squared kernel."""
    asset: Entity = env.scene[asset_cfg.name]
    link_quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids[0], :]
    link_projected_gravity = quat_apply_inverse(link_quat, asset.data.gravity_vec_w)
    return torch.sum(torch.square(link_projected_gravity[:, :2]), dim=1)


def feet_orientation_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward feet being oriented vertically when in contact with the ground."""
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]

    body_link_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
    num_envs, num_feet = body_link_quat_w.shape[:2]

    gravity_w = asset.data.gravity_vec_w.unsqueeze(1).expand(-1, num_feet, -1)
    projected_gravity = quat_apply_inverse(body_link_quat_w.reshape(-1, 4), gravity_w.reshape(-1, 3)).reshape(
        num_envs, num_feet, 3
    )
    orientation_error = torch.linalg.vector_norm(projected_gravity[:, :, :2], dim=-1)

    in_contact = (
        torch.max(torch.linalg.vector_norm(contact_sensor.data.force_history, dim=-1), dim=2)[0]
        > contact_force_threshold
    )

    return torch.sum(orientation_error * in_contact.float(), dim=1)


def feet_at_plane(
    env: ManagerBasedRlEnv,
    contact_sensor_name: str,
    left_height_scanner_name: str,
    right_height_scanner_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_offset: float = 0.035,
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward feet being at certain height above the ground plane."""
    asset: Entity = env.scene[asset_cfg.name]
    body_link_pos_w = asset.data.body_link_pos_w

    contact_sensor: ContactSensor = env.scene[contact_sensor_name]
    is_contact = (
        torch.max(torch.linalg.vector_norm(contact_sensor.data.force_history, dim=-1), dim=2)[0]
        > contact_force_threshold
    )

    left_sensor: RayCastSensor = env.scene[left_height_scanner_name]
    right_sensor: RayCastSensor = env.scene[right_height_scanner_name]
    left_hit_z = left_sensor.data.hit_pos_w[..., 2]
    right_hit_z = right_sensor.data.hit_pos_w[..., 2]
    left_hit_z = torch.where(left_sensor.data.distances < 0.0, 0.0, left_hit_z)
    right_hit_z = torch.where(right_sensor.data.distances < 0.0, 0.0, right_hit_z)

    left_height = body_link_pos_w[:, asset_cfg.body_ids[0], 2].unsqueeze(-1)
    right_height = body_link_pos_w[:, asset_cfg.body_ids[1], 2].unsqueeze(-1)

    left_contact = is_contact[:, 0:1].float()
    right_contact = is_contact[:, 1:2].float()

    left_reward = torch.clamp(left_height - left_hit_z - height_offset, min=0.0, max=0.3) * left_contact
    right_reward = torch.clamp(right_height - right_hit_z - height_offset, min=0.0, max=0.3) * right_contact
    return torch.sum(left_reward, dim=-1) + torch.sum(right_reward, dim=-1)


def feet_horizontal_impact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 30.0,
    vertical_ratio: float = 0.5,
) -> torch.Tensor:
    """Penalize horizontal ground-reaction force on the feet ("knocking" a riser).

    A foot resting flat on a tread produces a nearly vertical net force; a toe that
    strikes the vertical face of a step on touchdown produces a large horizontal force
    while the vertical support is still small. We therefore flag a knock only when the
    horizontal force exceeds both an absolute floor and a fraction of the vertical
    support, which keeps the normal braking/push-off shear of flat walking unpenalized.

    Uses the ``netforce`` contact sensor, whose force is the net wrench in the world
    frame. Returns a nonnegative magnitude per env; use a negative reward weight.
    """
    contact_sensor: ContactSensor = env.scene[sensor_name]
    force_history = contact_sensor.data.force_history
    assert force_history is not None

    f_horiz = torch.linalg.vector_norm(force_history[..., :2], dim=-1)
    f_vert = torch.abs(force_history[..., 2])
    knock = torch.clamp(f_horiz - vertical_ratio * f_vert - force_threshold, min=0.0)
    per_foot = torch.max(knock, dim=2)[0]
    return torch.sum(per_foot, dim=1)


def feet_unsupported(
    env: ManagerBasedRlEnv,
    contact_sensor_name: str,
    left_sole_scanner_name: str,
    right_sole_scanner_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_height_offset: float = 0.065,
    gap_threshold: float = 0.03,
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize a foot whose sole overhangs empty space while bearing weight.

    Each foot carries a grid of downward height rays tiling its sole. A ray is
    "unsupported" when the terrain beneath it sits more than ``gap_threshold`` below the
    sole plane (``foot_z - foot_height_offset``), i.e. that part of the foot is hanging
    over a step edge / drop. The penalty is the fraction of unsupported rays, gated by
    contact, so heel-only or edge landings on stairs are discouraged in favour of full
    foot support. Returns a nonnegative value per env; use a negative reward weight.
    """
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[contact_sensor_name]
    is_contact = (
        torch.max(torch.linalg.vector_norm(contact_sensor.data.force_history, dim=-1), dim=2)[0]
        > contact_force_threshold
    )

    left_sensor: RayCastSensor = env.scene[left_sole_scanner_name]
    right_sensor: RayCastSensor = env.scene[right_sole_scanner_name]

    foot_z = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
    left_sole_z = foot_z[:, 0:1] - foot_height_offset
    right_sole_z = foot_z[:, 1:2] - foot_height_offset

    left_hit_z = left_sensor.data.hit_pos_w[..., 2]
    right_hit_z = right_sensor.data.hit_pos_w[..., 2]
    left_valid = left_sensor.data.distances >= 0.0
    right_valid = right_sensor.data.distances >= 0.0

    left_unsupported = (left_sole_z - left_hit_z > gap_threshold) & left_valid
    right_unsupported = (right_sole_z - right_hit_z > gap_threshold) & right_valid
    left_frac = left_unsupported.float().sum(dim=1) / left_valid.float().sum(dim=1).clamp(min=1.0)
    right_frac = right_unsupported.float().sum(dim=1) / right_valid.float().sum(dim=1).clamp(min=1.0)

    return left_frac * is_contact[:, 0].float() + right_frac * is_contact[:, 1].float()


def feet_close_xy_gauss(
    env: ManagerBasedRlEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.1,
) -> torch.Tensor:
    """Penalize when feet are too close together in the y distance."""
    asset: Entity = env.scene[asset_cfg.name]
    body_link_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]

    left_foot_xy = body_link_pos_w[:, 0, :2]
    right_foot_xy = body_link_pos_w[:, 1, :2]
    heading_w = asset.data.heading_w

    cos_heading = torch.cos(heading_w)
    sin_heading = torch.sin(heading_w)

    left_y = -sin_heading * left_foot_xy[:, 0] + cos_heading * left_foot_xy[:, 1]
    right_y = -sin_heading * right_foot_xy[:, 0] + cos_heading * right_foot_xy[:, 1]
    feet_distance_y = torch.abs(left_y - right_y)

    return torch.exp(-torch.clamp(threshold - feet_distance_y, min=0.0) / std**2) - 1


def feet_heading_alignment(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cmd_threshold: float = 0.15,
    angle_scale: float = 0.10,
    yaw_cmd_threshold: float = 0.15,
) -> torch.Tensor:
    """Reward foot heading alignment with an angle-sensitive exponential shaping.

    Active under a linear command (walking) or a yaw command (in-place turning). The
    target is always the base forward direction, so during a turn the feet are pulled to
    rotate with the base instead of splaying out (hip-yaw toe-out) to pivot in place.
    """
    asset: Entity = env.scene[asset_cfg.name]
    command_b = env.command_manager.get_command(command_name)
    assert command_b is not None
    command_xy_b = command_b[:, :2]
    command_norm = torch.linalg.vector_norm(command_xy_b, dim=-1)
    yaw_cmd_abs = torch.abs(command_b[:, 2])
    is_active = (command_norm > cmd_threshold) | (yaw_cmd_abs > yaw_cmd_threshold)

    desired_dir_b = torch.zeros(env.num_envs, 3, device=command_b.device, dtype=command_b.dtype)
    desired_dir_b[:, 0] = 1.0
    desired_dir_w = quat_apply(asset.data.root_link_quat_w, desired_dir_b)[:, :2]
    desired_dir_w = desired_dir_w / torch.clamp(torch.linalg.vector_norm(desired_dir_w, dim=-1, keepdim=True), min=1e-6)

    if isinstance(asset_cfg.body_ids, slice):
        body_ids = list(range(asset.num_bodies))
    else:
        body_ids = list(asset_cfg.body_ids)
    body_link_quat_w = asset.data.body_link_quat_w[:, body_ids, :]
    num_envs, num_feet = body_link_quat_w.shape[:2]
    foot_forward_local = torch.zeros(num_envs, num_feet, 3, device=command_b.device, dtype=command_b.dtype)
    foot_forward_local[..., 0] = 1.0
    foot_forward_w = quat_apply(body_link_quat_w.reshape(-1, 4), foot_forward_local.reshape(-1, 3)).reshape(
        num_envs, num_feet, 3
    )
    foot_forward_xy = foot_forward_w[:, :, :2]
    foot_forward_xy = foot_forward_xy / torch.clamp(
        torch.linalg.vector_norm(foot_forward_xy, dim=-1, keepdim=True),
        min=1e-6,
    )

    alignment = torch.sum(foot_forward_xy * desired_dir_w.unsqueeze(1), dim=-1)
    cross = (
        foot_forward_xy[..., 0] * desired_dir_w[:, None, 1]
        - foot_forward_xy[..., 1] * desired_dir_w[:, None, 0]
    )
    angle_error = torch.abs(torch.atan2(cross, alignment))
    alignment_reward = torch.exp(-angle_error / angle_scale)
    return torch.mean(alignment_reward, dim=1) * is_active.float()


def target_heading_alignment(
    env: ManagerBasedRlEnv,
    command_name: str,
    patch_name: str = "target",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cmd_threshold: float = 0.05,
    std: float = 0.6,
) -> torch.Tensor:
    """Reward base heading alignment with the current flat-patch path tangent."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_norm = torch.linalg.vector_norm(command[:, :2], dim=-1)
    is_active = command_norm > cmd_threshold

    command_term = env.command_manager.get_term(command_name)
    patch_idx = getattr(command_term, "_flat_patch_idx", None)
    terrain = env.scene["terrain"]
    headings_by_name = getattr(terrain, "flat_patch_headings", None)
    if patch_idx is None or not isinstance(headings_by_name, dict) or patch_name not in headings_by_name:
        return torch.zeros(env.num_envs, device=env.device)

    headings = headings_by_name[patch_name]
    levels = terrain.terrain_levels
    cols = terrain.terrain_types
    patch_idx = torch.clamp(patch_idx, min=0, max=headings.shape[2] - 1)
    target_yaw = headings[levels, cols, patch_idx]

    yaw_error = torch.atan2(
        torch.sin(target_yaw - asset.data.heading_w),
        torch.cos(target_yaw - asset.data.heading_w),
    )
    return torch.exp(-torch.square(yaw_error) / std**2) * is_active.float()




def volume_points_penetration(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    tolerance: float = 0.0,
) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_name]
    penetration = sensor.data.penetration_offset
    points_vel = sensor.data.points_vel_w

    penetration_depth = torch.linalg.vector_norm(penetration.reshape(env.num_envs, -1, 3), dim=-1)
    in_obstacle = (penetration_depth > tolerance).float()
    points_vel_norm = torch.linalg.vector_norm(points_vel.reshape(env.num_envs, -1, 3), dim=-1)
    velocity_times_penetration = in_obstacle * (points_vel_norm + 1e-6) * penetration_depth
    return torch.sum(velocity_times_penetration, dim=-1)


def motors_power_square(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    normalize_by_stiffness: bool = True,
    normalize_by_num_joints: bool = False,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    # mjlab uses actuator_force instead of applied_torque
    power_j = asset.data.actuator_force * asset.data.joint_vel
    if normalize_by_stiffness:
        # mjlab: stiffness is configured on each BuiltinPositionActuator cfg.
        for actuator in asset.actuators:
            base_actuator = actuator.base_actuator
            target_ids = base_actuator.target_ids
            stiffness = base_actuator.cfg.stiffness
            power_j[:, target_ids] /= torch.as_tensor(stiffness, device=power_j.device, dtype=power_j.dtype)

    power_j = power_j[:, asset_cfg.joint_ids]
    power = torch.sum(torch.square(power_j), dim=-1)
    if normalize_by_num_joints:
        power = power / power_j.shape[-1]
    return power


def joint_vel_limits(
    env: ManagerBasedRlEnv,
    soft_ratio: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint velocities if they cross soft limits.

    Per-joint velocity limits are read from actuator cfg metadata
    (``velocity_limit``). Excess is clipped to [0, 1] rad/s per joint.
    """
    asset: Entity = env.scene[asset_cfg.name]

    joint_vel_limits = torch.zeros_like(asset.data.joint_vel)
    for actuator in asset.actuators:
        base_actuator = actuator.base_actuator
        target_ids = base_actuator.target_ids
        joint_vel_limits[:, target_ids] = float(base_actuator.cfg.velocity_limit)

    out_of_limits = (
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]) - joint_vel_limits[:, asset_cfg.joint_ids] * soft_ratio
    )
    # Clip to max error = 1 rad/s per joint to avoid huge penalties
    out_of_limits = out_of_limits.clip_(min=0.0, max=1.0)
    return torch.sum(out_of_limits, dim=1)


def applied_torque_limits_by_ratio(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    limit_ratio: float = 0.8,
) -> torch.Tensor:
    """Penalize when the applied torque exceeds a ratio of torque limits."""
    asset: Entity = env.scene[asset_cfg.name]

    if isinstance(asset_cfg.joint_ids, slice):
        selected_joint_ids = list(range(asset.num_joints))
    else:
        selected_joint_ids = list(asset_cfg.joint_ids)
    selected_joint_names = {asset.joint_names[j] for j in selected_joint_ids}

    actuator_forcerange = env.sim.model.actuator_forcerange
    if actuator_forcerange.ndim == 3:
        actuator_forcerange = actuator_forcerange[0]

    actuator_force = asset.data.actuator_force
    out_of_limits = []
    for actuator in asset.actuators:
        base_actuator = actuator.base_actuator
        target_names = list(base_actuator.target_names)
        ctrl_ids_local = base_actuator.ctrl_ids
        ctrl_ids_global = base_actuator.global_ctrl_ids
        for idx, joint_name in enumerate(target_names):
            if joint_name not in selected_joint_names:
                continue
            ctrl_id_local = int(ctrl_ids_local[idx])
            ctrl_id_global = int(ctrl_ids_global[idx])
            effort_limit = torch.max(torch.abs(actuator_forcerange[ctrl_id_global]))
            torque_abs = torch.abs(actuator_force[:, ctrl_id_local])
            out_of_limits.append(torch.clamp(torque_abs - effort_limit * limit_ratio, min=0.0))

    out_of_limits_tensor = torch.stack(out_of_limits, dim=-1)
    return torch.sum(torch.square(out_of_limits_tensor), dim=-1)


def on_deck(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_threshold: float = 0.10,
    contact_force_threshold: float = 1.0,
    command_name: str | None = None,
    near_target_xy_dist: float | None = None,
    spawn_patch_name: str | None = None,
    near_spawn_xy_dist: float | None = None,
    terminal_sequence_neutral: bool = False,
    terminal_sequence_neutral_count: int = 2,
    target_patch_name: str = "target",
) -> torch.Tensor:
    """Reward feet on elevated deck (+1/contact foot), penalize ground (-1/contact foot).

    With ``command_name``, within ``near_target_xy_dist`` of the pose target (defaults to
    ``target_dis_threshold``) off-deck feet score 0 so ground goals are not penalized.
    With ``spawn_patch_name``, the same exemption applies near the reset flat patch
    (e.g. ``start``) within ``near_spawn_xy_dist``.
    """
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    robot_xy = asset.data.root_link_pos_w[:, :2]

    foot_z = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
    in_contact = (
        torch.max(torch.linalg.vector_norm(contact_sensor.data.force_history, dim=-1), dim=2)[0]
        > contact_force_threshold
    ).float()

    on_deck_mask = (foot_z > height_threshold).float()
    per_foot = (on_deck_mask - (1.0 - on_deck_mask)) * in_contact
    no_ground_penalty = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    if command_name is not None:
        cmd_term = env.command_manager.get_term(command_name)
        pos_w = getattr(cmd_term, "pos_command_w", None)
        if pos_w is not None:
            dist_thresh = near_target_xy_dist
            if dist_thresh is None:
                cmd_cfg = getattr(cmd_term, "cfg", None)
                dist_thresh = getattr(cmd_cfg, "target_dis_threshold", None) if cmd_cfg is not None else None
            if dist_thresh is not None:
                target_dist = torch.norm(pos_w[:, :2] - robot_xy, dim=1)
                no_ground_penalty = no_ground_penalty | (target_dist <= float(dist_thresh))

    if spawn_patch_name is not None and near_spawn_xy_dist is not None:
        terrain = env.scene.terrain
        if terrain is not None and spawn_patch_name in terrain.flat_patches:
            spawn_patches = terrain.flat_patches[spawn_patch_name]
            levels = terrain.terrain_levels
            types = terrain.terrain_types
            spawn_xy = spawn_patches[levels, types, 0, :2]
            spawn_dist = torch.norm(robot_xy - spawn_xy, dim=1)
            no_ground_penalty = no_ground_penalty | (spawn_dist <= float(near_spawn_xy_dist))

    if bool(no_ground_penalty.any()):
        per_foot = torch.where(
            no_ground_penalty.unsqueeze(-1),
            on_deck_mask * in_contact,
            per_foot,
        )

    if terminal_sequence_neutral and command_name is not None:
        cmd_term = env.command_manager.get_term(command_name)
        patch_idx = getattr(cmd_term, "_flat_patch_idx", None)
        terrain = env.scene.terrain
        num_valid_by_name = getattr(terrain, "flat_patch_num_valid", None)
        if patch_idx is not None and isinstance(num_valid_by_name, dict) and target_patch_name in num_valid_by_name:
            levels = terrain.terrain_levels
            types = terrain.terrain_types
            num_valid = num_valid_by_name[target_patch_name][levels, types]
            terminal_count = max(1, int(terminal_sequence_neutral_count))
            terminal_start = torch.clamp(num_valid - terminal_count, min=0)
            terminal_segment = patch_idx >= terminal_start
            if bool(terminal_segment.any()):
                per_foot = torch.where(
                    terminal_segment.unsqueeze(-1),
                    torch.zeros_like(per_foot),
                    per_foot,
                )

    return torch.sum(per_foot, dim=1)


def undesired_contacts(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold: float,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_name]
    is_contact = torch.max(torch.linalg.vector_norm(contact_sensor.data.force_history, dim=-1), dim=2)[0] > threshold
    return torch.sum(is_contact.float(), dim=1)


def _column_subterrain_names(terrain_generator_cfg: object) -> list[str]:
    """Map each terrain column index to its sub-terrain name (proportion-weighted)."""
    sub_terrains = getattr(terrain_generator_cfg, "sub_terrains", None)
    num_cols = int(getattr(terrain_generator_cfg, "num_cols", 0) or 0)
    if not isinstance(sub_terrains, dict) or num_cols <= 0:
        return []

    names = list(sub_terrains.keys())
    proportions = [float(getattr(sub_cfg, "proportion", 0.0)) for sub_cfg in sub_terrains.values()]
    total = sum(proportions)
    if total <= 0.0:
        return names[:1] * num_cols

    cumulative: list[float] = []
    running = 0.0
    for proportion in proportions:
        running += proportion / total
        cumulative.append(running)

    column_names: list[str] = []
    for index in range(num_cols):
        ratio = index / num_cols + 0.001
        sub_index = next(i for i, threshold in enumerate(cumulative) if ratio < threshold)
        column_names.append(names[sub_index])
    return column_names


def _stair_step_height_phys(sub_cfg: object) -> float:
    step_height_range = getattr(sub_cfg, "step_height_range", (0.05, 0.20))
    lo = float(step_height_range[0])
    hi = float(step_height_range[1])
    difficulty = float(getattr(sub_cfg, "difficulty", 0.5))
    step_height = lo + difficulty * (hi - lo)
    if bool(getattr(sub_cfg, "inverted", False)):
        step_height *= -1.0
    return step_height


def _stair_platform_width_phys(sub_cfg: object) -> float:
    platform_width = getattr(sub_cfg, "platform_width", 1.5)
    if isinstance(platform_width, tuple):
        lo = float(platform_width[0])
        hi = float(platform_width[1])
        if lo > hi:
            lo, hi = hi, lo
        return 0.5 * (lo + hi)
    return float(platform_width)


def _tread_candidates_xz_local_neg_x_updown(sub_cfg: object) -> list[tuple[float, float]]:
    """Tread center (x, z) in terrain-local coordinates for neg_x up_down one-sided stairs.

    Matches ``_one_sided_up_down_stairs_heightfield_x`` on the inner stair run
    (``size_x - 2 * border_width``), not the full bordered tile extent.
    """
    face = getattr(sub_cfg, "stair_face", "neg_x")
    if face != "neg_x" or not bool(getattr(sub_cfg, "up_down", False)):
        return []

    step_width = float(getattr(sub_cfg, "step_width", 0.3))
    platform_width = _stair_platform_width_phys(sub_cfg)
    size = getattr(sub_cfg, "size", (8.0, 8.0))
    border_width = float(getattr(sub_cfg, "border_width", 1.0))
    if step_width <= 0.0:
        return []

    size_x = float(size[0])
    inner_x = size_x - 2.0 * border_width
    if inner_x <= 0.0:
        return []

    inner_half = 0.5 * inner_x
    step_height = _stair_step_height_phys(sub_cfg)
    rise_span = max(0.0, 0.5 * (inner_x - platform_width))
    max_steps = int(rise_span // step_width)

    treads: list[tuple[float, float]] = []
    for step_idx in range(max_steps):
        x_local = -inner_half + (step_idx + 0.5) * step_width
        treads.append((x_local, float(step_idx) * step_height))

    treads.append((0.0, float(max_steps) * step_height))

    for step_idx in range(max_steps - 1, -1, -1):
        x_local = inner_half - (step_idx + 0.5) * step_width
        treads.append((x_local, float(step_idx) * step_height))

    return treads


def _row_difficulty(row: int, num_rows: int) -> float:
    if num_rows <= 1:
        return 0.5
    return float(row) / float(num_rows - 1)


def _sub_cfg_for_cell(
    terrain_generator: object | None,
    terrain_generator_cfg: object,
    terrain_name: str,
    row: int,
    num_rows: int,
    col: int,
) -> object | None:
    sub_cfg: object | None = None
    get_subterrain = (
        getattr(terrain_generator, "get_subterrain_cfg", None) if terrain_generator is not None else None
    )
    if callable(get_subterrain):
        sub_cfg = get_subterrain(row, col)
    if sub_cfg is None:
        sub_terrains = getattr(terrain_generator_cfg, "sub_terrains", None)
        if isinstance(sub_terrains, dict) and terrain_name in sub_terrains:
            sub_cfg = sub_terrains[terrain_name]
    if sub_cfg is None:
        return None
    if getattr(sub_cfg, "difficulty", None) is None:
        sub_cfg = copy.copy(sub_cfg)
        setattr(sub_cfg, "difficulty", _row_difficulty(row, num_rows))
    return sub_cfg


def _min_xz_distance_to_candidates(
    foot_xz: torch.Tensor,
    candidates_xz: torch.Tensor,
    candidate_counts: torch.Tensor,
) -> torch.Tensor:
    """Minimum x-z distance from each foot position to its frozen candidate set."""
    diff = candidates_xz - foot_xz.unsqueeze(2)
    dist = torch.linalg.vector_norm(diff, dim=-1)
    max_candidates = candidates_xz.shape[2]
    valid = torch.arange(max_candidates, device=dist.device).view(1, 1, -1) < candidate_counts.unsqueeze(-1)
    dist = dist.masked_fill(~valid, float("inf"))
    return dist.min(dim=2).values


class cref_stair_foothold_reward(ManagerTermBase):
    """CReF-style terrain-aware foothold reward for one_sided_stairs (neg_x, up_down).

    Per foot: on liftoff refresh and freeze tread candidates; on touchdown reward once with
    ``exp(-d_xz / s_xz)`` where ``d_xz`` is the minimum distance to the frozen set in the
    terrain-local x-z plane. Inactive on non-stair terrains.
    """

    _STAIRS_TERRAIN_NAME = "one_sided_stairs"

    def __init__(self, cfg: Any, env: ManagerBasedRlEnv):
        super().__init__(env)
        self.asset_cfg: SceneEntityCfg = cfg.params.get(
            "asset_cfg",
            SceneEntityCfg("robot", body_names=("left_ankle_roll_link", "right_ankle_roll_link")),
        )
        self.sensor_name = str(cfg.params.get("sensor_name", "contact_forces"))
        self.contact_force_threshold = float(cfg.params.get("contact_force_threshold", 1.0))
        self.s_xz = float(cfg.params.get("s_xz", 0.15))
        self.min_forward_margin = float(cfg.params.get("min_forward_margin", 0.02))
        self.max_candidates_per_foot = int(cfg.params.get("max_candidates_per_foot", 2))
        self.foot_height_offset = float(cfg.params.get("foot_height_offset", 0.065))

        asset: Entity = env.scene[self.asset_cfg.name]
        if isinstance(self.asset_cfg.body_ids, slice):
            self.body_ids = list(range(asset.num_bodies))
        else:
            self.body_ids = list(self.asset_cfg.body_ids)
        self.num_feet = len(self.body_ids)

        self._enabled = False
        terrain = env.scene.terrain
        terrain_origins = getattr(terrain, "terrain_origins", None)
        terrain_cfg = getattr(terrain, "cfg", None)
        terrain_generator_cfg = getattr(terrain_cfg, "terrain_generator", None) if terrain_cfg is not None else None
        terrain_generator = getattr(terrain, "terrain_generator", None)

        if terrain_origins is None or terrain_generator_cfg is None:
            return

        terrain_origins_t = torch.as_tensor(terrain_origins, dtype=torch.float32, device=env.device)
        num_rows = int(terrain_origins_t.shape[0])
        num_cols = int(terrain_origins_t.shape[1])
        column_names = _column_subterrain_names(terrain_generator_cfg)
        if len(column_names) != num_cols:
            return

        max_treads = 0
        treads_by_cell: list[list[list[tuple[float, float]]]] = []
        for row in range(num_rows):
            row_treads: list[list[tuple[float, float]]] = []
            for col in range(num_cols):
                if column_names[col] != self._STAIRS_TERRAIN_NAME:
                    row_treads.append([])
                    continue
                sub_cfg = _sub_cfg_for_cell(
                    terrain_generator,
                    terrain_generator_cfg,
                    self._STAIRS_TERRAIN_NAME,
                    row,
                    num_rows,
                    col,
                )
                treads = _tread_candidates_xz_local_neg_x_updown(sub_cfg) if sub_cfg is not None else []
                row_treads.append(treads)
                max_treads = max(max_treads, len(treads))
            treads_by_cell.append(row_treads)

        if max_treads <= 0:
            return

        treads_tensor = torch.zeros((num_rows, num_cols, max_treads, 2), dtype=torch.float32, device=env.device)
        tread_counts = torch.zeros((num_rows, num_cols), dtype=torch.long, device=env.device)
        for row in range(num_rows):
            for col in range(num_cols):
                treads = treads_by_cell[row][col]
                count = len(treads)
                tread_counts[row, col] = count
                if count > 0:
                    treads_tensor[row, col, :count] = torch.tensor(treads, dtype=torch.float32, device=env.device)

        stairs_cols = {col for col, name in enumerate(column_names) if name == self._STAIRS_TERRAIN_NAME}
        self._enabled = len(stairs_cols) > 0
        self.terrain_origins = terrain_origins_t
        self.treads_xz_local = treads_tensor
        self.tread_counts = tread_counts
        self.stairs_col_mask = torch.zeros(num_cols, dtype=torch.bool, device=env.device)
        for col in stairs_cols:
            self.stairs_col_mask[col] = True

        self.prev_in_contact = torch.zeros(env.num_envs, self.num_feet, dtype=torch.bool, device=env.device)
        self.foot_armed = torch.zeros(env.num_envs, self.num_feet, dtype=torch.bool, device=env.device)
        max_frozen = self.max_candidates_per_foot
        self.frozen_candidates_xz = torch.zeros(
            env.num_envs, self.num_feet, max_frozen, 2, dtype=torch.float32, device=env.device
        )
        self.frozen_candidate_counts = torch.zeros(env.num_envs, self.num_feet, dtype=torch.long, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if not self._enabled:
            return
        if env_ids is None:
            env_ids = slice(None)
        in_contact = self._in_contact(self._env)
        self.prev_in_contact[env_ids] = in_contact[env_ids]
        self.foot_armed[env_ids] = False
        self.frozen_candidate_counts[env_ids] = 0

    def _in_contact(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        contact_sensor: ContactSensor = env.scene[self.sensor_name]
        force_history = contact_sensor.data.force_history
        if force_history is None:
            return torch.zeros(env.num_envs, self.num_feet, dtype=torch.bool, device=env.device)
        return (
            torch.max(torch.linalg.vector_norm(force_history, dim=-1), dim=2)[0] > self.contact_force_threshold
        )

    def _stairs_active_mask(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        terrain = env.scene.terrain
        cols = terrain.terrain_types
        return self.stairs_col_mask[cols]

    def _refresh_frozen_candidates(
        self,
        env_ids: torch.Tensor,
        foot_idx: int,
        foot_x_local: torch.Tensor,
    ) -> None:
        terrain = self._env.scene.terrain
        levels = terrain.terrain_levels[env_ids]
        cols = terrain.terrain_types[env_ids]
        batch_treads = self.treads_xz_local[levels, cols]
        batch_counts = self.tread_counts[levels, cols]

        max_all = batch_treads.shape[1]
        max_keep = self.max_candidates_per_foot
        device = batch_treads.device
        valid = torch.arange(max_all, device=device).view(1, -1) < batch_counts.unsqueeze(-1)
        forward = batch_treads[..., 0] > (foot_x_local.unsqueeze(-1) + self.min_forward_margin)
        mask = valid & forward

        x_ahead = batch_treads[..., 0] - foot_x_local.unsqueeze(-1)
        x_ahead = x_ahead.masked_fill(~mask, float("inf"))
        k = min(max_keep, max_all)
        _, topk_idx = torch.topk(x_ahead, k=k, dim=-1, largest=False)
        gathered = torch.gather(batch_treads, 1, topk_idx.unsqueeze(-1).expand(-1, -1, 2))
        topk_x_ahead = torch.gather(x_ahead, 1, topk_idx)
        keep_counts = torch.sum(torch.isfinite(topk_x_ahead), dim=-1)

        self.frozen_candidates_xz[env_ids, foot_idx, :k] = gathered
        if k < self.frozen_candidates_xz.shape[2]:
            self.frozen_candidates_xz[env_ids, foot_idx, k:] = 0.0
        self.frozen_candidate_counts[env_ids, foot_idx] = keep_counts

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg(
            "robot", body_names=("left_ankle_roll_link", "right_ankle_roll_link")
        ),
        sensor_name: str = "contact_forces",
        contact_force_threshold: float = 1.0,
        s_xz: float = 0.15,
        min_forward_margin: float = 0.02,
        max_candidates_per_foot: int = 2,
        foot_height_offset: float = 0.065,
    ) -> torch.Tensor:
        del (
            asset_cfg,
            sensor_name,
            contact_force_threshold,
            s_xz,
            min_forward_margin,
            max_candidates_per_foot,
            foot_height_offset,
        )
        if not self._enabled:
            return torch.zeros(env.num_envs, device=env.device)

        asset: Entity = env.scene[self.asset_cfg.name]
        in_contact = self._in_contact(env)
        stairs_active = self._stairs_active_mask(env)

        terrain = env.scene.terrain
        levels = terrain.terrain_levels
        cols = terrain.terrain_types
        origins = self.terrain_origins[levels, cols]
        foot_pos_w = asset.data.body_link_pos_w[:, self.body_ids, :]
        foot_x_local = foot_pos_w[..., 0] - origins[:, None, 0]
        foot_z = foot_pos_w[..., 2] - self.foot_height_offset
        foot_xz = torch.stack((foot_x_local, foot_z), dim=-1)

        liftoff = self.prev_in_contact & ~in_contact & stairs_active.unsqueeze(-1)
        touchdown = ~self.prev_in_contact & in_contact & self.foot_armed
        touchdown &= stairs_active.unsqueeze(-1)
        touchdown &= self.frozen_candidate_counts > 0

        for foot_idx in range(self.num_feet):
            liftoff_ids = liftoff[:, foot_idx].nonzero(as_tuple=False).flatten()
            if liftoff_ids.numel() > 0:
                self.foot_armed[liftoff_ids, foot_idx] = True
                self._refresh_frozen_candidates(liftoff_ids, foot_idx, foot_x_local[liftoff_ids, foot_idx])

        min_dist = _min_xz_distance_to_candidates(
            foot_xz,
            self.frozen_candidates_xz,
            self.frozen_candidate_counts,
        )
        foot_reward = torch.exp(-min_dist / self.s_xz) * touchdown.float()
        reward = torch.sum(foot_reward, dim=1) * stairs_active.float()

        self.foot_armed = torch.where(touchdown, torch.zeros_like(self.foot_armed), self.foot_armed)
        self.frozen_candidate_counts = torch.where(
            touchdown,
            torch.zeros_like(self.frozen_candidate_counts),
            self.frozen_candidate_counts,
        )
        self.prev_in_contact = in_contact
        return reward


def _min_xy_distance_to_candidates(
    foot_xy: torch.Tensor,
    candidates_xy: torch.Tensor,
    candidate_counts: torch.Tensor,
) -> torch.Tensor:
    """Minimum horizontal x-y distance from each foot position to its frozen candidate set."""
    diff = candidates_xy - foot_xy.unsqueeze(2)
    dist = torch.linalg.vector_norm(diff, dim=-1)
    max_candidates = candidates_xy.shape[2]
    valid = torch.arange(max_candidates, device=dist.device).view(1, 1, -1) < candidate_counts.unsqueeze(-1)
    dist = dist.masked_fill(~valid, float("inf"))
    return dist.min(dim=2).values


class cref_stepping_stone_foothold_reward(ManagerTermBase):
    """CReF-style terrain-aware foothold reward for plum_blossom_stakes.

    Per foot: on liftoff refresh and freeze stake candidates; on touchdown reward once with
    ``exp(-d_xy / s_xy)`` where ``d_xy`` is the minimum distance to the frozen set in the
    terrain-local horizontal plane. Inactive on non-stake terrains.
    """

    _STAKES_TERRAIN_NAME = "plum_blossom_stakes"

    def __init__(self, cfg: Any, env: ManagerBasedRlEnv):
        super().__init__(env)
        self.asset_cfg: SceneEntityCfg = cfg.params.get(
            "asset_cfg",
            SceneEntityCfg("robot", body_names=("left_ankle_roll_link", "right_ankle_roll_link")),
        )
        self.sensor_name = str(cfg.params.get("sensor_name", "contact_forces"))
        self.contact_force_threshold = float(cfg.params.get("contact_force_threshold", 1.0))
        self.s_xy = float(cfg.params.get("s_xy", 0.15))
        self.min_forward_margin = float(cfg.params.get("min_forward_margin", 0.02))
        self.max_candidates_per_foot = int(cfg.params.get("max_candidates_per_foot", 2))
        self.foot_height_offset = float(cfg.params.get("foot_height_offset", 0.065))

        asset: Entity = env.scene[self.asset_cfg.name]
        if isinstance(self.asset_cfg.body_ids, slice):
            self.body_ids = list(range(asset.num_bodies))
        else:
            self.body_ids = list(self.asset_cfg.body_ids)
        self.num_feet = len(self.body_ids)

        self._enabled = False
        terrain = env.scene.terrain
        terrain_origins = getattr(terrain, "terrain_origins", None)
        terrain_cfg = getattr(terrain, "cfg", None)
        terrain_generator_cfg = getattr(terrain_cfg, "terrain_generator", None) if terrain_cfg is not None else None
        terrain_generator = getattr(terrain, "terrain_generator", None)

        if terrain_origins is None or terrain_generator_cfg is None:
            return

        terrain_origins_t = torch.as_tensor(terrain_origins, dtype=torch.float32, device=env.device)
        num_rows = int(terrain_origins_t.shape[0])
        num_cols = int(terrain_origins_t.shape[1])
        column_names = _column_subterrain_names(terrain_generator_cfg)
        if len(column_names) != num_cols:
            return

        max_stakes = 0
        stakes_by_cell: list[list[list[tuple[float, float, float]]]] = []
        for row in range(num_rows):
            row_stakes: list[list[tuple[float, float, float]]] = []
            for col in range(num_cols):
                if column_names[col] != self._STAKES_TERRAIN_NAME:
                    row_stakes.append([])
                    continue
                sub_cfg = _sub_cfg_for_cell(
                    terrain_generator,
                    terrain_generator_cfg,
                    self._STAKES_TERRAIN_NAME,
                    row,
                    num_rows,
                    col,
                )
                if sub_cfg is None:
                    row_stakes.append([])
                    continue
                difficulty = float(getattr(sub_cfg, "difficulty", 0.5))
                stakes = plum_blossom_stake_centers_local(sub_cfg, difficulty, apply_border=True)
                row_stakes.append(stakes)
                max_stakes = max(max_stakes, len(stakes))
            stakes_by_cell.append(row_stakes)

        if max_stakes <= 0:
            return

        stakes_xy_tensor = torch.zeros((num_rows, num_cols, max_stakes, 2), dtype=torch.float32, device=env.device)
        stakes_z_tensor = torch.zeros((num_rows, num_cols, max_stakes), dtype=torch.float32, device=env.device)
        stake_counts = torch.zeros((num_rows, num_cols), dtype=torch.long, device=env.device)
        for row in range(num_rows):
            for col in range(num_cols):
                stakes = stakes_by_cell[row][col]
                count = len(stakes)
                stake_counts[row, col] = count
                if count > 0:
                    stakes_xy_tensor[row, col, :count] = torch.tensor(
                        [(x, y) for x, y, _z in stakes],
                        dtype=torch.float32,
                        device=env.device,
                    )
                    stakes_z_tensor[row, col, :count] = torch.tensor(
                        [z for _x, _y, z in stakes],
                        dtype=torch.float32,
                        device=env.device,
                    )

        stakes_cols = {col for col, name in enumerate(column_names) if name == self._STAKES_TERRAIN_NAME}
        self._enabled = len(stakes_cols) > 0
        self.terrain_origins = terrain_origins_t
        self.stake_centers_xy_local = stakes_xy_tensor
        self.stake_z_local = stakes_z_tensor
        self.stake_counts = stake_counts
        self.stakes_col_mask = torch.zeros(num_cols, dtype=torch.bool, device=env.device)
        for col in stakes_cols:
            self.stakes_col_mask[col] = True

        self.prev_in_contact = torch.zeros(env.num_envs, self.num_feet, dtype=torch.bool, device=env.device)
        self.foot_armed = torch.zeros(env.num_envs, self.num_feet, dtype=torch.bool, device=env.device)
        max_frozen = self.max_candidates_per_foot
        self.frozen_candidates_xy = torch.zeros(
            env.num_envs, self.num_feet, max_frozen, 2, dtype=torch.float32, device=env.device
        )
        self.frozen_candidate_counts = torch.zeros(env.num_envs, self.num_feet, dtype=torch.long, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if not self._enabled:
            return
        if env_ids is None:
            env_ids = slice(None)
        in_contact = self._in_contact(self._env)
        self.prev_in_contact[env_ids] = in_contact[env_ids]
        self.foot_armed[env_ids] = False
        self.frozen_candidate_counts[env_ids] = 0

    def _in_contact(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        contact_sensor: ContactSensor = env.scene[self.sensor_name]
        force_history = contact_sensor.data.force_history
        if force_history is None:
            return torch.zeros(env.num_envs, self.num_feet, dtype=torch.bool, device=env.device)
        return (
            torch.max(torch.linalg.vector_norm(force_history, dim=-1), dim=2)[0] > self.contact_force_threshold
        )

    def _stakes_active_mask(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        terrain = env.scene.terrain
        cols = terrain.terrain_types
        return self.stakes_col_mask[cols]

    def _refresh_frozen_candidates(
        self,
        env_ids: torch.Tensor,
        foot_idx: int,
        foot_x_local: torch.Tensor,
    ) -> None:
        terrain = self._env.scene.terrain
        levels = terrain.terrain_levels[env_ids]
        cols = terrain.terrain_types[env_ids]
        batch_stakes = self.stake_centers_xy_local[levels, cols]
        batch_counts = self.stake_counts[levels, cols]

        max_all = batch_stakes.shape[1]
        max_keep = self.max_candidates_per_foot
        device = batch_stakes.device
        valid = torch.arange(max_all, device=device).view(1, -1) < batch_counts.unsqueeze(-1)
        forward = batch_stakes[..., 0] > (foot_x_local.unsqueeze(-1) + self.min_forward_margin)
        mask = valid & forward

        x_ahead = batch_stakes[..., 0] - foot_x_local.unsqueeze(-1)
        x_ahead = x_ahead.masked_fill(~mask, float("inf"))
        k = min(max_keep, max_all)
        _, topk_idx = torch.topk(x_ahead, k=k, dim=-1, largest=False)
        gathered = torch.gather(batch_stakes, 1, topk_idx.unsqueeze(-1).expand(-1, -1, 2))
        topk_x_ahead = torch.gather(x_ahead, 1, topk_idx)
        keep_counts = torch.sum(torch.isfinite(topk_x_ahead), dim=-1)

        self.frozen_candidates_xy[env_ids, foot_idx, :k] = gathered
        if k < self.frozen_candidates_xy.shape[2]:
            self.frozen_candidates_xy[env_ids, foot_idx, k:] = 0.0
        self.frozen_candidate_counts[env_ids, foot_idx] = keep_counts

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg(
            "robot", body_names=("left_ankle_roll_link", "right_ankle_roll_link")
        ),
        sensor_name: str = "contact_forces",
        contact_force_threshold: float = 1.0,
        s_xy: float = 0.15,
        min_forward_margin: float = 0.02,
        max_candidates_per_foot: int = 2,
        foot_height_offset: float = 0.065,
    ) -> torch.Tensor:
        del (
            asset_cfg,
            sensor_name,
            contact_force_threshold,
            s_xy,
            min_forward_margin,
            max_candidates_per_foot,
            foot_height_offset,
        )
        if not self._enabled:
            return torch.zeros(env.num_envs, device=env.device)

        asset: Entity = env.scene[self.asset_cfg.name]
        in_contact = self._in_contact(env)
        stakes_active = self._stakes_active_mask(env)

        terrain = env.scene.terrain
        levels = terrain.terrain_levels
        cols = terrain.terrain_types
        origins = self.terrain_origins[levels, cols]
        foot_pos_w = asset.data.body_link_pos_w[:, self.body_ids, :]
        foot_x_local = foot_pos_w[..., 0] - origins[:, None, 0]
        foot_y_local = foot_pos_w[..., 1] - origins[:, None, 1]
        foot_xy = torch.stack((foot_x_local, foot_y_local), dim=-1)

        liftoff = self.prev_in_contact & ~in_contact & stakes_active.unsqueeze(-1)
        touchdown = ~self.prev_in_contact & in_contact & self.foot_armed
        touchdown &= stakes_active.unsqueeze(-1)
        touchdown &= self.frozen_candidate_counts > 0

        for foot_idx in range(self.num_feet):
            liftoff_ids = liftoff[:, foot_idx].nonzero(as_tuple=False).flatten()
            if liftoff_ids.numel() > 0:
                self.foot_armed[liftoff_ids, foot_idx] = True
                self._refresh_frozen_candidates(liftoff_ids, foot_idx, foot_x_local[liftoff_ids, foot_idx])

        min_dist = _min_xy_distance_to_candidates(
            foot_xy,
            self.frozen_candidates_xy,
            self.frozen_candidate_counts,
        )
        foot_reward = torch.exp(-min_dist / self.s_xy) * touchdown.float()
        reward = torch.sum(foot_reward, dim=1) * stakes_active.float()

        self.foot_armed = torch.where(touchdown, torch.zeros_like(self.foot_armed), self.foot_armed)
        self.frozen_candidate_counts = torch.where(
            touchdown,
            torch.zeros_like(self.frozen_candidate_counts),
            self.frozen_candidate_counts,
        )
        self.prev_in_contact = in_contact
        return reward
