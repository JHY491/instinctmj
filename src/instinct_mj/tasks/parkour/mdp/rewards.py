from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from mjlab.managers import SceneEntityCfg
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.sensor import ContactSensor, RayCastSensor
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

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


def ang_vel_z_excess_when_no_yaw_cmd(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cmd_epsilon: float = 0.01,
    ang_vel_tolerance: float = 0.15,
) -> torch.Tensor:
    """Penalize |ωz| above tolerance when yaw angular velocity command is near zero (body frame).

    Returns a nonnegative penalty magnitude per env; use a negative reward weight.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    cmd_z_abs = torch.abs(cmd[:, 2])
    wz_abs = torch.abs(asset.data.root_link_ang_vel_b[:, 2])
    no_yaw_cmd = cmd_z_abs < cmd_epsilon
    excess = torch.clamp(wz_abs - ang_vel_tolerance, min=0.0)
    return torch.square(excess) * no_yaw_cmd.float()


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
) -> torch.Tensor:
    """Reward foot heading alignment with an angle-sensitive exponential shaping."""
    asset: Entity = env.scene[asset_cfg.name]
    command_b = env.command_manager.get_command(command_name)
    assert command_b is not None
    command_xy_b = command_b[:, :2]
    command_norm = torch.linalg.vector_norm(command_xy_b, dim=-1)
    is_active = command_norm > cmd_threshold

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
