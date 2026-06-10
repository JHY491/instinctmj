from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from instinct_mj.tasks.parkour.mdp.commands.pose_velocity_command import PoseVelocityCommand


@dataclass(kw_only=True)
class _Cfg:
  velocity_control_stiffness: float = 2.0
  heading_control_stiffness: float = 2.0
  only_positive_lin_vel_x: bool = True
  yaw_vel_gate_threshold: float | None = None
  lin_vel_threshold: float = 0.0
  ang_vel_threshold: float = 0.0
  target_dis_threshold: float = 0.4


def _make_command(*, random_velocity: bool) -> PoseVelocityCommand:
  command = PoseVelocityCommand.__new__(PoseVelocityCommand)
  command._env = SimpleNamespace(num_envs=1, device="cpu")
  command.cfg = _Cfg()
  command.pos_command_w = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
  command.pos_command_b = torch.zeros(1, 3)
  command.heading_command_w = torch.zeros(1)
  command.vel_command_b = torch.zeros(1, 3)
  command.max_command_b = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
  command.ang_vel_z_range = torch.tensor([[-1.0, 1.0]], dtype=torch.float32)
  command.is_standing_env = torch.zeros(1, dtype=torch.bool)
  command._target_reached = torch.zeros(1, dtype=torch.bool)
  command.random_velocity_indices = torch.full((1,), random_velocity, dtype=torch.bool)
  command.random_lin_vel_x = torch.tensor([0.6], dtype=torch.float32)
  command.random_lin_vel_y = torch.tensor([0.0], dtype=torch.float32)
  command.random_ang_vel_z = torch.tensor([0.2], dtype=torch.float32)
  command.robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float32),
      root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
      heading_w=torch.zeros(1, dtype=torch.float32),
    )
  )
  command._maybe_advance_guided_waypoints = lambda: None
  command._maybe_advance_flat_patch_by_arrival = lambda: None
  command._velocity_stop_xy_dist = lambda: torch.full((1,), command.cfg.target_dis_threshold)
  return command


def test_random_velocity_command_stops_inside_target_radius() -> None:
  command = _make_command(random_velocity=True)

  command._update_command()

  assert torch.allclose(command.vel_command_b, torch.zeros_like(command.vel_command_b))


def test_command_remains_stopped_after_target_radius_is_crossed() -> None:
  command = _make_command(random_velocity=False)

  command._update_command()
  command.robot.data.root_link_pos_w[:, :2] = torch.tensor([[0.5, 0.0]], dtype=torch.float32)
  command._update_command()

  assert torch.allclose(command.vel_command_b, torch.zeros_like(command.vel_command_b))
