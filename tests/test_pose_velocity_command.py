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
  descend_vz_ema_alpha: float = 0.1
  descend_vz_threshold: float = -0.1
  descend_speed_scale: float = 0.5


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
  command._base_vz_ema = torch.zeros(1)
  command.descend_speed_indices = torch.zeros(1, dtype=torch.bool)
  command.random_velocity_indices = torch.full((1,), random_velocity, dtype=torch.bool)
  command.random_lin_vel_x = torch.tensor([0.6], dtype=torch.float32)
  command.random_lin_vel_y = torch.tensor([0.0], dtype=torch.float32)
  command.random_ang_vel_z = torch.tensor([0.2], dtype=torch.float32)
  command.robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float32),
      root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
      root_link_lin_vel_w=torch.zeros(1, 3),
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


def test_target_reached_clears_when_flat_patch_advances() -> None:
  command = PoseVelocityCommand.__new__(PoseVelocityCommand)
  command._env = SimpleNamespace(num_envs=1, device="cpu", step_dt=0.02, max_episode_length=1000)
  command.cfg = SimpleNamespace(
    flat_patch_advance_on_arrival=True,
    flat_patch_arrival_xy_dist=0.4,
    flat_patch_arrival_hold_steps=1,
    flat_patch_resample_velocity_on_advance=False,
    guided_resample_velocity_on_advance=False,
    rel_standing_envs=0.0,
  )
  command._target_reached = torch.ones(1, dtype=torch.bool)
  command._flat_patch_idx = torch.zeros(1, dtype=torch.long)
  command._flat_patch_arrival_hold = torch.ones(1, dtype=torch.long)
  command._flat_patch_seq_terminal = torch.zeros(1, dtype=torch.bool)
  command.pos_command_w = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
  command.valid_targets = torch.tensor([[[[0.0, 0.0, 0.0], [3.7, 0.0, 0.0]]]], dtype=torch.float32)
  command._flat_patch_target_fixed_sequence_by_col = torch.ones(1, dtype=torch.bool)
  command._flat_patch_reset_index_by_col = torch.zeros(1, dtype=torch.long)
  command._flat_patch_num_valid = None
  command._guided_mode = False
  command._guided_counts = None
  command.robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
    )
  )
  command.terrain = SimpleNamespace(
    terrain_levels=torch.zeros(1, dtype=torch.long),
    terrain_types=torch.zeros(1, dtype=torch.long),
  )
  command.lin_vel_x_range = torch.tensor([[0.3, 1.0]], dtype=torch.float32)
  command.lin_vel_y_range = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
  command.ang_vel_z_range = torch.tensor([[-0.5, 0.5]], dtype=torch.float32)
  command.random_velocity_indices = torch.zeros(1, dtype=torch.bool)
  command.is_standing_env = torch.zeros(1, dtype=torch.bool)

  command._maybe_advance_flat_patch_by_arrival()

  assert not command._target_reached.item()
  assert command._flat_patch_idx.item() == 1
  assert torch.allclose(command.pos_command_w[0], torch.tensor([3.7, 0.0, 0.0]))


def _make_metrics_command() -> PoseVelocityCommand:
  command = _make_command(random_velocity=False)
  command.cfg.resampling_time_range = (8.0, 12.0)
  command.cfg.lin_vel_metrics_std = 0.5
  command.cfg.ang_vel_metrics_std = 0.5
  command._env.step_dt = 0.02
  command._env.max_episode_length = 1000
  command.robot.data.root_link_lin_vel_b = torch.tensor([[0.6, 0.0, 0.0]])
  command.robot.data.root_link_ang_vel_b = torch.zeros(1, 3)
  command.vel_command_b = torch.tensor([[0.6, 0.0, 0.0]])
  command.pos_command_w = torch.tensor([[7.0, 0.0, 0.0]])
  keys = (
    "error_vel_xy", "error_vel_yaw", "tracking_exp_vel_xy", "tracking_exp_vel_yaw",
    "command_nonzero_ratio", "target_near_ratio", "standing_env_ratio", "random_velocity_env_ratio",
  )
  command.metrics = {key: torch.zeros(1) for key in keys}
  command.metrics.update({f"mean_{key}": torch.zeros(1) for key in keys})
  command.metrics["episode_steps"] = torch.zeros(1)
  return command


def test_short_episode_exposes_true_tracking_without_changing_curriculum_score() -> None:
  command = _make_metrics_command()
  for _ in range(100):
    command._update_metrics()

  assert torch.allclose(command.metrics["tracking_exp_vel_xy"], torch.tensor([0.1]))
  assert torch.allclose(command.metrics["mean_tracking_exp_vel_xy"], torch.ones(1))
  assert torch.allclose(command.metrics["mean_command_nonzero_ratio"], torch.ones(1))
  assert command.metrics["episode_steps"].item() == 100


def test_tracking_means_measure_error_and_command_duty_cycle() -> None:
  command = _make_metrics_command()
  command._update_metrics()
  # The second sample has a 0.6 m/s error and a zero command.
  command.vel_command_b.zero_()
  command._update_metrics()

  assert torch.allclose(command.metrics["mean_error_vel_xy"], torch.tensor([0.3]))
  expected_score = (1 + torch.exp(torch.tensor(-0.6**2 / 0.5**2))) / 2
  assert torch.allclose(command.metrics["mean_tracking_exp_vel_xy"], expected_score.view(1))
  assert torch.allclose(command.metrics["mean_command_nonzero_ratio"], torch.tensor([0.5]))


def test_metric_means_reset_only_selected_environments() -> None:
  from mjlab.managers.command_manager import CommandTerm

  command = _make_metrics_command()
  command._env.num_envs = 2
  for name in ("pos_command_w", "vel_command_b", "is_standing_env", "random_velocity_indices"):
    value = getattr(command, name)
    setattr(command, name, value.repeat(2, 1) if value.ndim == 2 else value.repeat(2))
  for name, value in vars(command.robot.data).items():
    setattr(command.robot.data, name, value.repeat(2, 1) if value.ndim == 2 else value.repeat(2))
  for name, value in command.metrics.items():
    command.metrics[name] = value.repeat(2)
  for _ in range(4):
    command._update_metrics()
  command.command_counter = torch.zeros(2, dtype=torch.long)
  command._resample = lambda env_ids: None

  extras = CommandTerm.reset(command, torch.tensor([0]))
  assert extras["mean_tracking_exp_vel_xy"] == 1.0
  assert extras["episode_steps"] == 4
  command.vel_command_b.zero_()
  command._update_metrics()

  assert torch.allclose(command.metrics["episode_steps"], torch.tensor([1.0, 5.0]))
  assert torch.allclose(command.metrics["mean_error_vel_xy"], torch.tensor([0.6, 0.12]))
  assert torch.allclose(command.metrics["mean_command_nonzero_ratio"], torch.tensor([0.0, 0.8]))
