"""Foothold / SSR Sec. 3.2 privileged observation terms.

Provides the ``foothold`` observation group consumed by
``instinct_rl.algorithms.FootholdAlgoMixin``:

- ``foot_region_scan_{l,r}``: absolute hit height z on a dense sole-centered grid
- ``foot_contact``, ``foot_pos_{b,w}``, ``foot_vel_b``, ``foot_sole_z``
- ``base_pose_w`` ([x, y, yaw]), ``base_lin_vel``, ``last_action``, ``terrain_level``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers import SceneEntityCfg
from mjlab.sensor import ContactSensor, RayCastSensor
from mjlab.utils.lab_api import math as math_utils

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_DEFAULT_FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")


def foot_region_hit_z(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Absolute world hit height z for a foot-region raycast grid.

  Misses (distance < 0) are filled with ``frame_z - max_distance`` so unsupported
  cells look like deep gaps to support-deficiency math.
  """
  sensor: RayCastSensor = env.scene[sensor_name]
  hit_z = sensor.data.hit_pos_w[..., 2]
  miss = sensor.data.distances < 0.0
  miss_z = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.cfg.max_distance
  return torch.where(miss, miss_z.expand_as(hit_z), hit_z)


def foot_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str = "contact_forces",
  force_threshold: float = 1.0,
) -> torch.Tensor:
  """Binary left/right foot contact from contact sensor forces. Shape [N, 2]."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  force = contact_sensor.data.force
  if force is None:
    found = contact_sensor.data.found
    assert found is not None
    return (found > 0).to(dtype=torch.float32)
  force_norm = torch.linalg.vector_norm(force, dim=-1)
  return (force_norm > force_threshold).to(dtype=torch.float32)


def _resolve_foot_body_ids(
  asset: Entity,
  asset_cfg: SceneEntityCfg,
) -> list[int]:
  if asset_cfg.body_ids is not None and len(asset_cfg.body_ids) > 0:
    return list(asset_cfg.body_ids)
  # Fallback: resolve by default ankle names if cfg was not pre-resolved.
  body_ids: list[int] = []
  for name in _DEFAULT_FOOT_BODIES:
    body_ids.append(int(asset.find_bodies(name)[0][0]))
  return body_ids


def foot_pos_w(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg(
    "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
  ),
) -> torch.Tensor:
  """Foot xy positions in world frame. Shape [N, 4] = (L_x, L_y, R_x, R_y)."""
  asset: Entity = env.scene[asset_cfg.name]
  body_ids = _resolve_foot_body_ids(asset, asset_cfg)
  pos = asset.data.body_link_pos_w[:, body_ids, :2]
  return pos.reshape(env.num_envs, -1)


def foot_pos_b(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg(
    "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
  ),
) -> torch.Tensor:
  """Foot xy positions in yaw-aligned base frame. Shape [N, 4]."""
  asset: Entity = env.scene[asset_cfg.name]
  body_ids = _resolve_foot_body_ids(asset, asset_cfg)
  root_pos = asset.data.root_link_pos_w
  root_quat = asset.data.root_link_quat_w
  yaw_q = math_utils.yaw_quat(root_quat)
  foot_pos = asset.data.body_link_pos_w[:, body_ids]
  delta = foot_pos - root_pos.unsqueeze(1)
  num_feet = delta.shape[1]
  yaw_q_feet = yaw_q.unsqueeze(1).expand(-1, num_feet, -1)
  foot_b = math_utils.quat_apply_inverse(yaw_q_feet, delta)
  return foot_b[..., :2].reshape(env.num_envs, -1)


def foot_vel_b(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg(
    "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
  ),
) -> torch.Tensor:
  """Foot xy linear velocities in yaw-aligned base frame. Shape [N, 4]."""
  asset: Entity = env.scene[asset_cfg.name]
  body_ids = _resolve_foot_body_ids(asset, asset_cfg)
  root_quat = asset.data.root_link_quat_w
  yaw_q = math_utils.yaw_quat(root_quat)
  foot_vel_w = asset.data.body_link_lin_vel_w[:, body_ids]
  num_feet = foot_vel_w.shape[1]
  yaw_q_feet = yaw_q.unsqueeze(1).expand(-1, num_feet, -1)
  foot_vel_b = math_utils.quat_apply_inverse(yaw_q_feet, foot_vel_w)
  return foot_vel_b[..., :2].reshape(env.num_envs, -1)


def foot_sole_z(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg(
    "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
  ),
  sole_z_offset: float = -0.055,
) -> torch.Tensor:
  """Approximate sole plane world height per foot. Shape [N, 2].

  Uses ankle link z plus a fixed offset matching Adam-SP foot volume points.
  """
  asset: Entity = env.scene[asset_cfg.name]
  body_ids = _resolve_foot_body_ids(asset, asset_cfg)
  return asset.data.body_link_pos_w[:, body_ids, 2] + sole_z_offset


def base_pose_w_xy_yaw(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Base pose as [x, y, yaw] in world frame. Shape [N, 3]."""
  asset: Entity = env.scene[asset_cfg.name]
  pos = asset.data.root_link_pos_w
  quat = asset.data.root_link_quat_w
  _, _, yaw = math_utils.euler_xyz_from_quat(quat)
  return torch.stack((pos[:, 0], pos[:, 1], yaw), dim=-1)


def terrain_level(
  env: ManagerBasedRlEnv,
) -> torch.Tensor:
  """Current terrain curriculum level per env. Shape [N, 1]."""
  terrain = env.scene.terrain
  levels = terrain.terrain_levels.float().unsqueeze(-1)
  return levels
