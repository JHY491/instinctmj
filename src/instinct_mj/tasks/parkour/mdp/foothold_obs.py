"""foothold 观测组用到的 privileged 观测函数（base 系脚状态、世界系脚/base 位姿、地形等级）。"""

from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse


def _foot_ids(asset_cfg: SceneEntityCfg) -> list[int]:
    return list(asset_cfg.body_ids)  # [left_ankle_roll, right_ankle_roll]


def foot_pos_b(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """两脚位置在 base 系 xy，(N,4)=[lx,ly,rx,ry]。"""
    asset: Entity = env.scene[asset_cfg.name]
    ids = _foot_ids(asset_cfg)
    root_pos = asset.data.root_link_pos_w
    root_quat = asset.data.root_link_quat_w
    out = []
    for i in ids:
        rel_w = asset.data.body_link_pos_w[:, i, :] - root_pos
        rel_b = quat_apply_inverse(root_quat, rel_w)
        out.append(rel_b[:, :2])
    return torch.cat(out, dim=-1)


def foot_sole_z(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, foot_height_offset: float = 0.065) -> torch.Tensor:
    """两脚鞋底平面世界高度 (N,2)=[left,right]，与 feet_unsupported 一致。"""
    asset: Entity = env.scene[asset_cfg.name]
    ids = _foot_ids(asset_cfg)
    z = torch.stack([asset.data.body_link_pos_w[:, i, 2] for i in ids], dim=-1)
    return z - foot_height_offset


def foot_vel_b(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """两脚线速度在 base 系 xy，(N,4)。"""
    asset: Entity = env.scene[asset_cfg.name]
    ids = _foot_ids(asset_cfg)
    root_quat = asset.data.root_link_quat_w
    out = []
    for i in ids:
        vel_b = quat_apply_inverse(root_quat, asset.data.body_link_lin_vel_w[:, i, :])
        out.append(vel_b[:, :2])
    return torch.cat(out, dim=-1)


def foot_pos_w_xy(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """两脚位置世界系 xy，(N,4)=[lx,ly,rx,ry]（供 GT 回填）。"""
    asset: Entity = env.scene[asset_cfg.name]
    ids = _foot_ids(asset_cfg)
    out = [asset.data.body_link_pos_w[:, i, :2] for i in ids]
    return torch.cat(out, dim=-1)


def base_xy_yaw_w(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """base 世界系 [x, y, yaw]，(N,3)（供 GT 回填的世界→base 变换）。"""
    asset: Entity = env.scene[asset_cfg.name]
    xy = asset.data.root_link_pos_w[:, :2]
    yaw = asset.data.heading_w.unsqueeze(-1)
    return torch.cat([xy, yaw], dim=-1)


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str, contact_force_threshold: float = 1.0) -> torch.Tensor:
    """两脚接触 0/1，(N,2)=[left,right]。与 rewards.py 中接触判定一致。"""
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    force = torch.max(torch.linalg.vector_norm(sensor.data.force_history, dim=-1), dim=2)[0]
    return (force > contact_force_threshold).float()


def terrain_level(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """该 env 当前地形等级，(N,1) float。"""
    del asset_cfg
    level = env.scene.terrain.terrain_levels
    return level.unsqueeze(-1).float()


def foot_region_hit_z(env: ManagerBasedRlEnv, sensor_name: str, miss_value: float = -10.0) -> torch.Tensor:
    """区域扫描每条 ray 命中点的世界绝对高度 z，(N, num_rays)。

    与 height_scan（返回相对下方距离）不同：ρ 数学需要绝对高度。miss（无命中）用极低值，
    视作深坑=无支撑。
    """
    from mjlab.sensor import RayCastSensor

    sensor: RayCastSensor = env.scene[sensor_name]
    z = sensor.data.hit_pos_w[..., 2]
    miss = sensor.data.distances < 0
    return torch.where(miss, torch.full_like(z, miss_value), z)
