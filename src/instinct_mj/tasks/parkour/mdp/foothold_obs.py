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
