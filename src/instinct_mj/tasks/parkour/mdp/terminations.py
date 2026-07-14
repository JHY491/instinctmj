from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers import SceneEntityCfg
from mjlab.sensor import ContactSensor, RayCastSensor

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv


def sub_terrain_out_of_bounds(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_buffer: float = 3.0,
) -> torch.Tensor:
    """Terminate when the actor move too close to the edge of the sub terrain.

    If the actor moves too close to the edge of the sub terrain, the termination is activated. The distance
    to the edge of the sub terrain is calculated based on the size of the sub terrain and the distance buffer.
    """
    terrain_type = env.cfg.scene.terrain.terrain_type
    if terrain_type == "plane":
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    elif terrain_type == "hacked_generator":
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        # extract the used quantities (to enable type-hinting)
        asset: Entity = env.scene[asset_cfg.name]

        root_pos_w = asset.data.root_link_pos_w
        terrain_origins = env.scene.terrain.env_origins

        # check if the agent is out of bounds
        x_out_of_bounds = torch.abs(root_pos_w[:, 0] - terrain_origins[:, 0]) > 0.5 * grid_width - distance_buffer
        y_out_of_bounds = torch.abs(root_pos_w[:, 1] - terrain_origins[:, 1]) > 0.5 * grid_length - distance_buffer
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    else:
        raise ValueError("Received unsupported terrain type, must be one of: 'plane', 'hacked_generator'.")


def root_height_below_env_origin_minimum(
    env: ManagerBasedRlEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when root height drops below env origin + minimum height."""
    asset: Entity = env.scene[asset_cfg.name]
    terrain_base_height = torch.clamp(env.scene.env_origins[:, 2], max=0.0)
    root_pos_w = asset.data.root_link_pos_w
    return root_pos_w[:, 2] - terrain_base_height < minimum_height


def feet_on_pit_bottom(
    env: ManagerBasedRlEnv,
    contact_sensor_name: str,
    left_sole_scanner_name: str,
    right_sole_scanner_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_height_offset: float = 0.065,
    ground_contact_tolerance: float = 0.04,
    pit_ground_height_max: float = -0.04,
    contact_force_threshold: float = 1.0,
    active_terrain_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Terminate when a loaded foot supports on terrain below stake/plate height.

    If ``active_terrain_names`` is set, only envs whose current sub-terrain name is in
    that set can terminate (avoids false positives on stairs/slopes in mixed curricula).
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

    origin_z = env.scene.env_origins[:, 2].unsqueeze(-1)
    left_local_ground_z = left_hit_z - origin_z
    right_local_ground_z = right_hit_z - origin_z

    left_on_ground = (left_sole_z - left_hit_z).abs() < ground_contact_tolerance
    right_on_ground = (right_sole_z - right_hit_z).abs() < ground_contact_tolerance

    left_pit = left_on_ground & left_valid & (left_local_ground_z < pit_ground_height_max)
    right_pit = right_on_ground & right_valid & (right_local_ground_z < pit_ground_height_max)

    left_fail = is_contact[:, 0:1] & left_pit.any(dim=-1, keepdim=True)
    right_fail = is_contact[:, 1:2] & right_pit.any(dim=-1, keepdim=True)
    fail = torch.any(left_fail | right_fail, dim=1)

    if active_terrain_names is not None:
        from instinct_mj.tasks.parkour.mdp.foothold_obs import _column_subterrain_names

        terrain = env.scene.terrain
        col_names = _column_subterrain_names(terrain.cfg.terrain_generator)
        allow = frozenset(active_terrain_names)
        active_by_col = torch.tensor(
            [name in allow for name in col_names],
            device=env.device,
            dtype=torch.bool,
        )
        fail = fail & active_by_col[terrain.terrain_types]

    return fail


def illegal_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold: float = 1.0,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_name]
    in_contact = torch.max(torch.linalg.vector_norm(contact_sensor.data.force_history, dim=-1), dim=2)[0] > threshold
    return torch.any(in_contact, dim=1)
