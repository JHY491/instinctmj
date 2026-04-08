from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import TYPE_CHECKING, TypedDict, cast

import torch
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers import SceneEntityCfg

from instinct_mj.terrains import TerrainImporter

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv

    from mjlab.managers import CurriculumTermCfg
    from instinct_mj.tasks.parkour.mdp.commands import PoseVelocityCommand
    from instinct_mj.tasks.parkour.mdp.commands.commands_cfg import PoseVelocityCommandCfg


class PoseVelocityStage(TypedDict, total=False):
    step: int
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]
    velocity_ranges: dict[str, dict[str, tuple[float, float]]]


def tracking_exp_vel(
    env: ManagerBasedRlEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lin_vel_threshold: tuple = (0.3, 0.6),
    ang_vel_threshold: tuple = (0.3, 0.5),
) -> torch.Tensor:
    """Curriculum based on the velocity tracking performance (exponential score) of the robot.

    This term is used to increase the difficulty of the terrain when the robot tracks its commanded velocity well
    (high score). It decreases the difficulty when the robot tracks its commanded velocity poorly (low score).

    Args:
        env: The learning environment.
        env_ids: The environment ids for which the curriculum should be computed.
        asset_cfg: The configuration of the robot articulation in the scene.
        lin_vel_threshold: A tuple specifying the lower and upper threshold for the linear velocity tracking
            score (exponential kernel).
            If the score is below the lower threshold (poor tracking), the terrain difficulty is decreased.
            If the score is above the upper threshold (good tracking), the terrain difficulty is increased.
        ang_vel_threshold: A tuple specifying the lower and upper threshold for the angular velocity tracking
            score (exponential kernel).
            Similar logic applies as lin_vel_threshold.
    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    env_ids_tensor = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    asset: Entity = env.scene[asset_cfg.name]
    terrain = cast(TerrainImporter, env.scene.terrain)
    command = env.command_manager.get_term("base_velocity")
    assert command is not None
    tracking_exp_vel_xy = command.metrics["tracking_exp_vel_xy"][env_ids_tensor]
    tracking_exp_vel_yaw = command.metrics["tracking_exp_vel_yaw"][env_ids_tensor]
    move_up = (tracking_exp_vel_xy > lin_vel_threshold[1]) * (tracking_exp_vel_yaw > ang_vel_threshold[1])
    move_down = tracking_exp_vel_xy < lin_vel_threshold[0]
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids_tensor, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def commands_vel(
    env: ManagerBasedRlEnv,
    env_ids: Sequence[int],
    command_name: str,
    velocity_stages: list[PoseVelocityStage],
) -> dict[str, torch.Tensor]:
    del env_ids  # Unused.
    command_term = cast("PoseVelocityCommand", env.command_manager.get_term(command_name))
    cfg = cast("PoseVelocityCommandCfg", command_term.cfg)
    active_stage: PoseVelocityStage | None = None
    for stage in velocity_stages:
        if env.common_step_counter >= stage.get("step", 0):
            active_stage = stage
    if active_stage is None:
        active_stage = velocity_stages[0]

    _apply_velocity_stage(command_term, cfg, active_stage)

    return {
        "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
        "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
        "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
        "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
        "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
        "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
    }


class adaptive_commands_vel_by_tracking(ManagerTermBase):
    """Adapt command velocity ranges based on delayed tracking performance."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        self.command_name = cfg.params.get("command_name", "base_velocity")
        self.delayed_lin_tracking = float(cfg.params.get("init_lin_tracking", 0.0))
        self.delayed_ang_tracking = float(cfg.params.get("init_ang_tracking", 0.0))
        self._command_term = cast("PoseVelocityCommand", env.command_manager.get_term(self.command_name))
        self._command_cfg = cast("PoseVelocityCommandCfg", self._command_term.cfg)
        self._base_ranges = {
            "lin_vel_x": self._command_cfg.ranges.lin_vel_x,
            "lin_vel_y": self._command_cfg.ranges.lin_vel_y,
            "ang_vel_z": self._command_cfg.ranges.ang_vel_z,
        }
        default_stage: PoseVelocityStage = {
            "lin_vel_x": cfg.params.get("default_lin_vel_x", self._base_ranges["lin_vel_x"]),
            "lin_vel_y": cfg.params.get("default_lin_vel_y", self._base_ranges["lin_vel_y"]),
            "ang_vel_z": cfg.params.get("default_ang_vel_z", self._base_ranges["ang_vel_z"]),
        }
        self._base_velocity_ranges = (
            copy.deepcopy(self._command_cfg.velocity_ranges) if self._command_cfg.velocity_ranges is not None else None
        )

        self.lin_velocity_stages = cast(
            list[PoseVelocityStage],
            cfg.params.get("lin_velocity_stages", cfg.params.get("velocity_stages", [])),
        )
        if not self.lin_velocity_stages:
            self.lin_velocity_stages = [copy.deepcopy(default_stage)]

        self.ang_velocity_stages = cast(list[PoseVelocityStage], cfg.params.get("ang_velocity_stages", []))
        if not self.ang_velocity_stages:
            self.ang_velocity_stages = [{"ang_vel_z": self._base_ranges["ang_vel_z"]}]

        self.current_lin_stage = min(cfg.params.get("init_lin_stage", cfg.params.get("init_stage", 0)), len(self.lin_velocity_stages) - 1)
        self.current_ang_stage = min(cfg.params.get("init_ang_stage", cfg.params.get("init_stage", 0)), len(self.ang_velocity_stages) - 1)
        effective_stage = _compose_velocity_stage(
            self._base_ranges,
            self._base_velocity_ranges,
            self.lin_velocity_stages[self.current_lin_stage],
            self.ang_velocity_stages[self.current_ang_stage],
        )
        _apply_velocity_stage(self._command_term, self._command_cfg, effective_stage)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: Sequence[int],
        command_name: str = "base_velocity",
        velocity_stages: list[PoseVelocityStage] | None = None,
        lin_velocity_stages: list[PoseVelocityStage] | None = None,
        ang_velocity_stages: list[PoseVelocityStage] | None = None,
        lin_vel_threshold: tuple[float, float] = (0.45, 0.7),
        ang_vel_threshold: tuple[float, float] = (0.35, 0.6),
        smoothing: float = 0.2,
        min_envs: int = 1,
        allow_lin_stage_regression: bool = True,
        allow_ang_stage_regression: bool = True,
        init_stage: int = 0,
        init_lin_stage: int = 0,
        init_ang_stage: int = 0,
        init_lin_tracking: float = 0.0,
        init_ang_tracking: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        del command_name, init_stage, init_lin_stage, init_ang_stage, init_lin_tracking, init_ang_tracking  # Stored during initialization.
        if lin_velocity_stages is not None:
            self.lin_velocity_stages = lin_velocity_stages
        elif velocity_stages is not None:
            self.lin_velocity_stages = velocity_stages
        if ang_velocity_stages is not None:
            self.ang_velocity_stages = ang_velocity_stages

        if len(env_ids) >= min_envs:
            tracking_exp_vel_xy = self._command_term.metrics["tracking_exp_vel_xy"][env_ids]
            tracking_exp_vel_yaw = self._command_term.metrics["tracking_exp_vel_yaw"][env_ids]
            current_lin_tracking = float(tracking_exp_vel_xy.mean())
            current_ang_tracking = float(tracking_exp_vel_yaw.mean())
            self.delayed_lin_tracking = (1.0 - smoothing) * self.delayed_lin_tracking + smoothing * current_lin_tracking
            self.delayed_ang_tracking = (1.0 - smoothing) * self.delayed_ang_tracking + smoothing * current_ang_tracking

            lin_move_up = self.delayed_lin_tracking > lin_vel_threshold[1]
            lin_move_down = self.delayed_lin_tracking < lin_vel_threshold[0]
            if lin_move_up and self.current_lin_stage < len(self.lin_velocity_stages) - 1:
                self.current_lin_stage += 1
            elif allow_lin_stage_regression and lin_move_down and self.current_lin_stage > 0:
                self.current_lin_stage -= 1

            ang_move_up = self.delayed_ang_tracking > ang_vel_threshold[1]
            ang_move_down = self.delayed_ang_tracking < ang_vel_threshold[0]
            if ang_move_up and self.current_ang_stage < len(self.ang_velocity_stages) - 1:
                self.current_ang_stage += 1
            elif allow_ang_stage_regression and ang_move_down and self.current_ang_stage > 0:
                self.current_ang_stage -= 1

        effective_stage = _compose_velocity_stage(
            self._base_ranges,
            self._base_velocity_ranges,
            self.lin_velocity_stages[self.current_lin_stage],
            self.ang_velocity_stages[self.current_ang_stage],
        )
        _apply_velocity_stage(self._command_term, self._command_cfg, effective_stage)
        metrics = {
            "stage_index": torch.tensor(float(self.current_lin_stage)),
            "lin_stage_index": torch.tensor(float(self.current_lin_stage)),
            "ang_stage_index": torch.tensor(float(self.current_ang_stage)),
            "tracking_lin": torch.tensor(self.delayed_lin_tracking),
            "tracking_ang": torch.tensor(self.delayed_ang_tracking),
            "lin_vel_x_min": torch.tensor(self._command_cfg.ranges.lin_vel_x[0]),
            "lin_vel_x_max": torch.tensor(self._command_cfg.ranges.lin_vel_x[1]),
            "lin_vel_y_min": torch.tensor(self._command_cfg.ranges.lin_vel_y[0]),
            "lin_vel_y_max": torch.tensor(self._command_cfg.ranges.lin_vel_y[1]),
            "ang_vel_z_min": torch.tensor(self._command_cfg.ranges.ang_vel_z[0]),
            "ang_vel_z_max": torch.tensor(self._command_cfg.ranges.ang_vel_z[1]),
        }
        if self._command_cfg.velocity_ranges is not None:
            for terrain_name, terrain_ranges in self._command_cfg.velocity_ranges.items():
                sanitized_name = terrain_name.replace("-", "_")
                metrics[f"{sanitized_name}_lin_vel_x_max"] = torch.tensor(terrain_ranges["lin_vel_x"][1])
                metrics[f"{sanitized_name}_ang_vel_z_max"] = torch.tensor(terrain_ranges["ang_vel_z"][1])
        return metrics


def _compose_velocity_stage(
    base_ranges: dict[str, tuple[float, float]],
    base_velocity_ranges: dict[str, dict[str, tuple[float, float]]] | None,
    *stages: PoseVelocityStage,
) -> PoseVelocityStage:
    effective_stage: PoseVelocityStage = {
        "lin_vel_x": base_ranges["lin_vel_x"],
        "lin_vel_y": base_ranges["lin_vel_y"],
        "ang_vel_z": base_ranges["ang_vel_z"],
    }
    if base_velocity_ranges is not None:
        effective_stage["velocity_ranges"] = copy.deepcopy(base_velocity_ranges)

    for stage in stages:
        if "lin_vel_x" in stage:
            effective_stage["lin_vel_x"] = stage["lin_vel_x"]
        if "lin_vel_y" in stage:
            effective_stage["lin_vel_y"] = stage["lin_vel_y"]
        if "ang_vel_z" in stage:
            effective_stage["ang_vel_z"] = stage["ang_vel_z"]
        if "velocity_ranges" in stage:
            if "velocity_ranges" not in effective_stage:
                effective_stage["velocity_ranges"] = {}
            for terrain_name, terrain_ranges in stage["velocity_ranges"].items():
                if terrain_name not in effective_stage["velocity_ranges"]:
                    effective_stage["velocity_ranges"][terrain_name] = {}
                effective_stage["velocity_ranges"][terrain_name].update(terrain_ranges)
    return effective_stage


def _apply_velocity_stage(
    command_term: "PoseVelocityCommand",
    cfg: "PoseVelocityCommandCfg",
    stage: PoseVelocityStage,
) -> None:
    if "lin_vel_x" in stage:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
        command_term.random_lin_vel_x_range[:, 0] = stage["lin_vel_x"][0]
        command_term.random_lin_vel_x_range[:, 1] = stage["lin_vel_x"][1]
    if "lin_vel_y" in stage:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
        command_term.random_lin_vel_y_range[:, 0] = stage["lin_vel_y"][0]
        command_term.random_lin_vel_y_range[:, 1] = stage["lin_vel_y"][1]
    if "ang_vel_z" in stage:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
        command_term.random_ang_vel_z_range[:, 0] = stage["ang_vel_z"][0]
        command_term.random_ang_vel_z_range[:, 1] = stage["ang_vel_z"][1]

    if "velocity_ranges" in stage and stage["velocity_ranges"] is not None:
        cfg.velocity_ranges = copy.deepcopy(stage["velocity_ranges"])
        _update_velocity_ranges_by_terrain(command_term, cfg.velocity_ranges)


def _update_velocity_ranges_by_terrain(
    command_term: "PoseVelocityCommand",
    velocity_ranges: dict[str, dict[str, tuple[float, float]]],
) -> None:
    terrain = command_term.terrain
    terrain_generator_cfg = terrain.cfg.terrain_generator
    if terrain_generator_cfg is None:
        return

    sub_terrain_names = list(terrain_generator_cfg.sub_terrains.keys())
    proportions = [sub_cfg.proportion for sub_cfg in terrain_generator_cfg.sub_terrains.values()]
    total_proportion = sum(proportions)
    cumulative = []
    running = 0.0
    for proportion in proportions:
        running += proportion / total_proportion
        cumulative.append(running)

    sub_indices: list[int] = []
    for index in range(terrain_generator_cfg.num_cols):
        ratio = index / terrain_generator_cfg.num_cols + 0.001
        sub_index = next(i for i, threshold in enumerate(cumulative) if ratio < threshold)
        sub_indices.append(sub_index)

    for terrain_name, terrain_velocity_range in velocity_ranges.items():
        if terrain_name not in sub_terrain_names:
            raise RuntimeError(f"Terrain type {terrain_name} not found in the terrain generator sub-terrain names.")

        terrain_type_index = sub_terrain_names.index(terrain_name)
        type_indices = [index for index, sub_index in enumerate(sub_indices) if sub_index == terrain_type_index]
        if not type_indices:
            continue

        env_mask = torch.zeros_like(terrain.terrain_types, dtype=torch.bool)
        for type_index in type_indices:
            env_mask |= terrain.terrain_types == type_index
        env_indices = torch.where(env_mask)[0]
        if len(env_indices) == 0:
            continue

        command_term.lin_vel_x_range[env_indices, 0] = terrain_velocity_range["lin_vel_x"][0]
        command_term.lin_vel_x_range[env_indices, 1] = terrain_velocity_range["lin_vel_x"][1]
        command_term.lin_vel_y_range[env_indices, 0] = terrain_velocity_range["lin_vel_y"][0]
        command_term.lin_vel_y_range[env_indices, 1] = terrain_velocity_range["lin_vel_y"][1]
        command_term.ang_vel_z_range[env_indices, 0] = terrain_velocity_range["ang_vel_z"][0]
        command_term.ang_vel_z_range[env_indices, 1] = terrain_velocity_range["ang_vel_z"][1]
