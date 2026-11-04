"""G1 parkour env cfg builders (mjlab-native style).

Provides a small flat base `ManagerBasedRlEnvCfg` for the 29-DoF torso-base
popsicle G1 that downstream parkour factories extend with terrain, sensors,
and rewards.
"""

from __future__ import annotations

import copy
import math

import mjlab.envs.mdp as mdp
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg, TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.viewer.viewer_config import ViewerConfig

import instinct_mj.envs.mdp as instinct_mdp
from instinct_mj.assets.unitree_g1 import G1_29DOF_TORSOBASE_POPSICLE_CFG, beyondmimic_action_scale


def g1_flat_parkour_env_cfg(*, play: bool = False, has_state_estimation: bool = True) -> ManagerBasedRlEnvCfg:
  """A minimal flat tracking env config for the G1 29-DoF popsicle model.

  Notes:
  - This is intentionally a small, stable base config that downstream task
    factories (e.g. parkour / AMP) can mutate.
  - `has_state_estimation` is kept for API parity with the Adam SP base cfg;
    it does not change the core observation terms here.
  """

  robot_cfg = copy.deepcopy(G1_29DOF_TORSOBASE_POPSICLE_CFG)

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      num_envs=1 if play else 2048,
      env_spacing=2.5,
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": robot_cfg},
      sensors=tuple(),
      extent=2.0,
    ),
    actions={
      # G1 用逐关节 action scale（0.25·effort/stiffness，BeyondMimic 惯例），
      # 与 locomotion flat G1 任务一致；Adam SP 统一 0.5 不适用于 G1 的电机分组。
      "joint_pos": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=dict(beyondmimic_action_scale),
        use_default_offset=True,
      ),
    },
    commands={
      "base_velocity": UniformVelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=0.5,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=UniformVelocityCommandCfg.Ranges(
          lin_vel_x=(-0.5, 1.0),
          lin_vel_y=(-0.5, 0.5),
          ang_vel_z=(-1.5, 1.5),
          heading=(-math.pi, math.pi),
        ),
      ),
    },
    observations={
      "policy": ObservationGroupCfg(
        terms={
          "base_ang_vel": ObservationTermCfg(
            func=instinct_mdp.base_ang_vel,
            noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
            scale=0.25,
          ),
          "projected_gravity": ObservationTermCfg(
            func=instinct_mdp.projected_gravity,
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
          ),
          "velocity_commands": ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "base_velocity"}),
          "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=".*")},
            noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
          ),
          "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=".*")},
            noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
            scale=0.05,
          ),
          "actions": ObservationTermCfg(func=instinct_mdp.last_action),
        },
        concatenate_terms=False,
        enable_corruption=True,
      ),
      "critic": ObservationGroupCfg(
        terms={
          "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
          "base_ang_vel": ObservationTermCfg(func=instinct_mdp.base_ang_vel, scale=0.25),
          "projected_gravity": ObservationTermCfg(func=instinct_mdp.projected_gravity),
          "velocity_commands": ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "base_velocity"}),
          "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=".*")},
          ),
          "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=".*")},
            scale=0.05,
          ),
          "actions": ObservationTermCfg(func=instinct_mdp.last_action),
        },
        concatenate_terms=False,
        enable_corruption=False,
      ),
    },
    rewards={},
    curriculum={},
    terminations={
      "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    },
    events={
      "reset_base": EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
          "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.1, 0.1)},
          "velocity_range": {
            "x": (-0.2, 0.2),
            "y": (-0.2, 0.2),
            "z": (-0.2, 0.2),
            "roll": (-0.2, 0.2),
            "pitch": (-0.2, 0.2),
            "yaw": (-0.2, 0.2),
          },
        },
      ),
      "reset_robot_joints": EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
          "position_range": (-0.15, 0.15),
          "velocity_range": (0.0, 0.0),
        },
      ),
    },
    viewer=ViewerConfig(
      lookat=(0.0, 0.0, 0.0),
      distance=3.5,
      elevation=-12.0,
      azimuth=45.0,
      origin_type=ViewerConfig.OriginType.ASSET_ROOT,
      entity_name="robot",
    ),
    sim=SimulationCfg(
      mujoco=MujocoCfg(
        timestep=0.005,
        solver="newton",
        iterations=10,
        ls_iterations=20,
        ccd_iterations=500 if not play else 50,
      ),
    ),
    decimation=4,
    episode_length_s=20.0,
  )

  return cfg


__all__ = ["g1_flat_parkour_env_cfg"]
