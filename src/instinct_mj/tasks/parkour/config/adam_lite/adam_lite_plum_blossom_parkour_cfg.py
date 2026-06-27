"""Adam Lite plum-blossom stakes parkour: stepping stones, AMP + vision."""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field
from typing import Any, cast

import mjlab.envs.mdp as envs_mdp
from mjlab.envs.mdp import dr
import instinct_mj.tasks.parkour.mdp.events as parkour_events

import mujoco
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg, GridPatternCfg, ObjRef, PinholeCameraPatternCfg, RayCastSensorCfg
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.viewer.viewer_config import ViewerConfig

import instinct_mj.envs.mdp as instinct_envs_mdp
import instinct_mj.tasks.parkour.mdp as parkour_mdp
from instinct_mj.assets.adam_lite import (
    ADAM_LITE_MJCF_PATH,
    adam_lite_symmetric_augmentation_joint_mapping,
    adam_lite_symmetric_augmentation_joint_reverse_buf,
)
from instinct_mj.motion_reference import MotionReferenceManagerCfg
from instinct_mj.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg as AmassMotionCfgBase
from instinct_mj.motion_reference.utils import motion_interpolate_bilinear
from instinct_mj.tasks.mdp import (
    parkour_amp_reference_base_ang_vel,
    parkour_amp_reference_base_lin_vel,
    parkour_amp_reference_joint_pos_rel,
    parkour_amp_reference_joint_vel_rel,
    parkour_amp_reference_projected_gravity,
)
from instinct_mj.tasks.parkour.config.adam_lite.env_cfgs import adam_sp_flat_parkour_env_cfg
from instinct_mj.tasks.parkour.config.parkour_env_cfg import (
    PLUM_BLOSSOM_TERRAINS_CFG,
    PLUM_BLOSSOM_TERRAINS_CFG_PLAY,
    _edit_parkour_scene_spec,
)
from instinct_mj.tasks.parkour.mdp.commands import PoseVelocityCommandCfg
from instinct_mj.terrains.terrain_importer_cfg import TerrainImporterCfg as InstinctTerrainImporterCfg
from instinct_mj.terrains.virtual_obstacle.edge_cylinder_cfg import GreedyconcatEdgeCylinderCfg
from instinct_mj.utils.noise import CropAndResizeCfg, DepthNormalizationCfg, GaussianBlurNoiseCfg
from instinct_mj.sensors.volume_points import Grid3dPointsGeneratorCfg, VolumePointsCfg
from instinct_mj.sensors.noisy_camera import NoisyGroupedRayCasterCameraCfg




__file_dir__ = os.path.dirname(os.path.realpath(__file__))
# NOTE: Change this to your local parkour dataset root before training / play.
_PARKOUR_DATASET_DIR = os.path.expanduser("src/instinct_mj/motions")


# ---------------------------------------------------------------------------
# Motion reference configs
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class AmassMotionCfg(AmassMotionCfgBase):
    """Parkour AMASS motion buffer config for Adam Lite."""

    path: str = _PARKOUR_DATASET_DIR
    retargetting_func: object | None = None
    filtered_motion_selection_filepath: str | None = os.path.join(
        _PARKOUR_DATASET_DIR, "motion_reference_plum_blossom.yaml"
    )
    motion_start_from_middle_range: list[float] = field(default_factory=lambda: [0.0, 0.9])
    motion_start_height_offset: float = 0.0
    ensure_link_below_zero_ground: bool = False
    buffer_device: str = "output_device"
    motion_interpolate_func: object = field(default_factory=lambda: motion_interpolate_bilinear)
    velocity_estimation_method: str = "frontward"


motion_reference_cfg = MotionReferenceManagerCfg(
    name="motion_reference",
    entity_name="robot",
    robot_model_path=ADAM_LITE_MJCF_PATH,
    link_of_interests=[
        "pelvis",
        "torso_link",
        "left_shoulder_roll_link",
        "right_shoulder_roll_link",
        "left_elbow_link",
        "right_elbow_link",
        "left_hip_roll_link",
        "right_hip_roll_link",
        "left_knee_link",
        "right_knee_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ],
    symmetric_augmentation_link_mapping=[0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10],
    symmetric_augmentation_joint_mapping=list(adam_lite_symmetric_augmentation_joint_mapping),
    symmetric_augmentation_joint_reverse_buf=list(adam_lite_symmetric_augmentation_joint_reverse_buf),
    frame_interval_s=0.02,
    update_period=0.02,
    num_frames=10,
    motion_buffers={"run_walk": AmassMotionCfg()},
    mp_split_method="None",
)


def instinct_adam_lite_plum_blossom_parkour_env_cfg(
    *,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Adam Lite parkour on plum-blossom stepping stakes, AMP + vision."""
    cfg = adam_sp_flat_parkour_env_cfg(play=play, has_state_estimation=True)
    setattr(cfg, "monitors", {})
    cfg.viewer.origin_type = ViewerConfig.OriginType.WORLD
    cfg.viewer.entity_name = None
    cfg.viewer.body_name = None
    cfg.scene.entities["robot"].init_state.pos = (-4.0, 0.0, 0.89)

    cfg.scene.num_envs = 2048
    cfg.scene.env_spacing = 2.5
    cfg.episode_length_s = 20.0
    cfg.sim.nconmax = 128
    cfg.sim.njmax = 700
    cfg.sim.mujoco.iterations = 10
    cfg.sim.mujoco.ls_iterations = 20
    cfg.sim.mujoco.ccd_iterations = 128
    cfg.sim.mujoco.multiccd = False


    terrain_gen = copy.deepcopy(PLUM_BLOSSOM_TERRAINS_CFG_PLAY if play else PLUM_BLOSSOM_TERRAINS_CFG)
    virtual_obstacles = {}
    if not play:
        edge_obstacle_cfg = GreedyconcatEdgeCylinderCfg(
            cylinder_radius=0.03,
            min_points=2,
            component_workers=0,
            merge_collinear_gap=0.09,
            merge_collinear_angle_threshold=30.0,
            merge_collinear_line_distance=0.04,
        )
        virtual_obstacles["edges"] = edge_obstacle_cfg
    cfg.scene.terrain = InstinctTerrainImporterCfg(
        terrain_type="hacked_generator",
        terrain_generator=terrain_gen,
        max_init_terrain_level=5,
        virtual_obstacle_source="mesh",
        virtual_obstacle_hfield_height_threshold=None if play else 0.04,
        collision_debug_vis=False if play else True,
        collision_debug_rgba=(0.62, 0.2, 0.9, 0.35),
        virtual_obstacles=virtual_obstacles,
    )
    cfg.scene.spec_fn = _edit_parkour_scene_spec
    geom_names = tuple(
        f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 9)
    )

    cfg.scene.sensors = (
        ContactSensorCfg(
            name="contact_forces",
            primary=ContactMatch(
                mode="body",
                pattern=("left_ankle_roll_link", "right_ankle_roll_link"),
                entity="robot",
            ),
            fields=("found", "force"),
            reduce="netforce",
            track_air_time=True,
            history_length=3,
        ),
        ContactSensorCfg(
            name="torso_contact_forces",
            primary=ContactMatch(mode="body", pattern="torso_link", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            track_air_time=False,
            history_length=3,
        ),
        ContactSensorCfg(
            name="undesired_contact_forces",
            primary=ContactMatch(
                mode="body",
                pattern=".*",
                entity="robot",
                exclude=("left_ankle_roll_link", "right_ankle_roll_link"),
            ),
            fields=("found", "force"),
            reduce="netforce",
            track_air_time=False,
            history_length=3,
        ),
        RayCastSensorCfg(
            name="left_height_scanner",
            frame=ObjRef(type="body", name="left_ankle_roll_link", entity="robot"),
            pattern=GridPatternCfg(resolution=0.12, size=(0.12, 0.0)),
            ray_alignment="yaw",
            max_distance=10.0,
            debug_vis=False,
        ),
        RayCastSensorCfg(
            name="right_height_scanner",
            frame=ObjRef(type="body", name="right_ankle_roll_link", entity="robot"),
            pattern=GridPatternCfg(resolution=0.12, size=(0.12, 0.0)),
            ray_alignment="yaw",
            max_distance=10.0,
            debug_vis=False,
        ),
        RayCastSensorCfg(
            name="left_foot_sole_scanner",
            frame=ObjRef(type="site", name="left_foot_scan", entity="robot"),
            pattern=GridPatternCfg(resolution=0.04, size=(0.24, 0.08)),
            ray_alignment="yaw",
            max_distance=10.0,
            debug_vis=False,
        ),
        RayCastSensorCfg(
            name="right_foot_sole_scanner",
            frame=ObjRef(type="site", name="right_foot_scan", entity="robot"),
            pattern=GridPatternCfg(resolution=0.04, size=(0.24, 0.08)),
            ray_alignment="yaw",
            max_distance=10.0,
            debug_vis=False,
        ),
        RayCastSensorCfg(
            name="base_heightmap_scanner",
            frame=ObjRef(type="site", name="base_heightmap_scan", entity="robot"),
            pattern=GridPatternCfg(resolution=0.1, size=(1.0, 0.8)),
            ray_alignment="yaw",
            max_distance=10.0,
            debug_vis=False,
        ),
        VolumePointsCfg(
            name="leg_volume_points",
            entity_name="robot",
            body_names=".*_ankle_roll_link",
            points_generator=Grid3dPointsGeneratorCfg(
                x_min=-0.06,
                x_max=0.15,
                x_num=10,
                y_min=-0.05,
                y_max=0.05,
                y_num=5,
                z_min=-0.060,
                z_max=-0.050,
                z_num=2,
            ),
            debug_vis=False,
        ),
        NoisyGroupedRayCasterCameraCfg(
            name="camera",
            frame=ObjRef(type="body", name="waist_pitch_link", entity="robot"),
            pattern=PinholeCameraPatternCfg(
                width=64,
                height=36,
                fovy=57.74,
            ),
            focal_length=1.0,
            horizontal_aperture=2 * math.tan(math.radians( 88.85) / 2.0),
            vertical_aperture=2 * math.tan(math.radians(57.74) / 2.0),
            ray_alignment="base",
            offset=NoisyGroupedRayCasterCameraCfg.OffsetCfg(
                # Torso-mounted depth camera nominal pose (world convention)
                pos=(
                    0.0958,
                    0.0475,
                    0.0095,
                ),
                rot=(
                    0.9063078,
                    0.0,
                    0.4226183,
                    0.0,
                ),
                
                convention="world",
            ),
            data_types=["distance_to_image_plane"],
            depth_clipping_behavior="max",
            noise_pipeline={
                # crop_region=(上, 下, 左, 右)。少裁顶部(18→6)保留更多前方远景，深度图 18x32 → 30x32；
                # PerceptiveEncoder 用注意力池化按 obs 形状自适应，无需改网络。
                "crop_and_resize": CropAndResizeCfg(crop_region=(6, 0, 16, 16)),
                "gaussian_blur": GaussianBlurNoiseCfg(kernel_size=3, sigma=1),
                "depth_normalization": DepthNormalizationCfg(
                    depth_range=(0.0, 2.5),
                    normalize=True,
                    output_range=(0.0, 1.0),
                ),
            },
            data_histories={"distance_to_image_plane_noised": 37},
            min_distance=0.1,
            max_distance=2.5,
            debug_vis=False,
        ),
    )
    motion_reference_sensor_cfg = copy.deepcopy(motion_reference_cfg)
    existing_sensors = tuple(
        sensor_cfg for sensor_cfg in cfg.scene.sensors if sensor_cfg.name != motion_reference_sensor_cfg.name
    )
    cfg.scene.sensors = existing_sensors + (motion_reference_sensor_cfg,)

    cfg.commands = {
        "base_velocity": PoseVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(8.0, 12.0),
            debug_vis=False,
            velocity_control_stiffness=2.0,
            heading_control_stiffness=2.0,
            rel_standing_envs=0.05,
            ranges=PoseVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 0.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-1.0, 1.0),
            ),
            random_velocity_terrain=[],
            velocity_ranges={
                "plum_blossom_stakes": {"lin_vel_x": (0.3, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-0.5, 0.5)},
            },
            only_positive_lin_vel_x=True,
            lin_vel_threshold=0.0,
            ang_vel_threshold=0.0,
            target_dis_threshold=0.4,
            flat_patch_advance_on_arrival=True,
        ),
    }

    policy_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=envs_mdp.base_ang_vel,
            noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
            history_length=8,
            flatten_history_dim=True,
            scale=0.25,
        ),
        "projected_gravity": ObservationTermCfg(
            func=envs_mdp.projected_gravity,
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
            history_length=8,
            flatten_history_dim=True,
        ),
        "velocity_commands": ObservationTermCfg(
            func=envs_mdp.generated_commands,
            params={"command_name": "base_velocity"},
            history_length=8,
            flatten_history_dim=True,
            noise=None,
        ),
        "joint_pos": ObservationTermCfg(
            func=envs_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=".*")},
            noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
            history_length=8,
            flatten_history_dim=True,
        ),
        "joint_vel": ObservationTermCfg(
            func=envs_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=".*")},
            noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5),
            scale=0.05,
            history_length=8,
            flatten_history_dim=True,
        ),
        "actions": ObservationTermCfg(
            func=envs_mdp.last_action,
            history_length=8,
            flatten_history_dim=True,
        ),
        "depth_image": ObservationTermCfg(
            func=instinct_envs_mdp.delayed_visualizable_image,
            params={
                "data_type": "distance_to_image_plane_noised_history",
                "sensor_cfg": SceneEntityCfg("camera"),
                "history_skip_frames": 5,
                "num_output_frames": 8,
                "delayed_frame_ranges": (0, 1),
                "debug_vis": False,
            },
            noise=None,
        ),
    }
    critic_terms = {
        "base_lin_vel": ObservationTermCfg(
            func=envs_mdp.base_lin_vel,
            history_length=8,
            flatten_history_dim=True,
        ),
        "base_ang_vel": ObservationTermCfg(
            func=envs_mdp.base_ang_vel,
            history_length=8,
            flatten_history_dim=True,
            scale=0.25,
        ),
        "projected_gravity": ObservationTermCfg(
            func=envs_mdp.projected_gravity,
            history_length=8,
            flatten_history_dim=True,
        ),
        "velocity_commands": ObservationTermCfg(
            func=envs_mdp.generated_commands,
            params={"command_name": "base_velocity"},
            history_length=8,
            flatten_history_dim=True,
            noise=None,
        ),
        "joint_pos": ObservationTermCfg(
            func=envs_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=".*")},
            history_length=8,
            flatten_history_dim=True,
        ),
        "joint_vel": ObservationTermCfg(
            func=envs_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=".*")},
            scale=0.05,
            history_length=8,
            flatten_history_dim=True,
        ),
        "actions": ObservationTermCfg(
            func=envs_mdp.last_action,
            history_length=8,
            flatten_history_dim=True,
        ),
        # Critic sees the privileged ground-truth heightmap, so it has no use for the
        # noisy depth image (that is the actor's deploy-time-only sensor).
        "height_map_priv": ObservationTermCfg(
            func=envs_mdp.height_scan,
            params={"sensor_name": "base_heightmap_scanner"},
        ),
    }
    cfg.observations["policy"] = ObservationGroupCfg(
        terms=policy_terms,
        concatenate_terms=False,
        enable_corruption=True,
    )
    cfg.observations["critic"] = ObservationGroupCfg(
        terms=critic_terms,
        concatenate_terms=False,
        enable_corruption=False,
    )
    cfg.observations.pop("actor", None)

    amp_policy_terms = {
        "projected_gravity": ObservationTermCfg(
            func=envs_mdp.projected_gravity,
            history_length=10,
            flatten_history_dim=True,
        ),
        "joint_pos_rel": ObservationTermCfg(
            func=envs_mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot",
                    joint_names=".*",
                )
            },
            history_length=10,
            flatten_history_dim=True,
        ),
        "joint_vel": ObservationTermCfg(
            func=envs_mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot",
                    joint_names=".*",
                )
            },
            scale=0.05,
            history_length=10,
            flatten_history_dim=True,
        ),
        "base_lin_vel": ObservationTermCfg(
            func=envs_mdp.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg(name="robot")},
            history_length=10,
            flatten_history_dim=True,
        ),
        "base_ang_vel": ObservationTermCfg(
            func=envs_mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg(name="robot")},
            history_length=10,
            flatten_history_dim=True,
        ),
    }
    amp_reference_terms = {
        "projected_gravity": ObservationTermCfg(
            func=parkour_amp_reference_projected_gravity,
            params={"asset_cfg": SceneEntityCfg(name="motion_reference")},
            history_length=10,
            flatten_history_dim=True,
        ),
        "joint_pos_rel": ObservationTermCfg(
            func=parkour_amp_reference_joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="motion_reference",
                    joint_names=".*",
                )
            },
            history_length=10,
            flatten_history_dim=True,
        ),
        "joint_vel": ObservationTermCfg(
            func=parkour_amp_reference_joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="motion_reference",
                    joint_names=".*",
                )
            },
            scale=0.05,
            history_length=10,
            flatten_history_dim=True,
        ),
        "base_lin_vel": ObservationTermCfg(
            func=parkour_amp_reference_base_lin_vel,
            params={"asset_cfg": SceneEntityCfg(name="motion_reference")},
            history_length=10,
            flatten_history_dim=True,
        ),
        "base_ang_vel": ObservationTermCfg(
            func=parkour_amp_reference_base_ang_vel,
            params={"asset_cfg": SceneEntityCfg(name="motion_reference")},
            history_length=10,
            flatten_history_dim=True,
        ),
    }
    cfg.observations["amp_policy"] = ObservationGroupCfg(
        terms=amp_policy_terms,
        concatenate_terms=False,
        enable_corruption=False,
    )
    cfg.observations["amp_reference"] = ObservationGroupCfg(
        terms=amp_reference_terms,
        concatenate_terms=False,
        enable_corruption=False,
    )

    cfg.rewards = {
        "track_lin_vel_xy_exp": RewardTermCfg(
            func=parkour_mdp.track_lin_vel_xy_exp,
            weight=2.0,
            params={"command_name": "base_velocity", "std": 0.5},
        ),
        "track_ang_vel_z_exp": RewardTermCfg(
            func=parkour_mdp.track_ang_vel_z_exp,
            weight=2.0,
            params={"command_name": "base_velocity", "std": 0.35},
        ),
        "ang_vel_z_overshoot": RewardTermCfg(
            func=parkour_mdp.ang_vel_z_overshoot,
            weight=-1.0,
            params={
                "command_name": "base_velocity",
                "ang_vel_tolerance": 0.1,
            },
        ),
        "heading_error": RewardTermCfg(
            func=parkour_mdp.heading_error,
            weight=-1.0,
            params={"command_name": "base_velocity"},
        ),
        "dont_wait": RewardTermCfg(
            func=parkour_mdp.dont_wait,
            weight=-0.5,
            params={"command_name": "base_velocity"},
        ),
        "is_alive": RewardTermCfg(func=envs_mdp.is_alive, weight=3.0),
        "stand_still": RewardTermCfg(
            func=parkour_mdp.stand_still,
            weight=-0.5,
            params={"command_name": "base_velocity", "offset": 4.0},
        ),
        "volume_points_penetration": RewardTermCfg(
            func=parkour_mdp.volume_points_penetration,
            weight=-1.0,
            params={"sensor_name": "leg_volume_points"},
        ),
        "feet_air_time": RewardTermCfg(
            func=parkour_mdp.feet_air_time,
            weight=0.5,
            params={
                "command_name": "base_velocity",
                "sensor_name": "contact_forces",
                "vel_threshold": 0.1,
            },
        ),
        "cref_stepping_stone_foothold": RewardTermCfg(
            func=parkour_mdp.cref_stepping_stone_foothold_reward,
            weight=1.5,
            params={
                "sensor_name": "contact_forces",
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
                ),
                "contact_force_threshold": 1.0,
                "s_xy": 0.15,
                "min_forward_margin": 0.02,
                "max_candidates_per_foot": 2,
                "foot_height_offset": 0.065,
            },
        ),
        "yaw_cmd_double_support": RewardTermCfg(
            func=parkour_mdp.yaw_cmd_double_support,
            weight=-1.2,
            params={
                "command_name": "base_velocity",
                "sensor_name": "contact_forces",
                "yaw_threshold": 0.15,
                "grace_period": 0.08,
                "max_penalty_time": 0.4,
                "random_velocity_only": False,
            },
        ),
        # Reserved for stair tuning; not active in baseline training (see 7f15824 reward set).
        # "feet_horizontal_impact": RewardTermCfg(
        #     func=parkour_mdp.feet_horizontal_impact,
        #     weight=-0.02,
        #     params={
        #         "sensor_name": "contact_forces",
        #         "force_threshold": 30.0,
        #         "vertical_ratio": 0.5,
        #     },
        # ),
        # "feet_unsupported": RewardTermCfg(
        #     func=parkour_mdp.feet_unsupported,
        #     weight=-1.0,
        #     params={
        #         "contact_sensor_name": "contact_forces",
        #         "left_sole_scanner_name": "left_foot_sole_scanner",
        #         "right_sole_scanner_name": "right_foot_sole_scanner",
        #         "asset_cfg": SceneEntityCfg(
        #             "robot",
        #             body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
        #         ),
        #         "foot_height_offset": 0.065,
        #         "gap_threshold": 0.03,
        #         "contact_force_threshold": 1.0,
        #     },
        # ),
        "feet_slide": RewardTermCfg(
            func=parkour_mdp.feet_slide,
            weight=-0.4,
            params={
                "sensor_name": "contact_forces",
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
                ),
                "threshold": 1.0,
            },
        ),
        # "joint_deviation_hip": RewardTermCfg(
        #     func=parkour_mdp.joint_deviation_square,
        #     weight=-0.5,
        #     params={
        #         "asset_cfg": SceneEntityCfg(
        #             "robot",
        #             joint_names=(".*_hip_yaw_joint", ".*_hip_roll_joint"),
        #         )
        #     },
        # ),
        "joint_deviation_ankle": RewardTermCfg(
            func=parkour_mdp.joint_deviation_square,
            weight=-0.5,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=(".*_ankle_roll_joint"),
                )
            },
        ),
        "ang_vel_xy_l2": RewardTermCfg(func=parkour_mdp.ang_vel_xy_l2, weight=-0.05),
        "dof_torques_l2": RewardTermCfg(
            func=parkour_mdp.joint_torques_l2,
            weight=-1.5e-7,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=(".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"),
                )
            },
        ),
        "dof_acc_l2": RewardTermCfg(
            func=envs_mdp.joint_acc_l2,
            weight=-1.25e-7,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
        ),
        "dof_vel_l2": RewardTermCfg(
            func=envs_mdp.joint_vel_l2,
            weight=-1e-4,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
        ),
        "action_rate_l2": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.1),
        "flat_orientation_l2": RewardTermCfg(func=envs_mdp.flat_orientation_l2, weight=-3.0),
        "torso_orientation_l2": RewardTermCfg(
            func=parkour_mdp.link_orientation,
            weight=-5.0,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
        ),
        "feet_flat_ori": RewardTermCfg(
            func=parkour_mdp.feet_orientation_contact,
            weight=-0.4,
            params={
                "sensor_name": "contact_forces",
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
                ),
            },
        ),
        "feet_heading_alignment": RewardTermCfg(
            func=parkour_mdp.feet_heading_alignment,
            weight=2.0,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
                ),
                "cmd_threshold": 0.0,
                "angle_scale": 0.05,
                # In-place yaw turns have zero linear command; keep feet aligned to base
                # forward so the robot steps to turn instead of splaying hip-yaw to pivot.
                "yaw_cmd_threshold": 0.10,
            },
        ),
        "feet_at_plane": RewardTermCfg(
            func=parkour_mdp.feet_at_plane,
            weight=-0.1,
            params={
                "contact_sensor_name": "contact_forces",
                "left_height_scanner_name": "left_height_scanner",
                "right_height_scanner_name": "right_height_scanner",
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
                ),
                "height_offset": 0.065,
            },
        ),
        "feet_close_xy": RewardTermCfg(
            func=parkour_mdp.feet_close_xy_gauss,
            # 梅花桩 quincunx 落点天然横向错开 ~0.3m，并拢奖励会把脚挤进孔里，故降权让 cref 主导。
            weight=0.1,
            params={
                "threshold": 0.24,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
                ),
                "std": math.sqrt(0.05),
            },
        ),
        "energy": RewardTermCfg(
            func=parkour_mdp.motors_power_square,
            weight=-5e-5,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=(".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"),
                ),
                "normalize_by_stiffness": True,
            },
        ),
        "freeze_waist_joints": RewardTermCfg(
            func=parkour_mdp.joint_deviation_l1,
            weight=-1.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    # Adam SP has shoulders/elbows + waist, no wrist joints in 23DOF model.
                    joint_names=("waist_.*"),
                )
            },
        ),
        "freeze_arm_joints": RewardTermCfg(
            func=parkour_mdp.joint_deviation_l1,
            weight=-1.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    # Adam SP has shoulders/elbows + waist, no wrist joints in 23DOF model.
                    joint_names=(".*_shoulder_.*", ".*_elbow_.*"),
                )
            },
        ),
        "dof_pos_limits": RewardTermCfg(
            func=envs_mdp.joint_pos_limits,
            weight=-1.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
        ),
        "dof_vel_limits": RewardTermCfg(
            func=parkour_mdp.joint_vel_limits,
            weight=-1.0,
            params={
                "soft_ratio": 0.9,
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        "torque_limits": RewardTermCfg(
            func=parkour_mdp.applied_torque_limits_by_ratio,
            weight=-0.01,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "limit_ratio": 0.8,
            },
        ),
        "undesired_contacts": RewardTermCfg(
            func=parkour_mdp.undesired_contacts,
            weight=-1.0,
            params={"sensor_name": "undesired_contact_forces", "threshold": 1.0},
        ),
    }
    cfg.curriculum = {
        "terrain_levels": CurriculumTermCfg(
            func=parkour_mdp.tracking_exp_vel,
            params={
                "lin_vel_threshold": (0.3, 0.6),
                "ang_vel_threshold": (0.0, 0.0),
            },
        ),
    }
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "terrain_out_bound": TerminationTermCfg(
            func=instinct_envs_mdp.terrain_out_of_bounds,
            time_out=True,
            params={"distance_buffer": 2.0},
        ),
        "base_contact": TerminationTermCfg(
            func=parkour_mdp.illegal_contact,
            params={"sensor_name": "torso_contact_forces", "threshold": 1.0},
        ),
        "bad_orientation": TerminationTermCfg(
            func=envs_mdp.bad_orientation,
            params={"limit_angle": 1.0},
        ),
        "root_height": TerminationTermCfg(
            func=parkour_mdp.root_height_below_env_origin_minimum,
            params={"minimum_height": 0.5},
        ),
        "feet_on_pit_bottom": TerminationTermCfg(
            func=parkour_mdp.feet_on_pit_bottom,
            params={
                "contact_sensor_name": "contact_forces",
                "left_sole_scanner_name": "left_foot_sole_scanner",
                "right_sole_scanner_name": "right_foot_sole_scanner",
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
                ),
                "foot_height_offset": 0.065,
                "pit_ground_height_max": -0.04,
            },
        ),
        "dataset_exhausted": TerminationTermCfg(
            func=instinct_envs_mdp.dataset_exhausted,
            time_out=True,
            params={
                "reference_cfg": SceneEntityCfg(name="motion_reference"),
                "print_reason": False,
                "reset_without_notice": True,
            },
        ),
    }
    cfg.events = {
        "physics_material": EventTermCfg(
            func=parkour_mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=".*"),
                "static_friction_range": (0.3, 1.6),
                "dynamic_friction_range": (0.3, 1.6),
                "make_consistent": True,
            },
        ),
        "register_virtual_obstacles": EventTermCfg(
            func=instinct_envs_mdp.register_virtual_obstacle_to_sensor,
            mode="startup",
            params={
                "sensor_cfgs": SceneEntityCfg("leg_volume_points"),
                "enable_debug_vis": False,
            },
        ),
        "reset_base": EventTermCfg(
            func=parkour_events.reset_root_state_on_target_patch,
            
            mode="reset",
            params={
                "patch_name": "spawn",
                "align_z_to_patch": True,
                "pose_range": {"x": (-0.1, 0.1), "y": (-0.5, 0.5), "yaw": (-0.5, 0.5)},
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
            func=envs_mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.15, 0.15),
                "velocity_range": (0.0, 0.0),
            },
        ),
        "encoder_bias": EventTermCfg(
            mode="startup",
            func=dr.encoder_bias,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "bias_range": (-0.015, 0.015),
            },
        ),
        "base_com": EventTermCfg(
            mode="startup",
            func=instinct_envs_mdp.randomize_rigid_body_com,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "com_range": {
                    "x": (-0.025, 0.025),
                    "y": (-0.025, 0.025),
                    "z": (-0.03, 0.03),
                },
            },
        ),
        "push_robot": EventTermCfg(
            func=envs_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(5.0, 8.0),
            params={
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.4, 0.4),
                    "roll": (-0.52, 0.52),
                    "pitch": (-0.52, 0.52),
                    "yaw": (-0.78, 0.78),
                },
            },
        ),
        "foot_friction": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),  # Set per-robot.
                "operation": "abs",
                "ranges": (0.3, 1.2),
                "shared_random": True,  # All foot geoms share the same friction.
            },
        ),
        "randomize_actuator_gains": EventTermCfg(
            func=dr.pd_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "kp_range": (0.8, 1.2), 
                "kd_range": (0.8, 1.2),
                "operation": "scale",
                "distribution": "uniform",
            },
        ),
    }

    setattr(cfg, "run_name", "AdamLitePlumBlossomParkour")

    if play:
        cfg.scene.num_envs = 10
        cfg.scene.env_spacing = 2.5
        cfg.episode_length_s = 20.0
        terrain_generator = cast(Any, cfg.scene.terrain.terrain_generator)
        terrain_generator.num_rows = 4
        leg_volume_points_sensor = cast(
            Any,
            next(sensor_cfg for sensor_cfg in cfg.scene.sensors if sensor_cfg.name == "leg_volume_points"),
        )
        leg_volume_points_sensor.debug_vis = True
        for sensor_cfg in cfg.scene.sensors:
            if isinstance(sensor_cfg, NoisyGroupedRayCasterCameraCfg):
                sensor_cfg.debug_vis = True
            if isinstance(sensor_cfg, RayCastSensorCfg) and sensor_cfg.name in (
                "left_foot_sole_scanner",
                "right_foot_sole_scanner",
                "base_heightmap_scanner",
            ):
                sensor_cfg.debug_vis = True
        cfg.scene.terrain.collision_debug_vis = False
        base_velocity_cmd = cast(PoseVelocityCommandCfg, cfg.commands["base_velocity"])
        base_velocity_cmd.debug_vis = True
        base_velocity_cmd.show_cref_foothold_candidates = True
        base_velocity_cmd.cref_foothold_reward_name = "cref_stepping_stone_foothold"
        cfg.terminations.pop("root_height", None)
        cfg.events.pop("physics_material", None)
        cfg.events.pop("register_virtual_obstacles", None)
        cfg.events.pop("show_virtual_obstacles", None)
        cfg.events["reset_robot_joints"].params = {
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        }
    return cfg


def instinct_adam_lite_plum_blossom_parkour_final_cfg(
    *,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Final Adam Lite plum-blossom parkour env config (with optional play viewer overrides)."""
    cfg = instinct_adam_lite_plum_blossom_parkour_env_cfg(play=play)
    if play:
        cfg.viewer = ViewerConfig(
            lookat=(0.0, 0.75, 0.0),
            distance=4.123105625617661,
            elevation=-14.036243467926479,
            azimuth=180.0,
            origin_type=ViewerConfig.OriginType.WORLD,
            entity_name=None,
        )
        cfg.viewer.origin_type = ViewerConfig.OriginType.WORLD
        cfg.viewer.entity_name = None
        cfg.viewer.body_name = None
    return cfg
