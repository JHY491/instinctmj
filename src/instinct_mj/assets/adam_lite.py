"""PND Adam Lite asset entry (MJCF under resources/adam_lite).

Delayed actuators, INIT_STATE, and symmetric augmentation mirror
`instinct_mj.assets.adam_sp` so runtime tuning matches Adam SP when the Lite
XML is kept aligned with `adam_sp_23_dof.xml`.
"""

from __future__ import annotations

import os

import mujoco
from mjlab.actuator import ActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets

from instinct_mj.actuators import DelayedInstinctActuatorCfg, InstinctActuatorCfg

__file_dir__ = os.path.dirname(os.path.realpath(__file__))

ADAM_LITE_MJCF_PATH: str = os.path.join(__file_dir__, "resources/adam_lite/xml/adam_lite.xml")
ADAM_LITE_MESH_COLLISION_MJCF_PATH: str = os.path.join(
    __file_dir__, "resources/adam_lite/xml/adam_lite_mesh_collision.xml"
)
ADAM_LITE_ASSETS_DIR: str = os.path.join(__file_dir__, "resources/adam_lite/assets")

##
# MJCF and assets.
##
"""
joint name order:
[

    'left_hip_pitch_joint',
    'left_hip_roll_joint',
    'left_hip_yaw_joint',
    'left_knee_joint',
    'left_ankle_pitch_joint',
    'left_ankle_roll_joint',
    'right_hip_pitch_joint',
    'right_hip_roll_joint',
    'right_hip_yaw_joint',
    'right_knee_joint',
    'right_ankle_pitch_joint',
    'right_ankle_roll_joint',
    'waist_roll_joint',
    'waist_pitch_joint',
    'waist_yaw_joint',
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_joint',
    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_joint',

]
"""


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, ADAM_LITE_ASSETS_DIR, meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(ADAM_LITE_MJCF_PATH))
  spec.assets = get_assets(spec.meshdir)
  return spec


def get_mesh_collision_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(ADAM_LITE_MESH_COLLISION_MJCF_PATH))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

ADAM_LITE_HIPPITCH_KNEEPITCH_ACTUATOR = DelayedInstinctActuatorCfg(
    base_cfg=InstinctActuatorCfg(
        target_names_expr=(".*_hip_pitch_joint", ".*_knee_joint"),
        effort_limit=230.0,
        velocity_limit=60.0,
        stiffness=300.0,
        damping=7.0,
        armature=0.13426,
    ),
    delay_target="position",
    delay_min_lag=0,
    delay_max_lag=1,
)

ADAM_LITE_HIPROLL_ACTUATOR = DelayedInstinctActuatorCfg(
    base_cfg=InstinctActuatorCfg(
        target_names_expr=(".*_hip_roll_joint",),
        effort_limit=180.0,
        velocity_limit=60.0,
        stiffness=600.0,
        damping=10.0,
        armature=0.281573,
    ),
    delay_target="position",
    delay_min_lag=0,
    delay_max_lag=1,
)

ADAM_LITE_HIPYAW_ACTUATOR = DelayedInstinctActuatorCfg(
    base_cfg=InstinctActuatorCfg(
        target_names_expr=(".*_hip_yaw_joint",),
        effort_limit=105.0,
        velocity_limit=60.0,
        stiffness=300.0,
        damping=2.0,
        armature=0.23409,
    ),
    delay_target="position",
    delay_min_lag=0,
    delay_max_lag=1,
)

ADAM_LITE_ANKLEPITCH_ACTUATOR = DelayedInstinctActuatorCfg(
    base_cfg=InstinctActuatorCfg(
        target_names_expr=(".*_ankle_pitch_joint",),
        effort_limit=80.0,
        velocity_limit=60.0,
        stiffness=130.0,
        damping=3.5,
        armature=0.0549,
    ),
    delay_target="position",
    delay_min_lag=0,
    delay_max_lag=1,
)

ADAM_LITE_ANKLEROLL_ACTUATOR = DelayedInstinctActuatorCfg(
    base_cfg=InstinctActuatorCfg(
        target_names_expr=(".*_ankle_roll_joint",),
        effort_limit=40.0,
        velocity_limit=60.0,
        stiffness=70.0,
        damping=2.0,
        armature=0.0549,
    ),
    delay_target="position",
    delay_min_lag=0,
    delay_max_lag=1,
)

ADAM_LITE_WAIST_ACTUATOR = DelayedInstinctActuatorCfg(
  base_cfg=InstinctActuatorCfg(
    target_names_expr=("waist_roll_joint", "waist_pitch_joint", "waist_yaw_joint"),
    effort_limit=150.0,
    velocity_limit=60.0,
    stiffness=400.0,
    damping=11.0,
    armature=0.23409,
  ),
  delay_target="position",
  delay_min_lag=0,
  delay_max_lag=1,
)

ADAM_LITE_SHOULDER_PITCH_ACTUATOR = DelayedInstinctActuatorCfg(
  base_cfg=InstinctActuatorCfg(
    target_names_expr=(".*_shoulder_pitch_joint",),
    effort_limit=65.0,
    velocity_limit=60.0,
    stiffness=60.0,
    damping=3.0,
    armature=0.01,
  ),
  delay_target="position",
  delay_min_lag=0,
  delay_max_lag=1,
)

ADAM_LITE_SHOULDER_ROLL_ACTUATOR = DelayedInstinctActuatorCfg(
  base_cfg=InstinctActuatorCfg(
    target_names_expr=(".*_shoulder_roll_joint",),
    effort_limit=65.0,
    velocity_limit=60.0,
    stiffness=60.0,
    damping=3.0,
    armature=0.01,
  ),
  delay_target="position",
  delay_min_lag=0,
  delay_max_lag=1,
)

ADAM_LITE_SHOULDER_YAW_ACTUATOR = DelayedInstinctActuatorCfg(
  base_cfg=InstinctActuatorCfg(
    target_names_expr=(".*_shoulder_yaw_joint",),
    effort_limit=65.0,
    velocity_limit=60.0,
    stiffness=60.0,
    damping=3.0,
    armature=0.01,
  ),
  delay_target="position",
  delay_min_lag=0,
  delay_max_lag=1,
)

ADAM_LITE_ARM_ELBOW_ACTUATOR = DelayedInstinctActuatorCfg(
  base_cfg=InstinctActuatorCfg(
    target_names_expr=(".*_elbow_joint",),
    effort_limit=30.0,
    velocity_limit=60.0,
    stiffness=60.0,
    damping=3.0,
    armature=0.01,
  ),
  delay_target="position",
  delay_min_lag=0,
  delay_max_lag=1,
)


adam_lite_delayed_actuator_cfgs: tuple[ActuatorCfg, ...] = (
    ADAM_LITE_HIPPITCH_KNEEPITCH_ACTUATOR,
    ADAM_LITE_HIPROLL_ACTUATOR,
    ADAM_LITE_HIPYAW_ACTUATOR,
    ADAM_LITE_ANKLEPITCH_ACTUATOR,
    ADAM_LITE_ANKLEROLL_ACTUATOR,
    ADAM_LITE_WAIST_ACTUATOR,
    ADAM_LITE_SHOULDER_PITCH_ACTUATOR,
    ADAM_LITE_SHOULDER_ROLL_ACTUATOR,
    ADAM_LITE_SHOULDER_YAW_ACTUATOR,
    ADAM_LITE_ARM_ELBOW_ACTUATOR,
)

beyondmimic_action_scale: dict[str, float] = {}
for actuator_cfg in adam_lite_delayed_actuator_cfgs:
  base_cfg = actuator_cfg.base_cfg
  effort = base_cfg.effort_limit
  stiffness = base_cfg.stiffness
  if effort is None or stiffness == 0.0:
    continue
  for joint_expr in base_cfg.target_names_expr:
    beyondmimic_action_scale[joint_expr] = 0.25 * effort / stiffness

##
# Keyframe config.
##
# Initial state matching InstinctLab G1_29DOF_TORSOBASE_POPSICLE_CFG.
# NOTE: pos is the root (torso_link) world position.
INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.89),
  joint_pos={
    ".*_hip_pitch_joint": -0.32,
    ".*_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": -0.18,
    "right_hip_yaw_joint": 0.18,
    ".*_knee_joint": 0.66,
    ".*_ankle_pitch_joint": -0.39,
    ".*_shoulder_pitch_joint": 0.0,
    ".*_elbow_joint": -0.3,
    "left_shoulder_roll_joint": 0.1,
    "right_shoulder_roll_joint": -0.1,
  },
  joint_vel={".*": 0.0},
)

##
# Final config.
##

ADAM_LITE_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    ADAM_LITE_HIPPITCH_KNEEPITCH_ACTUATOR,
    ADAM_LITE_HIPROLL_ACTUATOR,
    ADAM_LITE_HIPYAW_ACTUATOR,
    ADAM_LITE_ANKLEPITCH_ACTUATOR,
    ADAM_LITE_ANKLEROLL_ACTUATOR,
    ADAM_LITE_WAIST_ACTUATOR,
    ADAM_LITE_SHOULDER_PITCH_ACTUATOR,
    ADAM_LITE_SHOULDER_ROLL_ACTUATOR,
    ADAM_LITE_SHOULDER_YAW_ACTUATOR,
    ADAM_LITE_ARM_ELBOW_ACTUATOR,
  ),
  soft_joint_pos_limit_factor=0.9,
)

ADAM_LITE_ROBOT_CFG = EntityCfg(
  init_state=INIT_STATE,
  spec_fn=get_spec,
  articulation=ADAM_LITE_ARTICULATION,
)

ADAM_LITE_MESH_COLLISION_ROBOT_CFG = EntityCfg(
  init_state=INIT_STATE,
  spec_fn=get_mesh_collision_spec,
  articulation=ADAM_LITE_ARTICULATION,
)

# Symmetric augmentation mapping (same 23 DOF order as SP).
adam_lite_symmetric_augmentation_joint_mapping = [
    6,   # 0 left_hip_pitch <- right_hip_pitch
    7,   # 1 left_hip_roll <- right_hip_roll
    8,   # 2 left_hip_yaw <- right_hip_yaw
    9,   # 3 left_knee <- right_knee
    10,  # 4 left_ankle_pitch <- right_ankle_pitch
    11,  # 5 left_ankle_roll <- right_ankle_roll
    0,   # 6 right_hip_pitch <- left_hip_pitch
    1,   # 7 right_hip_roll <- left_hip_roll
    2,   # 8 right_hip_yaw <- left_hip_yaw
    3,   # 9 right_knee <- left_knee
    4,   # 10 right_ankle_pitch <- left_ankle_pitch
    5,   # 11 right_ankle_roll <- left_ankle_roll
    12,  # 12 waist_roll <- self
    13,  # 13 waist_pitch <- self
    14,  # 14 waist_yaw <- self
    19,  # 15 left_shoulder_pitch <- right_shoulder_pitch
    20,  # 16 left_shoulder_roll <- right_shoulder_roll
    21,  # 17 left_shoulder_yaw <- right_shoulder_yaw
    22,  # 18 left_elbow <- right_elbow
    15,  # 19 right_shoulder_pitch <- left_shoulder_pitch
    16,  # 20 right_shoulder_roll <- left_shoulder_roll
    17,  # 21 right_shoulder_yaw <- left_shoulder_yaw
    18,  # 22 right_elbow <- left_elbow
]

adam_lite_symmetric_augmentation_joint_reverse_buf = [
    1,   # 0 left_hip_pitch
    -1,  # 1 left_hip_roll
    -1,  # 2 left_hip_yaw
    1,   # 3 left_knee
    1,   # 4 left_ankle_pitch
    -1,  # 5 left_ankle_roll
    1,   # 6 right_hip_pitch
    -1,  # 7 right_hip_roll
    -1,  # 8 right_hip_yaw
    1,   # 9 right_knee
    1,   # 10 right_ankle_pitch
    -1,  # 11 right_ankle_roll
    -1,  # 12 waist_roll
    1,   # 13 waist_pitch
    -1,  # 14 waist_yaw
    1,   # 15 left_shoulder_pitch
    -1,  # 16 left_shoulder_roll
    -1,  # 17 left_shoulder_yaw
    1,   # 18 left_elbow
    1,   # 19 right_shoulder_pitch
    -1,  # 20 right_shoulder_roll
    -1,  # 21 right_shoulder_yaw
    1,   # 22 right_elbow
]


__all__ = [
    "ADAM_LITE_MJCF_PATH",
    "ADAM_LITE_ROBOT_CFG",
    "adam_lite_symmetric_augmentation_joint_mapping",
    "adam_lite_symmetric_augmentation_joint_reverse_buf",
    "adam_lite_delayed_actuator_cfgs",
    "beyondmimic_action_scale",
    "get_spec",
    "get_assets",
]
