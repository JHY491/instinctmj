from dataclasses import MISSING, dataclass

from instinct_mj.managers import CommandTermCfg

from .pose_velocity_command import PoseVelocityCommand


@dataclass(kw_only=True)
class PoseVelocityCommandCfg(CommandTermCfg):
    """Configuration for the position command generator."""

    class_type: type = PoseVelocityCommand

    entity_name: str = MISSING
    """Name of the entity in the environment for which commands are generated."""

    velocity_control_stiffness: float = 1.0
    """Scale factor to convert the position error to linear velocity command. Defaults to 1.0."""

    heading_control_stiffness: float = 1.0
    """Scale factor to convert the heading error to angular velocity command. Defaults to 1.0."""

    only_positive_lin_vel_x: bool = False
    """Whether to only sample positive linear x velocity commands. Defaults to False."""

    @dataclass(kw_only=True)
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        lin_vel_x: tuple[float, float] = MISSING
        """Range for the linear-x velocity command (in m/s)."""

        lin_vel_y: tuple[float, float] = MISSING
        """Range for the linear-y velocity command (in m/s)."""

        ang_vel_z: tuple[float, float] = MISSING
        """Range for the angular-z velocity command (in rad/s)."""

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands. Only used in random_velocity_terrains."""

    random_velocity_terrain: list[str] = None
    """List of terrain types for which the velocity commands should be randomized."""

    velocity_ranges: dict = None
    """Dictionary containing velocity ranges for different terrains. If not None, the velocity ranges will be set based on the terrain type."""

    lin_vel_threshold: float = 0.15
    """Minimal threshold for the linear velocity command (in m/s)."""

    ang_vel_threshold: float = 0.15
    """Minimal threshold for the angular velocity command (in rad/s)."""

    yaw_vel_gate_threshold: float | None = None
    """If set, linear velocity is zeroed when |heading_command_w| exceeds this threshold (rad)."""

    descend_speed_scale: float = 1.0
    """Multiplier applied to the forward speed cap (lin_vel_x) while the base is descending.

    1.0 disables descent slowdown. Used to slow down stair descent, where a fast command
    produces long steps that skip a tread or land on an edge. Only the forward cap is scaled.
    """

    descend_vz_threshold: float = -0.15
    """Smoothed world-frame base vertical velocity (m/s) below which the robot counts as descending."""

    descend_vz_ema_alpha: float = 0.1
    """Per-step EMA factor for the base vertical velocity used in descent detection (0..1)."""

    descend_speed_terrain: list[str] | None = None
    """Sub-terrain names where descent slowdown applies. None means all terrains."""

    lin_vel_metrics_std: float = 0.5
    """Standard deviation for the linear velocity metrics."""

    ang_vel_metrics_std: float = 0.5
    """Standard deviation for the angular velocity metrics."""

    rel_standing_envs: float = 0.0
    """The sampled probability of environments that should be standing still. Defaults to 0.0."""

    target_dis_threshold: float = 0.4
    """The distance threshold to the target position below which the command is set to zero. Defaults to 0.4."""

    patch_vis: bool = False
    """If True, draw flat-patch candidates (same markers as ``show_flat_patch_candidates``). Kept for compatibility."""

    show_flat_patch_candidates: bool = True
    """When True and command ``debug_vis`` is on, draw ``terrain.flat_patches[\"target\"]`` for this env's cell."""

    show_cref_foothold_candidates: bool = False
    """When True and command ``debug_vis`` is on, draw CReF-style stair foothold candidates from the reward term."""

    show_cref_foothold_all_treads: bool = True
    """When True, also draw all stair tread candidates used by ``cref_stair_foothold_reward``."""

    cref_foothold_reward_name: str = "cref_stair_foothold"
    """Reward term name used to look up the CReF-style foothold state for debug visualization."""

    flat_patch_target_count: int | None = None
    """If set, draw at most this many candidate patches (index order). Use for fixed-length targets (e.g. 4)."""

    flat_patch_random_index: bool = True
    """If True (default), pick a random patch index on each flat-patch resample. If False, always use index 0."""

    flat_patch_advance_on_arrival: bool = False
    """If True, advance patch index on arrival for columns using ``FixedSequenceFlatPatchSamplingCfg``."""

    flat_patch_arrival_xy_dist: float = 0.4
    """Horizontal distance threshold (m) for flat-patch arrival."""

    flat_patch_arrival_hold_steps: int = 3
    """Consecutive steps within ``flat_patch_arrival_xy_dist`` before advancing to the next patch."""

    flat_patch_resample_velocity_on_advance: bool = True
    """If True, call velocity resampling when advancing to the next flat-patch target."""

    guided_waypoints_local: dict[str, list[tuple[float, float, float]]] | None = None
    """Per sub-terrain name, waypoints in the sub-terrain cell frame (add cell world origin)."""

    guided_arrival_xy_dist: float = 0.35
    """Advance to the next waypoint when horizontal distance falls below this value."""

    guided_arrival_hold_steps: int = 3
    """Consecutive steps within ``guided_arrival_xy_dist`` required before advancing."""

    guided_loop_waypoints: bool = True
    """If True, wrap to the first waypoint after the last; otherwise stay on the last point."""

    guided_resample_velocity_on_advance: bool = True
    """If True, re-sample velocity limits when advancing to the next waypoint."""

    def build(self, env):
        return PoseVelocityCommand(self, env)
