"""Parkour terrain presets: legacy rough/noise, full sub-terrain grid, training mixes, single-terrain play."""

from __future__ import annotations

import copy

from mjlab.terrains import FlatPatchSamplingCfg

from instinct_mj.terrains.height_field.hf_terrains_cfg import (
    FixedSequenceFlatPatchSamplingCfg,
    PerlinContinuousHurdleTerrainCfg,
    PerlinContinuousSlopeTerrainCfg,
    PerlinCrossSlopeTerrainCfg,
    PerlinCurvedBridgeTerrainCfg,
    PerlinDiscreteObstaclesTerrainCfg,
    PerlinInvertedPyramidSlopedTerrainCfg,
    PerlinInvertedPyramidStairsTerrainCfg,
    PerlinNarrowLCornerTerrainCfg,
    PerlinOneSidedStairsTerrainCfg,
    PerlinPlaneTerrainCfg,
    PerlinPyramidSlopedTerrainCfg,
    PerlinPyramidStairsTerrainCfg,
    PerlinSlalomPoleTerrainCfg,
    PerlinSpiralStairsTerrainCfg,
    PerlinSquareGapTerrainCfg,
    PerlinSymmetricalSlopeTerrainCfg,
)
from instinct_mj.terrains.terrain_generator_cfg import FiledTerrainGeneratorCfg

# ---------------------------------------------------------------------------
# Legacy presets (multi-type rough curriculum, walls enabled)
# ---------------------------------------------------------------------------

ROUGH_TERRAINS_CFG = FiledTerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=3.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.07,
    vertical_scale=0.005,
    slope_threshold=1.0,
    curriculum=True,
    add_lights=True,
    sub_terrains={
        "perlin_rough": PerlinPlaneTerrainCfg(
            proportion=0.05,
            noise_scale=[0.0, 0.1],
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
            border_width=1.0,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50, patch_radius=[0.05, 0.10, 0.15, 0.20], max_height_diff=0.05
                ),
            },
        ),
        "perlin_rough_stand": PerlinPlaneTerrainCfg(
            proportion=0.05,
            noise_scale=[0.0, 0.1],
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
            border_width=1.0,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50, patch_radius=[0.05, 0.10, 0.15, 0.20], max_height_diff=0.05
                ),
            },
        ),
        "square_gaps": PerlinSquareGapTerrainCfg(
            proportion=0.10,
            gap_distance_range=(0.1, 0.7),
            gap_depth=(0.4, 0.6),
            platform_width=2.5,
            border_width=1.0,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(-0.0, 0.0),
                ),
            },
        ),
        "pyramid_stairs": PerlinPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=2.5,
            border_width=1.0,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            perlin_cfg=PerlinPlaneTerrainCfg(
                noise_scale=0.05,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(-0.0, 0.0),
                ),
            },
        ),
        "pyramid_stairs_high": PerlinPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.05, 0.45),
            step_width=1.5,
            platform_width=4.0,
            border_width=1.0,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            perlin_cfg=PerlinPlaneTerrainCfg(
                noise_scale=0.05,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(-0.0, 0.0),
                ),
            },
        ),
        "pyramid_stairs_inv": PerlinInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=2.5,
            border_width=1.0,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            perlin_cfg=PerlinPlaneTerrainCfg(
                noise_scale=0.05,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(-0.0, 0.0),
                ),
            },
        ),
        "pyramid_stairs_inv_high": PerlinInvertedPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.05, 0.45),
            step_width=1.5,
            platform_width=4.0,
            border_width=1.0,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            perlin_cfg=PerlinPlaneTerrainCfg(
                noise_scale=0.05,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(-1.2, 1.2),
                ),
            },
        ),
        "boxes": PerlinDiscreteObstaclesTerrainCfg(
            proportion=0.10,
            num_obstacles=20,
            obstacle_height_mode="fixed",
            obstacle_width_range=(0.8, 1.5),
            obstacle_height_range=(0.05, 0.45),
            platform_width=1.5,
            border_width=1.0,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            perlin_cfg=PerlinPlaneTerrainCfg(
                noise_scale=0.05,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50, patch_radius=[0.05, 0.10, 0.15, 0.20], max_height_diff=0.05
                ),
            },
        ),
        "mesh_boxes": PerlinDiscreteObstaclesTerrainCfg(
            proportion=0.10,
            num_obstacles=120,
            obstacle_height_mode="fixed",
            obstacle_width_range=(0.30, 0.50),
            obstacle_height_range=(0.05, 0.45),
            platform_width=1.5,
            border_width=1.0,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            perlin_cfg=PerlinPlaneTerrainCfg(
                noise_scale=0.05,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=50, patch_radius=[0.05, 0.10, 0.15], max_height_diff=0.05),
            },
        ),
        "hf_pyramid_slope_inv": PerlinInvertedPyramidSlopedTerrainCfg(
            proportion=0.10,
            slope_range=(0.0, 0.7),
            platform_width=1.5,
            border_width=1.0,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            perlin_cfg=PerlinPlaneTerrainCfg(
                noise_scale=0.00,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50, patch_radius=[0.05, 0.10, 0.15, 0.20], max_height_diff=0.05
                ),
            },
        ),
    },
)
ROUGH_TERRAINS_CFG_PLAY = copy.deepcopy(ROUGH_TERRAINS_CFG)
ROUGH_TERRAINS_CFG_PLAY.num_rows = 4
ROUGH_TERRAINS_CFG_PLAY.num_cols = 10

NOISE_ONLY_TERRAINS_CFG = FiledTerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=3.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.07,
    vertical_scale=0.005,
    slope_threshold=1.0,
    curriculum=True,
    add_lights=True,
    sub_terrains={
        "perlin_rough": PerlinPlaneTerrainCfg(
            proportion=1.0,
            noise_scale=[0.0, 0.1],
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
            border_width=0.0,
            wall_prob=[0.0, 0.0, 0.0, 0.0],
            wall_height=0.0,
            wall_thickness=0.05,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50, patch_radius=[0.05, 0.10, 0.15, 0.20], max_height_diff=0.05
                ),
            },
        ),
    },
)
NOISE_ONLY_TERRAINS_CFG_PLAY = copy.deepcopy(NOISE_ONLY_TERRAINS_CFG)
NOISE_ONLY_TERRAINS_CFG_PLAY.num_rows = 4
NOISE_ONLY_TERRAINS_CFG_PLAY.num_cols = 10

# ---------------------------------------------------------------------------
# Parkour terrain grid (all sub-types registered with proportion=0 until a preset activates them)
# ---------------------------------------------------------------------------

PARKOUR_TERRAIN_CATALOG_CFG = FiledTerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=5.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=1.0,
    curriculum=True,
    add_lights=True,
    sub_terrains={
        "perlin_rough": PerlinPlaneTerrainCfg(
            proportion=0.0,
            noise_scale=[0.0, 0.1],
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
            border_width=1.0,
            wall_prob=[0.0, 0.0, 0.0, 0.0],
            wall_height=0.0,
            wall_thickness=0.05,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(0.0, 0.0),
                ),
            },
        ),
        "perlin_rough_stand": PerlinPlaneTerrainCfg(
            proportion=0.0,
            noise_scale=[0.0, 0.1],
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
            border_width=1.0,
            wall_prob=[0.0, 0.0, 0.0, 0.0],
            wall_height=0.0,
            wall_thickness=0.05,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(0.0, 0.0),
                ),
            },
        ),
        "pyramid_stairs": PerlinPyramidStairsTerrainCfg(
            proportion=0.0,
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            platform_width=2.5,
            border_width=1.0,
            wall_prob=[0.0, 0.0, 0.0, 0.0],
            wall_height=0.0,
            wall_thickness=0.05,
            perlin_cfg=PerlinPlaneTerrainCfg(
                noise_scale=0.05,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(0.0, 0.0),
                ),
            },
        ),
        "pyramid_stairs_inv": PerlinInvertedPyramidStairsTerrainCfg(
            proportion=0.0,
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            platform_width=2.5,
            border_width=1.0,
            wall_prob=[0.0, 0.0, 0.0, 0.0],
            wall_height=0.0,
            wall_thickness=0.05,
            perlin_cfg=PerlinPlaneTerrainCfg(
                noise_scale=0.05,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(0.0, 0.0),
                ),
            },
        ),
        "one_sided_stairs": PerlinOneSidedStairsTerrainCfg(
            proportion=0.0,
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            platform_width=1.5,
            lateral_width=(2.0, 4.0),
            stair_face="neg_x",
            up_down=True,
            border_width=1.0,
            wall_prob=[0.0, 0.0, 0.0, 0.0],
            wall_height=0.0,
            wall_thickness=0.05,
            virtual_obstacle_keys=("stairs_edges",),
            perlin_cfg=PerlinPlaneTerrainCfg(
                noise_scale=0.05,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(0.0, 0.0),
                ),
            },
        ),
        "continuous_slope": PerlinContinuousSlopeTerrainCfg(
            proportion=0.0,
            slope_angle_range=(12.0, 20.0),
            total_length=7.4,
            terrain_width=(2.0, 2.8),
            max_height=0.39,
            border_width=1.0,
            wall_prob=[0.0, 0.0, 0.0, 0.0],
            wall_height=0.0,
            wall_thickness=0.05,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(0.0, 0.0),
                ),
            },
        ),
        "cross_slope": PerlinCrossSlopeTerrainCfg(
            proportion=0.0,
            total_length=6.0,
            terrain_width=3.0,
            segment_length=1.5,
            num_segments=4,
            slope_angle_range=(5.0, 15.0),
            border_width=1.0,
            wall_prob=[0.0, 0.0, 0.0, 0.0],
            wall_height=0.0,
            wall_thickness=0.05,
            flat_patch_sampling={
                "target": FixedSequenceFlatPatchSamplingCfg(
                    num_patches=6,
                    patch_radius=[0.10],
                    max_height_diff=0.05,
                    x_range=(0.0, 0.0),
                    y_range=(0.0, 0.0),
                    fixed_xy_points=(
                        (-3.5, -1.0),
                        (-2.25, -1.0),
                        (-2.25, 0.0),
                        (2.25, 0.0),
                        (2.25, 1.0),
                        (3.75, 1.0),
                    ),
                ),
            },
        ),
        "symmetrical_slope": PerlinSymmetricalSlopeTerrainCfg(
            proportion=0.0,
            total_length=3.0,
            terrain_width=3.5,
            slope_angle=20.0,
            flat_bottom=0.1,
            border_width=1.0,
            wall_prob=[0.0, 0.0, 0.0, 0.0],
            wall_height=0.0,
            wall_thickness=0.05,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(4.0, 4.0),
                    y_range=(4.0, 4.0),
                ),
            },
        ),
        "curved_bridge": PerlinCurvedBridgeTerrainCfg(
            proportion=0.0,
            bridge_length=10.0,
            bridge_width=0.5,
            bridge_width_range=(1.0, 0.5),
            bridge_height=0.1,
            curve_radius_range=(2.0, 1.0),
            straight_lead_in_m=0.0,
            straight_lead_out_m=0.0,
            border_width=1.0,
            wall_prob=[0.0, 0.0, 0.0, 0.0],
            wall_height=0.0,
            wall_thickness=0.05,
            flat_patch_sampling={
                "start": FixedSequenceFlatPatchSamplingCfg(
                    num_patches=1,
                    patch_radius=[0.10],
                    max_height_diff=0.05,
                    x_range=(0.0, 0.0),
                    y_range=(0.0, 0.0),
                    centerline_endpoint="start",
                    centerline_arc_offset_m=0.0,
                    centerline_spawn_on_ground=True,
                    centerline_ground_approach_m=0.75,
                    centerline_ground_offset_m=(0.0, 0.0),
                ),
                "target": FixedSequenceFlatPatchSamplingCfg(
                    num_patches=1,
                    patch_radius=[0.10],
                    max_height_diff=0.05,
                    x_range=(0.0, 0.0),
                    y_range=(0.0, 0.0),
                    centerline_endpoint="end",
                    centerline_spawn_on_ground=True,
                    centerline_tangent_offset_m=1.0,
                    centerline_ground_approach_m=0.75,
                    centerline_ground_offset_m=(0.0, 0.0),
                ),
            },
        ),
        "narrow_l_corner": PerlinNarrowLCornerTerrainCfg(
            proportion=0.0,
            leg_length=4.0,
            channel_width=0.6,
            channel_width_range=(1.0, 0.6),
            deck_height=0.3,
            deck_height_range=(0.1, 0.3),
            turn_angle_range=(0.0, 90.0),
            railing_side="random",
            railing_height=1.2,
            railing_spacing=0.25,
            railing_bar_radius=0.02,
            outer_curb_height=0.15,
            step_height=0.15,
            step_width=0.30,
            border_width=1.0,
            wall_prob=[0.0, 0.0, 0.0, 0.0],
            wall_height=0.0,
            wall_thickness=0.05,
            flat_patch_sampling={
                "start": FixedSequenceFlatPatchSamplingCfg(
                    num_patches=1,
                    patch_radius=[0.10],
                    max_height_diff=0.05,
                    x_range=(0.0, 0.0),
                    y_range=(0.0, 0.0),
                    centerline_endpoint="start",
                    centerline_spawn_on_ground=True,
                    centerline_ground_approach_m=0.75,
                    centerline_ground_offset_m=(0.0, 0.0),
                ),
                "target": FixedSequenceFlatPatchSamplingCfg(
                    num_patches=1,
                    patch_radius=[0.10],
                    max_height_diff=0.05,
                    x_range=(0.0, 0.0),
                    y_range=(0.0, 0.0),
                    centerline_endpoint="end",
                    centerline_spawn_on_ground=True,
                    centerline_tangent_offset_m=0.75,
                    centerline_ground_approach_m=0.75,
                    centerline_ground_offset_m=(0.0, 0.0),
                ),
            },
        ),
        "narrow_l_corner_right": PerlinNarrowLCornerTerrainCfg(
            proportion=0.0,
            leg_length=4.0,
            channel_width_range=(1.0, 0.6),
            deck_height=0.3,
            deck_height_range=(0.1, 0.3),
            turn="right",
            turn_angle_range=(0.0, 90.0),
            railing_side="random",
            railing_height=1.2,
            railing_spacing=0.25,
            railing_bar_radius=0.02,
            outer_curb_height=0.15,
            step_height=0.15,
            step_width=0.30,
            border_width=1.0,
            wall_prob=[0.0, 0.0, 0.0, 0.0],
            wall_height=0.0,
            wall_thickness=0.05,
            flat_patch_sampling={
                "start": FixedSequenceFlatPatchSamplingCfg(
                    num_patches=1,
                    patch_radius=[0.10],
                    max_height_diff=0.05,
                    x_range=(0.0, 0.0),
                    y_range=(0.0, 0.0),
                    centerline_endpoint="start",
                    centerline_spawn_on_ground=True,
                    centerline_ground_approach_m=0.75,
                    centerline_ground_offset_m=(0.0, 0.0),
                ),
                "target": FixedSequenceFlatPatchSamplingCfg(
                    num_patches=1,
                    patch_radius=[0.10],
                    max_height_diff=0.05,
                    x_range=(0.0, 0.0),
                    y_range=(0.0, 0.0),
                    centerline_endpoint="end",
                    centerline_spawn_on_ground=True,
                    centerline_tangent_offset_m=0.75,
                    centerline_ground_approach_m=0.75,
                    centerline_ground_offset_m=(0.0, 0.0),
                ),
            },
        ),
        "continuous_hurdle": PerlinContinuousHurdleTerrainCfg(
            proportion=0.0,
            num_hurdles=5,
            hurdle_width=2.4,
            hurdle_height=0.3,
            hurdle_height_range=(0.02, 0.30),
            hurdle_thickness=0.05,
            hurdle_spacing=1.1,
            hurdle_spacing_range=(1.5, 1.0),
            perlin_cfg=PerlinPlaneTerrainCfg(
                noise_scale=(0.10, 0.30),
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            border_width=1.0,
            wall_prob=[0.0, 0.0, 0.0, 0.0],
            wall_height=0.0,
            wall_thickness=0.05,
            virtual_obstacle_keys=("hurdle_edges",),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(4.0, 4.0),
                    y_range=(4.0, 4.0),
                ),
            },
        ),
        "slalom_pole": PerlinSlalomPoleTerrainCfg(
            proportion=0.0,
            num_poles=5,
            pole_height=1.5,
            pole_radius=0.015,
            pole_spacing=1.1,
            border_width=1.0,
            wall_prob=[0.0, 0.0, 0.0, 0.0],
            wall_height=0.0,
            wall_thickness=0.05,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(4.0, 4.0),
                    y_range=(4.0, 4.0),
                ),
            },
        ),
        "spiral_stairs": PerlinSpiralStairsTerrainCfg(
            proportion=0.0,
            obstacle_length=5.04,
            obstacle_width=3.0,
            step_height=0.15,
            num_steps=6,
            platform_length=1.44,
            stair_width=1.3,
            flight_angle_range=(0.0, 30.0),
            edge_resolution=0.02,
            spiral_radius=1.5,
            spiral_step_arc_length=0.30,
            spiral_entrance_along=0.0,
            spiral_entrance_inset_x=0.7,
            spiral_pivot_backoff_x=0.0,
            spiral_center_offset=(0.0, 0.0),
            spiral_up_start_angle_deg=270.0,
            spiral_total_angle_deg=30.0,
            add_down_spiral=True,
            border_width=1.0,
            wall_prob=[0.0, 0.0, 0.0, 0.0],
            wall_height=0.0,
            wall_thickness=0.05,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(0.0, 0.0),
                ),
            },
        ),
    },
)
PARKOUR_TERRAIN_CATALOG_CFG_PLAY = copy.deepcopy(PARKOUR_TERRAIN_CATALOG_CFG)
PARKOUR_TERRAIN_CATALOG_CFG_PLAY.num_rows = 4
PARKOUR_TERRAIN_CATALOG_CFG_PLAY.num_cols = 10

ADAM_LITE_PARKOUR_TERRAINS_CFG = copy.deepcopy(PARKOUR_TERRAIN_CATALOG_CFG)
for _k in ADAM_LITE_PARKOUR_TERRAINS_CFG.sub_terrains:
    ADAM_LITE_PARKOUR_TERRAINS_CFG.sub_terrains[_k].proportion = (
        0.5 if _k in ("narrow_l_corner", "continuous_hurdle") else 0.0
    )

STAIRS_ONLY_TERRAINS_CFG = FiledTerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=5.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=1.0,
    curriculum=True,
    add_lights=True,
    sub_terrains={
        "perlin_rough": PerlinPlaneTerrainCfg(
            proportion=0.1,
            noise_scale=[0.0, 0.1],
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
            border_width=1.0,
            wall_prob=[0.0, 0.0, 0.0, 0.0],
            wall_height=0.0,
            wall_thickness=0.05,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50, patch_radius=[0.05, 0.10, 0.15, 0.20], max_height_diff=0.05
                ),
            },
        ),
        "perlin_rough_stand": PerlinPlaneTerrainCfg(
            proportion=0.1,
            noise_scale=[0.0, 0.1],
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
            border_width=1.0,
            wall_prob=[0.0, 0.0, 0.0, 0.0],
            wall_height=0.0,
            wall_thickness=0.05,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50, patch_radius=[0.05, 0.10, 0.15, 0.20], max_height_diff=0.05
                ),
            },
        ),
        "pyramid_stairs": PerlinPyramidStairsTerrainCfg(
            proportion=0.0,
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            platform_width=2.5,
            border_width=1.0,
            wall_prob=[0.0, 0.0, 0.0, 0.0],
            wall_height=0.0,
            wall_thickness=0.05,
            perlin_cfg=PerlinPlaneTerrainCfg(
                noise_scale=0.05,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(-0.0, 0.0),
                ),
            },
        ),
        "pyramid_stairs_inv": PerlinInvertedPyramidStairsTerrainCfg(
            proportion=0.0,
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            platform_width=2.5,
            border_width=1.0,
            wall_prob=[0.0, 0.0, 0.0, 0.0],
            wall_height=0.0,
            wall_thickness=0.05,
            perlin_cfg=PerlinPlaneTerrainCfg(
                noise_scale=0.05,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(-0.0, 0.0),
                ),
            },
        ),
        "one_sided_stairs": PerlinOneSidedStairsTerrainCfg(
            proportion=0.8,
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            platform_width=1.5,
            lateral_width=(2.0, 4.0),
            stair_face="neg_x",
            up_down=True,
            border_width=1.0,
            wall_prob=[0.0, 0.0, 0.0, 0.0],
            wall_height=0.0,
            wall_thickness=0.05,
            perlin_cfg=PerlinPlaneTerrainCfg(
                noise_scale=0.05,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(0.0, 0.0),
                ),
            },
        ),
    },
)
STAIRS_ONLY_TERRAINS_CFG_PLAY = copy.deepcopy(STAIRS_ONLY_TERRAINS_CFG)
STAIRS_ONLY_TERRAINS_CFG_PLAY.num_rows = 4
STAIRS_ONLY_TERRAINS_CFG_PLAY.num_cols = 10

# ---------------------------------------------------------------------------
# Three slopes (training num_cols=20)
# ---------------------------------------------------------------------------

THREE_SLOPES_TERRAINS_CFG = FiledTerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=3.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=1.0,
    curriculum=True,
    add_lights=True,
    sub_terrains={
        "perlin_rough_stand": PerlinPlaneTerrainCfg(
            proportion=0.1,
            noise_scale=[0.0, 0.1],
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
            border_width=1.0,
            wall_prob=[0.0, 0.0, 0.0, 0.0],
            wall_height=0.0,
            wall_thickness=0.05,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(0.0, 0.0),
                ),
            },
        ),
        "continuous_slope": PerlinContinuousSlopeTerrainCfg(
            proportion=0.2,
            slope_angle_range=(12.0, 20.0),
            total_length=7.4,
            terrain_width=(1.2, 2.8),
            max_height=0.39,
            border_width=1.0,
            wall_prob=[0.0, 0.0, 0.0, 0.0],
            wall_height=0.0,
            wall_thickness=0.05,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(0.0, 0.0),
                ),
            },
        ),
        "cross_slope": PerlinCrossSlopeTerrainCfg(
            proportion=0.5,
            total_length=6.0,
            terrain_width=3.0,
            segment_length=1.5,
            num_segments=4,
            slope_angle_range=(5.0, 15.0),
            border_width=1.0,
            wall_prob=[0.0, 0.0, 0.0, 0.0],
            wall_height=0.0,
            wall_thickness=0.05,
            flat_patch_sampling={
                "target": FixedSequenceFlatPatchSamplingCfg(
                    num_patches=6,
                    patch_radius=[0.10],
                    max_height_diff=0.05,
                    x_range=(0.0, 0.0),
                    y_range=(0.0, 0.0),
                    fixed_xy_points=(
                        (-3.25, -1.0),
                        (-2.25, -1.0),
                        (-2.25, 0.0),
                        # (-0.75, 0.0),
                        # (0.75, 0.0),
                        (2.25, 0.0),
                        (2.25, 1.0),
                        (3.75, 1.0),
                    ),
                ),
            },
        ),
        "symmetrical_slope": PerlinSymmetricalSlopeTerrainCfg(
            proportion=0.2,
            total_length=3.0,
            terrain_width=3.5,
            slope_angle_range=(10.0, 20.0),
            flat_bottom=0.1,
            border_width=1.0,
            wall_prob=[0.0, 0.0, 0.0, 0.0],
            wall_height=0.0,
            wall_thickness=0.05,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(0.0, 0.0),
                ),
            },
        ),
    },
)
THREE_SLOPES_TERRAINS_CFG_PLAY = copy.deepcopy(THREE_SLOPES_TERRAINS_CFG)
THREE_SLOPES_TERRAINS_CFG_PLAY.num_rows = 4
THREE_SLOPES_TERRAINS_CFG_PLAY.num_cols = 10

# ---------------------------------------------------------------------------
# Adam Lite parkour base (rough / stand / pyramid stairs + inv / pyramid slopes)
# ---------------------------------------------------------------------------

_BASE_LIGHT_PERLIN_CFG = PerlinPlaneTerrainCfg(
    noise_scale=[0.0, 0.10],
    noise_frequency=20,
    fractal_octaves=2,
    fractal_lacunarity=2.0,
    fractal_gain=0.25,
    centering=True,
)

ADAM_LITE_BASE_PARKOUR_TERRAINS_CFG = FiledTerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=3.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=1.0,
    curriculum=True,
    add_lights=True,
    sub_terrains={
        "perlin_rough": PerlinPlaneTerrainCfg(
            proportion=0.22,
            noise_scale=[0.10, 0.20],
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
            border_width=1.0,
            wall_prob=[0.0, 0.0, 0.0, 0.0],
            wall_height=0.0,
            wall_thickness=0.05,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(2.0, 7.5),
                    y_range=(-2.0, 2.0),
                ),
            },
        ),
        "perlin_rough_stand": PerlinPlaneTerrainCfg(
            proportion=0.10,
            noise_scale=[0.20, 0.30],
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
            border_width=1.0,
            wall_prob=[0.0, 0.0, 0.0, 0.0],
            wall_height=0.0,
            wall_thickness=0.05,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(2.0, 4.5),
                    y_range=(-2.0, 2.0),
                ),
            },
        ),
        "pyramid_stairs": PerlinPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.02, 0.15),
            step_width=0.3,
            platform_width=2.5,
            border_width=1.0,
            wall_prob=[0.0, 0.0, 0.0, 0.0],
            wall_height=0.0,
            wall_thickness=0.05,
            virtual_obstacle_keys=("stairs_edges",),
            perlin_cfg=copy.deepcopy(_BASE_LIGHT_PERLIN_CFG),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(0.0, 0.0),
                ),
            },
        ),
        "pyramid_stairs_inv": PerlinInvertedPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.02, 0.15),
            step_width=0.3,
            platform_width=2.5,
            border_width=1.0,
            wall_prob=[0.0, 0.0, 0.0, 0.0],
            wall_height=0.0,
            wall_thickness=0.05,
            virtual_obstacle_keys=("stairs_edges",),
            perlin_cfg=copy.deepcopy(_BASE_LIGHT_PERLIN_CFG),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(0.0, 0.0),
                ),
            },
        ),
        "hf_pyramid_slope": PerlinPyramidSlopedTerrainCfg(
            proportion=0.14,
            slope_range=(0.09, 0.27),
            platform_width=2.0,
            inverted=False,
            border_width=1.0,
            wall_prob=[0.0, 0.0, 0.0, 0.0],
            wall_height=0.0,
            wall_thickness=0.05,
            perlin_cfg=copy.deepcopy(_BASE_LIGHT_PERLIN_CFG),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(0.0, 0.0),
                ),
            },
        ),
        "hf_pyramid_slope_inv": PerlinInvertedPyramidSlopedTerrainCfg(
            proportion=0.14,
            slope_range=(0.09, 0.27),
            platform_width=2.0,
            inverted=True,
            border_width=1.0,
            wall_prob=[0.0, 0.0, 0.0, 0.0],
            wall_height=0.0,
            wall_thickness=0.05,
            perlin_cfg=copy.deepcopy(_BASE_LIGHT_PERLIN_CFG),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(0.0, 0.0),
                ),
            },
        ),
    },
)
ADAM_LITE_BASE_PARKOUR_TERRAINS_CFG_PLAY = copy.deepcopy(ADAM_LITE_BASE_PARKOUR_TERRAINS_CFG)
ADAM_LITE_BASE_PARKOUR_TERRAINS_CFG_PLAY.num_rows = 4
ADAM_LITE_BASE_PARKOUR_TERRAINS_CFG_PLAY.num_cols = 10

# ---------------------------------------------------------------------------
# Adam Lite mixed parkour (base finetune: rough / three slopes / one-sided stairs / slalom deck)
# ---------------------------------------------------------------------------

_mixed_catalog = PARKOUR_TERRAIN_CATALOG_CFG.sub_terrains
_mixed_three_slopes = THREE_SLOPES_TERRAINS_CFG.sub_terrains
_mixed_base = ADAM_LITE_BASE_PARKOUR_TERRAINS_CFG.sub_terrains

ADAM_LITE_MIXED_PARKOUR_TERRAINS_CFG = FiledTerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=3.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=1.0,
    curriculum=True,
    add_lights=True,
    sub_terrains={
        "perlin_rough": copy.deepcopy(_mixed_base["perlin_rough"]),
        "perlin_rough_stand": copy.deepcopy(_mixed_base["perlin_rough_stand"]),
        "continuous_slope": copy.deepcopy(_mixed_catalog["continuous_slope"]),
        "cross_slope": copy.deepcopy(_mixed_catalog["cross_slope"]),
        "symmetrical_slope": copy.deepcopy(_mixed_three_slopes["symmetrical_slope"]),
        "one_sided_stairs": copy.deepcopy(_mixed_catalog["one_sided_stairs"]),
        "slalom_pole": copy.deepcopy(_mixed_catalog["slalom_pole"]),
    },
)
_mixed_props = {
    "perlin_rough": 0.18,
    "perlin_rough_stand": 0.08,
    "continuous_slope": 0.13,
    "cross_slope": 0.18,
    "symmetrical_slope": 0.10,
    "one_sided_stairs": 0.22,
    "slalom_pole": 0.11,
}
for _k, _p in _mixed_props.items():
    ADAM_LITE_MIXED_PARKOUR_TERRAINS_CFG.sub_terrains[_k].proportion = _p

# Light Perlin on slopes + slalom deck/approach; amplitude scales with terrain row difficulty.
for _mixed_perlin_key in ("continuous_slope", "cross_slope", "symmetrical_slope", "slalom_pole"):
    ADAM_LITE_MIXED_PARKOUR_TERRAINS_CFG.sub_terrains[_mixed_perlin_key].perlin_cfg = copy.deepcopy(
        _BASE_LIGHT_PERLIN_CFG
    )

_mixed_slalom_fp = ADAM_LITE_MIXED_PARKOUR_TERRAINS_CFG.sub_terrains["slalom_pole"].flat_patch_sampling
if _mixed_slalom_fp is not None and "target" in _mixed_slalom_fp:
    _mixed_slalom_fp["target"].x_range = (0.0, 1.5)
    _mixed_slalom_fp["target"].y_range = (3.0, 5.0)

ADAM_LITE_MIXED_PARKOUR_TERRAINS_CFG_PLAY = copy.deepcopy(ADAM_LITE_MIXED_PARKOUR_TERRAINS_CFG)
ADAM_LITE_MIXED_PARKOUR_TERRAINS_CFG_PLAY.num_rows = 4
ADAM_LITE_MIXED_PARKOUR_TERRAINS_CFG_PLAY.num_cols = 10

# ---------------------------------------------------------------------------
# Precision navigation: curved bridge + narrow L-corner (X-only velocity)
# ---------------------------------------------------------------------------

BRIDGE_L_CORNER_TERRAINS_CFG = copy.deepcopy(PARKOUR_TERRAIN_CATALOG_CFG)
BRIDGE_L_CORNER_TERRAINS_CFG.size = (12.0, 10.0)
for _k in BRIDGE_L_CORNER_TERRAINS_CFG.sub_terrains:
    BRIDGE_L_CORNER_TERRAINS_CFG.sub_terrains[_k].proportion = (
        0.5
        if _k == "curved_bridge"
        else 0.25
        if _k in ("narrow_l_corner", "narrow_l_corner_right")
        else 0.0
    )
_curved_bridge_cfg = BRIDGE_L_CORNER_TERRAINS_CFG.sub_terrains["curved_bridge"]
_curved_bridge_cfg.straight_at_zero_difficulty = True
_curved_bridge_cfg.straight_max_difficulty = 1.0 / float(BRIDGE_L_CORNER_TERRAINS_CFG.num_rows)
BRIDGE_L_CORNER_TERRAINS_CFG_PLAY = copy.deepcopy(BRIDGE_L_CORNER_TERRAINS_CFG)
BRIDGE_L_CORNER_TERRAINS_CFG_PLAY.size = (12.0, 10.0)
BRIDGE_L_CORNER_TERRAINS_CFG_PLAY.num_rows = 4
BRIDGE_L_CORNER_TERRAINS_CFG_PLAY.num_cols = 10
_play_curved_bridge = BRIDGE_L_CORNER_TERRAINS_CFG_PLAY.sub_terrains["curved_bridge"]
_play_curved_bridge.straight_max_difficulty = 1.0 / float(BRIDGE_L_CORNER_TERRAINS_CFG_PLAY.num_rows)

# ---------------------------------------------------------------------------
# Hurdle terrain (continuous hurdle-only, training num_cols=20)
# ---------------------------------------------------------------------------

HURDLE_TERRAINS_CFG = FiledTerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=3.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=1.0,
    curriculum=True,
    add_lights=True,
    sub_terrains={
        "continuous_hurdle": PerlinContinuousHurdleTerrainCfg(
            proportion=1.0,
            num_hurdles=5,
            hurdle_width=2.4,
            hurdle_height=0.3,
            hurdle_height_range=(0.02, 0.30),
            hurdle_thickness=0.05,
            hurdle_spacing=1.1,
            hurdle_spacing_range=(1.5, 1.0),
            perlin_cfg=PerlinPlaneTerrainCfg(
                noise_scale=(0.02, 0.10),
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            border_width=1.0,
            wall_prob=[0.0, 0.0, 0.0, 0.0],
            wall_height=0.0,
            wall_thickness=0.05,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(7.5, 7.5),
                    y_range=(0.0, 0.0),
                ),
            },
        ),
    },
)
HURDLE_TERRAINS_CFG_PLAY = copy.deepcopy(HURDLE_TERRAINS_CFG)
HURDLE_TERRAINS_CFG_PLAY.num_rows = 4
HURDLE_TERRAINS_CFG_PLAY.num_cols = 10

# ---------------------------------------------------------------------------
# Single-terrain presets (play / ablation), cloned from PARKOUR_TERRAIN_CATALOG_CFG
# ---------------------------------------------------------------------------

def _with_single_terrain(base: FiledTerrainGeneratorCfg, terrain_key: str) -> FiledTerrainGeneratorCfg:
    cfg = copy.deepcopy(base)
    for k in cfg.sub_terrains:
        cfg.sub_terrains[k].proportion = 1.0 if k == terrain_key else 0.0
    return cfg


CONTINUOUS_SLOPE_TERRAIN_CFG = _with_single_terrain(PARKOUR_TERRAIN_CATALOG_CFG, "continuous_slope")
CONTINUOUS_SLOPE_TERRAIN_CFG_PLAY = _with_single_terrain(PARKOUR_TERRAIN_CATALOG_CFG_PLAY, "continuous_slope")
SYMMETRICAL_SLOPE_TERRAIN_CFG = _with_single_terrain(PARKOUR_TERRAIN_CATALOG_CFG, "symmetrical_slope")
SYMMETRICAL_SLOPE_TERRAIN_CFG_PLAY = _with_single_terrain(PARKOUR_TERRAIN_CATALOG_CFG_PLAY, "symmetrical_slope")
CURVED_BRIDGE_TERRAIN_CFG = _with_single_terrain(PARKOUR_TERRAIN_CATALOG_CFG, "curved_bridge")
CURVED_BRIDGE_TERRAIN_CFG_PLAY = _with_single_terrain(PARKOUR_TERRAIN_CATALOG_CFG_PLAY, "curved_bridge")
CONTINUOUS_HURDLE_TERRAIN_CFG = _with_single_terrain(PARKOUR_TERRAIN_CATALOG_CFG, "continuous_hurdle")
CONTINUOUS_HURDLE_TERRAIN_CFG_PLAY = _with_single_terrain(PARKOUR_TERRAIN_CATALOG_CFG_PLAY, "continuous_hurdle")
SLALOM_POLE_TERRAIN_CFG = _with_single_terrain(PARKOUR_TERRAIN_CATALOG_CFG, "slalom_pole")
SLALOM_POLE_TERRAIN_CFG_PLAY = _with_single_terrain(PARKOUR_TERRAIN_CATALOG_CFG_PLAY, "slalom_pole")
CROSS_SLOPE_TERRAIN_CFG = _with_single_terrain(PARKOUR_TERRAIN_CATALOG_CFG, "cross_slope")
CROSS_SLOPE_TERRAIN_CFG_PLAY = _with_single_terrain(PARKOUR_TERRAIN_CATALOG_CFG_PLAY, "cross_slope")
NARROW_L_CORNER_TERRAIN_CFG = _with_single_terrain(PARKOUR_TERRAIN_CATALOG_CFG, "narrow_l_corner")
NARROW_L_CORNER_TERRAIN_CFG_PLAY = _with_single_terrain(PARKOUR_TERRAIN_CATALOG_CFG_PLAY, "narrow_l_corner")
NARROW_L_CORNER_RIGHT_TERRAIN_CFG = _with_single_terrain(PARKOUR_TERRAIN_CATALOG_CFG, "narrow_l_corner_right")
NARROW_L_CORNER_RIGHT_TERRAIN_CFG_PLAY = _with_single_terrain(PARKOUR_TERRAIN_CATALOG_CFG_PLAY, "narrow_l_corner_right")
PYRAMID_STAIRS_TERRAIN_CFG = _with_single_terrain(ADAM_LITE_BASE_PARKOUR_TERRAINS_CFG, "pyramid_stairs")
PYRAMID_STAIRS_TERRAIN_CFG_PLAY = _with_single_terrain(
    ADAM_LITE_BASE_PARKOUR_TERRAINS_CFG_PLAY, "pyramid_stairs"
)
PYRAMID_STAIRS_INV_TERRAIN_CFG = _with_single_terrain(ADAM_LITE_BASE_PARKOUR_TERRAINS_CFG, "pyramid_stairs_inv")
PYRAMID_STAIRS_INV_TERRAIN_CFG_PLAY = _with_single_terrain(
    ADAM_LITE_BASE_PARKOUR_TERRAINS_CFG_PLAY, "pyramid_stairs_inv"
)
