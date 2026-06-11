"""Parkour environment scene helpers and terrain preset re-exports."""

from __future__ import annotations

import mujoco
from mjlab.utils.spec_config import MaterialCfg, TextureCfg

from instinct_mj.tasks.parkour.config.terrains.presets import (  # noqa: F401
    ADAM_LITE_BASE_PARKOUR_TERRAINS_CFG,
    ADAM_LITE_BASE_PARKOUR_TERRAINS_CFG_PLAY,
    ADAM_LITE_MIXED_PARKOUR_TERRAINS_CFG,
    ADAM_LITE_MIXED_PARKOUR_TERRAINS_CFG_PLAY,
    ADAM_LITE_PARKOUR_TERRAINS_CFG,
    BRIDGE_L_CORNER_TERRAINS_CFG,
    BRIDGE_L_CORNER_TERRAINS_CFG_PLAY,
    CHAINED_STRAIGHT_BRIDGE_TERRAIN_CFG,
    CHAINED_STRAIGHT_BRIDGE_TERRAIN_CFG_PLAY,
    CHAINED_STRAIGHT_BRIDGE_TERRAINS_CFG,
    CHAINED_STRAIGHT_BRIDGE_TERRAINS_CFG_PLAY,
    CONTINUOUS_HURDLE_TERRAIN_CFG,
    CONTINUOUS_HURDLE_TERRAIN_CFG_PLAY,
    CONTINUOUS_SLOPE_TERRAIN_CFG,
    CONTINUOUS_SLOPE_TERRAIN_CFG_PLAY,
    CROSS_SLOPE_TERRAIN_CFG,
    CROSS_SLOPE_TERRAIN_CFG_PLAY,
    CURVED_BRIDGE_TERRAIN_CFG,
    CURVED_BRIDGE_TERRAIN_CFG_PLAY,
    HURDLE_TERRAINS_CFG,
    HURDLE_TERRAINS_CFG_PLAY,
    MIXED_HURDLE_TERRAINS_CFG,
    MIXED_HURDLE_TERRAINS_CFG_PLAY,
    NARROW_L_CORNER_TERRAIN_CFG,
    NARROW_L_CORNER_TERRAIN_CFG_PLAY,
    NARROW_L_CORNER_RIGHT_TERRAIN_CFG,
    NARROW_L_CORNER_RIGHT_TERRAIN_CFG_PLAY,
    PYRAMID_STAIRS_TERRAIN_CFG,
    PYRAMID_STAIRS_TERRAIN_CFG_PLAY,
    PYRAMID_STAIRS_INV_TERRAIN_CFG,
    PYRAMID_STAIRS_INV_TERRAIN_CFG_PLAY,
    NOISE_ONLY_TERRAINS_CFG,
    NOISE_ONLY_TERRAINS_CFG_PLAY,
    PARKOUR_TERRAIN_CATALOG_CFG,
    PARKOUR_TERRAIN_CATALOG_CFG_PLAY,
    ROUGH_TERRAINS_CFG,
    ROUGH_TERRAINS_CFG_PLAY,
    SLALOM_POLE_TERRAIN_CFG,
    SLALOM_POLE_TERRAIN_CFG_PLAY,
    STAIRS_ONLY_TERRAINS_CFG,
    STAIRS_ONLY_TERRAINS_CFG_PLAY,
    SYMMETRICAL_SLOPE_TERRAIN_CFG,
    SYMMETRICAL_SLOPE_TERRAIN_CFG_PLAY,
    THREE_SLOPES_TERRAINS_CFG,
    THREE_SLOPES_TERRAINS_CFG_PLAY,
)


def _edit_parkour_scene_spec(spec: mujoco.MjSpec) -> None:
    """Apply matte parkour scene visual style directly on the MuJoCo spec."""
    ground_texture_name = "parkour_groundplane"
    ground_material_name = "parkour_groundplane"

    existing_skybox = None
    for tex in spec.textures:
        if tex.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            existing_skybox = tex
            break

    if existing_skybox is not None:
        existing_skybox.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
        existing_skybox.rgb1[:] = (0.95, 0.97, 0.99)
        existing_skybox.rgb2[:] = (0.83, 0.89, 0.95)
        existing_skybox.width = 512
        existing_skybox.height = 3072
    else:
        TextureCfg(
            name="parkour_skybox",
            type="skybox",
            builtin="gradient",
            rgb1=(0.95, 0.97, 0.99),
            rgb2=(0.83, 0.89, 0.95),
            width=512,
            height=3072,
        ).edit_spec(spec)

    TextureCfg(
        name=ground_texture_name,
        type="2d",
        builtin="checker",
        mark="edge",
        rgb1=(0.72, 0.72, 0.72),
        rgb2=(0.62, 0.62, 0.62),
        markrgb=(0.50, 0.50, 0.50),
        width=300,
        height=300,
    ).edit_spec(spec)
    MaterialCfg(
        name=ground_material_name,
        texuniform=True,
        texrepeat=(4, 4),
        reflectance=0.0,
        texture=ground_texture_name,
    ).edit_spec(spec)

    spec.visual.rgba.haze[:] = (0.95, 0.97, 0.99, 1.0)
    spec.visual.headlight.ambient[:] = (0.24, 0.24, 0.24)
    spec.visual.headlight.diffuse[:] = (0.34, 0.34, 0.34)
    spec.visual.headlight.specular[:] = (0.0, 0.0, 0.0)

    terrain_body = spec.body("terrain")

    for geom in terrain_body.geoms:
        geom.material = ground_material_name
        geom.rgba[:] = (0.72, 0.72, 0.72, 1.0)

    for light in terrain_body.lights:
        light.castshadow = False
        light.ambient[:] = (0.08, 0.08, 0.08)
        light.diffuse[:] = (0.16, 0.16, 0.16)
        light.specular[:] = (0.0, 0.0, 0.0)
