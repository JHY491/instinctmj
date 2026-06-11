from __future__ import annotations

from instinct_mj.tasks.parkour.config.terrains.presets import HURDLE_TERRAINS_CFG, MIXED_HURDLE_TERRAINS_CFG
from instinct_mj.terrains.height_field.hf_terrains_cfg import (
    _subterrain_reset_root_xy_offset,
    terrain_column_reset_root_xy_offset,
)


def test_mixed_hurdle_subterrain_proportions() -> None:
    subs = MIXED_HURDLE_TERRAINS_CFG.sub_terrains
    assert subs["perlin_rough_stand"].proportion == 0.1
    assert subs["perlin_rough"].proportion == 0.1
    assert subs["continuous_hurdle"].proportion == 0.8


def test_mixed_hurdle_spawn_offsets() -> None:
    subs = MIXED_HURDLE_TERRAINS_CFG.sub_terrains
    assert _subterrain_reset_root_xy_offset("perlin_rough_stand", subs["perlin_rough_stand"]) == (0.0, 0.0)
    assert _subterrain_reset_root_xy_offset("perlin_rough", subs["perlin_rough"]) == (0.0, 0.0)
    assert _subterrain_reset_root_xy_offset("continuous_hurdle", subs["continuous_hurdle"]) == (-4.0, 0.0)


def test_mixed_hurdle_column_offsets() -> None:
    offsets = terrain_column_reset_root_xy_offset(MIXED_HURDLE_TERRAINS_CFG)
    assert len(offsets) == MIXED_HURDLE_TERRAINS_CFG.num_cols
    assert offsets.count((0.0, 0.0)) == 4
    assert offsets.count((-4.0, 0.0)) == 16


def test_mixed_hurdle_geometry() -> None:
    hurdle = MIXED_HURDLE_TERRAINS_CFG.sub_terrains["continuous_hurdle"]
    assert hurdle.hurdle_thickness == 0.05
    assert hurdle.hurdle_height_range == (0.05, 0.30)
    assert hurdle.virtual_obstacle_keys == ("hurdle_edges",)


def test_hurdle_only_subterrain_geometry() -> None:
    subs = HURDLE_TERRAINS_CFG.sub_terrains
    hurdle = subs["continuous_hurdle"]
    assert hurdle.proportion == 1.0
    assert _subterrain_reset_root_xy_offset("continuous_hurdle", hurdle) == (-4.0, 0.0)
    assert hurdle.hurdle_thickness == 0.05
    assert hurdle.hurdle_height_range == (0.05, 0.30)
    assert hurdle.hurdle_spacing_range == (1.5, 1.0)
    assert hurdle.virtual_obstacle_keys == ("hurdle_edges",)
