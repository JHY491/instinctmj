from __future__ import annotations

from instinct_mj.tasks.parkour.config.terrains.presets import (
    ADAM_LITE_BASE_PARKOUR_TERRAINS_CFG,
    ADAM_LITE_MIXED_PARKOUR_TERRAINS_CFG,
    PARKOUR_TERRAIN_CATALOG_CFG,
    STAIRS_ONLY_TERRAINS_CFG,
)
from instinct_mj.terrains.height_field.hf_terrains_cfg import (
    _subterrain_reset_root_xy_offset,
    terrain_column_reset_root_xy_offset,
)


def test_perlin_spawns_center_one_sided_stairs_spawn_edge() -> None:
    base = ADAM_LITE_BASE_PARKOUR_TERRAINS_CFG.sub_terrains
    assert _subterrain_reset_root_xy_offset("perlin_rough", base["perlin_rough"]) == (0.0, 0.0)
    assert _subterrain_reset_root_xy_offset("perlin_rough_stand", base["perlin_rough_stand"]) == (0.0, 0.0)
    assert _subterrain_reset_root_xy_offset("one_sided_stairs", PARKOUR_TERRAIN_CATALOG_CFG.sub_terrains["one_sided_stairs"]) == (
        -4.0,
        0.0,
    )


def test_fixed_sequence_stairs_spawn_at_explicit_ground_patch() -> None:
    sub = STAIRS_ONLY_TERRAINS_CFG.sub_terrains["one_sided_stairs"]
    assert _subterrain_reset_root_xy_offset("one_sided_stairs", sub) == (-3.6, 0.0)


def test_mixed_parkour_columns_include_center_and_edge_offsets() -> None:
    offsets = terrain_column_reset_root_xy_offset(ADAM_LITE_MIXED_PARKOUR_TERRAINS_CFG)
    assert len(offsets) == ADAM_LITE_MIXED_PARKOUR_TERRAINS_CFG.num_cols
    assert (0.0, 0.0) in offsets
    assert (-4.0, 0.0) in offsets
