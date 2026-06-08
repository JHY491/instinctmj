from __future__ import annotations

from mjlab.terrains import FlatPatchSamplingCfg

from instinct_mj.tasks.parkour.config.terrains.presets import ADAM_LITE_BASE_PARKOUR_TERRAINS_CFG
from instinct_mj.terrains.height_field.hf_terrains_cfg import (
    PerlinInvertedPyramidStairsTerrainCfg,
    PerlinPyramidStairsTerrainCfg,
)


def test_adam_lite_base_pyramid_stairs_flat_patch_on_center_platform() -> None:
    for key, cfg_type in (
        ("pyramid_stairs", PerlinPyramidStairsTerrainCfg),
        ("pyramid_stairs_inv", PerlinInvertedPyramidStairsTerrainCfg),
    ):
        sub = ADAM_LITE_BASE_PARKOUR_TERRAINS_CFG.sub_terrains[key]
        assert isinstance(sub, cfg_type)
        assert sub.step_height_range == (0.02, 0.15)
        assert sub.step_width == 0.3
        assert sub.platform_width == 2.5
        assert sub.virtual_obstacle_keys == ("stairs_edges",)
        tgt = sub.flat_patch_sampling["target"]
        assert isinstance(tgt, FlatPatchSamplingCfg)
        assert tgt.num_patches == 50
        assert tgt.x_range == (3.7, 3.7)
        assert tgt.y_range == (0.0, 0.0)
