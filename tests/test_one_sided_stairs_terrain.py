from __future__ import annotations

import numpy as np

from instinct_mj.tasks.parkour.config.parkour_env_cfg import PARKOUR_TERRAIN_CATALOG_CFG
from instinct_mj.terrains.height_field.hf_terrains_cfg import PerlinOneSidedStairsTerrainCfg


def test_one_sided_stairs_parkour_runs_along_world_x() -> None:
    cfg = PARKOUR_TERRAIN_CATALOG_CFG.sub_terrains["one_sided_stairs"]

    assert isinstance(cfg, PerlinOneSidedStairsTerrainCfg)
    assert cfg.stair_face == "neg_x"


def test_one_sided_up_down_stairs_are_center_high_along_world_x() -> None:
    cfg = PerlinOneSidedStairsTerrainCfg(
        size=(4.0, 6.0),
        horizontal_scale=0.1,
        vertical_scale=0.01,
        step_height_range=(0.1, 0.1),
        step_width=0.5,
        platform_width=1.0,
        lateral_width=None,
        stair_face="neg_x",
        up_down=True,
        perlin_cfg=None,
    )

    height_field = cfg._generate_height_field(difficulty=1.0, cfg_for_gen=cfg)

    assert height_field.shape == (60, 40)
    center_row = height_field.shape[0] // 2
    center_col = height_field.shape[1] // 2
    assert int(height_field[center_row, center_col]) > int(height_field[center_row, 0])
    assert int(height_field[center_row, center_col]) > int(height_field[center_row, -1])
    assert np.all(height_field[:, center_col] == height_field[center_row, center_col])
