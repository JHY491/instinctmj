from __future__ import annotations

import mujoco
import numpy as np

from instinct_mj.tasks.parkour.config.parkour_env_cfg import PARKOUR_TERRAIN_CATALOG_CFG
from instinct_mj.tasks.parkour.config.terrains.presets import STAIRS_ONLY_TERRAINS_CFG
from instinct_mj.terrains.height_field.hf_terrains_cfg import (
    FixedSequenceFlatPatchSamplingCfg,
    PerlinOneSidedStairsTerrainCfg,
)


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


def test_one_sided_up_down_stairs_accepts_platform_width_range() -> None:
    cfg = PerlinOneSidedStairsTerrainCfg(
        size=(4.0, 6.0),
        horizontal_scale=0.1,
        vertical_scale=0.01,
        step_height_range=(0.1, 0.1),
        step_width=0.5,
        platform_width=(1.0, 1.0),
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


def test_stairs_only_one_sided_stairs_uses_fixed_sequence_target_patches() -> None:
    sub = STAIRS_ONLY_TERRAINS_CFG.sub_terrains["one_sided_stairs"]
    assert isinstance(sub, PerlinOneSidedStairsTerrainCfg)
    assert sub.platform_width == (1.0, 1.5)
    tgt = sub.flat_patch_sampling["target"]
    assert isinstance(tgt, FixedSequenceFlatPatchSamplingCfg)
    assert tgt.num_patches == 2
    assert tgt.patch_radius == [0.10]
    assert tgt.max_height_diff == 0.05
    assert tgt.fixed_xy_points == ((0.0, 0.0), (3.7, 0.0))
    assert tgt.fixed_y_offset_range == (-0.4, 0.4)
    assert tgt.reset_command_patch_index == 0
    assert sub.reset_root_xy_offset == (-3.6, 0.0)


def test_stairs_only_one_sided_stairs_fixed_sequence_target_patches_on_terrain() -> None:
    cfg = STAIRS_ONLY_TERRAINS_CFG.sub_terrains["one_sided_stairs"]
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    output = cfg.function(difficulty=0.0, spec=spec, rng=np.random.default_rng(0))
    assert output.flat_patches is not None
    spawn = output.flat_patches["spawn"]
    target = output.flat_patches["target"]
    assert spawn.shape == (50, 3)
    assert target.shape == (2, 3)
    assert np.all(np.abs(spawn[:, 2]) <= 0.05)
    assert np.all(spawn[:, 0] < target[0, 0])
    assert target[1, 0] > target[0, 0]
    assert np.isclose(target[0, 1], target[1, 1])
    y_center = 0.5 * float(cfg.size[1])
    assert y_center - 0.4 <= target[0, 1] <= y_center + 0.4
    assert target[0, 2] >= target[1, 2]
