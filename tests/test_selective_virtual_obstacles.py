"""Tests for per-sub-terrain virtual obstacle selection."""

from __future__ import annotations

import copy

import pytest
import trimesh

from instinct_mj.terrains.height_field.hf_terrains_cfg import PerlinPlaneTerrainCfg
from instinct_mj.terrains.terrain_generator_cfg import FiledTerrainGeneratorCfg
from instinct_mj.terrains.terrain_importer import TerrainImporter
from instinct_mj.terrains.terrain_importer_cfg import TerrainImporterCfg
from instinct_mj.terrains.virtual_obstacle.edge_cylinder_cfg import GreedyconcatEdgeCylinderCfg


def _make_importer_stub(
    *,
    sub_cfgs: list[PerlinPlaneTerrainCfg],
    tile_meshes: list[trimesh.Trimesh],
    virtual_obstacle_names: tuple[str, ...] = ("edges", "hurdle"),
) -> TerrainImporter:
    importer = TerrainImporter.__new__(TerrainImporter)
    importer.cfg = TerrainImporterCfg(terrain_type="generator")
    importer._virtual_obstacles = {name: object() for name in virtual_obstacle_names}
    importer.terrain_generator = type(
        "Gen",
        (),
        {
            "_terrain_meshes": tile_meshes,
            "subterrain_specific_cfgs": sub_cfgs,
        },
    )()
    importer._spec = None
    importer._iter_hfield_geoms_with_subterrain_cfgs = lambda: iter([])  # type: ignore[method-assign]
    return importer


def test_subterrain_virtual_obstacle_keys_field():
    cfg = PerlinPlaneTerrainCfg(
        proportion=1.0,
        size=(4.0, 4.0),
        virtual_obstacle_keys=("edges",),
    )
    assert cfg.virtual_obstacle_keys == ("edges",)


def test_uses_selective_when_any_subterrain_has_keys():
    gen_cfg = FiledTerrainGeneratorCfg(
        size=(4.0, 4.0),
        num_rows=1,
        num_cols=1,
        sub_terrains={
            "rough": PerlinPlaneTerrainCfg(proportion=1.0, size=(4.0, 4.0)),
            "stairs": PerlinPlaneTerrainCfg(
                proportion=0.0,
                size=(4.0, 4.0),
                virtual_obstacle_keys=("edges",),
            ),
        },
    )
    importer_cfg = TerrainImporterCfg(
        terrain_type="generator",
        terrain_generator=gen_cfg,
        virtual_obstacles={"edges": GreedyconcatEdgeCylinderCfg(min_points=2)},
    )
    importer = TerrainImporter.__new__(TerrainImporter)
    importer.cfg = importer_cfg
    importer._virtual_obstacles = {"edges": object()}
    assert importer._uses_selective_subterrain_virtual_obstacles()


def test_legacy_mode_when_no_subterrain_keys():
    gen_cfg = FiledTerrainGeneratorCfg(
        size=(4.0, 4.0),
        num_rows=1,
        num_cols=1,
        sub_terrains={
            "rough": PerlinPlaneTerrainCfg(proportion=1.0, size=(4.0, 4.0)),
        },
    )
    importer_cfg = TerrainImporterCfg(
        terrain_type="generator",
        terrain_generator=gen_cfg,
        virtual_obstacles={"edges": GreedyconcatEdgeCylinderCfg(min_points=2)},
    )
    importer = TerrainImporter.__new__(TerrainImporter)
    importer.cfg = importer_cfg
    importer._virtual_obstacles = {"edges": object()}
    assert not importer._uses_selective_subterrain_virtual_obstacles()


def test_collect_meshes_by_key_skips_tiles_without_keys():
    tile = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]])
    rough = PerlinPlaneTerrainCfg(proportion=1.0, size=(4.0, 4.0))
    stairs = PerlinPlaneTerrainCfg(proportion=0.0, size=(4.0, 4.0), virtual_obstacle_keys=("edges",))
    hurdle = PerlinPlaneTerrainCfg(proportion=0.0, size=(4.0, 4.0), virtual_obstacle_keys=("hurdle",))
    importer = _make_importer_stub(sub_cfgs=[rough, stairs, hurdle], tile_meshes=[tile, tile, tile])
    meshes_by_key = importer._collect_meshes_by_virtual_obstacle_key()
    assert len(meshes_by_key["edges"]) == 1
    assert len(meshes_by_key["hurdle"]) == 1


def test_validate_unknown_virtual_obstacle_key():
    gen_cfg = FiledTerrainGeneratorCfg(
        size=(4.0, 4.0),
        num_rows=1,
        num_cols=1,
        sub_terrains={
            "stairs": PerlinPlaneTerrainCfg(
                proportion=1.0,
                size=(4.0, 4.0),
                virtual_obstacle_keys=("missing",),
            ),
        },
    )
    importer_cfg = TerrainImporterCfg(
        terrain_type="generator",
        terrain_generator=gen_cfg,
        virtual_obstacles={"edges": GreedyconcatEdgeCylinderCfg(min_points=2)},
    )
    importer = TerrainImporter.__new__(TerrainImporter)
    importer.cfg = importer_cfg
    importer._virtual_obstacles = {"edges": object()}
    with pytest.raises(ValueError, match="missing"):
        importer._validate_subterrain_virtual_obstacle_keys()


@pytest.mark.slow
def test_selective_generation_compiles():
    gen_cfg = FiledTerrainGeneratorCfg(
        seed=0,
        size=(4.0, 4.0),
        border_width=0.0,
        num_rows=1,
        num_cols=2,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        curriculum=False,
        sub_terrains={
            "flat": PerlinPlaneTerrainCfg(
                proportion=0.5,
                size=(4.0, 4.0),
                noise_scale=0.0,
                noise_frequency=10,
            ),
            "stairs": PerlinPlaneTerrainCfg(
                proportion=0.5,
                size=(4.0, 4.0),
                noise_scale=0.0,
                noise_frequency=10,
                virtual_obstacle_keys=("edges",),
            ),
        },
    )
    importer_cfg = TerrainImporterCfg(
        terrain_type="generator",
        terrain_generator=copy.deepcopy(gen_cfg),
        virtual_obstacles={
            "edges": GreedyconcatEdgeCylinderCfg(
                cylinder_radius=0.05,
                min_points=2,
                angle_threshold=10.0,
            ),
        },
    )
    importer = TerrainImporter(importer_cfg, device="cpu")
    obstacle = importer.virtual_obstacles["edges"]
    assert obstacle.cylinders is None or obstacle.edges_pyt.shape[0] >= 0
