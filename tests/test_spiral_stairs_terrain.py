from __future__ import annotations

import mujoco
import numpy as np

from instinct_mj.terrains.height_field.hf_terrains_cfg import PerlinSpiralStairsTerrainCfg


def _cfg(*, flight_angle_range: tuple[float, float] | float = (0.0, 30.0)) -> PerlinSpiralStairsTerrainCfg:
    return PerlinSpiralStairsTerrainCfg(
        size=(4.0, 6.0),
        horizontal_scale=0.05,
        vertical_scale=0.005,
        obstacle_length=5.04,
        obstacle_width=3.0,
        step_height=0.15,
        num_steps=6,
        platform_length=1.44,
        stair_width=1.2,
        flight_angle_range=flight_angle_range,
        edge_resolution=0.02,
        perlin_cfg=None,
    )


def _spiral_layout(cfg: PerlinSpiralStairsTerrainCfg, height_field: np.ndarray) -> dict[str, int]:
    obstacle_width_px = int(round(cfg.obstacle_width / cfg.horizontal_scale))
    obstacle_length_px = int(round(cfg.obstacle_length / cfg.horizontal_scale))
    stair_width_px = int(round(cfg.stair_width / cfg.horizontal_scale))
    platform_length_px = int(round(cfg.platform_length / cfg.horizontal_scale))

    x0 = (height_field.shape[0] - obstacle_width_px) // 2
    x1 = x0 + obstacle_width_px
    x_mid = x0 + obstacle_width_px // 2
    y0 = (height_field.shape[1] - obstacle_length_px) // 2
    y1 = y0 + obstacle_length_px
    platform_start_y = y0 + (obstacle_length_px - platform_length_px) // 2
    platform_end_y = platform_start_y + platform_length_px
    entrance_margin = max(0, (x_mid - x0 - stair_width_px) // 2)
    entrance_lane_x0 = x0 + entrance_margin
    entrance_lane_x1 = entrance_lane_x0 + stair_width_px
    exit_margin = max(0, (x1 - x_mid - stair_width_px) // 2)
    exit_lane_x0 = x_mid + exit_margin
    exit_lane_x1 = exit_lane_x0 + stair_width_px

    return {
        "x0": x0,
        "x1": x1,
        "x_mid": x_mid,
        "y0": y0,
        "y1": y1,
        "platform_start_y": platform_start_y,
        "platform_end_y": platform_end_y,
        "entrance_lane_x0": entrance_lane_x0,
        "entrance_lane_x1": entrance_lane_x1,
        "exit_lane_x0": exit_lane_x0,
        "exit_lane_x1": exit_lane_x1,
    }


def test_spiral_stairs_height_field_keeps_image_dimensions_and_step_height() -> None:
    cfg = _cfg(flight_angle_range=(0.0, 0.0))
    height_field = cfg._generate_height_field(difficulty=1.0, cfg_for_gen=cfg)

    assert height_field.shape == (80, 120)
    assert int(np.max(height_field)) == int(round((6 * 0.15) / 0.005))

    obstacle_rows = np.argwhere(height_field > 0)[:, 0]
    obstacle_cols = np.argwhere(height_field > 0)[:, 1]
    obstacle_width = (obstacle_rows.max() - obstacle_rows.min() + 1) * cfg.horizontal_scale
    obstacle_length = (obstacle_cols.max() - obstacle_cols.min() + 1) * cfg.horizontal_scale

    assert np.isclose(obstacle_length, 5.05)
    assert np.isclose(obstacle_width, 3.0)


def test_spiral_stairs_height_field_has_walls_platform_and_edge_start() -> None:
    cfg = _cfg(flight_angle_range=(0.0, 0.0))
    height_field = cfg._generate_height_field(difficulty=1.0, cfg_for_gen=cfg)
    step_height_i = int(round(0.15 / cfg.vertical_scale))
    max_height = cfg.num_steps * step_height_i
    layout = _spiral_layout(cfg, height_field)

    first_step = height_field[layout["entrance_lane_x0"] : layout["entrance_lane_x1"], layout["y0"]]
    assert first_step.min() == step_height_i
    assert first_step.max() == step_height_i

    entrance_left_wall = height_field[layout["x0"] : layout["entrance_lane_x0"], layout["y0"] : layout["platform_start_y"]]
    entrance_right_wall = height_field[
        layout["entrance_lane_x1"] : layout["x_mid"], layout["y0"] : layout["platform_start_y"]
    ]
    for wall in (entrance_left_wall, entrance_right_wall):
        assert wall.size > 0
        assert wall.min() == max_height

    platform = height_field[layout["x0"] : layout["x1"], layout["platform_start_y"] : layout["platform_end_y"]]
    assert np.isclose(platform.shape[1] * cfg.horizontal_scale, cfg.platform_length, atol=cfg.horizontal_scale)
    assert platform.min() == max_height


def test_spiral_stairs_height_field_rotates_edges_in_xy_plane() -> None:
    cfg = _cfg(flight_angle_range=(0.0, 30.0))
    straight_height_field = cfg._generate_height_field(difficulty=0.0, cfg_for_gen=cfg)
    angled_height_field = cfg._generate_height_field(difficulty=1.0, cfg_for_gen=cfg)
    layout = _spiral_layout(cfg, angled_height_field)
    step_height_i = int(round(0.15 / cfg.vertical_scale))

    positive_heights = angled_height_field[(angled_height_field > 0) & (angled_height_field < cfg.num_steps * step_height_i)]
    assert np.all(positive_heights % step_height_i == 0)

    first_step = angled_height_field[layout["entrance_lane_x0"] : layout["entrance_lane_x1"], layout["y0"]]
    assert first_step.min() == step_height_i
    assert first_step.max() == step_height_i

    def first_step_end_offsets(height_field: np.ndarray) -> np.ndarray:
        lane = height_field[
            layout["entrance_lane_x0"] : layout["entrance_lane_x1"], layout["y0"] : layout["platform_start_y"]
        ]
        offsets = []
        for row in lane:
            matches = np.flatnonzero(row == step_height_i)
            assert matches.size > 0
            offsets.append(int(matches[-1]))
        return np.asarray(offsets)

    straight_offsets = first_step_end_offsets(straight_height_field)
    angled_offsets = first_step_end_offsets(angled_height_field)

    assert int(straight_offsets.max() - straight_offsets.min()) == 0
    assert int(angled_offsets.max() - angled_offsets.min()) > 0
    assert int(angled_offsets[0]) > int(angled_offsets[-1])


def test_spiral_stairs_height_field_rotates_side_wall_edges_with_fixed_stair_width() -> None:
    cfg = _cfg(flight_angle_range=(0.0, 30.0))
    straight_height_field = cfg._generate_height_field(difficulty=0.0, cfg_for_gen=cfg)
    angled_height_field = cfg._generate_height_field(difficulty=1.0, cfg_for_gen=cfg)
    layout = _spiral_layout(cfg, angled_height_field)
    step_height_i = int(round(0.15 / cfg.vertical_scale))
    max_height = cfg.num_steps * step_height_i

    def wall_front_offsets(height_field: np.ndarray) -> tuple[int, int]:
        left_wall = height_field[
            layout["x0"] : layout["entrance_lane_x0"], layout["y0"] : layout["platform_start_y"]
        ]
        right_wall = height_field[
            layout["entrance_lane_x1"] : layout["x_mid"], layout["y0"] : layout["platform_start_y"]
        ]
        assert left_wall.size > 0
        assert right_wall.size > 0
        left_matches = np.flatnonzero(left_wall.max(axis=0) == max_height)
        right_matches = np.flatnonzero(right_wall.max(axis=0) == max_height)
        assert left_matches.size > 0
        assert right_matches.size > 0
        return int(left_matches[0]), int(right_matches[0])

    straight_left_offset, straight_right_offset = wall_front_offsets(straight_height_field)
    angled_left_offset, angled_right_offset = wall_front_offsets(angled_height_field)

    assert straight_left_offset == straight_right_offset
    assert angled_left_offset > angled_right_offset

    first_step = angled_height_field[layout["entrance_lane_x0"] : layout["entrance_lane_x1"], layout["y0"]]
    assert first_step.shape[0] == int(round(cfg.stair_width / cfg.horizontal_scale))
    assert first_step.min() == step_height_i
    assert first_step.max() == step_height_i


def test_spiral_stairs_function_uses_hfield_at_two_centimeter_resolution() -> None:
    cfg = _cfg()
    spec = mujoco.MjSpec()  # pyright: ignore[reportAttributeAccessIssue]
    spec.worldbody.add_body(name="terrain")

    output = cfg.function(1.0, spec, np.random.default_rng(0))
    hfield = output.geometries[0].hfield

    assert len(spec.meshes) == 0
    assert len(spec.hfields) == 1
    assert hfield is not None
    assert hfield.nrow == int(cfg.size[0] / cfg.edge_resolution)
    assert hfield.ncol == int(cfg.size[1] / cfg.edge_resolution)
