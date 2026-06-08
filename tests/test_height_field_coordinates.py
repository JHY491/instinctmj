from __future__ import annotations

import numpy as np

from instinct_mj.terrains.height_field.utils import convert_height_field_to_mesh


def test_convert_height_field_to_mesh_maps_rows_to_y_and_cols_to_x() -> None:
    height_field = np.zeros((3, 4), dtype=np.int16)
    height_field[2, 3] = 7

    vertices, _ = convert_height_field_to_mesh(
        height_field,
        horizontal_scale=0.5,
        vertical_scale=0.1,
    )

    vertex = vertices[2 * 4 + 3]
    assert np.allclose(vertex, np.array([1.5, 1.0, 0.7], dtype=np.float32))


def test_convert_height_field_to_mesh_projects_row_cliffs_along_y() -> None:
    height_field = np.array(
        [
            [0, 0, 0],
            [10, 10, 10],
        ],
        dtype=np.int16,
    )

    vertices, _ = convert_height_field_to_mesh(
        height_field,
        horizontal_scale=1.0,
        vertical_scale=1.0,
        slope_threshold=1.0,
    )

    assert np.allclose(vertices[2], np.array([2.0, 1.0, 0.0], dtype=np.float32))
