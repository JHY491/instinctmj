from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

_command_path = Path(__file__).resolve().parents[1] / "src/instinct_mj/tasks/parkour/mdp/commands/pose_velocity_command.py"
_command_spec = importlib.util.spec_from_file_location("pose_velocity_command_test", _command_path)
assert _command_spec is not None and _command_spec.loader is not None
_command_mod = importlib.util.module_from_spec(_command_spec)
_command_spec.loader.exec_module(_command_mod)
_cref_all_tread_marker_centers = _command_mod._cref_all_tread_marker_centers


def test_cref_all_tread_marker_centers_cover_every_valid_cell() -> None:
    origins = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [0.0, 8.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    treads_xz = torch.zeros((1, 2, 3, 2), dtype=torch.float32)
    treads_xz[0, 0, :2] = torch.tensor([[-1.0, 0.0], [0.0, 0.1]], dtype=torch.float32)
    treads_xz[0, 1, :3] = torch.tensor([[-1.0, 0.0], [0.0, 0.1], [1.0, 0.0]], dtype=torch.float32)
    tread_counts = torch.tensor([[2, 3]], dtype=torch.long)

    centers = _cref_all_tread_marker_centers(origins, treads_xz, tread_counts)

    assert len(centers) == 5
    assert {label for label, _ in centers} == {
        "cref_tread_0_0_0",
        "cref_tread_0_0_1",
        "cref_tread_0_1_0",
        "cref_tread_0_1_1",
        "cref_tread_0_1_2",
    }
    assert any(center[1] == 8.0 for _, center in centers)
    assert all(center[2] >= 0.15 for _, center in centers)
