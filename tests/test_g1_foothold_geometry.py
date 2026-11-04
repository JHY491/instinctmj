"""Regressions for G1 sole coordinates and shallow-gap support detection."""

from __future__ import annotations

from types import SimpleNamespace

import mujoco
import torch

from instinct_mj.assets.unitree_g1 import get_g1_spec
from instinct_mj.tasks.parkour.config.g1.agents.instinct_rl_amp_cfg import (
    g1_plum_blossom_foothold_amp_runner_cfg,
)
from instinct_mj.tasks.parkour.config.g1.g1_plum_blossom_foothold_parkour_cfg import (
    instinct_g1_plum_blossom_foothold_env_cfg,
)
from instinct_rl.algorithms.foothold_math import support_deficiency_at_center


def _sole_observations():
    cfg = instinct_g1_plum_blossom_foothold_env_cfg()
    model = get_g1_spec().compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # Put the nominal, horizontal sole plane at ground level.
    data.qpos[2] -= data.xpos[model.body("LL_FOOT").id, 2]
    mujoco.mj_forward(model, data)
    robot = SimpleNamespace(data=SimpleNamespace(
        body_link_pos_w=torch.tensor(data.xpos, dtype=torch.float32).unsqueeze(0),
        root_link_pos_w=torch.tensor(data.xpos[1], dtype=torch.float32).unsqueeze(0),
        root_link_quat_w=torch.tensor(data.xquat[1], dtype=torch.float32).unsqueeze(0),
    ))
    env = SimpleNamespace(scene={"robot": robot})
    terms = cfg.observations["foothold"].terms
    for name in ("foot_pos_b", "foot_pos_w", "foot_sole_z"):
        asset = terms[name].params["asset_cfg"]
        asset.body_ids = [model.body(body).id for body in asset.body_names]
    return terms, env, model, data


def test_g1_imagination_and_touchdown_use_sole_center() -> None:
    terms, env, model, data = _sole_observations()
    pos = terms["foot_pos_w"].func(env, **terms["foot_pos_w"].params).view(2, 2)
    expected_xy = torch.tensor([
        data.site_xpos[model.site(f"{side}_foot_scan").id, :2].tolist()
        for side in ("left", "right")
    ])
    assert torch.allclose(pos, expected_xy, atol=1e-6)
    pos_b = terms["foot_pos_b"].func(env, **terms["foot_pos_b"].params).view(2, 2)
    assert torch.allclose(pos_b, expected_xy, atol=1e-6)
    sole_z = terms["foot_sole_z"].func(env, **terms["foot_sole_z"].params)
    assert torch.allclose(sole_z, torch.zeros(1, 2), atol=1e-6)


def test_g1_shallow_gap_edge_is_not_marked_fully_supported() -> None:
    terms, env, _, _ = _sole_observations()
    sole_z = terms["foot_sole_z"].func(env, **terms["foot_sole_z"].params)[:, 0]
    algo = g1_plum_blossom_foothold_amp_runner_cfg().algorithm
    height, width = algo.region_grid
    scan = torch.zeros(1, height, width)
    scan[:, :, width // 2 + 1:] = -0.05  # easy-level pit next to a stake
    rho = support_deficiency_at_center(
        scan.flatten(1), sole_z, algo.region_grid, algo.region_res, algo.sole_grid, algo.epsilon_h,
    )
    assert rho.item() > 0.3
    # Reproduce the old Adam-to-G1 offset error: a 5 cm hole disappears within
    # the 3 cm support tolerance when the inferred sole is 2.8 cm too low.
    old_rho = support_deficiency_at_center(
        scan.flatten(1), sole_z - 0.028, algo.region_grid, algo.region_res, algo.sole_grid, algo.epsilon_h,
    )
    assert old_rho.item() == 0.0
