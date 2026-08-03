"""Play-time debug drawing for SSR imagined foothold (mu + sole-grid patch)."""

from __future__ import annotations

from typing import Any

import torch

from mjlab.viewer.debug_visualizer import DebugVisualizer

_MU_RGBA = (1.0, 0.05, 0.05, 1.0)  # center mu: red
_SOLE_RGBA = (0.1, 0.35, 1.0, 0.95)  # sole_grid cells around mu: blue


class FootholdImaginationDebugVis:
    """Registers on ``env.manager_visualizers`` and draws via ``DebugVisualizer``.

    - Red sphere: imagined foothold mean ``mu`` (swing foot only)
    - Blue spheres: ``sole_grid`` cells centered at ``mu`` (swing foot only)
    """

    def __init__(
        self,
        alg: Any,
        vec_env: Any,
        *,
        mean_radius: float = 0.018,
        sole_radius: float = 0.008,
        contact_threshold: float = 0.5,
        num_sample_nodes: int | None = None,
        draw_all_envs: bool = True,
    ) -> None:
        self.alg = alg
        self.vec_env = vec_env
        self.mean_radius = mean_radius
        self.sole_radius = sole_radius
        self.contact_threshold = contact_threshold
        # Kept for call-site compatibility; sole-patch drawing ignores GH nodes.
        self.num_sample_nodes = num_sample_nodes
        # True: always draw every env's foothold; False: follow viewer env selection / A-key.
        self.draw_all_envs = draw_all_envs
        self.enabled = True
        self._last_obs_pack: dict[str, torch.Tensor] | None = None

    def cache_obs_pack(self, obs_pack: dict[str, torch.Tensor] | None) -> None:
        self._last_obs_pack = obs_pack

    def debug_vis(self, visualizer: DebugVisualizer) -> None:
        if not self.enabled:
            return
        if not hasattr(self.alg, "compute_foothold_imagination_visualization"):
            return

        obs_pack = self._last_obs_pack
        if obs_pack is None or "foothold" not in obs_pack:
            try:
                _, extras = self.vec_env.get_observations()
                obs_pack = extras.get("observations")
            except Exception:
                return
        if obs_pack is None or "foothold" not in obs_pack:
            return

        vis = self.alg.compute_foothold_imagination_visualization(
            obs_pack, num_sample_nodes=self.num_sample_nodes
        )
        points_w = vis["points_w"]  # (N, 2, 3)
        sole_patch_w = vis["sole_patch_w"]  # (N, 2, P, 3)
        foot_contact = vis["foot_contact"]  # (N, 2)
        num_envs = points_w.shape[0]
        if self.draw_all_envs:
            env_ids = list(range(num_envs))
        else:
            env_ids = list(visualizer.get_env_indices(num_envs))
        if not env_ids:
            return

        for env_idx in env_ids:
            # Prefer a single swing foot. If both swing, take the first; if none, skip.
            swing_feet = [
                foot
                for foot in range(points_w.shape[1])
                if float(foot_contact[env_idx, foot]) < self.contact_threshold
            ]
            if not swing_feet:
                continue
            foot = swing_feet[0]

            for pi in range(sole_patch_w.shape[2]):
                visualizer.add_sphere(
                    center=sole_patch_w[env_idx, foot, pi],
                    radius=self.sole_radius,
                    color=_SOLE_RGBA,
                )
            visualizer.add_sphere(
                center=points_w[env_idx, foot],
                radius=self.mean_radius,
                color=_MU_RGBA,
            )
