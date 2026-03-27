from __future__ import annotations

from mjlab.scene import Scene as SceneBase


class Scene(SceneBase):
    """Instinct scene extension that honors terrain cfg.class_type."""

    def _add_terrain(self) -> None:
        terrain_cfg = self._cfg.terrain
        terrain_cfg.num_envs = self._cfg.num_envs
        terrain_cfg.env_spacing = self._cfg.env_spacing
        terrain_cls = terrain_cfg.class_type
        terrain = terrain_cls(terrain_cfg, device=self._device)
        self._terrain = terrain
        self._entities["terrain"] = terrain
        frame = self._spec.worldbody.add_frame()
        self._spec.attach(terrain.spec, prefix="", frame=frame)
