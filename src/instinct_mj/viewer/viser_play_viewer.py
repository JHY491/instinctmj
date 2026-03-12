"""Viser play viewer that shows ray caster camera feeds (e.g. NoisyGroupedRayCasterCamera)."""

from __future__ import annotations

from typing import Any

import numpy as np
import viser
from mjlab.viewer.viser.viewer import ViserPlayViewer
from typing_extensions import override

from instinct_mj.sensors.grouped_ray_caster import GroupedRayCasterCamera


def _get_ray_caster_camera_sensors(env: Any) -> list[tuple[str, GroupedRayCasterCamera]]:
    """Return list of (name, sensor) for scene sensors that are GroupedRayCasterCamera."""
    out: list[tuple[str, GroupedRayCasterCamera]] = []
    scene = getattr(env, "scene", None)
    if scene is None:
        return out
    sensors = getattr(scene, "sensors", None)
    if sensors is None:
        return out
    for name, sensor in sensors.items():
        if isinstance(sensor, GroupedRayCasterCamera):
            out.append((name, sensor))
    return out


def _depth_tensor_to_display_image(tensor: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Convert depth tensor (H, W) or (H, W, 1) to uint8 (H, W, 3) for viser image."""
    if tensor.ndim == 3:
        tensor = tensor.squeeze(-1)
    valid = np.isfinite(tensor) & (tensor > 0)
    if not np.any(valid):
        out = np.zeros((*tensor.shape, 3), dtype=np.uint8)
        return out
    lo, hi = np.nanmin(tensor[valid]), np.nanmax(tensor[valid])
    if hi <= lo:
        hi = lo + 1.0
    normalized = np.clip((tensor - lo) / (hi - lo), 0.0, 1.0)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    if scale != 1.0:
        normalized = np.clip(normalized * scale, 0.0, 1.0)
    gray = (normalized * 255).astype(np.uint8)
    return np.repeat(gray[:, :, np.newaxis], 3, axis=-1)


class _RayCasterFeed:
    """One ray caster camera feed: output key, image handle, depth scale slider."""

    __slots__ = ("key", "handle", "depth_slider")

    def __init__(
        self,
        key: str,
        handle: viser.GuiImageHandle,
        depth_slider: viser.GuiSliderHandle | None,
    ):
        self.key = key
        self.handle = handle
        self.depth_slider = depth_slider


class InstinctViserPlayViewer(ViserPlayViewer):
    """Viser play viewer with Camera Feeds for ray caster cameras (e.g. Parkour depth)."""

    _ray_caster_feeds: list[tuple[str, GroupedRayCasterCamera, _RayCasterFeed]]

    @override
    def setup(self) -> None:
        super().setup()
        self._ray_caster_feeds = []
        env = self.env.unwrapped
        sensors = _get_ray_caster_camera_sensors(env)
        if not sensors:
            return
        with self._server.gui.add_folder("Camera Feeds (Ray Caster)"):
            for name, sensor in sensors:
                cam_data = getattr(sensor, "_camera_data", None)
                if cam_data is None:
                    continue
                output = getattr(cam_data, "output", None)
                if not output:
                    continue
                key = "distance_to_camera" if "distance_to_camera" in output else "distance_to_image_plane"
                if key not in output:
                    continue
                shape = getattr(sensor, "image_shape", (64, 64))
                if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                    h, w = int(shape[0]), int(shape[1])
                else:
                    h, w = 64, 64
                img = np.zeros((h, w, 3), dtype=np.uint8)
                handle = self._server.gui.add_image(
                    image=img,
                    label=f"{name}_{key}",
                    format="jpeg",
                )
                depth_scale = self._server.gui.add_slider(
                    label=f"{name} depth scale",
                    min=0.1,
                    max=10.0,
                    step=0.1,
                    initial_value=1.0,
                )
                self._ray_caster_feeds.append((name, sensor, _RayCasterFeed(key, handle, depth_scale)))

    @override
    def sync_env_to_viewer(self) -> None:
        super().sync_env_to_viewer()
        if not self._ray_caster_feeds:
            return
        env_idx = self._scene.env_idx
        for _name, sensor, feed in self._ray_caster_feeds:
            cam_data = getattr(sensor, "_camera_data", None)
            if cam_data is None:
                continue
            output = getattr(cam_data, "output", None)
            if not output or feed.key not in output:
                continue
            tensor = output[feed.key]
            if tensor is None:
                continue
            try:
                arr = tensor[env_idx].cpu().numpy()
            except Exception:
                continue
            scale = float(feed.depth_slider.value) if feed.depth_slider is not None else 1.0
            img = _depth_tensor_to_display_image(arr, scale=scale)
            feed.handle.image = img

    @override
    def close(self) -> None:
        for _name, _sensor, feed in self._ray_caster_feeds:
            try:
                feed.handle.remove()
            except Exception:
                pass
            if feed.depth_slider is not None:
                try:
                    feed.depth_slider.remove()
                except Exception:
                    pass
        self._ray_caster_feeds.clear()
        super().close()
