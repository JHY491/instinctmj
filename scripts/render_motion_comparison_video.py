#!/usr/bin/env python
"""Render retargeted motion npz files as a labeled side-by-side mp4.

Both robots live in one combined floor scene rendered through a single
offscreen renderer; each panel hides the other robot via geom/site groups
(G1 -> group 4, Adam -> group 5). The camera tracks each root (yaw-following
for non-straight motions), frames are interpolated for smoothness, and the
look is bright and shadowless. The result is a reference clip to compare
trained policies against.

Example:
  MUJOCO_GL=egl .venv/bin/python -I scripts/render_motion_comparison_video.py \
      -o videos/motion_reference_walk_g1_vs_adam.mp4
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")  # must precede import mujoco

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

G1_XML = "src/instinct_mj/assets/resources/unitree_g1/xml/g1_29dof_torsobase_popsicle.xml"
G1_NPZ = "src/instinct_mj/motions/walk_g1_retargeted.npz"
ADAM_XML = "src/instinct_mj/assets/resources/adam_lite/xml/adam_lite.xml"
ADAM_NPZ = "src/instinct_mj/motions/walk_retargeted.npz"

G1_GROUP, ADAM_GROUP = 4, 5


def quat_slerp(q0: np.ndarray, q1: np.ndarray, a: float) -> np.ndarray:
    """wxyz slerp; falls back to lerp+normalize for near-identical inputs."""
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = (1 - a) * q0 + a * q1
        return q / np.linalg.norm(q)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    return (np.sin((1 - a) * theta) * q0 + np.sin(a * theta) * q1) / np.sin(theta)


class TrackedMotion:
    """A motion npz bound to one prefixed robot in the combined model."""

    def __init__(self, npz_path: str):
        raw = np.load(npz_path, allow_pickle=True)
        self.joint_names = raw["joint_names"].tolist()
        self.joint_pos = np.asarray(raw["joint_pos"], dtype=np.float64)
        self.base_pos = np.asarray(raw["base_pos_w"], dtype=np.float64)
        self.base_quat = np.asarray(raw["base_quat_w"], dtype=np.float64)  # wxyz
        self.fps = float(raw["framerate"].item())
        self.free_adr = None
        self.joint_adrs: list[tuple[int, int]] = []

    @property
    def duration(self) -> float:
        return len(self.joint_pos) / self.fps

    def bind(self, model: mujoco.MjModel, prefix: str) -> None:
        for j in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name is None:
                continue
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and name.startswith(prefix):
                self.free_adr = int(model.jnt_qposadr[j])
            elif name.startswith(prefix) and name[len(prefix) :] in self.joint_names:
                self.joint_adrs.append((int(model.jnt_qposadr[j]), self.joint_names.index(name[len(prefix) :])))
        assert self.free_adr is not None, f"freejoint not found for prefix {prefix!r}"

    def write_qpos(self, data: mujoco.MjData, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Write the interpolated frame at time t; return (root_pos, root_quat)."""
        x = min(t * self.fps, len(self.joint_pos) - 1)
        i0 = int(np.floor(x))
        i1 = min(i0 + 1, len(self.joint_pos) - 1)
        a = x - i0
        pos = (1 - a) * self.base_pos[i0] + a * self.base_pos[i1]
        quat = quat_slerp(self.base_quat[i0], self.base_quat[i1], a)
        joints = (1 - a) * self.joint_pos[i0] + a * self.joint_pos[i1]

        adr = self.free_adr
        data.qpos[adr : adr + 3] = pos
        data.qpos[adr + 3 : adr + 7] = quat
        for qadr, col in self.joint_adrs:
            data.qpos[qadr] = joints[col]
        return pos, quat


def assign_vis_groups(model: mujoco.MjModel, prefix_to_group: dict[str, int]) -> None:
    """Route each robot's geoms and sites to its visualization group."""
    for gid in range(model.ngeom):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid])) or ""
        for prefix, group in prefix_to_group.items():
            if body.startswith(prefix):
                model.geom_group[gid] = group
                break
    for sid in range(model.nsite):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.site_bodyid[sid])) or ""
        for prefix, group in prefix_to_group.items():
            if body.startswith(prefix):
                model.site_group[sid] = group
                break


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output", default="videos/motion_reference_walk_g1_vs_adam.mp4")
    parser.add_argument("--fps", type=float, default=50.0, help="output video framerate")
    parser.add_argument("--duration", type=float, default=None, help="seconds (default: shortest motion)")
    parser.add_argument("--width", type=int, default=960, help="per-panel width")
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    motions = [
        ("g1_", G1_XML, G1_NPZ, "G1 29-DoF  walk_g1_retargeted", (40, 180, 40), G1_GROUP, False),
        ("adam_", ADAM_XML, ADAM_NPZ, "Adam Lite 23-DoF  walk_retargeted (loop)", (235, 120, 20), ADAM_GROUP, True),
    ]

    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth = args.width
    spec.visual.global_.offheight = args.height
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[40, 40, 0.1], rgba=[0.82, 0.83, 0.86, 1])
    spec.worldbody.add_light(pos=[3, 0, 5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    spec.worldbody.add_light(pos=[-3, 0, 4], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, intensity=0.6)
    tracked = []
    for prefix, xml, npz, *_ in motions:
        spec.attach(mujoco.MjSpec.from_file(xml), prefix=prefix, frame=spec.worldbody.add_frame())
        tracked.append(TrackedMotion(npz))
    model = spec.compile()
    assign_vis_groups(model, {"g1_": G1_GROUP, "adam_": ADAM_GROUP})
    for (prefix, *_), tm in zip(motions, tracked, strict=True):
        tm.bind(model, prefix)
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()

    duration = args.duration or min(tm.duration for tm in tracked)
    n_frames = int(duration * args.fps)
    print(f"[render] {n_frames} frames @ {args.fps:.0f} fps ({duration:.1f} s)")

    font = load_font(22)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    writer = imageio.get_writer(args.output, fps=args.fps, codec="libx264", quality=8, macro_block_size=None)

    for f in range(n_frames):
        t = f / args.fps
        states = [tm.write_qpos(data, t) for tm in tracked]
        mujoco.mj_forward(model, data)

        panels = []
        for k, (prefix, _xml, _npz, label, color, group, yaw_follow) in enumerate(motions):
            for g in (G1_GROUP, ADAM_GROUP):
                opt.geomgroup[g] = g == group
                opt.sitegroup[g] = g == group
            pos, quat = states[k]
            yaw = np.degrees(2 * np.arctan2(quat[3], quat[0]))
            cam.lookat[:] = [pos[0], pos[1], 0.85]
            cam.distance = 3.2
            cam.azimuth = (yaw - 90.0) if yaw_follow else -90.0  # walk left->right in frame
            cam.elevation = -10.0
            renderer.update_scene(data, cam, scene_option=opt)
            panels.append(Image.fromarray(renderer.render()))  # dev build already returns top-down rows

        img = Image.new("RGB", (args.width * 2 + 4, args.height), "black")
        for k, (frame, (prefix, _xml, _npz, label, color, _g, _y)) in enumerate(zip(panels, motions, strict=True)):
            draw = ImageDraw.Draw(frame)
            draw.rectangle([0, 0, args.width, 32], fill=(15, 15, 15))
            draw.text((10, 4), label, font=font, fill=color)
            draw.text((args.width - 130, args.height - 30), f"t = {t:5.2f} s", font=font, fill=(40, 40, 40))
            img.paste(frame, (k * (args.width + 4), 0))
        writer.append_data(np.asarray(img))

    writer.close()
    renderer.close()
    print(f"[render] wrote {args.output}")


if __name__ == "__main__":
    main()
