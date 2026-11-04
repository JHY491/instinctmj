#!/usr/bin/env python
"""Convert Unitree Motion Studio exports into instinct ``*_retargeted.npz``.

Two input layouts are understood (auto-detected by extension):

- ``.npz`` — keys ``qpos [N, 3+1+num_joints]``, ``fps``, ``joint_names``,
  ``motion_studio_qpos_layout == "quaternion_first"``: each qpos row is
  ``[quat_xyzw(4), root_pos(3), joint_pos(num_joints)]``.
- ``.csv`` — columns ``root_pos_{x,y,z}(m)``, ``root_rot_{w,x,y,z}``,
  ``dof_<joint_name>(rad)``: quaternion is already wxyz and the root
  position comes first.

Output matches what ``AmassMotion._read_retargetted_motion_file`` expects:
``{framerate, joint_names, joint_pos, base_pos_w, base_quat_w}`` with
quaternions in wxyz, data in float32, the xy trajectory re-started at the
origin (root height kept), and trailing loop-closure teleports trimmed.

Example:
  .venv/bin/python -I scripts/convert_motion_studio_to_retargeted.py \
      ~/下载/BG_Normal_Walking_00003.npz \
      -o src/instinct_mj/motions/walk_g1_retargeted.npz \
      --xml src/instinct_mj/assets/resources/unitree_g1/xml/g1_29dof_torsobase_popsicle.xml
"""

from __future__ import annotations

import argparse
import os

import numpy as np

TELEPORT_METERS_PER_FRAME = 0.3  # per-frame root jump that counts as a teleport
TELEPORT_TAIL_FRACTION = 0.1  # only trim teleports in the last 10% of frames


def rebase_root_to_sim(
    xml_path: str,
    ref_root_body: str,
    base_pos: np.ndarray,
    base_quat_wxyz: np.ndarray,
    joint_names: list[str],
    joint_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-express the root pose from the data's root body to the sim root body.

    Motion Studio exports the pelvis pose, but the sim's freejoint is on
    torso_link, offset from pelvis by the waist joints (T_tp, frame-dependent).
    The equivalent torso pose is T_t = T_p . T_tp^-1.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(xml_path)
    hinge = [i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    hinge_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in hinge]
    qadr = [int(model.jnt_qposadr[i]) for i in hinge]
    remap = [joint_names.index(n) for n in hinge_names]
    ref_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ref_root_body)
    sim_body = int(model.jnt_bodyid[int(np.argmax(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE))])
    if ref_body == sim_body:
        return base_pos, base_quat_wxyz

    data = mujoco.MjData(model)
    out_pos = np.empty_like(base_pos)
    out_quat = np.empty_like(base_quat_wxyz)
    for i in range(len(base_pos)):
        # FK with the sim root at the origin to get T_tp (sim root -> ref body).
        data.qpos[:3] = 0.0
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qpos[qadr] = joint_pos[i, remap]
        mujoco.mj_forward(model, data)
        tp_pos = data.xpos[ref_body].copy()
        tp_quat = data.xquat[ref_body].copy()  # wxyz

        # T_t = T_p . T_tp^-1  (quaternion convention wxyz throughout).
        inv_tp_quat = tp_quat * np.array([1.0, -1.0, -1.0, -1.0])
        out_quat[i] = quat_mul(base_quat_wxyz[i], inv_tp_quat)
        out_pos[i] = base_pos[i] - quat_rotate(out_quat[i], tp_pos)
    out_quat /= np.linalg.norm(out_quat, axis=1, keepdims=True)
    print(f"[convert] rebased root body '{ref_root_body}' -> sim root (body id {sim_body})")
    return out_pos, out_quat


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of wxyz quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by wxyz quaternion q."""
    w, xyz = q[0], q[1:]
    return v * (2 * w * w - 1) + 2 * np.dot(xyz, v) * xyz + 2 * w * np.cross(xyz, v)


def load_motion_studio_npz(path: str) -> tuple[np.ndarray, float, list[str]]:
    """Return (qpos [N, 4+3+J] as [quat_xyzw, pos, joints], fps, joint_names)."""
    data = np.load(path, allow_pickle=True)
    layout = str(data["motion_studio_qpos_layout"].item())
    if layout != "quaternion_first":
        raise ValueError(f"Unsupported qpos layout: {layout!r} (expected 'quaternion_first')")
    qpos = np.asarray(data["qpos"], dtype=np.float64)
    joint_names = [str(n) for n in data["joint_names"].tolist()]
    if qpos.shape[1] != 4 + 3 + len(joint_names):
        raise ValueError(
            f"qpos has {qpos.shape[1]} cols, expected {4 + 3 + len(joint_names)} "
            "for [quat(4), pos(3), joints]"
        )
    return qpos, float(data["fps"].item()), joint_names


def load_motion_studio_csv(path: str) -> tuple[np.ndarray, float, list[str]]:
    """Return (qpos [N, 4+3+J] as [quat_xyzw, pos, joints], fps, joint_names)."""
    import pandas as pd

    frame = pd.read_csv(path)
    pos_cols = ["root_pos_x(m)", "root_pos_y(m)", "root_pos_z(m)"]
    quat_cols = ["root_rot_w", "root_rot_x", "root_rot_y", "root_rot_z"]
    joint_cols = [c for c in frame.columns if c.startswith("dof_")]
    joint_names = [c[len("dof_") : -len("(rad)")] for c in joint_cols]

    pos = frame[pos_cols].to_numpy(dtype=np.float64)
    quat_wxyz = frame[quat_cols].to_numpy(dtype=np.float64)
    joints = frame[joint_cols].to_numpy(dtype=np.float64)
    qpos = np.concatenate([quat_wxyz[:, [1, 2, 3, 0]], pos, joints], axis=1)

    # The CSV export has no fps column; Motion Studio records at 120 Hz.
    return qpos, 120.0, joint_names


def trim_trailing_teleport(qpos: np.ndarray) -> np.ndarray:
    """Drop loop-closure frames whose root teleports back to the start."""
    pos = qpos[:, 4:7]
    steps = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1)
    tail_start = int(len(steps) * (1.0 - TELEPORT_TAIL_FRACTION))
    teleports = np.where(steps > TELEPORT_METERS_PER_FRAME)[0]
    tail_teleports = teleports[teleports >= tail_start]
    if len(tail_teleports) > 0 and len(tail_teleports) == len(teleports):
        cut = tail_teleports[0]
        print(f"[convert] trimming {len(qpos) - cut} trailing loop-closure frame(s) at frame {cut}")
        return qpos[: cut + 1]
    if len(teleports) > 0:
        print(f"[convert] WARNING: {len(teleports)} mid-motion teleport frame(s) kept: {teleports[:10]}")
    return qpos


def check_joint_limits(xml_path: str, joint_names: list[str], joint_pos: np.ndarray) -> None:
    """Report motion frames outside the model's joint position limits."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(xml_path)
    sim_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    violations = []
    for name in sim_names:
        if name not in joint_names:
            continue  # free joint / joints absent from the motion (held at default)
        j = model.joint(name)
        col = joint_names.index(name)
        lo, hi = np.asarray(j.range).reshape(-1)[:2]
        out = ((joint_pos[:, col] < lo) | (joint_pos[:, col] > hi)).mean()
        if out > 0:
            violations.append(f"  {name}: {out:.1%} frames outside [{lo:.3f}, {hi:.3f}] rad")
    if violations:
        print("[convert] joint-limit violations vs XML:")
        print("\n".join(violations))
    else:
        print("[convert] all motion joints within XML limits")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Motion Studio export (.npz or .csv)")
    parser.add_argument("-o", "--output", required=True, help="output *_retargeted.npz path")
    parser.add_argument("--xml", default=None, help="robot XML used to sanity-check joint limits")
    parser.add_argument(
        "--ref-root-body",
        default="pelvis",
        help="body the exported root pose refers to; rebased to the sim freejoint body "
        "(no rebase if identical or --xml is omitted)",
    )
    args = parser.parse_args()

    if args.input.endswith(".npz"):
        qpos, fps, joint_names = load_motion_studio_npz(args.input)
    elif args.input.endswith(".csv"):
        qpos, fps, joint_names = load_motion_studio_csv(args.input)
    else:
        raise ValueError(f"Unsupported input: {args.input}")

    qpos = trim_trailing_teleport(qpos)

    quat_xyzw = qpos[:, :4]
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
    quat_wxyz /= np.linalg.norm(quat_wxyz, axis=1, keepdims=True)

    joint_pos = qpos[:, 7:]
    base_pos_w = qpos[:, 4:7].copy()
    if args.xml and args.ref_root_body:
        base_pos_w, quat_wxyz = rebase_root_to_sim(
            args.xml, args.ref_root_body, base_pos_w, quat_wxyz, joint_names, joint_pos
        )
    base_pos_w[:, :2] -= base_pos_w[0, :2]  # re-start the walk at the origin

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(
        args.output,
        framerate=np.float64(fps),
        joint_names=np.array(joint_names, dtype=object),
        joint_pos=joint_pos.astype(np.float32),
        base_pos_w=base_pos_w.astype(np.float32),
        base_quat_w=quat_wxyz.astype(np.float32),
    )

    duration = len(qpos) / fps
    walked = np.linalg.norm(base_pos_w[-1, :2] - base_pos_w[0, :2])
    print(f"[convert] wrote {args.output}")
    print(
        f"[convert] {len(qpos)} frames @ {fps:.0f} Hz ({duration:.1f} s), "
        f"{len(joint_names)} joints, walked {walked:.2f} m ({walked / duration:.2f} m/s), "
        f"root z in [{base_pos_w[:, 2].min():.3f}, {base_pos_w[:, 2].max():.3f}] m, "
        f"yaw0 = {np.degrees(2 * np.arctan2(quat_wxyz[0, 3], quat_wxyz[0, 0])):.1f} deg"
    )

    if args.xml:
        check_joint_limits(args.xml, joint_names, joint_pos)


if __name__ == "__main__":
    main()
