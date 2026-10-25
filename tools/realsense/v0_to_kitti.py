#!/usr/bin/env python3
"""Convert RealSense v0 sequence (color/depth/calib.json) to KITTI-like layout
for DegrafFlowGPU (image_2, disp_*, flow_*, obj_map, calib_cam_to_cam).

Example:
  python3 v0_to_kitti.py \\
    --src /root/autodl-tmp/realsense_seq/seq_box_slide_002 \\
    --out /root/autodl-tmp/data/realsense_kitti/seq_box_slide_002 \\
    --stride 3 --max-pairs 8
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RealSense v0 -> KITTI-like for DegrafFlowGPU")
    p.add_argument("--src", required=True, help="v0 sequence dir with color/ depth/ calib.json")
    p.add_argument("--out", required=True, help="Output dataset root (will contain training/)")
    p.add_argument("--stride", type=int, default=4, help="Frame gap between i1 and i2")
    p.add_argument(
        "--step",
        type=int,
        default=0,
        help="Advance start index by this many frames (0=same as --stride, non-overlapping scan)",
    )
    p.add_argument("--max-pairs", type=int, default=1000, help="Max number of pairs to export")
    p.add_argument("--start", type=int, default=0, help="Start frame index in source sequence")
    p.add_argument(
        "--sample",
        choices=("scan", "even", "head"),
        default="scan",
        help="scan=walk start..end with --step (default); even=uniform subsample; head=dense from start",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing training/ under --out before writing",
    )
    return p.parse_args()


def list_frames(color_dir: Path) -> list[int]:
    ids = []
    for p in color_dir.glob("*.png"):
        if p.name.startswith("._"):
            continue
        try:
            ids.append(int(p.stem))
        except ValueError:
            continue
    return sorted(ids)


def depth_mm_to_kitti_disp(depth_mm: np.ndarray, fx: float, baseline_m: float) -> np.ndarray:
    """KITTI uint16 disparity: round(disp_px * 256), 0 = invalid."""
    z_m = depth_mm.astype(np.float32) / 1000.0
    valid = depth_mm > 0
    disp = np.zeros(depth_mm.shape, dtype=np.float32)
    disp[valid] = (fx * baseline_m) / np.maximum(z_m[valid], 1e-6)
    out = np.zeros(depth_mm.shape, dtype=np.uint16)
    scaled = np.clip(np.round(disp * 256.0), 0, 65535).astype(np.uint16)
    out[valid] = scaled[valid]
    # Also zero out absurd disparities (near-zero depth noise).
    out[disp > 512] = 0
    return out


def write_kitti_flow_png(path: Path, h: int, w: int) -> None:
    """Placeholder GT: zero flow, all pixels valid. OpenCV BGR = (valid, v, u)."""
    flow = np.zeros((h, w, 3), dtype=np.uint16)
    flow[:, :, 0] = 1  # valid
    flow[:, :, 1] = 32768  # v = 0
    flow[:, :, 2] = 32768  # u = 0
    cv2.imwrite(str(path), flow)


def write_calib(path: Path, fx: float, fy: float, cx: float, cy: float, baseline_m: float, w: int, h: int) -> None:
    # P_rect_02 / P_rect_03 only fields required by EvaluateSceneFlow::loadCameraIntrinsics.
    # baseline = -(Tx3 - Tx2) / fx  with Tx2=0 => Tx3 = -baseline * fx
    tx2 = 0.0
    tx3 = -baseline_m * fx
    p2 = [fx, 0.0, cx, tx2, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    p3 = [fx, 0.0, cx, tx3, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]

    def fmt(vals: list[float]) -> str:
        return " ".join(f"{v:.6e}" for v in vals)

    lines = [
        "calib_time: realsense_v0_to_kitti",
        f"S_rect_02: {w:.6e} {h:.6e}",
        f"P_rect_02: {fmt(p2)}",
        f"S_rect_03: {w:.6e} {h:.6e}",
        f"P_rect_03: {fmt(p3)}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    src = Path(args.src).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    color_dir = src / "color"
    depth_dir = src / "depth"
    calib_path = src / "calib.json"
    if not color_dir.is_dir() or not depth_dir.is_dir() or not calib_path.is_file():
        raise SystemExit(f"Missing color/depth/calib.json under {src}")

    calib = json.loads(calib_path.read_text(encoding="utf-8"))
    aligned = calib.get("aligned") or calib.get("color")
    fx = float(aligned["fx"])
    fy = float(aligned["fy"])
    cx = float(aligned["cx"])
    cy = float(aligned["cy"])
    baseline = float(aligned.get("baseline_m") or calib.get("stereo_baseline_m") or 0.05)

    frame_ids = list_frames(color_dir)
    if not frame_ids:
        raise SystemExit(f"No frames in {color_dir}")

    pairs: list[tuple[int, int]] = []
    if args.start in frame_ids:
        start_idx = frame_ids.index(args.start)
    else:
        start_idx = 0
    step = args.step if args.step > 0 else args.stride

    if args.sample == "even":
        usable = len(frame_ids) - args.stride
        if usable <= 0:
            raise SystemExit("Sequence too short for --stride")
        n = min(args.max_pairs, usable)
        if n == 1:
            starts = [start_idx]
        else:
            starts = [
                start_idx + int(round(i * (usable - 1 - start_idx) / (n - 1)))
                for i in range(n)
            ]
        for s in starts:
            if s + args.stride < len(frame_ids):
                pairs.append((frame_ids[s], frame_ids[s + args.stride]))
    elif args.sample == "scan":
        i = start_idx
        while i + args.stride < len(frame_ids) and len(pairs) < args.max_pairs:
            pairs.append((frame_ids[i], frame_ids[i + args.stride]))
            i += step
    else:  # head: dense starts from beginning
        i = start_idx
        while i + args.stride < len(frame_ids) and len(pairs) < args.max_pairs:
            pairs.append((frame_ids[i], frame_ids[i + args.stride]))
            i += 1

    if not pairs:
        raise SystemExit("No pairs produced; check --start/--stride/--max-pairs")
    print(f"[INFO] sample={args.sample} stride={args.stride} step={step} pairs={len(pairs)}")

    train = out / "training"
    calib_root = out / "calib" / "training" / "calib_cam_to_cam"
    if args.overwrite and out.exists():
        shutil.rmtree(out)
    dirs = [
        train / "image_2",
        train / "disp_noc_0",
        train / "disp_noc_1",
        train / "disp_occ_0",
        train / "disp_occ_1",
        train / "flow_noc",
        train / "flow_occ",
        train / "obj_map",
        calib_root,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] src={src}")
    print(f"[INFO] fx={fx:.3f} fy={fy:.3f} cx={cx:.3f} cy={cy:.3f} baseline={baseline:.4f}m")
    print(f"[INFO] exporting {len(pairs)} pairs -> {out}")

    for kitti_id, (id0, id1) in enumerate(pairs):
        stem = f"{kitti_id:06d}"
        # Source color is RGB; OpenCV pipeline expects BGR files via imread.
        c0 = cv2.cvtColor(
            np.array(Image.open(color_dir / f"{id0:06d}.png").convert("RGB")),
            cv2.COLOR_RGB2BGR,
        )
        c1 = cv2.cvtColor(
            np.array(Image.open(color_dir / f"{id1:06d}.png").convert("RGB")),
            cv2.COLOR_RGB2BGR,
        )
        d0 = np.array(Image.open(depth_dir / f"{id0:06d}.png"))
        d1 = np.array(Image.open(depth_dir / f"{id1:06d}.png"))
        if d0.ndim != 2:
            d0 = d0[:, :, 0]
        if d1.ndim != 2:
            d1 = d1[:, :, 0]
        h, w = c0.shape[:2]
        if c1.shape[:2] != (h, w) or d0.shape != (h, w) or d1.shape != (h, w):
            raise SystemExit(f"Size mismatch at pair {id0}-{id1}")

        cv2.imwrite(str(train / "image_2" / f"{stem}_10.png"), c0)
        cv2.imwrite(str(train / "image_2" / f"{stem}_11.png"), c1)

        disp0 = depth_mm_to_kitti_disp(d0, fx, baseline)
        disp1 = depth_mm_to_kitti_disp(d1, fx, baseline)
        cv2.imwrite(str(train / "disp_noc_0" / f"{stem}_10.png"), disp0)
        cv2.imwrite(str(train / "disp_noc_1" / f"{stem}_10.png"), disp1)
        cv2.imwrite(str(train / "disp_occ_0" / f"{stem}_10.png"), disp0)
        cv2.imwrite(str(train / "disp_occ_1" / f"{stem}_10.png"), disp1)

        write_kitti_flow_png(train / "flow_noc" / f"{stem}_10.png", h, w)
        write_kitti_flow_png(train / "flow_occ" / f"{stem}_10.png", h, w)

        # Background-only obj_map (uint8)
        cv2.imwrite(str(train / "obj_map" / f"{stem}_10.png"), np.zeros((h, w), dtype=np.uint8))

        write_calib(calib_root / f"{stem}.txt", fx, fy, cx, cy, baseline, w, h)
        print(f"[OK] {stem}: src {id0:06d}->{id1:06d}  disp0_valid={(disp0 > 0).mean():.1%}")

    # Convenience: also expose calib at DEGRAF_CALIB_PATH style sibling if user points out/calib
    print(f"[OK] Done. Set:")
    print(f"  export DEGRAF_DATA_PATH={out}")
    print(f"  export DEGRAF_CALIB_PATH={out / 'calib'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
