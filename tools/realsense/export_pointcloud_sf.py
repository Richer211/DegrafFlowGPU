#!/usr/bin/env python3
"""Export RGB + industrial RGB+scene-flow point clouds (PLY) from RealSense data.

For each frame pair index:
  - ply_rgb: geometry colored by true RGB
  - ply_sf:  same points — RGB base, warm tint only where |SF| is strong
             (industrial demo look; static objects stay recognizable)

Also writes a 3-view preview JPG (front / top / side).

For the preferred JPG demo panel (tabletop crop + legend), use:
  export_pointcloud_easy.py

Example:
  python3 export_pointcloud_sf.py \\
    --seq-kitti .../seq_box_slide_002 \\
    --flow-dir .../degraf_flow_interponet \\
    --out .../seq_box_slide_002_pointcloud \\
    --count 29 --stride-indices 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from viz_scene_flow import compute_scene_flow, load_calib, read_kitti_disp, read_kitti_flow


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seq-kitti", required=True)
    p.add_argument("--flow-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--count", type=int, default=29)
    p.add_argument("--stride-indices", type=int, default=1, help="export every N pair index")
    p.add_argument("--max-depth-m", type=float, default=1.8)
    p.add_argument("--min-depth-m", type=float, default=0.2)
    p.add_argument("--sf-max-m", type=float, default=0.08)
    p.add_argument("--sf-thresh-m", type=float, default=0.02)
    p.add_argument("--blend", type=float, default=0.72)
    p.add_argument("--max-points", type=int, default=200000)
    return p.parse_args()


def depth_from_disp(disp: np.ndarray, fx: float, baseline: float) -> np.ndarray:
    z = np.zeros_like(disp, dtype=np.float32)
    valid = (disp > 1.0) & (disp < 300.0)
    z[valid] = (fx * baseline) / disp[valid]
    return z


def backproject(z: np.ndarray, fx, fy, cx, cy):
    h, w = z.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    return X, Y, z


def write_ply_binary(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    n = xyz.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    # Structured array for fast dump
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    arr = np.empty(n, dtype=dt)
    arr["x"] = xyz[:, 0].astype(np.float32)
    arr["y"] = xyz[:, 1].astype(np.float32)
    arr["z"] = xyz[:, 2].astype(np.float32)
    arr["r"] = rgb[:, 0].astype(np.uint8)
    arr["g"] = rgb[:, 1].astype(np.uint8)
    arr["b"] = rgb[:, 2].astype(np.uint8)
    with path.open("wb") as f:
        f.write(header)
        arr.tofile(f)


def subsample(xyz, *extras, max_points: int, seed: int):
    if xyz.shape[0] <= max_points:
        return (xyz,) + extras
    sel = np.random.default_rng(seed).choice(xyz.shape[0], max_points, replace=False)
    return (xyz[sel],) + tuple(e[sel] for e in extras)


def sf_heat_rgb(mag: np.ndarray, max_sf: float) -> np.ndarray:
    n = np.clip(mag / max(max_sf, 1e-6), 0, 1)
    heat = cv2.applyColorMap((n * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO)
    return heat.reshape(-1, 3)[:, ::-1]  # BGR→RGB


def rgb_with_motion(rgb: np.ndarray, mag: np.ndarray, thresh: float, max_sf: float, blend: float) -> np.ndarray:
    """Industrial: keep RGB for static; warm-blend moving points."""
    out = rgb.astype(np.float32).copy()
    heat = sf_heat_rgb(mag, max_sf).astype(np.float32)
    moving = mag >= thresh
    if np.any(moving):
        w = np.clip((mag[moving] - thresh) / (max_sf - thresh + 1e-6), 0, 1)[:, None]
        w = w * float(blend)
        out[moving] = (1.0 - w) * out[moving] + w * heat[moving]
    return np.clip(out, 0, 255).astype(np.uint8)


def render_views(xyz: np.ndarray, rgb: np.ndarray, size=480) -> np.ndarray:
    """Cheap orthographic previews: front (X-Y), top (X-Z), side (Z-Y)."""
    if xyz.shape[0] == 0:
        return np.zeros((size, size * 3, 3), np.uint8)

    def paint(xs, ys, title):
        canvas = np.zeros((size, size, 3), np.uint8)
        # normalize to canvas
        xmin, xmax = np.percentile(xs, [2, 98])
        ymin, ymax = np.percentile(ys, [2, 98])
        if xmax - xmin < 1e-3:
            xmax = xmin + 1e-3
        if ymax - ymin < 1e-3:
            ymax = ymin + 1e-3
        u = ((xs - xmin) / (xmax - xmin) * (size - 3)).astype(np.int32)
        v = ((ys - ymin) / (ymax - ymin) * (size - 3)).astype(np.int32)
        v = size - 1 - v
        u = np.clip(u, 0, size - 1)
        v = np.clip(v, 0, size - 1)
        canvas[v, u] = rgb
        cv2.putText(canvas, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return canvas

    front = paint(xyz[:, 0], -xyz[:, 1], "front (X-Y)")  # flip Y for upright
    top = paint(xyz[:, 0], xyz[:, 2], "top (X-Z)")
    side = paint(xyz[:, 2], -xyz[:, 1], "side (Z-Y)")
    return np.hstack([front, top, side])


def main() -> int:
    args = parse_args()
    root = Path(args.seq_kitti)
    flow_dir = Path(args.flow_dir)
    out = Path(args.out)
    (out / "ply_rgb").mkdir(parents=True, exist_ok=True)
    (out / "ply_sf").mkdir(parents=True, exist_ok=True)
    (out / "preview").mkdir(parents=True, exist_ok=True)

    img_dir = root / "training" / "image_2"
    d0_dir = root / "training" / "disp_noc_0"
    d1_dir = root / "training" / "disp_noc_1"
    calib_dir = root / "calib" / "training" / "calib_cam_to_cam"

    exported = []
    for i in range(0, args.count, args.stride_indices):
        stem = f"{i:06d}"
        img_p = img_dir / f"{stem}_10.png"
        d0_p = d0_dir / f"{stem}_10.png"
        d1_p = d1_dir / f"{stem}_10.png"
        flow_p = flow_dir / f"{stem}_10.png"
        cal_p = calib_dir / f"{stem}.txt"
        if not all(p.exists() for p in (img_p, d0_p, d1_p, flow_p, cal_p)):
            continue

        fx, fy, cx, cy, baseline = load_calib(cal_p)
        bgr = cv2.imread(str(img_p))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        disp0 = read_kitti_disp(d0_p)
        disp1 = read_kitti_disp(d1_p)
        flow, _ = read_kitti_flow(flow_p)
        z = depth_from_disp(disp0, fx, baseline)
        X, Y, Z = backproject(z, fx, fy, cx, cy)

        sf, sf_valid = compute_scene_flow(flow, disp0, disp1, fx, fy, cx, cy, baseline)
        mag = np.linalg.norm(sf, axis=2)

        geom = (
            (Z > args.min_depth_m)
            & (Z < args.max_depth_m)
            & np.isfinite(Z)
            & (disp0 > 1.0)
        )
        ys, xs = np.where(geom)
        if ys.size == 0:
            continue

        xyz = np.stack([X[ys, xs], Y[ys, xs], Z[ys, xs]], axis=1)
        col = rgb[ys, xs]
        mag_pts = mag[ys, xs]
        xyz, col, mag_pts = subsample(xyz, col, mag_pts, max_points=args.max_points, seed=0)

        write_ply_binary(out / "ply_rgb" / f"{stem}_rgb.ply", xyz, col)
        col_sf = rgb_with_motion(col, mag_pts, args.sf_thresh_m, args.sf_max_m, args.blend)
        write_ply_binary(out / "ply_sf" / f"{stem}_sf.ply", xyz, col_sf)

        prev_rgb = render_views(xyz, col, size=420)
        prev_sf = render_views(xyz, col_sf, size=420)
        both = np.vstack([prev_rgb, prev_sf])
        cv2.putText(
            both,
            f"{stem} top: RGB cloud | bottom: RGB + motion highlight (industrial)",
            (10, both.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.imwrite(str(out / "preview" / f"{stem}_views.jpg"), both, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        exported.append(stem)
        print(f"[OK] {stem} points={xyz.shape[0]}  -> ply_rgb/ ply_sf/ preview/")

    html = [
        "<html><head><meta charset='utf-8'><title>point cloud + scene flow</title>",
        "<style>body{font-family:sans-serif;background:#111;color:#eee}img{max-width:100%;border:1px solid #333}",
        ".card{margin:16px 0;padding:12px;background:#1a1a1a} code{color:#9cf}</style></head><body>",
        "<h1>Point cloud + scene flow (industrial)</h1>",
        "<p><b>ply_rgb</b>: 3D shape/pose (true RGB).<br>",
        "<b>ply_sf</b>: RGB base + warm tint only on moving points (|SF|).<br>",
        "Prefer <code>preview_easy/*_easy.jpg</code> for demos (tabletop crop + legend).<br>",
        "Open <code>.ply</code> in MeshLab / CloudCompare / Open3D. Previews below are rough orthographic views.</p>",
    ]
    for stem in exported:
        html.append(
            f"<div class='card'><h3>{stem}</h3>"
            f"<p>files: <code>ply_rgb/{stem}_rgb.ply</code> , <code>ply_sf/{stem}_sf.ply</code></p>"
            f"<img src='preview/{stem}_views.jpg' loading='lazy'></div>"
        )
    html.append("</body></html>")
    (out / "index.html").write_text("\n".join(html), encoding="utf-8")
    print(f"[OK] exported {len(exported)} frames -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
