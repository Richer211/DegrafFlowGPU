#!/usr/bin/env python3
"""Scene-flow viz that separates lateral (XY) motion from depth (Z) motion.

Panel (L→R):
  t | XY magnitude | XY arrows on image | signed dZ

Also prints per-frame mean dX/dY/dZ (mm) so pure lateral push shows |dZ| << |dXY|.

Example:
  python3 viz_scene_flow_xyz.py \\
    --seq-kitti .../seq_box_slide_002 \\
    --flow-dir .../degraf_flow_interponet \\
    --out .../seq_box_slide_002_sceneflow_xyz \\
    --count 29
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from viz_scene_flow import (
    clean_components,
    compute_scene_flow,
    load_calib,
    read_kitti_disp,
    read_kitti_flow,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seq-kitti", required=True)
    p.add_argument("--flow-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--count", type=int, default=29)
    p.add_argument("--sf-thresh-m", type=float, default=0.006)
    p.add_argument("--sf-max-m", type=float, default=0.20)
    p.add_argument("--max-xy-m", type=float, default=0.08, help="color scale for |XY|")
    p.add_argument("--max-dz-m", type=float, default=0.04, help="color scale for dZ")
    p.add_argument("--photo-thresh", type=float, default=8.0)
    p.add_argument("--border-margin", type=int, default=40)
    p.add_argument("--min-component", type=int, default=500)
    p.add_argument("--arrow-step", type=int, default=24)
    return p.parse_args()


def label(im, text):
    out = im.copy()
    cv2.rectangle(out, (0, 0), (min(620, out.shape[1]), 36), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def xy_heat(xy_mag, mask, max_v):
    n = np.clip(xy_mag / max(max_v, 1e-6), 0, 1)
    heat = cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat[~mask] = 0
    return heat


def dz_heat(dz, mask, max_abs):
    t = np.clip(dz / max(max_abs, 1e-6), -1, 1)
    img = np.zeros((*dz.shape, 3), np.uint8)
    pos = mask & (t > 0)
    neg = mask & (t < 0)
    img[pos, 2] = (t[pos] * 255).astype(np.uint8)
    img[neg, 0] = ((-t[neg]) * 255).astype(np.uint8)
    return img


def draw_xy_arrows(img, sf, mask, step, max_xy):
    vis = img.copy()
    h, w = mask.shape
    scale = 80.0 / max(max_xy, 1e-6)  # meters -> pixels for arrow length
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            if not mask[y, x]:
                continue
            dx, dy = float(sf[y, x, 0]), float(sf[y, x, 1])
            # image y is down; camera Y often down too — keep same sign for viz
            x2 = int(x + dx * scale)
            y2 = int(y + dy * scale)
            cv2.arrowedLine(vis, (x, y), (x2, y2), (0, 255, 255), 1, tipLength=0.35)
    return vis


def main() -> int:
    args = parse_args()
    root = Path(args.seq_kitti)
    flow_dir = Path(args.flow_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    img_dir = root / "training" / "image_2"
    d0_dir = root / "training" / "disp_noc_0"
    d1_dir = root / "training" / "disp_noc_1"
    calib_dir = root / "calib" / "training" / "calib_cam_to_cam"

    rows = []
    for i in range(args.count):
        stem = f"{i:06d}"
        paths = {
            "flow": flow_dir / f"{stem}_10.png",
            "img": img_dir / f"{stem}_10.png",
            "img2": img_dir / f"{stem}_11.png",
            "d0": d0_dir / f"{stem}_10.png",
            "d1": d1_dir / f"{stem}_10.png",
            "cal": calib_dir / f"{stem}.txt",
        }
        if not all(paths[k].exists() for k in ("flow", "img", "d0", "d1", "cal")):
            continue

        fx, fy, cx, cy, baseline = load_calib(paths["cal"])
        flow, _ = read_kitti_flow(paths["flow"])
        disp0 = read_kitti_disp(paths["d0"])
        disp1 = read_kitti_disp(paths["d1"])
        img = cv2.imread(str(paths["img"]))
        img2 = cv2.imread(str(paths["img2"])) if paths["img2"].exists() else img

        sf, valid = compute_scene_flow(flow, disp0, disp1, fx, fy, cx, cy, baseline)
        mag3 = np.linalg.norm(sf, axis=2)
        xy = np.sqrt(sf[..., 0] ** 2 + sf[..., 1] ** 2)
        dz = sf[..., 2]

        show = valid & (mag3 >= args.sf_thresh_m) & (mag3 <= args.sf_max_m)
        show &= np.abs(dz) <= args.sf_max_m

        g0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        photo = cv2.GaussianBlur(cv2.absdiff(g0, g1).astype(np.float32), (5, 5), 0)
        show &= photo >= args.photo_thresh
        m = args.border_margin
        show[:m, :] = False
        show[-m:, :] = False
        show[:, :m] = False
        show[:, -m:] = False
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        show = cv2.morphologyEx((show.astype(np.uint8) * 255), cv2.MORPH_CLOSE, k) > 0
        show = clean_components(show, args.min_component)

        # Show XY wherever motion evidence exists; Z panel uses a higher bar so
        # depth-noise at edges does not look as strong as real lateral push.
        xy_mask = show & (xy >= args.sf_thresh_m)
        z_mask = show & (np.abs(dz) >= max(0.015, args.max_dz_m * 0.45)) & (np.abs(dz) > 0.35 * np.maximum(xy, 1e-6))

        heat_xy = xy_heat(xy, xy_mask, args.max_xy_m)
        arrows = draw_xy_arrows(img, sf, xy_mask, args.arrow_step, args.max_xy_m)
        heat_z = dz_heat(dz, z_mask, args.max_dz_m)

        # HUD text with mean components (mm)
        if show.any():
            mx = float(sf[show, 0].mean()) * 1000
            my = float(sf[show, 1].mean()) * 1000
            mz = float(sf[show, 2].mean()) * 1000
            mxy = float(xy[show].mean()) * 1000
            mz_abs = float(np.abs(dz[show]).mean()) * 1000
        else:
            mx = my = mz = mxy = mz_abs = 0.0

        hud = img.copy()
        lines = [
            f"{stem}  mean|XY|={mxy:.1f}mm  mean|dZ|={mz_abs:.1f}mm",
            f"mean dX={mx:.1f}mm  dY={my:.1f}mm  dZ={mz:.1f}mm",
            "pure lateral push => |dZ| << |XY|",
        ]
        y0 = 28
        for li, t in enumerate(lines):
            cv2.putText(hud, t, (12, y0 + li * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(hud, t, (12, y0 + li * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        panel = np.hstack(
            [
                label(hud, "t + stats"),
                label(heat_xy, "XY only (lateral)"),
                label(arrows, "XY arrows"),
                label(heat_z, "Z only (blue=closer red=farther)"),
            ]
        )
        panel = cv2.resize(panel, None, fx=0.52, fy=0.52, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / f"{stem}_xyz.jpg"), panel, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        rows.append((stem, mxy, mz_abs, mx, my, mz, float(show.mean() * 100)))
        print(f"[OK] {stem} |XY|={mxy:.1f}mm |dZ|={mz_abs:.1f}mm  ratio_Z/XY={mz_abs/max(mxy,1e-3):.2f}")

    with (out / "xy_z_stats.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stem", "mean_abs_xy_mm", "mean_abs_dz_mm", "mean_dx_mm", "mean_dy_mm", "mean_dz_mm", "show_pct"])
        w.writerows(rows)

    if rows:
        xy_m = np.mean([r[1] for r in rows if r[1] > 0] or [0])
        z_m = np.mean([r[2] for r in rows if r[1] > 0] or [0])
        print(f"[SUMMARY] mean |XY|={xy_m:.1f}mm  mean |dZ|={z_m:.1f}mm  (Z/XY~{z_m/max(xy_m,1e-3):.2f})")

    html = [
        "<html><head><meta charset='utf-8'><title>SF XY vs Z</title>",
        "<style>body{font-family:sans-serif;background:#111;color:#eee}img{max-width:100%;border:1px solid #333}",
        ".card{margin:16px 0;padding:12px;background:#1a1a1a}</style></head><body>",
        "<h1>Scene Flow — XY vs Z separated</h1>",
        "<p>L→R: stats | <b>XY lateral magnitude</b> | <b>XY arrows</b> | <b>Z only</b> (blue closer / red farther). "
        "For a left-right table push, |dZ| should be much smaller than |XY|.</p>",
    ]
    for stem, mxy, mz_abs, mx, my, mz, pct in rows:
        html.append(
            f"<div class='card'><h3>{stem} &nbsp; |XY|={mxy:.1f}mm &nbsp; |dZ|={mz_abs:.1f}mm "
            f"&nbsp; dX={mx:.1f} dY={my:.1f} dZ={mz:.1f}</h3>"
            f"<img src='{stem}_xyz.jpg' loading='lazy'></div>"
        )
    html.append("</body></html>")
    (out / "index.html").write_text("\n".join(html), encoding="utf-8")
    print(f"[OK] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
