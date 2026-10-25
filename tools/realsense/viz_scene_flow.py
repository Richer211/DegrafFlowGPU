#!/usr/bin/env python3
"""Compute and visualize scene flow for RealSense KITTI-like Degraf outputs.

Scene flow SF = P1 - P0 in camera coordinates (meters), matching SceneFlowReconstructor:
  P = ((u-cx)*Z/fx, (v-cy)*Z/fy, Z),  Z = fx*baseline/disp

Panels (left→right):
  t | |SF| heatmap | overlay | signed ΔZ (red=away, blue=closer)

Example:
  python3 viz_scene_flow.py \\
    --seq-kitti /root/autodl-tmp/data/realsense_kitti/seq_box_slide_002 \\
    --flow-dir /root/autodl-tmp/DegrafFlowGPU/data/outputs/degraf_flow_interponet \\
    --out /root/autodl-tmp/DegrafFlowGPU/data/outputs/realsense_viz/seq_box_slide_002_sceneflow \\
    --count 29 --apply-motion-filter
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seq-kitti", required=True)
    p.add_argument("--flow-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--count", type=int, default=29)
    p.add_argument("--sf-thresh-m", type=float, default=0.008, help="min |SF| to show (meters)")
    p.add_argument("--sf-max-m", type=float, default=0.25, help="reject |SF| above this as depth/flow outliers")
    p.add_argument("--max-sf-m", type=float, default=0.12, help="color scale for |SF| (meters)")
    p.add_argument("--max-dz-m", type=float, default=0.06, help="color scale for ΔZ (meters)")
    p.add_argument(
        "--apply-motion-filter",
        action="store_true",
        help="also require photometric change (same idea as OF filter)",
    )
    p.add_argument("--photo-thresh", type=float, default=8.0)
    p.add_argument("--border-margin", type=int, default=40)
    p.add_argument("--min-component", type=int, default=600)
    return p.parse_args()


def read_kitti_flow(path: Path):
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    valid = raw[:, :, 0] > 0
    u = (raw[:, :, 2].astype(np.float32) - 32768.0) / 64.0
    v = (raw[:, :, 1].astype(np.float32) - 32768.0) / 64.0
    flow = np.dstack([u, v]).astype(np.float32)
    flow[~valid] = 0
    return flow, valid


def read_kitti_disp(path: Path) -> np.ndarray:
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise RuntimeError(f"missing disp {path}")
    return d.astype(np.float32) / 256.0


def load_calib(calib_txt: Path):
    fx = fy = cx = cy = baseline = None
    p2 = p3 = None
    for line in calib_txt.read_text(encoding="utf-8").splitlines():
        if line.startswith("P_rect_02:"):
            p2 = [float(x) for x in line.split()[1:]]
        elif line.startswith("P_rect_03:"):
            p3 = [float(x) for x in line.split()[1:]]
    if not p2 or not p3 or len(p2) < 12 or len(p3) < 12:
        raise RuntimeError(f"bad calib {calib_txt}")
    fx, fy, cx, cy = p2[0], p2[5], p2[2], p2[6]
    baseline = -(p3[3] - p2[3]) / fx
    return float(fx), float(fy), float(cx), float(cy), float(baseline)


def compute_scene_flow(flow, disp0, disp1, fx, fy, cx, cy, baseline):
    """Vectorized scene flow in meters. Invalid -> 0 and mask False."""
    h, w = disp0.shape
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    fu = flow[:, :, 0]
    fv = flow[:, :, 1]

    valid0 = (disp0 > 1.0) & (disp0 < 300.0) & np.isfinite(disp0)
    z0 = np.zeros_like(disp0, dtype=np.float32)
    z0[valid0] = (fx * baseline) / disp0[valid0]
    valid0 &= (z0 > 0.05) & (z0 < 80.0)

    u1 = np.rint(uu + fu).astype(np.int32)
    v1 = np.rint(vv + fv).astype(np.int32)
    inb = (u1 >= 0) & (u1 < w) & (v1 >= 0) & (v1 < h) & valid0

    disp1_s = np.zeros_like(disp0, dtype=np.float32)
    disp1_s[inb] = disp1[v1[inb], u1[inb]]
    valid1 = inb & (disp1_s > 1.0) & (disp1_s < 300.0)
    z1 = np.zeros_like(disp0, dtype=np.float32)
    z1[valid1] = (fx * baseline) / disp1_s[valid1]
    valid = valid1 & (z1 > 0.05) & (z1 < 80.0)

    x0 = (uu - cx) * z0 / fx
    y0 = (vv - cy) * z0 / fy
    x1 = (u1.astype(np.float32) - cx) * z1 / fx
    y1 = (v1.astype(np.float32) - cy) * z1 / fy

    sf = np.zeros((h, w, 3), dtype=np.float32)
    sf[..., 0] = np.where(valid, x1 - x0, 0.0)
    sf[..., 1] = np.where(valid, y1 - y0, 0.0)
    sf[..., 2] = np.where(valid, z1 - z0, 0.0)
    return sf, valid


def clean_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    u8 = (mask.astype(np.uint8) * 255)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    keep = np.zeros_like(mask, dtype=bool)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep |= labels == i
    return keep


def mag_to_color(mag, mask, max_v):
    norm = np.clip(mag / max(max_v, 1e-6), 0, 1)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat[~mask] = 0
    return heat


def dz_to_color(dz, mask, max_abs):
    """Blue = closer to camera (dz<0), Red = farther (dz>0)."""
    t = np.clip(dz / max(max_abs, 1e-6), -1, 1)
    img = np.zeros((*dz.shape, 3), np.uint8)
    # BGR
    pos = t > 0
    neg = t < 0
    img[pos, 2] = (t[pos] * 255).astype(np.uint8)  # R
    img[neg, 0] = ((-t[neg]) * 255).astype(np.uint8)  # B
    img[~mask] = 0
    return img


def label(im, text):
    out = im.copy()
    cv2.rectangle(out, (0, 0), (min(560, out.shape[1]), 36), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


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
        flow_p = flow_dir / f"{stem}_10.png"
        img_p = img_dir / f"{stem}_10.png"
        img2_p = img_dir / f"{stem}_11.png"
        d0_p = d0_dir / f"{stem}_10.png"
        d1_p = d1_dir / f"{stem}_10.png"
        cal_p = calib_dir / f"{stem}.txt"
        if not all(p.exists() for p in [flow_p, img_p, d0_p, d1_p, cal_p]):
            print(f"[WARN] skip {stem}")
            continue

        fx, fy, cx, cy, baseline = load_calib(cal_p)
        flow, _ = read_kitti_flow(flow_p)
        disp0 = read_kitti_disp(d0_p)
        disp1 = read_kitti_disp(d1_p)
        img = cv2.imread(str(img_p))
        img2 = cv2.imread(str(img2_p)) if img2_p.exists() else img

        sf, valid = compute_scene_flow(flow, disp0, disp1, fx, fy, cx, cy, baseline)
        mag = np.linalg.norm(sf, axis=2)
        # Reject huge 3D jumps (usually bad disparity correspondence).
        show = valid & (mag >= args.sf_thresh_m) & (mag <= args.sf_max_m)
        show &= np.abs(sf[..., 2]) <= args.sf_max_m

        if args.apply_motion_filter:
            g0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            g1 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            photo = cv2.GaussianBlur(cv2.absdiff(g0, g1).astype(np.float32), (5, 5), 0)
            show &= photo >= args.photo_thresh
            m = args.border_margin
            if m > 0:
                show[:m, :] = False
                show[-m:, :] = False
                show[:, :m] = False
                show[:, -m:] = False
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            show = cv2.morphologyEx((show.astype(np.uint8) * 255), cv2.MORPH_CLOSE, k) > 0
            show = clean_components(show, args.min_component)

        heat = mag_to_color(mag, show, args.max_sf_m)
        dz = dz_to_color(sf[..., 2], show, args.max_dz_m)
        overlay = cv2.addWeighted(img, 0.55, heat, 0.45, 0)

        panel = np.hstack(
            [
                label(img, f"{stem} t"),
                label(heat, "|SF| (m)"),
                label(overlay, "overlay"),
                label(dz, "dZ blue=closer red=farther"),
            ]
        )
        panel = cv2.resize(panel, None, fx=0.55, fy=0.55, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / f"{stem}_panel.jpg"), panel, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        cv2.imwrite(str(out / f"{stem}_overlay.jpg"), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        cv2.imwrite(str(out / f"{stem}_dz.jpg"), dz, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        # save compact npz for later (optional analysis)
        np.savez_compressed(
            out / f"{stem}_sf.npz",
            sf=sf.astype(np.float16),
            valid=show.astype(np.uint8),
        )

        mean_m = float(mag[show].mean()) if show.any() else 0.0
        mean_dz = float(sf[..., 2][show].mean()) if show.any() else 0.0
        rows.append((stem, float(show.mean() * 100), mean_m, mean_dz, float(mag[valid].mean()) if valid.any() else 0.0))
        print(
            f"[OK] {stem} show={rows[-1][1]:.2f}% mean|SF|={mean_m*1000:.1f}mm "
            f"mean_dZ={mean_dz*1000:.1f}mm valid3d={valid.mean()*100:.1f}%"
        )

    with (out / "sceneflow_stats.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stem", "show_pct", "mean_sf_m", "mean_dz_m", "mean_sf_all_valid_m"])
        w.writerows(rows)

    html = [
        "<html><head><meta charset='utf-8'><title>scene flow viz</title>",
        "<style>body{font-family:sans-serif;background:#111;color:#eee}",
        "img{max-width:100%;border:1px solid #333}.card{margin:18px 0;padding:12px;background:#1a1a1a}</style></head><body>",
        "<h1>seq_box_slide_002 — Scene Flow</h1>",
        "<p>Left→right: t | 3D motion magnitude |overlay | ΔZ (blue=closer to camera, red=farther). "
        "Units meters. Optical flow + stereo depth → 3D displacement.</p>",
    ]
    for stem, pct, mean_m, mean_dz, _ in rows:
        html.append(
            f"<div class='card'><h3>{stem} &nbsp; shown={pct:.2f}% &nbsp; "
            f"mean|SF|={mean_m*1000:.1f}mm &nbsp; meanΔZ={mean_dz*1000:.1f}mm</h3>"
            f"<img src='{stem}_panel.jpg' loading='lazy'></div>"
        )
    html.append("</body></html>")
    (out / "index.html").write_text("\n".join(html), encoding="utf-8")
    print(f"[OK] wrote {len(rows)} panels -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
