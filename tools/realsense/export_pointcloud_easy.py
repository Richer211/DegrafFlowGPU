#!/usr/bin/env python3
"""Canonical industrial demo: RGB point cloud + scene-flow + IDLE/MOVING.

Phase A product viz (one 4-panel JPG per frame):
  photo(+boxes) | front RGB+filtered motion
  top RGB+filtered motion | status / legend

Motion cleaning:
  - |SF| gate
  - image-border drop
  - photo-consistency (|I0-I1|)
  - connected components + min area (kill edge speckles)

Example:
  python3 export_demo.py \\
    --seq-kitti .../seq_static_002 \\
    --flow-dir .../realsense_flow/seq_static_002 \\
    --out .../realsense_viz/seq_static_002/demo
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from viz_scene_flow import compute_scene_flow, load_calib, read_kitti_disp, read_kitti_flow


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-kitti", required=True)
    p.add_argument("--flow-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--count", type=int, default=29)
    p.add_argument("--indices", default="", help="comma list e.g. 8,15 ; empty=all")
    p.add_argument("--sf-thresh-m", type=float, default=0.02, help="motion gate (m), default 20mm")
    p.add_argument("--sf-max-m", type=float, default=0.08, help="heat full-scale (m)")
    p.add_argument("--blend", type=float, default=0.72, help="motion color mix into RGB")
    p.add_argument("--min-blob-px", type=int, default=2000, help="min connected motion area (px)")
    p.add_argument("--border-margin", type=int, default=56, help="ignore image border band")
    p.add_argument("--left-wall-frac", type=float, default=0.22, help="drop leftmost fraction (wall FP)")
    p.add_argument("--desk-y0-frac", type=float, default=0.28, help="alarm only below this image row frac")
    p.add_argument("--photo-thresh", type=float, default=4.0, help="min |I0-I1| gray diff")
    p.add_argument("--alarm-min-blobs", type=int, default=1, help="blobs needed for MOVING")
    p.add_argument("--merge-gap-px", type=int, default=40, help="merge blobs closer than this (px)")
    p.add_argument("--morph-close", type=int, default=9, help="close kernel to glue nearby motion")
    p.add_argument("--keep-top-blobs", type=int, default=1, help="draw/keep only largest N blobs (1=main)")
    return p.parse_args()


def depth_from_disp(disp, fx, baseline):
    z = np.zeros_like(disp, np.float32)
    m = (disp > 1.0) & (disp < 300)
    z[m] = (fx * baseline) / disp[m]
    return z


def splat(canvas, us, vs, cols, radius=2):
    for u, v, c in zip(us, vs, cols):
        cv2.circle(canvas, (int(u), int(v)), radius, (int(c[0]), int(c[1]), int(c[2])), -1)


def project_front(xyz, size):
    xs, ys = xyz[:, 0], -xyz[:, 1]
    xmin, xmax = np.percentile(xs, [1, 99])
    ymin, ymax = np.percentile(ys, [1, 99])
    pad = 0.05 * max(xmax - xmin, ymax - ymin, 1e-3)
    xmin, xmax, ymin, ymax = xmin - pad, xmax + pad, ymin - pad, ymax + pad
    u = (xs - xmin) / (xmax - xmin + 1e-9) * (size - 1)
    v = (1.0 - (ys - ymin) / (ymax - ymin + 1e-9)) * (size - 1)
    return u, v


def project_top(xyz, size):
    xs, zs = xyz[:, 0], xyz[:, 2]
    xmin, xmax = np.percentile(xs, [1, 99])
    zmin, zmax = np.percentile(zs, [1, 99])
    pad = 0.05 * max(xmax - xmin, zmax - zmin, 1e-3)
    xmin, xmax, zmin, zmax = xmin - pad, xmax + pad, zmin - pad, zmax + pad
    u = (xs - xmin) / (xmax - xmin + 1e-9) * (size - 1)
    v = (zs - zmin) / (zmax - zmin + 1e-9) * (size - 1)
    return u, v


def mag_to_bgr(mag, max_sf=0.08):
    n = np.clip(mag / max(max_sf, 1e-6), 0, 1)
    heat = cv2.applyColorMap((n * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO)
    return heat.reshape(-1, 3)


def rgb_with_motion(bgr_cols, mags, moving, thresh, max_sf, blend):
    out = bgr_cols.astype(np.float32).copy()
    heat = mag_to_bgr(mags, max_sf=max_sf).astype(np.float32)
    if np.any(moving):
        w = np.clip((mags[moving] - thresh) / (max_sf - thresh + 1e-6), 0, 1)[:, None]
        w = w * float(blend)
        out[moving] = (1.0 - w) * out[moving] + w * heat[moving]
    return np.clip(out, 0, 255).astype(np.uint8)


def render_cloud(xyz, cols, moving, mode="front", size=520):
    canvas = np.zeros((size, size, 3), np.uint8)
    if xyz.shape[0] == 0:
        return canvas
    u, v = project_front(xyz, size) if mode == "front" else project_top(xyz, size)
    static = ~moving
    if np.any(static):
        splat(canvas, u[static], v[static], cols[static], radius=2)
    if np.any(moving):
        splat(canvas, u[moving], v[moving], cols[moving], radius=3)
    return canvas


def label_bar(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(out, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def _bbox_gap(a, b) -> float:
    """Min distance between two axis-aligned boxes; 0 if overlap/touch."""
    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    if ax2 < bx:
        dx = bx - ax2
    elif bx2 < ax:
        dx = ax - bx2
    else:
        dx = 0
    if ay2 < by:
        dy = by - ay2
    elif by2 < ay:
        dy = ay - by2
    else:
        dy = 0
    return float(np.hypot(dx, dy))


def merge_blobs(blobs, labels_map, mag, merge_gap_px: int):
    """Greedy merge nearby blob boxes; return merged blob list + label-id groups."""
    if not blobs:
        return [], []
    # blobs items: (x,y,w,h,area,mean_sf_mm, lab_id)
    items = list(blobs)
    used = [False] * len(items)
    merged = []
    groups = []
    for i, bi in enumerate(items):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j, bj in enumerate(items):
                if used[j]:
                    continue
                # gap to any in group
                ok = any(_bbox_gap(items[g], bj) <= merge_gap_px for g in group)
                if ok:
                    used[j] = True
                    group.append(j)
                    changed = True
        labs = [items[g][6] for g in group]
        sel = np.isin(labels_map, labs)
        ys, xs = np.where(sel)
        if ys.size == 0:
            continue
        x, y = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())
        area = int(ys.size)
        mean_sf_mm = float(mag[sel].mean() * 1000.0)
        merged.append((x, y, x2 - x + 1, y2 - y + 1, area, mean_sf_mm))
        groups.append(labs)
    merged_sorted = sorted(zip(merged, groups), key=lambda t: t[0][4], reverse=True)
    return [m for m, _ in merged_sorted], [g for _, g in merged_sorted]


def filter_motion_blobs(
    raw_moving: np.ndarray,
    mag: np.ndarray,
    gray0: np.ndarray,
    gray1: np.ndarray,
    border_margin: int,
    photo_thresh: float,
    min_blob_px: int,
    morph_close: int,
    merge_gap_px: int,
    keep_top: int,
    left_wall_frac: float = 0.22,
    desk_y0_frac: float = 0.28,
):
    """Return cleaned mask + merged blob list (largest-first, truncated)."""
    h, w = raw_moving.shape
    mask = raw_moving.copy()
    if border_margin > 0:
        mask[:border_margin, :] = False
        mask[-border_margin:, :] = False
        mask[:, :border_margin] = False
        mask[:, -border_margin:] = False
    # Phase-B: kill left wall strip + upper wall band (common FP on belt demos)
    x0 = int(w * max(0.0, min(0.45, left_wall_frac)))
    y0 = int(h * max(0.0, min(0.6, desk_y0_frac)))
    if x0 > 0:
        mask[:, :x0] = False
    if y0 > 0:
        mask[:y0, :] = False

    photo = np.abs(gray0.astype(np.float32) - gray1.astype(np.float32))
    mask &= photo >= photo_thresh

    if morph_close and morph_close > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    raw_blobs = []
    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_blob_px:
            continue
        x = int(stats[lab, cv2.CC_STAT_LEFT])
        y = int(stats[lab, cv2.CC_STAT_TOP])
        bw = int(stats[lab, cv2.CC_STAT_WIDTH])
        bh = int(stats[lab, cv2.CC_STAT_HEIGHT])
        mean_sf_mm = float(mag[labels == lab].mean() * 1000.0)
        raw_blobs.append((x, y, bw, bh, area, mean_sf_mm, lab))

    merged, groups = merge_blobs(raw_blobs, labels, mag, merge_gap_px)
    if keep_top > 0:
        merged = merged[:keep_top]
        groups = groups[:keep_top]

    kept = np.zeros_like(mask)
    for labs in groups:
        kept[np.isin(labels, labs)] = True
    return kept, merged


def draw_photo_alarms(bgr, blobs, status: str, size=520):
    vis = bgr.copy()
    color = (0, 220, 0) if status == "IDLE" else (0, 80, 255)
    for i, (x, y, bw, bh, area, mean_sf_mm) in enumerate(blobs):
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 3)
        tag = f"MAIN {area}px |SF|~{mean_sf_mm:.0f}mm" if i == 0 else f"M{i+1} {area}px"
        cv2.putText(vis, tag, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 48), (0, 0, 0), -1)
    cv2.putText(vis, f"STATUS: {status}", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    return cv2.resize(vis, (size, size))


def status_panel(size, thresh, max_sf, min_blob_px, status, blobs, raw_pct, clean_pct):
    canvas = np.zeros((size, size, 3), np.uint8)
    st_color = (0, 220, 0) if status == "IDLE" else (0, 80, 255)
    cv2.putText(canvas, "Alarm status", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas, status, (16, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.6, st_color, 3)

    lines = [
        f"raw moving:   {raw_pct:.1f}%",
        f"after clean:  {clean_pct:.1f}%",
        f"main blobs:   {len(blobs)} (keep largest)",
        f"|SF| gate:    {thresh*1000:.0f} mm",
        f"min blob:     {min_blob_px} px",
        f"full heat:    {max_sf*1000:.0f} mm",
        "",
        "Rules:",
        "  photo + morph-close + merge",
        "  -> one MAIN box (largest)",
        "  MOVING if MAIN exists",
    ]
    y = 150
    for line in lines:
        cv2.putText(canvas, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        y += 28

    if blobs:
        y += 8
        x, y0, w, h, area, mean_sf_mm = blobs[0]
        cv2.putText(canvas, "MAIN motion:", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += 28
        cv2.putText(
            canvas,
            f"  {area}px  mean|SF|={mean_sf_mm:.0f}mm",
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 255),
            1,
        )
        y += 26
        cv2.putText(
            canvas,
            f"  box {w}x{h} @ ({x},{y0})",
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 255),
            1,
        )
    return canvas


def main():
    args = parse_args()
    root = Path(args.seq_kitti)
    flow_dir = Path(args.flow_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.indices.strip():
        indices = [int(x) for x in args.indices.split(",")]
    else:
        indices = list(range(args.count))

    written = []
    alarm_rows = []
    for i in indices:
        stem = f"{i:06d}"
        img_p = root / "training/image_2" / f"{stem}_10.png"
        img1_p = root / "training/image_2" / f"{stem}_11.png"
        d0_p = root / "training/disp_noc_0" / f"{stem}_10.png"
        d1_p = root / "training/disp_noc_1" / f"{stem}_10.png"
        flow_p = flow_dir / f"{stem}_10.png"
        cal_p = root / "calib/training/calib_cam_to_cam" / f"{stem}.txt"
        if not all(p.exists() for p in (img_p, img1_p, d0_p, d1_p, flow_p, cal_p)):
            print("skip", stem)
            continue

        fx, fy, cx, cy, baseline = load_calib(cal_p)
        bgr = cv2.imread(str(img_p))
        bgr1 = cv2.imread(str(img1_p))
        h, w = bgr.shape[:2]
        gray0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2GRAY)
        disp0 = read_kitti_disp(d0_p)
        disp1 = read_kitti_disp(d1_p)
        flow, _ = read_kitti_flow(flow_p)
        Z = depth_from_disp(disp0, fx, baseline)
        uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        X = (uu - cx) * Z / fx
        Y = (vv - cy) * Z / fy

        sf, _ = compute_scene_flow(flow, disp0, disp1, fx, fy, cx, cy, baseline)
        mag = np.linalg.norm(sf, axis=2)

        valid = (Z > 0.25) & (Z < 1.5) & (disp0 > 1.0)
        desk_samples = Z[valid & (vv > h * 0.35)]
        if desk_samples.size < 1000:
            desk_samples = Z[valid]
        desk = float(np.median(desk_samples))
        table = valid & (np.abs(Z - desk) < 0.28) & (Y > np.percentile(Y[valid], 35))
        table |= valid & (Z > desk - 0.35) & (Z < desk + 0.15) & (vv > h * 0.25) & (vv < h * 0.95)

        raw_moving = table & (mag >= args.sf_thresh_m)
        alarm_mask, blobs = filter_motion_blobs(
            raw_moving,
            mag,
            gray0,
            gray1,
            args.border_margin,
            args.photo_thresh,
            args.min_blob_px,
            args.morph_close,
            args.merge_gap_px,
            args.keep_top_blobs,
            args.left_wall_frac,
            args.desk_y0_frac,
        )
        status = "MOVING" if len(blobs) >= args.alarm_min_blobs else "IDLE"
        table_n = max(int(table.sum()), 1)
        raw_pct = 100.0 * float(raw_moving.sum()) / table_n
        clean_pct = 100.0 * float(alarm_mask.sum()) / table_n

        ys, xs = np.where(table)
        if ys.size < 500:
            print(f"[WARN] {stem} few table points {ys.size}")
        if ys.size > 80000:
            sel = np.random.default_rng(0).choice(ys.size, 80000, replace=False)
            ys, xs = ys[sel], xs[sel]

        xyz = np.stack([X[ys, xs], Y[ys, xs], Z[ys, xs]], axis=1)
        cols = bgr[ys, xs]
        mags = mag[ys, xs]
        moving = alarm_mask[ys, xs]
        demo_cols = rgb_with_motion(cols, mags, moving, args.sf_thresh_m, args.sf_max_m, args.blend)

        photo = label_bar(draw_photo_alarms(bgr, blobs, status), f"{stem} photo + alarm boxes")
        front = label_bar(
            render_cloud(xyz, demo_cols, moving, mode="front"),
            "front: RGB + cleaned motion",
        )
        top = label_bar(
            render_cloud(xyz, demo_cols, moving, mode="top"),
            "top-down: RGB + cleaned motion",
        )
        status_img = label_bar(
            status_panel(
                520,
                args.sf_thresh_m,
                args.sf_max_m,
                args.min_blob_px,
                status,
                blobs,
                raw_pct,
                clean_pct,
            ),
            "status",
        )

        panel = np.vstack([np.hstack([photo, front]), np.hstack([top, status_img])])
        note = (
            f"Phase-A | desk~{desk*1000:.0f}mm | {status} | blobs={len(blobs)} | "
            f"clean {clean_pct:.1f}% (raw {raw_pct:.1f}%) | min_blob={args.min_blob_px}px"
        )
        cv2.rectangle(panel, (0, panel.shape[0] - 36), (panel.shape[1], panel.shape[0]), (0, 0, 0), -1)
        cv2.putText(panel, note, (12, panel.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        out_p = out / f"{stem}.jpg"
        cv2.imwrite(str(out_p), panel, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        written.append(out_p.name)
        alarm_rows.append(
            {
                "stem": stem,
                "status": status,
                "n_blobs": len(blobs),
                "raw_pct": f"{raw_pct:.2f}",
                "clean_pct": f"{clean_pct:.2f}",
                "top_blob_px": blobs[0][4] if blobs else 0,
                "top_blob_sf_mm": f"{blobs[0][5]:.1f}" if blobs else "0",
            }
        )
        print(f"[OK] {out_p}  {status} blobs={len(blobs)} clean={clean_pct:.1f}% (raw={raw_pct:.1f}%)")

    csv_p = out / "alarms.csv"
    with csv_p.open("w", encoding="utf-8", newline="") as f:
        wri = csv.DictWriter(
            f,
            fieldnames=["stem", "status", "n_blobs", "raw_pct", "clean_pct", "top_blob_px", "top_blob_sf_mm"],
        )
        wri.writeheader()
        wri.writerows(alarm_rows)

    n_move = sum(1 for r in alarm_rows if r["status"] == "MOVING")
    n_idle = sum(1 for r in alarm_rows if r["status"] == "IDLE")
    html = [
        "<html><head><meta charset='utf-8'><title>industrial demo</title>",
        "<style>body{font-family:sans-serif;background:#111;color:#eee}",
        "img{max-width:100%;margin:12px 0;border:1px solid #333}</style></head><body>",
        "<h1>RGB point cloud + scene flow + IDLE/MOVING</h1>",
        f"<p>thresh={args.sf_thresh_m*1000:.0f}mm · min_blob={args.min_blob_px}px · "
        f"{len(written)} frames · MOVING={n_move} IDLE={n_idle}</p>",
        f"<p>See <code>alarms.csv</code></p>",
    ]
    for name in written:
        html.append(f"<h3>{name}</h3><img src='{name}' loading='lazy'>")
    html.append("</body></html>")
    (out / "index.html").write_text("\n".join(html), encoding="utf-8")

    print(f"[OK] demo JPGs: {out}  MOVING={n_move} IDLE={n_idle}  csv={csv_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
