#!/usr/bin/env python3
"""Compare raw optical flow vs filtered motion mask for RealSense Degraf runs.

Filters (product-oriented, not network retrain):
  1) magnitude threshold
  2) depth gate (drop invalid / far background)
  3) low-texture suppression (kill hallucinated flow on flat desk)
  4) small-component cleanup

Example:
  python3 viz_flow_filtered.py \\
    --seq-kitti /root/autodl-tmp/data/realsense_kitti/seq_box_slide_002 \\
    --seq-v0 /root/autodl-tmp/realsense_seq/seq_box_slide_002 \\
    --flow-dir /root/autodl-tmp/DegrafFlowGPU/data/outputs/degraf_flow_interponet \\
    --out /root/autodl-tmp/DegrafFlowGPU/data/outputs/realsense_viz/seq_box_slide_002_filtered \\
    --stride 4 --count 29
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seq-kitti", required=True)
    p.add_argument("--seq-v0", required=True, help="original v0 seq for depth mm PNGs")
    p.add_argument("--flow-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stride", type=int, default=4, help="src frame step used in v0_to_kitti scan")
    p.add_argument("--count", type=int, default=29)
    p.add_argument("--mag-thresh", type=float, default=4.0, help="min flow magnitude (px)")
    p.add_argument("--far-depth-mm", type=float, default=1400.0, help="drop farther background")
    p.add_argument("--near-depth-mm", type=float, default=200.0, help="drop too-near noise")
    p.add_argument("--grad-thresh", type=float, default=12.0, help="low-texture gradient threshold")
    p.add_argument("--photo-thresh", type=float, default=6.0, help="min |I0-I1| to trust motion")
    p.add_argument("--min-component", type=int, default=800, help="min motion blob area (px)")
    p.add_argument("--border-margin", type=int, default=40, help="zero out detections within image border")
    p.add_argument(
        "--wall-depth-slack-mm",
        type=float,
        default=250.0,
        help="drop pixels deeper than desk_median + slack (far wall)",
    )
    return p.parse_args()


def read_kitti_flow(path: Path):
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.ndim != 3:
        raise RuntimeError(f"bad flow {path}")
    valid = raw[:, :, 0] > 0
    u = (raw[:, :, 2].astype(np.float32) - 32768.0) / 64.0
    v = (raw[:, :, 1].astype(np.float32) - 32768.0) / 64.0
    u = np.where(valid, u, 0.0)
    v = np.where(valid, v, 0.0)
    return u, v, valid


def flow_to_color(u, v, mask, max_flow: float):
    mag = np.sqrt(u * u + v * v)
    ang = np.arctan2(v, u)
    hsv = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    hsv[..., 1] = 255
    val = np.clip(mag / max_flow * 255.0, 0, 255).astype(np.uint8)
    val = np.where(mask, val, 0)
    hsv[..., 2] = val
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def label(im, text):
    out = im.copy()
    cv2.rectangle(out, (0, 0), (min(520, out.shape[1]), 36), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def clean_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    mask_u8 = (mask.astype(np.uint8) * 255)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    keep = np.zeros_like(mask, dtype=bool)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep |= labels == i
    return keep


def main() -> int:
    args = parse_args()
    kitti = Path(args.seq_kitti)
    v0 = Path(args.seq_v0)
    flow_dir = Path(args.flow_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_dir = kitti / "training" / "image_2"
    depth_dir = v0 / "depth"
    calib = json.loads((v0 / "calib.json").read_text(encoding="utf-8"))
    # depth already in mm in v0 PNGs

    # global color scale from raw mags
    scales = []
    for i in range(min(args.count, 29)):
        fp = flow_dir / f"{i:06d}_10.png"
        if not fp.exists():
            continue
        u, v, _ = read_kitti_flow(fp)
        mag = np.sqrt(u * u + v * v)
        nz = mag[mag > 0.5]
        if nz.size:
            scales.append(float(np.percentile(nz, 95)))
    max_flow = float(np.clip(max(scales) if scales else 40.0, 8.0, 80.0))
    print(f"[INFO] max_flow scale={max_flow:.1f}")

    rows = []
    for i in range(args.count):
        stem = f"{i:06d}"
        flow_path = flow_dir / f"{stem}_10.png"
        img_path = img_dir / f"{stem}_10.png"
        img2_path = img_dir / f"{stem}_11.png"
        src_id = i * args.stride
        depth_path = depth_dir / f"{src_id:06d}.png"
        if not flow_path.exists() or not img_path.exists() or not depth_path.exists():
            print(f"[WARN] skip {stem}")
            continue

        u, v, valid = read_kitti_flow(flow_path)
        mag = np.sqrt(u * u + v * v)
        img = cv2.imread(str(img_path))
        img2 = cv2.imread(str(img2_path)) if img2_path.exists() else img
        depth = np.array(cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED))
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        if depth.shape[:2] != mag.shape[:2]:
            depth = cv2.resize(depth, (mag.shape[1], mag.shape[0]), interpolation=cv2.INTER_NEAREST)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)
        # Photometric change supports "something really moved here".
        photo = cv2.absdiff(gray, gray2).astype(np.float32)
        photo = cv2.GaussianBlur(photo, (5, 5), 0)

        raw_mask = valid & (mag > 0.5)
        mag_mask = valid & (mag >= args.mag_thresh)
        depth_ok = (depth > args.near_depth_mm) & (depth < args.far_depth_mm)
        desk = depth[(depth > 300) & (depth < 1200)]
        desk_med = float(np.median(desk)) if desk.size else 800.0
        # Keep tabletop band; aggressively drop far wall / door plane.
        band = (depth > args.near_depth_mm) & (depth < desk_med + args.wall_depth_slack_mm)
        depth_mask = mag_mask & depth_ok & band

        textured = grad >= args.grad_thresh
        photo_ok = photo >= args.photo_thresh
        # Stronger evidence: require photometric support always.
        # (Removes flicker/noise on static wall that still has weak gradients.)
        evidence = photo_ok & ((textured & (photo >= args.photo_thresh)) | (photo >= args.photo_thresh * 2.5))
        tex_mask = depth_mask & evidence

        # Kill image-border glitter (common RealSense / compression edge noise).
        m = args.border_margin
        if m > 0:
            tex_mask[:m, :] = False
            tex_mask[-m:, :] = False
            tex_mask[:, :m] = False
            tex_mask[:, -m:] = False

        # light morphology to fill gaps on moving hand/object
        tex_u8 = (tex_mask.astype(np.uint8) * 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tex_u8 = cv2.morphologyEx(tex_u8, cv2.MORPH_CLOSE, kernel)
        tex_mask = tex_u8 > 0

        final_mask = clean_components(tex_mask, args.min_component)

        flow_raw = flow_to_color(u, v, raw_mask, max_flow)
        flow_f = flow_to_color(u, v, final_mask, max_flow)
        ov_raw = cv2.addWeighted(img, 0.55, flow_raw, 0.45, 0)
        ov_f = cv2.addWeighted(img, 0.55, flow_f, 0.45, 0)

        # stats
        def frac(m):
            return float(m.mean()) * 100.0

        # rough box-ish central ROI vs left desk ROI for quick metric
        h, w = mag.shape
        box_roi = mag[int(0.35 * h) : int(0.70 * h), int(0.38 * w) : int(0.55 * w)]
        left_roi = mag[int(0.35 * h) : int(0.75 * h), int(0.05 * w) : int(0.22 * w)]
        box_keep = final_mask[int(0.35 * h) : int(0.70 * h), int(0.38 * w) : int(0.55 * w)].mean() * 100
        left_keep = final_mask[int(0.35 * h) : int(0.75 * h), int(0.05 * w) : int(0.22 * w)].mean() * 100

        panel = np.hstack(
            [
                label(img, f"{stem} t"),
                label(ov_raw, "raw overlay"),
                label(ov_f, "filtered overlay"),
                label(img2, "t+stride"),
            ]
        )
        panel = cv2.resize(panel, None, fx=0.55, fy=0.55, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / f"{stem}_compare.jpg"), panel, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        cv2.imwrite(str(out_dir / f"{stem}_raw.jpg"), ov_raw, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        cv2.imwrite(str(out_dir / f"{stem}_filtered.jpg"), ov_f, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        rows.append(
            {
                "stem": stem,
                "raw_pct": frac(raw_mask),
                "filt_pct": frac(final_mask),
                "left_keep": left_keep,
                "box_zone_keep": box_keep,
                "desk_med_mm": desk_med,
            }
        )
        print(
            f"[OK] {stem} raw={rows[-1]['raw_pct']:.1f}% -> filt={rows[-1]['filt_pct']:.1f}% "
            f"left_keep={left_keep:.1f}% box_zone={box_keep:.1f}%"
        )

    # summary csv + html
    csv_path = out_dir / "filter_stats.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("stem,raw_pct,filt_pct,left_keep_pct,box_zone_keep_pct,desk_med_mm\n")
        for r in rows:
            f.write(
                f"{r['stem']},{r['raw_pct']:.2f},{r['filt_pct']:.2f},"
                f"{r['left_keep']:.2f},{r['box_zone_keep']:.2f},{r['desk_med_mm']:.1f}\n"
            )

    html = [
        "<html><head><meta charset='utf-8'><title>filtered vs raw flow</title>",
        "<style>body{font-family:sans-serif;background:#111;color:#eee}",
        "img{max-width:100%;border:1px solid #333}.card{margin:20px 0;padding:12px;background:#1a1a1a}</style></head><body>",
        "<h1>seq_box_slide_002 — raw vs filtered motion</h1>",
        "<p>Filters: mag threshold + depth band + low-texture suppress + small blob removal. "
        "Left→right: t | raw overlay | <b>filtered overlay</b> | t+stride</p>",
        f"<p>mag_thresh={args.mag_thresh}px, depth=({args.near_depth_mm},{args.far_depth_mm})mm, "
        f"grad_thresh={args.grad_thresh}, min_component={args.min_component}</p>",
    ]
    for r in rows:
        html.append(
            f"<div class='card'><h3>{r['stem']} &nbsp; raw={r['raw_pct']:.1f}% → filt={r['filt_pct']:.1f}% "
            f"&nbsp; left_desk_keep={r['left_keep']:.1f}% &nbsp; center_zone_keep={r['box_zone_keep']:.1f}%</h3>"
            f"<img src='{r['stem']}_compare.jpg' loading='lazy'></div>"
        )
    html.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(html), encoding="utf-8")

    if rows:
        raw_m = np.mean([r["raw_pct"] for r in rows])
        filt_m = np.mean([r["filt_pct"] for r in rows])
        left_m = np.mean([r["left_keep"] for r in rows])
        print(
            f"[SUMMARY] mean coverage raw={raw_m:.1f}% filt={filt_m:.1f}% | "
            f"mean left-desk kept after filter={left_m:.1f}%"
        )
    print(f"[OK] outputs -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
