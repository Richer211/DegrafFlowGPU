#!/usr/bin/env python3
"""
Record a RealSense D435/D435I sequence into DegrafFlowGPU v0 layout.

Output:
  <out_root>/<seq_name>/
    meta.json
    calib.json
    timestamps.csv
    color/000000.png ...
    depth/000000.png ...   # uint16 millimeters, aligned to color

Mac example:
  python3 record_realsense_seq.py --out ~/realsense_seq --name seq_box_slide_001 --seconds 15
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


def _intrinsics_to_dict(intr: rs.intrinsics) -> dict:
    return {
        "width": int(intr.width),
        "height": int(intr.height),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "cx": float(intr.ppx),
        "cy": float(intr.ppy),
        "model": str(intr.model),
        "coeffs": [float(c) for c in intr.coeffs],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record RealSense color+depth sequence (v0)")
    p.add_argument("--out", required=True, help="Root output directory, e.g. ~/realsense_seq")
    p.add_argument("--name", required=True, help="Sequence name, e.g. seq_box_slide_001")
    p.add_argument("--seconds", type=float, default=15.0, help="Recording duration in seconds")
    p.add_argument("--width", type=int, default=848)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument(
        "--warmup",
        type=float,
        default=1.5,
        help="Discard first N seconds while auto-exposure settles",
    )
    p.add_argument(
        "--preview",
        action="store_true",
        help="Show live preview window (press q to stop early)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seq_dir = Path(args.out).expanduser().resolve() / args.name
    color_dir = seq_dir / "color"
    depth_dir = seq_dir / "depth"
    color_dir.mkdir(parents=True, exist_ok=False)
    depth_dir.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    profile = pipeline.start(config)
    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())  # meters per unit
    align = rs.align(rs.stream.color)

    # Warmup
    t_warm_end = time.time() + max(0.0, args.warmup)
    while time.time() < t_warm_end:
        pipeline.wait_for_frames()

    depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_intr = depth_stream.get_intrinsics()
    color_intr = color_stream.get_intrinsics()
    extrinsics = depth_stream.get_extrinsics_to(color_stream)

    # Stereo baseline from left/right infrared if available; fallback ~0.05m for D435.
    baseline_m = 0.05
    try:
        left = profile.get_stream(rs.stream.infrared, 1)
        right = profile.get_stream(rs.stream.infrared, 2)
        ir_ext = left.as_video_stream_profile().get_extrinsics_to(right.as_video_stream_profile())
        baseline_m = float(abs(ir_ext.translation[0]))
    except Exception:
        pass

    meta = {
        "device_name": device.get_info(rs.camera_info.name),
        "serial": device.get_info(rs.camera_info.serial_number),
        "firmware": device.get_info(rs.camera_info.firmware_version),
        "product_id": device.get_info(rs.camera_info.product_id),
        "usb_type": device.get_info(rs.camera_info.usb_type_descriptor),
        "color_size": [args.width, args.height],
        "depth_size": [args.width, args.height],
        "fps": args.fps,
        "align": "depth_to_color",
        "depth_unit": "millimeter_uint16_png",
        "note": "color is BGR8 PNG; depth is uint16 millimeters aligned to color",
    }
    calib = {
        "color": _intrinsics_to_dict(color_intr),
        "depth_before_align": _intrinsics_to_dict(depth_intr),
        "depth_to_color_extrinsics": {
            "rotation": [float(x) for x in extrinsics.rotation],
            "translation_m": [float(x) for x in extrinsics.translation],
        },
        "depth_scale_m_per_unit": depth_scale,
        "stereo_baseline_m": baseline_m,
        "aligned": {
            "fx": float(color_intr.fx),
            "fy": float(color_intr.fy),
            "cx": float(color_intr.ppx),
            "cy": float(color_intr.ppy),
            "width": int(color_intr.width),
            "height": int(color_intr.height),
            "baseline_m": baseline_m,
            "depth_scale_m_per_unit": depth_scale,
        },
    }
    (seq_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (seq_dir / "calib.json").write_text(json.dumps(calib, indent=2), encoding="utf-8")

    ts_path = seq_dir / "timestamps.csv"
    ts_file = ts_path.open("w", encoding="utf-8")
    ts_file.write("frame_id,color_t_ns,depth_t_ns\n")

    print(f"[INFO] Recording to {seq_dir}")
    print(f"[INFO] {meta['device_name']} serial={meta['serial']} usb={meta['usb_type']}")
    if "2." in str(meta.get("usb_type", "")):
        print("[WARN] USB2 detected. Prefer a USB3 port/cable for stable 30fps.")

    frame_id = 0
    t_end = time.time() + args.seconds
    try:
        while time.time() < t_end:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())  # Z16 in camera depth units
            # Convert to millimeters for portable uint16 PNG.
            depth_m = depth_raw.astype(np.float32) * depth_scale
            depth_mm = np.clip(np.round(depth_m * 1000.0), 0, 65535).astype(np.uint16)

            stem = f"{frame_id:06d}"
            cv2.imwrite(str(color_dir / f"{stem}.png"), color)
            cv2.imwrite(str(depth_dir / f"{stem}.png"), depth_mm)
            ts_file.write(
                f"{frame_id},{color_frame.get_timestamp() * 1e6:.0f},{depth_frame.get_timestamp() * 1e6:.0f}\n"
            )
            frame_id += 1

            if args.preview:
                depth_vis = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_mm, alpha=0.03), cv2.COLORMAP_JET
                )
                view = np.hstack([color, depth_vis])
                cv2.imshow("realsense color | depth", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Stopped early by user")
                    break

            if frame_id % args.fps == 0:
                print(f"[INFO] frames={frame_id}")
    finally:
        ts_file.close()
        pipeline.stop()
        if args.preview:
            cv2.destroyAllWindows()

    print(f"[OK] Saved {frame_id} frames to {seq_dir}")
    print("[OK] Next: copy this folder to AutoDL, then convert to KITTI-like inputs.")
    return 0 if frame_id > 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
