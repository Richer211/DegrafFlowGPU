#!/usr/bin/env python3
"""Realtime status window: replay (GPU flow) + live (Mac Farneback).

Replay mode (no camera):
  Plays KITTI-like pairs + precomputed Degraf/RAFT flow.

Live mode (Mac + RealSense + conveyor):
  C++ grabber writes latest color/depth to a slot dir; this script polls,
  runs Farneback optical flow, same IDLE/MOVING + MAIN box UI.

Controls (GUI):
  SPACE  pause/resume
  n / →  next frame (replay, when paused)
  p / ←  prev frame (replay, when paused)
  +/-    speed (replay)
  q/ESC  quit

Examples:
  # Replay
  python3 realtime_status.py --mode replay \\
    --seq-kitti /path/to/seq_box_slide_002 \\
    --flow-dir /path/to/realsense_flow/seq_box_slide_002

  # Live (Mac): terminal A
  sudo ./live_grab_rs --out /tmp/degraf_live
  # Live: terminal B
  python3 realtime_status.py --mode live --live-dir /tmp/degraf_live
"""

from __future__ import annotations

import argparse
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from motion_alarm import AlarmConfig, AlarmResult, StickyAlarm, run_alarm
from viz_scene_flow import compute_scene_flow, load_calib, read_kitti_disp, read_kitti_flow


@dataclass
class FramePacket:
    """One processing unit: pair of frames + geometry + flow."""

    index: int
    stem: str
    bgr0: np.ndarray
    bgr1: np.ndarray
    flow: np.ndarray
    disp0: np.ndarray
    disp1: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    baseline: float
    source_tag: str  # replay | live


class FrameSource(ABC):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def get(self, i: int) -> FramePacket: ...


class ReplaySource(FrameSource):
    """Offline source: KITTI-like RealSense export + Degraf flow PNGs."""

    def __init__(self, seq_kitti: Path, flow_dir: Path, count: int = 0):
        self.root = Path(seq_kitti)
        self.flow_dir = Path(flow_dir)
        img_dir = self.root / "training/image_2"
        stems = sorted(p.name.replace("_10.png", "") for p in img_dir.glob("*_10.png"))
        if count > 0:
            stems = stems[:count]
        self.stems = []
        for stem in stems:
            need = [
                img_dir / f"{stem}_10.png",
                img_dir / f"{stem}_11.png",
                self.root / "training/disp_noc_0" / f"{stem}_10.png",
                self.root / "training/disp_noc_1" / f"{stem}_10.png",
                self.flow_dir / f"{stem}_10.png",
                self.root / "calib/training/calib_cam_to_cam" / f"{stem}.txt",
            ]
            if all(p.exists() for p in need):
                self.stems.append(stem)
        if not self.stems:
            raise FileNotFoundError(f"No complete pairs under {self.root} + {self.flow_dir}")

    def __len__(self) -> int:
        return len(self.stems)

    def get(self, i: int) -> FramePacket:
        i = int(np.clip(i, 0, len(self.stems) - 1))
        stem = self.stems[i]
        bgr0 = cv2.imread(str(self.root / "training/image_2" / f"{stem}_10.png"))
        bgr1 = cv2.imread(str(self.root / "training/image_2" / f"{stem}_11.png"))
        disp0 = read_kitti_disp(self.root / "training/disp_noc_0" / f"{stem}_10.png")
        disp1 = read_kitti_disp(self.root / "training/disp_noc_1" / f"{stem}_10.png")
        flow, _ = read_kitti_flow(self.flow_dir / f"{stem}_10.png")
        fx, fy, cx, cy, baseline = load_calib(
            self.root / "calib/training/calib_cam_to_cam" / f"{stem}.txt"
        )
        return FramePacket(
            index=i,
            stem=stem,
            bgr0=bgr0,
            bgr1=bgr1,
            flow=flow,
            disp0=disp0,
            disp1=disp1,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            baseline=baseline,
            source_tag="replay",
        )


def _depth_mm_to_disp(depth_mm: np.ndarray, fx: float, baseline_m: float) -> np.ndarray:
    """Convert depth in millimetres to disparity (pixels)."""
    disp = np.zeros(depth_mm.shape, np.float32)
    m = depth_mm > 1
    z_m = depth_mm[m].astype(np.float32) * 0.001
    disp[m] = (fx * baseline_m) / np.maximum(z_m, 1e-4)
    return disp


def _farneback_flow(bgr0: np.ndarray, bgr1: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Lightweight dense OF for Mac live (no CUDA)."""
    g0 = cv2.cvtColor(bgr0, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2GRAY)
    if scale < 0.999:
        g0s = cv2.resize(g0, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        g1s = cv2.resize(g1, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        g0s, g1s = g0, g1
    # levels=3: mini belt is fast and cannot be slowed, so displacement per pair can be large.
    flow_s = cv2.calcOpticalFlowFarneback(
        g0s, g1s, None, pyr_scale=0.5, levels=3, winsize=21, iterations=3, poly_n=5, poly_sigma=1.1, flags=0
    )
    if scale < 0.999:
        flow = cv2.resize(flow_s, (g0.shape[1], g0.shape[0]), interpolation=cv2.INTER_LINEAR)
        flow *= 1.0 / scale
    else:
        flow = flow_s
    return flow.astype(np.float32)


class LiveSource:
    """Poll live_grab_rs slot directory; Farneback between frames pair_dt apart."""

    def __init__(self, live_dir: Path, of_scale: float = 0.5, poll_hz: float = 30.0, pair_dt: float = 0.08):
        self.root = Path(live_dir)
        self.of_scale = float(of_scale)
        self.poll_dt = 1.0 / max(poll_hz, 1.0)
        self.pair_dt = max(0.05, float(pair_dt))
        self.calib_path = self.root / "calib.json"
        self.color_path = self.root / "color.jpg"
        self.depth_path = self.root / "depth_mm.u16"
        self.id_path = self.root / "frame_id.txt"

        self.fx = self.fy = self.cx = self.cy = 0.0
        self.baseline = 0.05
        self.width = self.height = 0
        self._load_calib(wait=True)

        self.prev_bgr: np.ndarray | None = None
        self.prev_disp: np.ndarray | None = None
        self.prev_t = 0.0
        self.last_id = -1
        self.index = 0

    def _load_calib(self, wait: bool = False) -> None:
        deadline = time.time() + (30.0 if wait else 0.0)
        while True:
            if self.calib_path.exists():
                try:
                    data = json.loads(self.calib_path.read_text())
                    self.fx = float(data["fx"])
                    self.fy = float(data["fy"])
                    self.cx = float(data["cx"])
                    self.cy = float(data["cy"])
                    self.baseline = float(data.get("baseline_m", 0.05))
                    self.width = int(data["width"])
                    self.height = int(data["height"])
                    return
                except Exception:
                    pass
            if time.time() >= deadline:
                raise FileNotFoundError(
                    f"Waiting for calib.json in {self.root}. Start live_grab_rs first."
                )
            time.sleep(0.2)

    def _read_frame_id(self) -> tuple[int, int, int] | None:
        if not self.id_path.exists():
            return None
        try:
            text = self.id_path.read_text().strip().split()
            if len(text) < 3:
                return None
            return int(text[0]), int(text[1]), int(text[2])
        except Exception:
            return None

    def _read_depth(self, w: int, h: int) -> np.ndarray | None:
        try:
            raw = self.depth_path.read_bytes()
            need = w * h * 2
            if len(raw) < need:
                return None
            depth = np.frombuffer(raw[:need], dtype=np.uint16).reshape(h, w)
            return depth
        except Exception:
            return None

    def _read_color(self) -> np.ndarray | None:
        try:
            # OpenCV decodes JPEG → BGR
            img = cv2.imread(str(self.color_path), cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None

    def wait_packet(self, timeout_s: float = 5.0) -> FramePacket | None:
        """Block until a new consecutive pair is ready, or timeout."""
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            meta = self._read_frame_id()
            if meta is None:
                time.sleep(self.poll_dt)
                continue
            fid, w, h = meta
            if fid == self.last_id:
                time.sleep(self.poll_dt)
                continue
            if w != self.width or h != self.height:
                # rare: resolution change — reload calib
                try:
                    self._load_calib(wait=False)
                except Exception:
                    pass

            bgr = self._read_color()
            depth = self._read_depth(w, h)
            if bgr is None or depth is None or bgr.shape[0] != h or bgr.shape[1] != w:
                time.sleep(self.poll_dt)
                continue

            disp = _depth_mm_to_disp(depth, self.fx, self.baseline)
            now = time.time()
            self.last_id = fid

            if self.prev_bgr is None:
                self.prev_bgr = bgr
                self.prev_disp = disp
                self.prev_t = now
                time.sleep(self.poll_dt)
                continue

            # Fast belt: keep the gap short (~0.08s) so the object does not jump out of Farneback range.
            if now - self.prev_t < self.pair_dt:
                time.sleep(self.poll_dt)
                continue

            flow = _farneback_flow(self.prev_bgr, bgr, scale=self.of_scale)
            pkt = FramePacket(
                index=self.index,
                stem=f"live_{fid:06d}",
                bgr0=self.prev_bgr,
                bgr1=bgr,
                flow=flow,
                disp0=self.prev_disp,
                disp1=disp,
                fx=self.fx,
                fy=self.fy,
                cx=self.cx,
                cy=self.cy,
                baseline=self.baseline,
                source_tag="live",
            )
            self.prev_bgr = bgr
            self.prev_disp = disp
            self.prev_t = now
            self.index += 1
            return pkt
        return None


def depth_from_disp(disp, fx, baseline):
    z = np.zeros_like(disp, np.float32)
    m = (disp > 1.0) & (disp < 300)
    z[m] = (fx * baseline) / disp[m]
    return z


def process_packet(
    pkt: FramePacket,
    cfg: AlarmConfig,
    sticky: StickyAlarm,
    live: bool = False,
) -> tuple[AlarmResult, np.ndarray, dict]:
    sf, _ = compute_scene_flow(
        pkt.flow, pkt.disp0, pkt.disp1, pkt.fx, pkt.fy, pkt.cx, pkt.cy, pkt.baseline
    )
    mag_sf = np.linalg.norm(sf, axis=2)
    Z = depth_from_disp(pkt.disp0, pkt.fx, pkt.baseline)
    h, w = Z.shape
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    Y = (vv - pkt.cy) * Z / pkt.fy

    # Live consecutive frames: depth holes on the belt zero out SF — estimate from flow.
    # Sub-2px Farneback jitter on the dark table is noise (~3mm); ignore it.
    flow_mag_px = np.linalg.norm(pkt.flow, axis=2)
    z_fill = np.where(Z > 0.15, Z, 0.45)
    mag_flow = flow_mag_px * z_fill / max(float(pkt.fx), 1.0)
    # Depth on the green belt is full of holes, so scene flow there is 0.
    # Replay uses the same flow-to-metric estimate as live, then drops belt-colored pixels.
    if live or cfg.ignore_belt:
        mag_flow = np.where(flow_mag_px >= 2.0, mag_flow, 0.0)
        mag = np.maximum(mag_sf, mag_flow)
    else:
        mag = mag_sf

    alarm = run_alarm(pkt.bgr0, pkt.bgr1, mag, Z, Y, pkt.disp0, cfg, sticky)
    dbg = {
        "flow_p95_px": float(np.percentile(flow_mag_px, 95)),
        "mag_p95_mm": float(np.percentile(mag, 95) * 1000.0),
        "depth_valid_pct": 100.0 * float((Z > 0.15).mean()),
    }
    return alarm, mag, dbg


def _stamp(canvas: np.ndarray, u, v, cols) -> None:
    h, w = canvas.shape[:2]
    ui = np.clip(np.rint(u).astype(np.int32), 0, w - 1)
    vi = np.clip(np.rint(v).astype(np.int32), 0, h - 1)
    canvas[vi, ui] = cols
    canvas[np.clip(vi + 1, 0, h - 1), ui] = cols
    canvas[vi, np.clip(ui + 1, 0, w - 1)] = cols


def render_scene_cloud(pkt: FramePacket, alarm: AlarmResult, mag: np.ndarray, size: int) -> np.ndarray:
    """Front-view point cloud: RGB, motion points tinted by |scene flow|."""
    canvas = np.zeros((size, size, 3), np.uint8)
    Z = depth_from_disp(pkt.disp0, pkt.fx, pkt.baseline)
    h, w = Z.shape
    stride = 3 if max(h, w) <= 720 else 4
    ys = np.arange(0, h, stride)
    xs = np.arange(0, w, stride)
    zz = Z[np.ix_(ys, xs)]
    valid = (zz > 0.2) & (zz < 1.6)
    if not np.any(valid):
        cv2.putText(canvas, "no depth", (20, size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
        return canvas
    uu, vv = np.meshgrid(xs.astype(np.float32), ys.astype(np.float32))
    X = ((uu - pkt.cx) * zz / pkt.fx)[valid]
    Y = ((vv - pkt.cy) * zz / pkt.fy)[valid]
    Zv = zz[valid]
    cols = pkt.bgr0[np.ix_(ys, xs)][valid]
    mags = mag[np.ix_(ys, xs)][valid]
    moving = alarm.alarm_mask[np.ix_(ys, xs)][valid] if alarm.alarm_mask is not None else np.zeros(mags.shape, bool)

    ys_img = -Y
    xmin, xmax = np.percentile(X, [2, 98])
    ymin, ymax = np.percentile(ys_img, [2, 98])
    span = max(float(xmax - xmin), float(ymax - ymin), 1e-3)
    pad = 0.06 * span
    xmin, xmax, ymin, ymax = xmin - pad, xmax + pad, ymin - pad, ymax + pad
    u = (X - xmin) / (xmax - xmin + 1e-9) * (size - 1)
    v = (1.0 - (ys_img - ymin) / (ymax - ymin + 1e-9)) * (size - 1)
    order = np.argsort(Zv)  # far first
    u, v, cols, mags, moving = u[order], v[order], cols[order], mags[order], moving[order]
    _stamp(canvas, u[~moving], v[~moving], cols[~moving])
    if np.any(moving):
        heat = cv2.applyColorMap(
            np.clip(mags[moving] / 0.04 * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO
        ).reshape(-1, 3)
        tint = (0.35 * cols[moving].astype(np.float32) + 0.65 * heat.astype(np.float32)).astype(np.uint8)
        _stamp(canvas, u[moving], v[moving], tint)
    cv2.rectangle(canvas, (0, 0), (size, 36), (0, 0, 0), -1)
    cv2.putText(canvas, "POINT CLOUD + SCENE FLOW", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    return canvas


def render_panel(
    pkt: FramePacket,
    alarm: AlarmResult,
    fps_show: float,
    paused: bool,
    speed: float,
    mag: np.ndarray | None = None,
    dbg: dict | None = None,
) -> np.ndarray:
    vis = pkt.bgr0.copy()
    color = (0, 220, 0) if alarm.status == "IDLE" else (0, 80, 255)
    for i, (x, y, bw, bh, area, mean_sf_mm) in enumerate(alarm.blobs):
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 3)
        tag = f"MAIN {area}px |SF|~{mean_sf_mm:.0f}mm"
        cv2.putText(vis, tag, (x, max(28, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    card_w = 340
    h, w = vis.shape[:2]
    cloud = render_scene_cloud(pkt, alarm, mag if mag is not None else np.zeros((h, w), np.float32), h)
    dbg = dbg or {}
    card = np.zeros((h, card_w, 3), np.uint8)
    cv2.putText(card, "STATUS", (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)
    cv2.putText(card, alarm.status, (16, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    mean_sf = alarm.blobs[0][5] if alarm.blobs else 0.0
    lines = [
        f"mode:   {pkt.source_tag}",
        f"frame:  {pkt.index}",
        f"|SF|:   {mean_sf:.0f} mm",
        f"p95:    {dbg.get('mag_p95_mm', 0):.1f} mm",
        f"flow:   {dbg.get('flow_p95_px', 0):.1f} px",
        f"depth:  {dbg.get('depth_valid_pct', 0):.0f}%",
        f"raw:    {alarm.raw_pct:.1f}%",
        f"clean:  {alarm.clean_pct:.1f}%",
        f"blobs:  {len(alarm.blobs)}",
        f"fps:    {fps_show:.1f}",
        f"state:  {'PAUSE' if paused else 'PLAY'}",
        "",
        "q quit   SPACE pause",
    ]
    y = 140
    for line in lines:
        cv2.putText(card, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1)
        y += 24

    cv2.rectangle(vis, (0, 0), (w, 44), (0, 0, 0), -1)
    cv2.putText(
        vis,
        f"{pkt.source_tag.upper()} | {alarm.status} | {pkt.stem}",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )
    return np.hstack([vis, cloud, card])


def parse_args():
    p = argparse.ArgumentParser(description="Realtime status window (replay + live)")
    p.add_argument("--mode", choices=("replay", "live"), default="replay")
    p.add_argument("--seq-kitti", default="")
    p.add_argument("--flow-dir", default="")
    p.add_argument("--live-dir", default="/tmp/degraf_live", help="live_grab_rs output slot")
    p.add_argument("--of-scale", type=float, default=0.5, help="Farneback downsample (live)")
    p.add_argument("--pair-dt", type=float, default=0.08, help="seconds between the two live frames (short: belt is fast)")
    p.add_argument("--count", type=int, default=0)
    p.add_argument("--fps", type=float, default=8.0, help="replay playback fps")
    p.add_argument("--loop", action="store_true", help="loop replay")
    p.add_argument("--headless", action="store_true", help="no GUI; process & optional save")
    p.add_argument("--save-dir", default="", help="save rendered panels (esp. headless)")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0=all)")
    p.add_argument("--sf-thresh-m", type=float, default=None, help="default: 0.02 replay / 0.003 live")
    p.add_argument("--min-blob-px", type=int, default=None, help="default: 2000 replay / 400 live")
    p.add_argument("--sticky-frames", type=int, default=None, help="default: 2 replay / 1 live")
    p.add_argument("--left-wall-frac", type=float, default=None)
    p.add_argument("--desk-y0-frac", type=float, default=None)
    p.add_argument("--photo-thresh", type=float, default=None)
    p.add_argument("--debug", action="store_true", help="print live mag/flow stats")
    return p.parse_args()


def _make_alarm_cfg(args) -> AlarmConfig:
    live = args.mode == "live"
    # Live noise floor seen on a still table is ~3mm / <2px. Real belt motion is larger.
    sf = args.sf_thresh_m if args.sf_thresh_m is not None else (0.008 if live else 0.02)
    blob = args.min_blob_px if args.min_blob_px is not None else (1200 if live else 2000)
    sticky = args.sticky_frames if args.sticky_frames is not None else (2 if live else 2)
    left = args.left_wall_frac if args.left_wall_frac is not None else (0.02 if live else 0.22)
    desk = args.desk_y0_frac if args.desk_y0_frac is not None else (0.05 if live else 0.28)
    photo = args.photo_thresh if args.photo_thresh is not None else (8.0 if live else 4.0)
    return AlarmConfig(
        sf_thresh_m=sf,
        min_blob_px=blob,
        sticky_frames=sticky,
        left_wall_frac=left,
        desk_y0_frac=desk,
        photo_thresh=photo,
        border_margin=20 if live else 56,
        morph_close=7 if live else 9,
        release_frames=5 if live else 0,
        live_mode=live,
        ignore_belt=not live,
    )


def run_live(args) -> int:
    live = LiveSource(Path(args.live_dir), of_scale=args.of_scale, pair_dt=args.pair_dt)
    cfg = _make_alarm_cfg(args)
    sticky = StickyAlarm(cfg.sticky_frames, release=cfg.release_frames)
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[OK] source=live  dir={args.live_dir}  pair_dt={args.pair_dt:.2f}s  "
        f"sf>={cfg.sf_thresh_m*1000:.1f}mm  min_blob={cfg.min_blob_px}  hold={cfg.release_frames}"
    )
    print("[OK] keys: SPACE pause | q quit")
    print("[OK] Tip: put object on belt, start conveyor; watch IDLE→MOVING")

    win = "Degraf Status (live)"
    if not args.headless:
        try:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        except Exception as e:
            print(f"[WARN] GUI unavailable ({e}); falling back to --headless")
            args.headless = True

    paused = False
    last_panel = None
    n_done = 0
    t_prev = time.time()
    fps_ema = 0.0

    while True:
        if args.max_frames and n_done >= args.max_frames:
            break

        if paused and last_panel is not None and not args.headless:
            cv2.imshow(win, last_panel)
            key = cv2.waitKey(50) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = False
            continue

        pkt = live.wait_packet(timeout_s=2.0)
        if pkt is None:
            if args.headless:
                print("[WARN] no new live frame (is live_grab_rs running?)")
                continue
            # keep window alive
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" ") and last_panel is not None:
                paused = True
            continue

        alarm, mag, dbg = process_packet(pkt, cfg, sticky, live=True)
        now = time.time()
        dt = max(now - t_prev, 1e-6)
        fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt) if fps_ema > 0 else 1.0 / dt
        t_prev = now
        panel = render_panel(pkt, alarm, fps_ema, paused, speed=1.0, mag=mag, dbg=dbg)
        last_panel = panel
        n_done += 1

        if args.debug and (n_done % 15 == 1):
            print(
                f"[{pkt.stem}] {alarm.status} blobs={len(alarm.blobs)} "
                f"raw={alarm.raw_pct:.1f}% clean={alarm.clean_pct:.1f}% "
                f"flow_p95={dbg['flow_p95_px']:.2f}px mag_p95={dbg['mag_p95_mm']:.1f}mm "
                f"depth_ok={dbg['depth_valid_pct']:.0f}%"
            )

        if save_dir:
            cv2.imwrite(str(save_dir / f"{pkt.stem}.jpg"), panel, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        if args.headless:
            print(f"[{pkt.stem}] {alarm.status} blobs={len(alarm.blobs)} clean={alarm.clean_pct:.1f}%")
            continue

        cv2.imshow(win, panel)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = True

    if not args.headless:
        cv2.destroyAllWindows()
    print(f"[OK] done  processed={n_done}")
    if save_dir:
        print(f"[OK] panels: {save_dir}")
    return 0


def run_replay(args) -> int:
    if not args.seq_kitti or not args.flow_dir:
        print("[ERR] replay needs --seq-kitti and --flow-dir")
        return 2

    src: FrameSource = ReplaySource(Path(args.seq_kitti), Path(args.flow_dir), count=args.count)
    cfg = _make_alarm_cfg(args)
    sticky = StickyAlarm(cfg.sticky_frames)

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[OK] source=replay  frames={len(src)}  seq={args.seq_kitti}")
    print("[OK] keys: SPACE pause | n/p step | +/- speed | q quit")

    idx = 0
    paused = False
    speed = 1.0
    win = "Degraf Status (replay)"
    if not args.headless:
        try:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        except Exception as e:
            print(f"[WARN] GUI unavailable ({e}); falling back to --headless")
            args.headless = True

    n_done = 0
    t_prev = time.time()
    fps_ema = 0.0

    while True:
        if args.max_frames and n_done >= args.max_frames:
            break
        pkt = src.get(idx)
        alarm, mag, dbg = process_packet(pkt, cfg, sticky, live=False)
        now = time.time()
        dt = max(now - t_prev, 1e-6)
        fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt) if fps_ema > 0 else 1.0 / dt
        t_prev = now
        panel = render_panel(pkt, alarm, fps_ema, paused, speed, mag=mag, dbg=dbg)

        if save_dir:
            cv2.imwrite(str(save_dir / f"{pkt.stem}.jpg"), panel, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        if args.headless:
            print(f"[{pkt.stem}] {alarm.status} blobs={len(alarm.blobs)} clean={alarm.clean_pct:.1f}%")
            idx += 1
            n_done += 1
            if idx >= len(src):
                if args.loop:
                    idx = 0
                    sticky = StickyAlarm(cfg.sticky_frames)
                else:
                    break
            continue

        cv2.imshow(win, panel)
        wait_ms = max(1, int(1000.0 / max(args.fps * speed, 0.5)))
        key = cv2.waitKey(0 if paused else wait_ms) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
        elif key in (ord("n"), 83):
            idx = min(idx + 1, len(src) - 1)
            n_done += 1
            continue
        elif key in (ord("p"), 81):
            idx = max(idx - 1, 0)
            continue
        elif key in (ord("+"), ord("=")):
            speed = min(4.0, speed + 0.25)
        elif key in (ord("-"), ord("_")):
            speed = max(0.25, speed - 0.25)

        if not paused:
            idx += 1
            n_done += 1
            if idx >= len(src):
                if args.loop:
                    idx = 0
                    sticky = StickyAlarm(cfg.sticky_frames)
                else:
                    break

    if not args.headless:
        cv2.destroyAllWindows()
    print(f"[OK] done  processed={n_done}")
    if save_dir:
        print(f"[OK] panels: {save_dir}")
    return 0


def main() -> int:
    args = parse_args()
    if args.mode == "live":
        return run_live(args)
    return run_replay(args)


if __name__ == "__main__":
    raise SystemExit(main())
