#!/usr/bin/env python3
"""Shared IDLE/MOVING alarm logic for demo export and realtime status UI."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class AlarmConfig:
    sf_thresh_m: float = 0.02
    min_blob_px: int = 2000
    border_margin: int = 56
    left_wall_frac: float = 0.22
    desk_y0_frac: float = 0.28
    photo_thresh: float = 4.0
    morph_close: int = 9
    merge_gap_px: int = 40
    keep_top_blobs: int = 1
    sticky_frames: int = 2  # need N consecutive MOVING raw hits to show MOVING
    release_frames: int = 0  # after MOVING, keep it this many misses (live anti-flicker)
    live_mode: bool = False  # softer table / ROI for Mac consecutive-frame live
    ignore_belt: bool = False  # replay: drop motion whose color matches the belt


@dataclass
class AlarmResult:
    status: str  # IDLE | MOVING
    blobs: list  # list of (x,y,w,h,area,mean_sf_mm)
    alarm_mask: np.ndarray
    raw_pct: float
    clean_pct: float
    mag: np.ndarray


def _bbox_gap(a, b) -> float:
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
    if not blobs:
        return [], []
    items = list(blobs)
    used = [False] * len(items)
    merged, groups = [], []
    for i, _ in enumerate(items):
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
                if any(_bbox_gap(items[g], bj) <= merge_gap_px for g in group):
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
    ordered = sorted(zip(merged, groups), key=lambda t: t[0][4], reverse=True)
    return [m for m, _ in ordered], [g for _, g in ordered]


def table_mask(
    Z: np.ndarray,
    Y: np.ndarray,
    vv: np.ndarray,
    disp0: np.ndarray,
    h: int,
    live_mode: bool = False,
) -> np.ndarray:
    valid = (Z > 0.15) & (Z < 2.0) & (disp0 > 0.5)
    desk_samples = Z[valid & (vv > h * 0.35)]
    if desk_samples.size < 1000:
        desk_samples = Z[valid]
    if desk_samples.size == 0:
        table = np.zeros_like(valid)
    else:
        desk = float(np.median(desk_samples))
        table = valid & (np.abs(Z - desk) < 0.28) & (Y > np.percentile(Y[valid], 35))
        table |= valid & (Z > desk - 0.35) & (Z < desk + 0.15) & (vv > h * 0.25) & (vv < h * 0.95)

    # Live overhead belt: depth often holes on green belt → fall back to center band.
    if live_mode and float(table.mean()) < 0.03:
        hh, ww = Z.shape
        band = np.zeros((hh, ww), dtype=bool)
        band[int(0.18 * hh) : int(0.82 * hh), int(0.02 * ww) : int(0.98 * ww)] = True
        return band
    return table


def filter_motion_blobs(
    raw_moving: np.ndarray,
    mag: np.ndarray,
    gray0: np.ndarray,
    gray1: np.ndarray,
    cfg: AlarmConfig,
):
    h, w = raw_moving.shape
    mask = raw_moving.copy()
    m = cfg.border_margin
    if m > 0:
        mask[:m, :] = False
        mask[-m:, :] = False
        mask[:, :m] = False
        mask[:, -m:] = False
    x0 = int(w * max(0.0, min(0.45, cfg.left_wall_frac)))
    y0 = int(h * max(0.0, min(0.6, cfg.desk_y0_frac)))
    if x0 > 0:
        mask[:, :x0] = False
    if y0 > 0:
        mask[:y0, :] = False

    photo = np.abs(gray0.astype(np.float32) - gray1.astype(np.float32))
    mask &= photo >= cfg.photo_thresh
    # Live: ignore near-black table. IR speckle there jitters without a real object.
    if cfg.live_mode:
        mask &= gray0 >= 35

    if cfg.morph_close > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_close, cfg.morph_close))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    raw_blobs = []
    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < cfg.min_blob_px:
            continue
        x = int(stats[lab, cv2.CC_STAT_LEFT])
        y = int(stats[lab, cv2.CC_STAT_TOP])
        bw = int(stats[lab, cv2.CC_STAT_WIDTH])
        bh = int(stats[lab, cv2.CC_STAT_HEIGHT])
        mean_sf_mm = float(mag[labels == lab].mean() * 1000.0)
        raw_blobs.append((x, y, bw, bh, area, mean_sf_mm, lab))

    merged, groups = merge_blobs(raw_blobs, labels, mag, cfg.merge_gap_px)
    if cfg.keep_top_blobs > 0:
        merged = merged[: cfg.keep_top_blobs]
        groups = groups[: cfg.keep_top_blobs]
    kept = np.zeros_like(mask)
    for labs in groups:
        kept[np.isin(labels, labs)] = True
    return kept, merged


class StickyAlarm:
    """Turn on after N hits. Live can hold MOVING across brief misses."""

    def __init__(self, n: int = 2, release: int = 0):
        self.n = max(1, int(n))
        self.release = max(0, int(release))
        self.hit = 0
        self.hold = 0
        self.status = "IDLE"
        self.last_blobs: list = []

    def update(self, raw_moving: bool, blobs: list | None = None) -> str:
        if raw_moving:
            self.hit = min(self.n, self.hit + 1)
            if blobs:
                self.last_blobs = list(blobs)
            if self.hit >= self.n:
                self.status = "MOVING"
                self.hold = self.release
        else:
            self.hit = 0
            if self.status == "MOVING" and self.hold > 0:
                self.hold -= 1
            else:
                self.status = "IDLE"
                self.last_blobs = []
        return self.status


def belt_object_blobs(bgr: np.ndarray, mag: np.ndarray, cfg: AlarmConfig):
    """Motion on the green belt surface is the belt itself. Keep only other colors."""
    h, w = mag.shape
    empty = np.zeros((h, w), dtype=bool)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.int16)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    rough = (hue >= 35) & (hue <= 90) & (sat >= 40) & (val >= 40)
    row_frac = rough.mean(axis=1)
    rows = np.where(row_frac > 0.25)[0]
    if rows.size == 0:
        return empty, []
    splits = np.where(np.diff(rows) > 4)[0]
    seg = max(np.split(rows, splits + 1), key=len)
    y0, y1 = int(seg.min()), int(seg.max())
    col_frac = rough[y0 : y1 + 1].mean(axis=0)
    cols = np.where(col_frac > 0.45)[0]
    if cols.size == 0:
        return empty, []
    col_splits = np.where(np.diff(cols) > 8)[0]
    col_seg = max(np.split(cols, col_splits + 1), key=len)
    x0, x1 = int(col_seg.min()), int(min(col_seg.max(), int(0.82 * w)))
    if x1 <= x0 + 20:
        return empty, []
    band = np.zeros((h, w), dtype=bool)
    band[y0 : y1 + 1, x0:x1] = True
    hues = hue[band & rough]
    med = int(np.median(hues)) if hues.size else 55
    belt = band & (np.abs(hue - med) <= 10) & (sat >= 30) & (val >= 30)
    moving = (band & (mag >= cfg.sf_thresh_m) & ~belt).astype(np.uint8)
    moving = cv2.morphologyEx(
        moving, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    moving = cv2.morphologyEx(
        moving, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    )
    n, labels, stats, _ = cv2.connectedComponentsWithStats(moving, connectivity=8)
    raw = []
    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        bw = int(stats[lab, cv2.CC_STAT_WIDTH])
        bh = int(stats[lab, cv2.CC_STAT_HEIGHT])
        if area < max(700, cfg.min_blob_px // 2):
            continue
        # Belt rails show up as a long strip a few pixels tall.
        if bh < 32 or bw / max(bh, 1) > 8:
            continue
        if bw > 0.45 * w and bh > 0.7 * max(1, y1 - y0):
            continue
        x = int(stats[lab, cv2.CC_STAT_LEFT])
        y = int(stats[lab, cv2.CC_STAT_TOP])
        sel = labels == lab
        mean_sf_mm = float(mag[sel].mean() * 1000.0)
        raw.append((x, y, bw, bh, area, mean_sf_mm, lab))
    raw.sort(key=lambda t: t[4], reverse=True)
    if cfg.keep_top_blobs > 0:
        raw = raw[: cfg.keep_top_blobs]
    kept = np.zeros((h, w), dtype=bool)
    blobs = []
    for x, y, bw, bh, area, mean_sf_mm, lab in raw:
        kept[labels == lab] = True
        blobs.append((x, y, bw, bh, area, mean_sf_mm))
    return kept, blobs


def run_alarm(
    bgr0: np.ndarray,
    bgr1: np.ndarray,
    mag: np.ndarray,
    Z: np.ndarray,
    Y: np.ndarray,
    disp0: np.ndarray,
    cfg: AlarmConfig,
    sticky: StickyAlarm | None = None,
) -> AlarmResult:
    h, w = bgr0.shape[:2]
    vv = np.arange(h, dtype=np.float32)[:, None]
    table = table_mask(Z, Y, vv, disp0, h, live_mode=cfg.live_mode)
    raw_moving = table & (mag >= cfg.sf_thresh_m)
    gray0 = cv2.cvtColor(bgr0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2GRAY)
    if cfg.ignore_belt:
        alarm_mask, blobs = belt_object_blobs(bgr0, mag, cfg)
    else:
        alarm_mask, blobs = filter_motion_blobs(raw_moving, mag, gray0, gray1, cfg)
    table_n = max(int(table.sum()), 1)
    raw_pct = 100.0 * float(raw_moving.sum()) / table_n
    clean_pct = 100.0 * float(alarm_mask.sum()) / table_n
    raw_status = len(blobs) >= 1
    if sticky is None:
        status = "MOVING" if raw_status else "IDLE"
    else:
        status = sticky.update(raw_status, blobs)
        if status == "MOVING" and not blobs and sticky.last_blobs:
            blobs = list(sticky.last_blobs)
    return AlarmResult(status, blobs, alarm_mask, raw_pct, clean_pct, mag)
