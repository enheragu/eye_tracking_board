#!/usr/bin/env python3
# encoding: utf-8
"""
Per-participant gaze drift calibration from the in-video calibration panels.

Many recordings show a 9-dot (3x3) calibration panel between blocks that Pupil
never registered as a calibration (the operator did not always trigger it). The
recorded gaze therefore keeps an uncorrected, time-varying bias (drift). This
tool recovers that drift offline, purely from data we already have:

  world.mp4 + gaze.pldata + notify.pldata + blinks.pldata

It detects the panels, locates the 9 dots (3x3 homography), clusters the gaze
into fixations, assigns each to its dot, and measures the residual gaze - dot at
each panel over time. A leave-one-panel-out cross-validation (bootstrapped)
decides, per participant, whether a time-interpolated, calibration-segment-aware
offset actually reduces the error out of sample. If it does, an artifact is
written to calibration/gaze/<id>.json for GazeCorrectionHandler to apply at
runtime; if not, the artifact records apply=false (identity, never worsen).

Residuals are stored in NORMALISED image coordinates with origin at the TOP-LEFT
(i.e. the same frame the dataloader uses after its 1-Y flip), so the runtime
handler can subtract them directly.
"""
import os
import sys
import json
import argparse
import hashlib
import inspect

import cv2 as cv
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from src.core.deps.file_methods import load_pldata_file
from src.core.utils import log

try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None

W, H = 1280, 720
CONF_THRESHOLD = 0.6
BLINK_MARGIN_S = 0.05
SCAN_STEP = 15            # world-frame step for panel detection
MIN_PANEL_DOTS = 7
MIN_FIX_SAMPLES = 15
FIX_DISPERSION_PX = 20.0
ASSIGN_RADIUS_PX = 70.0
N_BOOTSTRAP = 500
# Gate: adopt the correction only if the gain is RELIABLY positive (the 5th
# percentile of the bootstrapped gain clears the margin) and the base error is
# worth correcting. Margin 0 = "lower bound > 0" (correct solid gains, never
# worsen); raise it to be more conservative.
GATE_MIN_GAIN_LOWER = 0.0
GATE_MIN_BASE_PX = 12.0


# --------------------------------------------------------------------------- #
#  Data loading
# --------------------------------------------------------------------------- #
def load_blink_intervals(data_path):
    try:
        bl = load_pldata_file(directory=data_path, topic="blinks", track_progress_in_console=False)
    except (FileNotFoundError, OSError):
        return np.zeros((0, 2))
    events = sorted((d["timestamp"], d["type"]) for d in bl.data)
    intervals, open_ts = [], None
    for ts, kind in events:
        if kind == "onset":
            open_ts = ts
        elif kind == "offset" and open_ts is not None:
            intervals.append((open_ts - BLINK_MARGIN_S, ts + BLINK_MARGIN_S))
            open_ts = None
    return np.array(intervals) if intervals else np.zeros((0, 2))


def load_gaze(data_path):
    """Returns (timestamps, xy_pixels_topleft) for valid, non-blink samples."""
    gz = load_pldata_file(directory=data_path, topic="gaze", track_progress_in_console=False)
    raw = np.array([(d["timestamp"], d["norm_pos"][0] * W, (1 - d["norm_pos"][1]) * H, d["confidence"])
                    for d in gz.data])
    blinks = load_blink_intervals(data_path)
    blink_mask = np.zeros(len(raw), bool)
    for s, e in blinks:
        blink_mask |= (raw[:, 0] >= s) & (raw[:, 0] <= e)
    keep = (raw[:, 3] > CONF_THRESHOLD) & (~blink_mask)
    sel = raw[keep]
    sel = sel[np.argsort(sel[:, 0])]
    return sel[:, 0], sel[:, 1:3]


def load_calibration_times(data_path, world_t0):
    """Registered calibration setup times (relative to world start)."""
    try:
        nf = load_pldata_file(directory=data_path, topic="notify", track_progress_in_console=False)
    except (FileNotFoundError, OSError):
        return []
    return sorted(d["timestamp"] - world_t0
                  for t, d in zip(nf.topics, nf.data) if "calibration.setup" in t)


# --------------------------------------------------------------------------- #
#  Panel / dot detection
# --------------------------------------------------------------------------- #
def detect_dots(gray):
    """Dark dots on a light background, exposure-adaptive."""
    th = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 31, 8)
    contours, _ = cv.findContours(th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    dots = []
    for c in contours:
        area = cv.contourArea(c)
        if area < 8 or area > 600:
            continue
        per = cv.arcLength(c, True)
        if per == 0 or 4 * np.pi * area / (per * per) < 0.6:
            continue
        (x, y), r = cv.minEnclosingCircle(c)
        rr = int(max(r, 3))
        xi, yi = int(x), int(y)
        center = gray[max(0, yi - rr):yi + rr, max(0, xi - rr):xi + rr]
        ring = gray[max(0, yi - 4 * rr):yi + 4 * rr, max(0, xi - 4 * rr):xi + 4 * rr]
        if center.size and ring.size and center.mean() < ring.mean() - 12 and ring.mean() > 90:
            dots.append((x, y))
    return np.array(dots)


def fit_grid(cands):
    """Fit a 3x3 grid to dot candidates and return all 9 sub-pixel positions."""
    if KMeans is None or len(cands) < 7 or len(cands) > 14:
        return None
    try:
        kx = KMeans(3, n_init=5, random_state=0).fit(cands[:, 0:1])
        ky = KMeans(3, n_init=5, random_state=0).fit(cands[:, 1:2])
    except Exception:
        return None
    col_order = np.argsort(kx.cluster_centers_.ravel())
    row_order = np.argsort(ky.cluster_centers_.ravel())
    col_map = {v: i for i, v in enumerate(col_order)}
    row_map = {v: i for i, v in enumerate(row_order)}
    grid_pos = np.array([[col_map[kx.labels_[k]], row_map[ky.labels_[k]]] for k in range(len(cands))], float)
    if len(set(grid_pos[:, 0])) < 3 or len(set(grid_pos[:, 1])) < 3:
        return None
    hom, _ = cv.findHomography(grid_pos, cands, cv.RANSAC, 6.0)
    if hom is None:
        return None
    canonical = np.array([[c, r] for r in range(3) for c in range(3)], float)
    pred = cv.perspectiveTransform(canonical.reshape(-1, 1, 2), hom).reshape(-1, 2)
    if pred[:, 0].min() < -60 or pred[:, 0].max() > W + 60 or pred[:, 1].min() < -60 or pred[:, 1].max() > H + 60:
        return None
    return pred


def grid_present(dots):
    return dots is not None and len(dots) >= MIN_PANEL_DOTS and np.ptp(dots[:, 0]) > 150 and np.ptp(dots[:, 1]) > 100


def detect_panels(cap, fps, n_frames):
    """Time segments (seconds) where the 9-dot panel is shown."""
    times, counts, i = [], [], 0
    while i < n_frames:
        cap.set(cv.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            break
        dots = detect_dots(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
        counts.append(len(dots) if grid_present(dots) else 0)
        times.append(i / fps)
        i += SCAN_STEP
    times, counts = np.array(times), np.array(counts)
    active = counts >= MIN_PANEL_DOTS
    segments, start = [], None
    for k in range(len(active)):
        if active[k] and start is None:
            start = k
        if (not active[k] or k == len(active) - 1) and start is not None:
            end = k - 1 if not active[k] else k
            segments.append([times[start], times[end]])
            start = None
    merged = []
    for seg in segments:
        if merged and seg[0] - merged[-1][1] < 6:
            merged[-1][1] = seg[1]
        else:
            merged.append(seg)
    return [(a, b) for a, b in merged if b - a >= 4]


# --------------------------------------------------------------------------- #
#  Fixations and per-dot residuals
# --------------------------------------------------------------------------- #
def fixations(ts, xy, a, b):
    mask = (ts >= a) & (ts <= b)
    sub_ts, sub_xy = ts[mask], xy[mask]
    if len(sub_ts) < MIN_FIX_SAMPLES:
        return []
    out, cur = [], [0]
    for k in range(1, len(sub_ts)):
        cx, cy = np.median(sub_xy[cur, 0]), np.median(sub_xy[cur, 1])
        if np.hypot(sub_xy[k, 0] - cx, sub_xy[k, 1] - cy) < FIX_DISPERSION_PX and sub_ts[k] - sub_ts[cur[-1]] < 0.25:
            cur.append(k)
        else:
            if len(cur) >= MIN_FIX_SAMPLES:
                out.append((np.median(sub_ts[cur]), np.median(sub_xy[cur, 0]), np.median(sub_xy[cur, 1])))
            cur = [k]
    if len(cur) >= MIN_FIX_SAMPLES:
        out.append((np.median(sub_ts[cur]), np.median(sub_xy[cur, 0]), np.median(sub_xy[cur, 1])))
    return out


def grid_at(cap, world_ts, n_frames, abs_ts):
    fi = int(np.searchsorted(world_ts, abs_ts))
    cap.set(cv.CAP_PROP_POS_FRAMES, max(0, min(fi, n_frames - 1)))
    ok, frame = cap.read()
    return fit_grid(detect_dots(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))) if ok else None


def measure_residuals(data_path):
    """Per-panel per-dot residual (gaze - dot) in pixels, plus calibration times."""
    world_ts = np.sort(np.load(os.path.join(data_path, "world_timestamps.npy")))
    t0 = world_ts[0]
    ts, xy = load_gaze(data_path)
    calib_times = load_calibration_times(data_path, t0)

    cap = cv.VideoCapture(os.path.join(data_path, "world.mp4"))
    fps = cap.get(cv.CAP_PROP_FPS)
    n_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    panels = detect_panels(cap, fps, n_frames)

    rows = []   # (panel_idx, panel_time_s, res_x_px, res_y_px)
    for pi, (a, b) in enumerate(panels):
        by_dot = {}
        for (fts, fx, fy) in fixations(ts, xy, a + t0, b + t0):
            grid = grid_at(cap, world_ts, n_frames, fts)
            if grid is None:
                continue
            dist = np.hypot(grid[:, 0] - fx, grid[:, 1] - fy)
            j = int(dist.argmin())
            if dist[j] < ASSIGN_RADIUS_PX:
                by_dot.setdefault(j, []).append((grid[j, 0], grid[j, 1], fx - grid[j, 0], fy - grid[j, 1]))
        for j, vals in by_dot.items():
            v = np.array(vals)   # cols: dot_x_px, dot_y_px, res_x_px, res_y_px
            rows.append((pi, (a + b) / 2.0,
                         float(np.median(v[:, 0])), float(np.median(v[:, 1])),
                         float(np.median(v[:, 2])), float(np.median(v[:, 3]))))
    cap.release()
    # rows columns: panel_idx, panel_time_s, dot_x_px, dot_y_px, res_x_px, res_y_px
    return np.array(rows) if rows else np.zeros((0, 6)), panels, calib_times


# --------------------------------------------------------------------------- #
#  Cross-validation gate (segment-aware, bootstrapped)
# --------------------------------------------------------------------------- #
def segment_of(t, calib_times):
    seg = 0
    for i, c in enumerate(calib_times):
        if t >= c:
            seg = i
    return seg


def cv_gain(rows, calib_times):
    """Leave-one-panel-out CV of a segment-aware, time-interpolated offset.
    Returns (base_err, corr_err) arrays of per-point errors."""
    P = rows[:, 0].astype(int)
    PT = rows[:, 1]
    R = rows[:, 4:6]
    panels = sorted(set(P))
    pan_mean = {p: R[P == p].mean(axis=0) for p in panels}
    pan_time = {p: PT[P == p][0] for p in panels}
    pan_seg = {p: segment_of(pan_time[p], calib_times) for p in panels}
    base, corr = [], []
    base_vec, corr_vec = [], []   # per-axis residual vectors [rx, ry] (for the anisotropic bias cov)
    for held in panels:
        held_mask = P == held
        if held_mask.sum() < 2:
            continue
        others = [p for p in panels if p != held and pan_seg[p] == pan_seg[held]]
        if not others:
            ox = oy = 0.0   # no in-segment reference -> no correction for this panel
        else:
            ot = np.array([pan_time[p] for p in others])
            order = np.argsort(ot)
            ox = np.interp(pan_time[held], ot[order], [pan_mean[p][0] for p in others])
            oy = np.interp(pan_time[held], ot[order], [pan_mean[p][1] for p in others])
        base += list(np.hypot(R[held_mask, 0], R[held_mask, 1]))
        corr += list(np.hypot(R[held_mask, 0] - ox, R[held_mask, 1] - oy))
        base_vec += [[float(rx), float(ry)] for rx, ry in R[held_mask]]
        corr_vec += [[float(rx - ox), float(ry - oy)] for rx, ry in R[held_mask]]
    return np.array(base), np.array(corr), np.array(base_vec), np.array(corr_vec)


def gate_decision(base, corr):
    """Bootstrap the relative median gain; adopt only if its lower bound clears
    the margin and the base error is worth correcting."""
    if len(base) < 6:
        return False, {}
    base_med = float(np.median(base))
    corr_med = float(np.median(corr))
    point_gain = 1.0 - corr_med / base_med if base_med > 0 else 0.0
    rng = np.random.default_rng(0)
    gains = []
    n = len(base)
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        bm, cm = np.median(base[idx]), np.median(corr[idx])
        gains.append(1.0 - cm / bm if bm > 0 else 0.0)
    gains = np.array(gains)
    lo, hi = float(np.percentile(gains, 5)), float(np.percentile(gains, 95))
    apply = (lo > GATE_MIN_GAIN_LOWER) and (base_med >= GATE_MIN_BASE_PX)
    stats = {"base_px": round(base_med, 2), "corr_px": round(corr_med, 2),
             "gain": round(point_gain, 3), "gain_lo": round(lo, 3), "gain_hi": round(hi, 3),
             "n_points": int(n)}
    return apply, stats


# --------------------------------------------------------------------------- #
#  Per-sample measurement-noise model (Phase 0 uncertainty)
# --------------------------------------------------------------------------- #
def _load_gaze_conf(data_path):
    """Like load_gaze but KEEPS the per-sample confidence: returns (ts, xy_px_topleft,
    conf) for valid (conf > threshold), non-blink samples."""
    gz = load_pldata_file(directory=data_path, topic="gaze", track_progress_in_console=False)
    raw = np.array([(d["timestamp"], d["norm_pos"][0] * W, (1 - d["norm_pos"][1]) * H, d["confidence"])
                    for d in gz.data])
    blinks = load_blink_intervals(data_path)
    bmask = np.zeros(len(raw), bool)
    for s, e in blinks:
        bmask |= (raw[:, 0] >= s) & (raw[:, 0] <= e)
    sel = raw[(raw[:, 3] > CONF_THRESHOLD) & (~bmask)]
    sel = sel[np.argsort(sel[:, 0])]
    return sel[:, 0], sel[:, 1:3], sel[:, 3]


def _cov_ellipse(xy):
    """2x2 covariance of (de-biased) jitter -> (sigma_major_px, sigma_minor_px, angle_deg).
    The angle is that of the MAJOR axis, in image degrees."""
    c = np.cov(xy.T)
    w, v = np.linalg.eigh(c)               # eigenvalues ascending
    major = float(np.sqrt(max(w[1], 0.0)))
    minor = float(np.sqrt(max(w[0], 0.0)))
    ang = float(np.degrees(np.arctan2(v[1, 1], v[0, 1])))
    return major, minor, ang


def _isotonic_nonincreasing(y):
    """Clamp a sequence to be non-increasing (running-min). Higher confidence cannot
    mean MORE error, so the high-confidence anomaly is capped to the previous value."""
    out = list(y)
    for i in range(1, len(out)):
        if out[i] > out[i - 1]:
            out[i] = out[i - 1]
    return out


def measure_uncertainty(data_path, panels=None):
    """Per-sample measurement-noise model from the calibration panels (Phase 0).
    In image pixels: (a) base 2x2 jitter covariance (anisotropic/tilted, de-biased per
    dot); (b) a MONOTONE confidence->variance factor; (c) a coarse radial (eccentricity)
    spatial->variance factor. The two factors average ~1 over the pool, so per sample:

        Sigma_meas(conf, pos) = jitter_cov * conf_var_factor(conf) * spatial_var_factor(ecc)

    The accuracy/bias (drift) is handled separately (panel_offsets / base_px)."""
    ts, xy, conf = _load_gaze_conf(data_path)
    if len(ts) < 50:
        return None
    cap = cv.VideoCapture(os.path.join(data_path, "world.mp4"))
    fps = cap.get(cv.CAP_PROP_FPS)
    n = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    world_ts = np.sort(np.load(os.path.join(data_path, "world_timestamps.npy")))
    t0 = world_ts[0]
    if panels is None:
        panels = detect_panels(cap, fps, n)
    raw = []   # conf, dot_x, dot_y, res_x, res_y, key(panel*100+dot)
    for pi, (a, b) in enumerate(panels):
        grid = grid_at(cap, world_ts, n, (a + b) / 2.0 + t0)
        if grid is None:
            continue
        m = (ts >= a + t0) & (ts <= b + t0)
        for x, y, c in zip(xy[m, 0], xy[m, 1], conf[m]):
            d = np.hypot(grid[:, 0] - x, grid[:, 1] - y)
            j = int(d.argmin())
            if d[j] < ASSIGN_RADIUS_PX:
                raw.append((c, grid[j, 0], grid[j, 1], x - grid[j, 0], y - grid[j, 1], pi * 100 + j))
    cap.release()
    if len(raw) < 50:
        return None
    raw = np.array(raw)
    # de-bias per (panel, dot) -> isolate jitter (precision) from the dot's offset (bias)
    jit = raw[:, 3:5].copy()
    for k in np.unique(raw[:, 5]):
        mk = raw[:, 5] == k
        jit[mk] -= jit[mk].mean(axis=0)
    confs = raw[:, 0]
    ecc = np.hypot(raw[:, 1] - W / 2.0, raw[:, 2] - H / 2.0)
    pool_var = float((jit[:, 0] ** 2 + jit[:, 1] ** 2).mean())
    if pool_var <= 0:
        return None
    major, minor, ang = _cov_ellipse(jit)
    # confidence -> variance factor (relative to pool, monotone non-increasing)
    cc, cf = [], []
    for lo, hi in [(0.6, 0.75), (0.75, 0.85), (0.85, 0.93), (0.93, 0.98), (0.98, 1.01)]:
        mk = (confs >= lo) & (confs < hi)
        if mk.sum() < 30:
            continue
        cc.append(round((lo + hi) / 2.0, 3))
        cf.append(float((jit[mk, 0] ** 2 + jit[mk, 1] ** 2).mean()) / pool_var)
    cf = [round(f, 3) for f in _isotonic_nonincreasing(cf)]
    # radial (eccentricity) -> variance factor (relative to pool)
    qs = np.quantile(ecc, [0.0, 0.33, 0.66, 1.0])
    sc, sf = [], []
    for i in range(3):
        mk = (ecc >= qs[i]) & (ecc <= qs[i + 1])
        if mk.sum() < 30:
            continue
        sc.append(round(float((qs[i] + qs[i + 1]) / 2.0), 1))
        sf.append(round(float((jit[mk, 0] ** 2 + jit[mk, 1] ** 2).mean()) / pool_var, 3))
    return {"jitter_cov_px": {"sigma_major": round(major, 2), "sigma_minor": round(minor, 2),
                              "angle_deg": round(ang, 1)},
            "conf_var_factor": {"conf": cc, "factor": cf},
            "spatial_var_factor": {"model": "radial_ecc_px", "ecc": sc, "factor": sf},
            "jitter_rms_px": round(float(np.sqrt(pool_var)), 2),
            "n_samples": int(len(raw)),
            "units": "image px (1280x720); Sigma_meas = jitter_cov * conf_var_factor(conf) * spatial_var_factor(ecc)"}


def calibration_fingerprint():
    """Stable hash of the whole calibration computation chain (source + key constants).
    Stored in the artifact so the runtime can recompute when the METHOD changes, not
    just when the file is missing. Covers the residual/drift path AND the uncertainty
    model (helpers included, since getsource of one function misses its callees)."""
    # NOTE: inspect.getsource only captures each function's OWN body, not its callees, so
    # EVERY function whose source affects the artifact must be listed explicitly (incl. the
    # data loaders and the panel/dot detection helpers). If you add a helper to this chain,
    # add it here too, or the fingerprint will not detect the change.
    fns = [measure_uncertainty, _load_gaze_conf, _cov_ellipse, _isotonic_nonincreasing,
           measure_residuals, cv_gain, gate_decision, build_artifact, fixations,
           detect_panels, grid_at, calibrate_participant,
           load_blink_intervals, load_gaze, load_calibration_times,
           detect_dots, fit_grid, grid_present, segment_of]
    parts = []
    for fn in fns:
        try:
            parts.append(inspect.getsource(fn))
        except Exception:
            parts.append(getattr(fn, "__name__", "?"))
    consts = f"{W}|{H}|{CONF_THRESHOLD}|{ASSIGN_RADIUS_PX}|{MIN_FIX_SAMPLES}|{FIX_DISPERSION_PX}|{GATE_MIN_GAIN_LOWER}|{GATE_MIN_BASE_PX}"
    raw = "||".join(parts) + "||" + consts
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- #
#  Artifact
# --------------------------------------------------------------------------- #
def build_artifact(participant, rows, panels, calib_times, apply, stats, uncertainty=None):
    P = rows[:, 0].astype(int)
    PT = rows[:, 1]
    R = rows[:, 4:6]
    # Per-panel offset in NORMALISED top-left image coords (what the runtime subtracts)
    panel_offsets = []
    for p in sorted(set(P)):
        m = R[P == p]
        panel_offsets.append({"t": round(float(PT[P == p][0]), 3),
                              "off_x": round(float(m[:, 0].mean()) / W, 6),
                              "off_y": round(float(m[:, 1].mean()) / H, 6),
                              "seg": segment_of(float(PT[P == p][0]), calib_times),
                              "n_dots": int((P == p).sum())})
    return {"participant": participant,
            "apply": bool(apply),
            "calib_segment_times": [round(c, 3) for c in calib_times],
            "panel_offsets": panel_offsets,
            "stats": stats,
            "uncertainty": uncertainty,
            "fingerprint": calibration_fingerprint(),
            "units": "normalized image coords, origin top-left; subtract off from [x, 1-y]"}


def calibrate_participant(participant, input_root, out_dir):
    data_path = os.path.join(input_root, participant)
    if not os.path.isdir(data_path):
        log(f"[gaze_calibration] {participant}: input folder not found ({data_path})")
        return None
    rows, panels, calib_times = measure_residuals(data_path)
    if len(rows) == 0 or len(set(rows[:, 0].astype(int))) < 3:
        log(f"[gaze_calibration] {participant}: not enough panel data ({len(panels)} panels) -> skipped")
        artifact = {"participant": participant, "apply": False, "stats": {"reason": "insufficient_panels"},
                    "uncertainty": None, "fingerprint": calibration_fingerprint()}
    else:
        base, corr, base_vec, corr_vec = cv_gain(rows, calib_times)
        apply, stats = gate_decision(base, corr)
        uncertainty = measure_uncertainty(data_path, panels=panels)
        # anisotropic accuracy/drift covariance (2x2 second moment of the residual
        # vectors: corrected if the drift gate passed, else raw). This is the bias floor.
        vec = corr_vec if apply else base_vec
        if uncertainty is not None and len(vec) > 0:
            bc = (vec.T @ vec) / len(vec)
            uncertainty["bias_cov_px"] = [[float(bc[0, 0]), float(bc[0, 1])],
                                          [float(bc[1, 0]), float(bc[1, 1])]]
        artifact = build_artifact(participant, rows, panels, calib_times, apply, stats, uncertainty)
        uj = uncertainty["jitter_rms_px"] if uncertainty else "n/a"
        log(f"[gaze_calibration] {participant}: panels={len(panels)} "
            f"base={stats['base_px']}px corr={stats['corr_px']}px gain={stats['gain']:+.0%} "
            f"[{stats['gain_lo']:+.0%},{stats['gain_hi']:+.0%}] -> apply={apply}; jitter_rms={uj}px")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{participant}.json")
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
    return artifact


def main():
    ap = argparse.ArgumentParser(description="Per-participant gaze drift calibration from video panels.")
    ap.add_argument("-p", "--participant", action="append", help="participant id (repeatable; default: all in input root)")
    ap.add_argument("--input-root", required=True, help="folder with per-participant InputData subfolders")
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "calibration", "gaze"), help="output folder for <id>.json artifacts")
    args = ap.parse_args()

    participants = args.participant
    if not participants:
        participants = sorted(d for d in os.listdir(args.input_root)
                              if os.path.isdir(os.path.join(args.input_root, d)))
    for participant in participants:
        try:
            calibrate_participant(participant, args.input_root, args.out_dir)
        except Exception as exc:  # noqa: BLE001 - one bad participant must not stop the batch
            log(f"[gaze_calibration] {participant}: ERROR {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
