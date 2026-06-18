#!/usr/bin/env python3
# encoding: utf-8
"""
Documentation figures for the v1.3 gaze drift correction (docs/documentacion_tecnica.md
section 13). Reproducible from the per-participant artifacts in calibration/gaze/
(and, for the residual quiver, by recomputing the measurement from the input video).

Produces three views (one separate, full-resolution figure per participant — no
mosaics, so resolution is not lost):
  1. drift_time_<id>  - the measured offset (px) per panel over time; vertical
                        lines mark registered recalibrations (the bias resets
                        there -> sawtooth).
  2. grid_shift_<id>  - a regular grid displaced by the correction at each panel
                        time, coloured by time (amplified for visibility). The
                        applied correction is a time-varying translation, so the
                        grid shifts rigidly; the colour shows the temporal drift.
  3. quiver_<id>      - the measured residual (gaze - dot) at the 9 panel dots,
                        panel by panel: the field is ~uniform and grows over time.

Examples use anonymous participant ids only (no named recordings).
"""
import os
import sys
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

W, H = 1280, 720
DPI = 200   # separate high-resolution figures, never mosaics
CALIB_DIR = os.path.join(REPO_ROOT, "calibration", "gaze")
# Default gallery: a big clean drift (1 calib), a sawtooth with resets (5 calib),
# a moderate/erratic one, and a gated-off case (identity) for contrast.
DEFAULT_GALLERY = ["044", "054", "064", "058"]


def _load(participant, calib_dir):
    with open(os.path.join(calib_dir, f"{participant}.json")) as f:
        return json.load(f)


def fig_drift_time(participant, calib_dir, out_path):
    """One figure PER participant (not a mosaic) to keep full resolution."""
    art = _load(participant, calib_dir)
    po = art["panel_offsets"]
    t = [d["t"] for d in po]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0, color="gray", lw=0.6)
    ax.plot(t, [d["off_x"] * W for d in po], "-o", color="#1f77b4", label="off x (px)")
    ax.plot(t, [d["off_y"] * H for d in po], "-s", color="#d62728", label="off y (px)")
    for c in art.get("calib_segment_times", []):
        if c > 0:
            ax.axvline(c, color="green", ls="--", lw=1, alpha=0.7)
    st = art.get("stats", {})
    tag = "APLICA" if art.get("apply") else "identidad"
    gain = st.get("gain")
    gtxt = f", ganancia {gain:+.0%}" if gain is not None else ""
    ax.set_title(f"Deriva de calibración en el tiempo — {participant} ({tag}{gtxt})\n"
                 "verde = recalibración registrada", fontsize=11)
    ax.set_xlabel("tiempo (s)"); ax.set_ylabel("offset (px)")
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out_path}")


def fig_grid_shift(participant, calib_dir, out_path, amp=8):
    art = _load(participant, calib_dir)
    po = art["panel_offsets"]
    gx, gy = np.meshgrid(np.linspace(120, W - 120, 7), np.linspace(90, H - 90, 5))
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.scatter(gx, gy, s=14, color="black", zorder=3, label="rejilla original")
    tmax = max(d["t"] for d in po) or 1.0
    for d in po:
        col = cm.viridis(d["t"] / tmax)
        sx = gx + amp * d["off_x"] * W
        sy = gy + amp * d["off_y"] * H
        ax.scatter(sx, sy, s=10, color=col, alpha=0.8)
        for i in range(gx.shape[0]):
            for j in range(gx.shape[1]):
                ax.plot([gx[i, j], sx[i, j]], [gy[i, j], sy[i, j]], color=col, lw=0.5, alpha=0.5)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal")
    sm = cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, tmax)); sm.set_array([])
    fig.colorbar(sm, ax=ax, label="tiempo (s)")
    ax.set_title(f"{participant}: desplazamiento de la rejilla por la corrección (×{amp} amplificado)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=DPI); plt.close(fig)
    print(f"wrote {out_path}")


def fig_quiver(participant, input_root, out_path):
    from src.tools.gaze_calibration import measure_residuals
    rows, _, _ = measure_residuals(os.path.join(input_root, participant))
    if len(rows) == 0:
        print(f"{participant}: no panel data, skipping quiver")
        return
    pans = sorted(set(rows[:, 0].astype(int)))
    n = len(pans)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.0))
    axes = np.atleast_1d(axes)
    for ax, pi in zip(axes, pans):
        m = rows[rows[:, 0].astype(int) == pi]
        ax.quiver(m[:, 2], m[:, 3], m[:, 4], m[:, 5], angles="xy", scale_units="xy",
                  scale=0.5, color="#d62728", width=0.012)
        ax.scatter(m[:, 2], m[:, 3], s=12, color="black")
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal")
        ax.set_title(f"t={m[0, 1]:.0f}s", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{participant}: residuo medido gaze→punto en los 9 puntos, panel a panel (flecha ×2)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92]); fig.savefig(out_path, dpi=DPI); plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Drift-correction documentation figures.")
    ap.add_argument("--calib-dir", default=CALIB_DIR, help="folder with <id>.json artifacts")
    ap.add_argument("--input-root", default=None, help="InputData root (needed only for the quiver)")
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "docs", "media", "documentation"))
    ap.add_argument("--gallery", nargs="+", default=DEFAULT_GALLERY, help="participants for drift_time")
    ap.add_argument("--quiver", nargs="+", default=["044", "058"], help="participants for the quiver")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for p in args.gallery:
        try:
            fig_drift_time(p, args.calib_dir, os.path.join(args.out_dir, f"drift_time_{p}.png"))
            fig_grid_shift(p, args.calib_dir, os.path.join(args.out_dir, f"drift_grid_shift_{p}.png"))
        except (FileNotFoundError, KeyError) as exc:
            print(f"{p}: skipped ({exc})")
    if args.input_root:
        for p in args.quiver:
            fig_quiver(p, args.input_root, os.path.join(args.out_dir, f"drift_quiver_{p}.png"))
    else:
        print("(--input-root not given: skipping quiver figures)")


if __name__ == "__main__":
    main()
