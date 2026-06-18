#!/usr/bin/env python3
# encoding: utf-8
"""Render debug trajectory figures for FLAGGED trials (likely wrong-piece / incongruent / reach
without target touch), into each participant's output folder (`<id>/debug_figures/`).

Reuses the doc-figure board alignment (crop only the BLACK outer border, keep the white cell
margins, map the non-black band to data [0,1]) and the block-3 180deg rotation handling, so the
trajectory AND the markers land on the right cells. Adds markers for the TARGET, the TOUCHED and
the gaze-VALIDATED cells (as small annotations, not a giant legend), plus a world-video thumbnail
at the press, so a human can VISUALLY validate the wrong-piece detection and label dubious cases.

Usage:
  debug_flagged_trials.py -p 002 --output_root .../OutputData_vX --data_root .../InputData
"""
import os
import sys
import pickle
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2 as cv

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
from src.core.version import __version__
from src.core.ArucoBoardHandler import detectAllArucos


def crop_to_board(im, margin=0.22):
    """Crop the world frame to the board region (ArUco bounding box + margin) so the touch is
    visible, instead of the small board lost in the whole frame. Falls back to the full frame if
    no markers are found (e.g. the hand is covering them)."""
    try:
        corners, ids = detectAllArucos(im)
    except Exception:
        corners = None
    if not corners:
        return im
    pts = np.concatenate([np.asarray(c).reshape(-1, 2) for c in corners], axis=0)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    H, W = im.shape[:2]
    x0, y0 = max(0, int(x0 - mx)), max(0, int(y0 - my))
    x1, y1 = min(W, int(x1 + mx)), min(H, int(y1 + my))
    return im[y0:y1, x0:x1] if (x1 - x0 > 20 and y1 - y0 > 20) else im

BOARD_IMG = os.path.join(REPO_ROOT, 'docs/media/TableroSinBordes.png')
DEFAULT_OUTPUT_ROOT = os.environ.get('EEHA_OUTPUT_ROOT', f'/media/quique/EXTERNAL_USB1/BusquedaVisualAnalysis/OutputData_v{__version__}')
DEFAULT_DATA_ROOT = os.environ.get('EEHA_DATA_ROOT', '/media/quique/EXTERNAL_USB1/BusquedaVisualAnalysis/InputData')


def board_extent(bgr):
    """imshow extent + aspect so the NON-BLACK board area (cells + their white margins) maps to
    data [0,1], cropping only the black outer border (same as the doc trajectory figures)."""
    ys, xs = np.where(bgr.max(axis=2) > 50)
    H, W = bgr.shape[:2]
    fx0, fx1 = xs.min() / W, xs.max() / W
    fy0, fy1 = ys.min() / H, ys.max() / H
    ext = [(0 - fx0) / (fx1 - fx0), (1 - fx0) / (fx1 - fx0),
           (1 - fy0) / (fy1 - fy0), (0 - fy0) / (fy1 - fy0)]
    aspect = ((fy1 - fy0) * H) / ((fx1 - fx0) * W)
    return ext, aspect


def fit_cell_to_norm(seq):
    """Linear map (col,row) -> (norm_x,norm_y) fitted from the trial's gaze samples. Captures the
    border offset AND the block-3 rotated frame automatically (the coords live in the trial frame),
    so a marker for a cell the gaze never settled on still lands correctly. None if too few cells."""
    cols, rows, nx, ny = [], [], [], []
    for s in seq:
        bc, nc = s.get('board_coord'), s.get('norm_board_coord')
        if bc and bc[0] is not None and nc and nc[0] is not None:
            cols.append(bc[0]); rows.append(bc[1]); nx.append(nc[0]); ny.append(nc[1])
    if len(set(cols)) < 2 or len(set(rows)) < 2:
        return None
    px = np.polyfit(cols, nx, 1)
    py = np.polyfit(rows, ny, 1)
    return lambda col, row: (float(px[0] * col + px[1]), float(py[0] * row + py[1]))


def read_frame_exact(video, f):
    """Read the EXACT frame f. cv.VideoCapture seek (CAP_PROP_POS_FRAMES) lands on the nearest
    keyframe, so a single set+read can return a different (often clean) frame; seek a little
    before and grab forward to the precise one."""
    start = max(0, int(f) - 20)
    video.set(cv.CAP_PROP_POS_FRAMES, start)
    for _ in range(int(f) - start):
        if not video.grab():
            break
    ok, im = video.read()
    return im if ok else None


def gaze_norm_on_cell(seq, cell):
    """Mean normalized gaze position over the samples that fell on `cell` -- the EXACT marker
    spot for the validated piece (the gaze was really there), no cell->norm estimation."""
    if not cell:
        return None
    xs, ys = [], []
    for s in seq:
        bc, nc = s.get('board_coord'), s.get('norm_board_coord')
        if bc and tuple(bc) == tuple(cell) and nc and nc[0] is not None:
            xs.append(nc[0]); ys.append(nc[1])
    return (float(np.mean(xs)), float(np.mean(ys))) if xs else None


def is_flagged(m):
    """A trial worth a debug look: an off-target / no-touch anomaly (a completed reach with no
    target touch) or an incongruent bump."""
    if m.get('error_type') in ('off_target', 'no_touch'):
        return True
    b = m.get('bump')
    if isinstance(b, dict) and b.get('congruent') is False:
        return True
    return False


def render(pid, m, name, block, trial, bgr, ext, aspect, video, out_dir):
    seq = m.get('sequence', [])
    raw = [(s['norm_board_coord'][0], s['norm_board_coord'][1]) for s in seq
           if s.get('norm_board_coord') and s['norm_board_coord'][0] is not None]
    if not raw:
        return None
    fit = fit_cell_to_norm(seq)
    rotated = (block == 3)                          # block 3 is presented 180deg-rotated (by design)

    def disp(p):                                    # rotated trial frame -> canonical image overlay
        return (1.0 - p[0], 1.0 - p[1]) if rotated else (p[0], p[1])

    tnorm = None
    tn = m.get('target_norm_coord')
    if tn:
        t = tn[0] if isinstance(tn[0], (list, tuple)) else tn
        if t and t[0] is not None:
            tnorm = (float(t[0]), float(t[1]))

    def cellnorm(cell):
        return fit(cell[0], cell[1]) if (cell and fit is not None) else None

    touched_n = cellnorm(m.get('touched_cell'))
    valid_n = gaze_norm_on_cell(seq, m.get('gaze_validated_cell')) or cellnorm(m.get('gaze_validated_cell'))

    bg = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
    has_v = video is not None
    fig, axes = plt.subplots(1, 2 if has_v else 1, figsize=(13.5 if has_v else 7, 4.8))
    ax = axes[0] if has_v else axes
    ax.imshow(bg, extent=ext, aspect='auto')
    ax.set_aspect(aspect)
    pts = [disp(p) for p in raw]
    x = [p[0] for p in pts]; y = [p[1] for p in pts]
    cmap = LinearSegmentedColormap.from_list('c', ['blue', 'cyan', 'lime', 'yellow', 'red'])
    ax.plot(x, y, '-', color='#222', lw=0.6, alpha=0.4, zorder=2)
    ax.scatter(x, y, c=np.linspace(0, 1, len(pts)), cmap=cmap, s=15,
               edgecolor='white', linewidths=0.3, zorder=3)

    def mark(p, color, label):
        if not p:
            return
        dp = disp(p)
        ax.scatter([dp[0]], [dp[1]], s=230, facecolors='none', edgecolors=color,
                   linewidths=2.2, zorder=5)
        ax.text(dp[0], dp[1] - 0.045, label, ha='center', va='bottom', fontsize=7,
                color=color, fontweight='bold', zorder=6,
                bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.7))

    mark(tnorm, 'limegreen', f'target: {name}')
    mark(touched_n, 'red', f"touched: {m.get('touched_piece')}")
    mark(valid_n, 'deepskyblue', f"gaze: {m.get('gaze_validated_piece')}")
    ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{pid}  b{block}t{trial}  target={name}  |  error_type={m.get('error_type')}"
                 + ("  |  ROT180" if rotated else ""), fontsize=9)

    if has_v:
        f = m.get('cell_occ_peak_frame') or m.get('target_touch_capture') or m.get('end_capture')
        if f is not None:
            im = read_frame_exact(video, f)
            if im is not None:
                axes[1].imshow(cv.cvtColor(crop_to_board(im), cv.COLOR_BGR2RGB))
        axes[1].set_title(f"board @ frame {f} (press)", fontsize=9)
        axes[1].axis('off')

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"flagged_b{block}t{trial}_{name}.png")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description="Debug figures for flagged (likely error) trials.")
    ap.add_argument('-p', dest='participant', required=True)
    ap.add_argument('--output_root', default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument('--data_root', default=DEFAULT_DATA_ROOT)
    ap.add_argument('-t', dest='topic', default='gaze')
    args = ap.parse_args()

    pdir = os.path.join(args.output_root, args.topic, args.participant)
    with open(os.path.join(pdir, f'data_{args.participant}.pkl'), 'rb') as f:
        d = pickle.load(f)
    bgr = cv.imread(BOARD_IMG)
    ext, aspect = board_extent(bgr)
    vpath = os.path.join(args.data_root, args.participant, 'world.mp4')
    video = cv.VideoCapture(vpath) if os.path.exists(vpath) else None
    out_dir = os.path.join(pdir, 'debug_figures')

    n = 0
    for k, t in sorted(d['trials_data'].items()):
        if k == 'latest':
            continue
        name = list(t.keys())[0]
        m = list(t.values())[0]
        if not isinstance(m, dict) or m.get('init_capture', -1) == -1 \
           or name.startswith(('missing', 'transition', 'end_of')):
            continue
        if is_flagged(m):
            render(args.participant, m, name, k[0], k[1], bgr, ext, aspect, video, out_dir)
            n += 1
    if video:
        video.release()
    print(f"[debug_flagged::{args.participant}] {n} figuras -> {out_dir}"
          + ("" if video else "  (sin vídeo: solo trayectoria)"))


if __name__ == '__main__':
    main()
