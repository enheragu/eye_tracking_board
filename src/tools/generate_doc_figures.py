#!/usr/bin/env python3
# encoding: utf-8

"""
    Regenerates the documentation figures in docs/media/documentation/ from a REAL run, so
    they stay consistent with the current visualization (gaze marker, overlays, colours) and
    can be reproduced on demand instead of being captured by hand.

    It runs process_video.py with the debug visualization over a reference trial and dumps the
    debug render at chosen WORLD frames (via --dump_frames, reproducible regardless of
    frame-skipping), captures gaze-classification close-ups, the colour/undistortion
    before-after pairs, and two data plots (occlusion-over-time and the phase timeline). Plots
    with text are written in BOTH Spanish and English (the English one gets an `_eng` suffix).
    Every figure is a SEPARATE full/native-resolution PNG (no mosaics).

    Usage:
        python3 src/tools/generate_doc_figures.py
"""

import os
import re
import sys
import shutil
import argparse
import subprocess

import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from src.core.ArucoBoardHandler import ARUCOColorCorrection, detectAllArucos
from src.core.DistortionHandler import DistortionHandler
from src.core.version import __version__

DEFAULT_DATA_ROOT = os.environ.get('EEHA_DATA_ROOT', '/media/quique/EXTERNAL_USB1/BusquedaVisualAnalysis/InputData')
# Default to the CURRENT version's output (never hardcode a version tag): figures are
# regenerated from the run that matches the code that draws them.
DEFAULT_OUTPUT_ROOT = os.environ.get('EEHA_OUTPUT_ROOT',
                                     f'/media/quique/EXTERNAL_USB1/BusquedaVisualAnalysis/OutputData_v{__version__}')
MEDIA = os.path.join(REPO_ROOT, 'docs', 'media', 'documentation')
CALIB = os.path.join(REPO_ROOT, 'calibration', 'camera_calib.json')

# Trajectory gallery: varied target pieces, each a single RICH-path trial (participant, (block,trial),
# piece), read straight from the reprocessed output. Picked as the trial with the most search gaze
# for that piece (one per colour/shape for variety).
# All from 049, and NONE from block 3: block 3 is presented with the board ROTATED 180° (by design,
# the SAME for every participant — 220/220 trials), so its gaze/target coords live in the rotated
# frame (internally consistent — cells/touch are correct — but they would overlay flipped on the
# canonical board background TableroSinBordes.png). Blocks 0–2,4,5 are upright and overlay correctly.
TRAJECTORY_GALLERY = [
    ('049', (5, 6), 'green_triangle'),  ('049', (4, 0), 'red_circle'),
    ('049', (1, 5), 'blue_triangle'),   ('049', (5, 2), 'yellow_hexagon'),
    ('049', (0, 1), 'red_hexagon'),     ('049', (5, 8), 'yellow_triangle'),
]

# Reference trial for the figures: participant 049, green_hexagon (block 0 / trial 7).
PARTICIPANT = '049'
SEGMENT = (9120, 9320)
REF_TRIAL_NAME = 'green_hexagon'
# Phase timeline: trial (5,0) green_hexagon -- textbook ordering (search<found<motor<touch<exit),
# panel 9236 -> next panel 9363, so the whole cycle 'panel shown -> next panel' is well defined.
TIMELINE_TRIAL = (5, 0)
# Bump + touch-mask trial: (5,5) yellow_circle -- a CLEAN complete reach with a strong, clearly
# bell-shaped target occlusion (fT~0.75) and a visible whole-board rise; non-triangle. Its touch
# masks (and the inter-trial gap up to the next panel at 9797) make the cleanest illustration.
CLEAN_TOUCH_TRIAL = (5, 5)
CLEAN_TOUCH_SEGMENT = (9700, 9790)
# gaze_anticipada (pre_start, COUNTS): the EARLY green_hexagon (block 0 trial 7) has a pre_start gaze
# on the red-hexagon cell at ~frame 1324. Rendered from the block-0 start (frame 400) so the board
# reference grid is warm (6 prior trials warm it up): a cold start mislabels this same gaze as
# not_board — an artefact the real reprocess (from frame 0) does not produce. Verified pre_start 1322-1325.
ANTICIPADA_SOURCE = ('049', (400, 1340), 1324)
# Debug-render figures (full canvas): name -> WORLD frame inside the matching state
STATE_FIGURES = {
    'estado_1_panel':    9242,   # get_test_name        : sample panel shown over the board
    # estado_2_retirada (test_start_execution, panel removal) is taken from the WARM anticipatory
    # run instead: a cold start mislabels its pre_start gaze as not_board (blue). See main().
    'estado_3_busqueda': 9278,   # test_execution       : search, gaze on cells + cenital PiP
    'estado_4_motora':   9288,   # test_motor_recovery  : hand on the board, reaching the piece
    'proceso_overview':  9278,   # reused as the pipeline overview
    'marca_target_found': 9274,  # first gaze on the target cell (target highlighted in the PiP)
}
# Figures from the FULL-RES cenital warp (grid + target/control occlusion ROIs)
WARP_FIGURES = {
    'reproyeccion_celdas': 9278,  # clean board: rejilla + objetivo + control
    'oclusion_areas':      9288,  # reach: la mano oclúye el objetivo (occl alto), control a 0
}
# Sub-marks (entra/toca/sale) need a trial whose target is FAR from the hand-entry edge, so the
# three frames look clearly different (in the 049 reference the target sits by the edge and the
# touch lands right after the onset). 064 block 0 / trial 2 (green_triangle) has a CENTRAL target
# (norm ~0.56,0.5) and a long, visible reach; the frames are its motor_onset/target_touch/hand_exit.
SUBMARK_SOURCE = ('054', (1050, 1200))   # blk0/tr3 blue_circle: long reach AND a clear withdraw
SUBMARK_FIGURES = {'submarca_entra': 1117, 'submarca_toca': 1148, 'submarca_sale': 1166}  # onset/touch/exit
COLOR_FRAME = 9278               # clean board frame for colour / undistortion before-after
# Gaze close-ups: scanned by the unified marker core colour (BGR), cropped at native res
GAZE_CASES = {  # name -> (BGR core colour, frame-selection strategy, avoid_panel)
    'gaze_tablero':    ((0, 255, 0),   'most',     True),    # execution -> cuenta (tablero despejado)
}
# gaze_anticipada (pre_start, COUNTS) and gaze_fuera (not_board, does NOT count) are taken from
# VERIFIED frames instead of a colour scan: a cold scan mislabels pre_start as not_board, and the
# 049 reference trial has no genuine off-board gaze at all (its gaze never leaves the board).
GAZE_PANEL_MAGENTA = ((255, 0, 255), 'most')   # on_panel -> no cuenta (needs a trial that looks at it)
PANEL_SOURCE = ('007', (1560, 1720))  # 007 yellow_hexagon ~1642: on_panel gaze on a 2-marker panel
                                       # (a clean polygon — single-marker panels like blue_circle look skewed)
GAZE_FUERA_SOURCE = ('009', (16050, 16400), 16364)  # 009: gaze clearly ABOVE the board during search (real not_board)
GAZE_PANELNO_SOURCE = ('002', (800, 1400), 1366)   # 002: gaze on the sample panel as it is withdrawn (no count)
GAZE_CROP = (300, 220)

# Bilingual strings for the two data plots
STR = {
    'es': {'fT_title': 'Oclusión del objetivo vs control en el tiempo (049, hexágono verde)',
           'fT_target': 'oclusión objetivo (fT)', 'fT_ctrl': 'oclusión control (fC)',
           'fT_thr': 'umbral rojo (0.13)', 'fT_x': 'frame (World)', 'fT_y': 'fracción ocluida',
           'tl_title': 'Fases de un trial y sus marcas (049, hexágono verde)', 'tl_x': 'frame (World)',
           'm_found': 'objetivo visto', 'm_in': 'mano entra', 'm_touch': 'toque', 'm_out': 'mano sale',
           'm_panel': 'aparece panel', 'm_next': 'aparece panel sig.',
           'p_search': 'búsqueda', 'p_verif': 'verificación', 'p_motor': 'motora (alcance)', 'p_with': 'retirada',
           'p_panel': 'panel de muestra', 'p_inter': 'intervalo · panel siguiente'},
    'en': {'fT_title': 'Target vs control occlusion over time (049, green hexagon)',
           'fT_target': 'target occlusion (fT)', 'fT_ctrl': 'control occlusion (fC)',
           'fT_thr': 'red threshold (0.13)', 'fT_x': 'frame (World)', 'fT_y': 'occluded fraction',
           'tl_title': 'Trial phases and their marks (049, green hexagon)', 'tl_x': 'frame (World)',
           'm_found': 'target seen', 'm_in': 'hand in', 'm_touch': 'touch', 'm_out': 'hand out',
           'm_panel': 'panel appears', 'm_next': 'next panel appears',
           'p_search': 'search', 'p_verif': 'verification', 'p_motor': 'motor (reach)', 'p_with': 'withdraw',
           'p_panel': 'sample panel', 'p_inter': 'inter-trial · next panel'},
}
MARK_COL = {'m_found': '#3b82f6', 'm_in': '#e0a040', 'm_touch': '#d33333', 'm_out': '#4a9933',
            'm_panel': '#8855cc', 'm_next': '#8855cc'}
STR_TRAJ = {
    'es': {'title': 'Recorrido de la mirada sobre el tablero (049, busca hexágono verde)',
           'target': 'objetivo', 'leg': {'pre_start': 'anticipada', 'search': 'búsqueda',
           'verification': 'verificación', 'motor': 'motora', 'withdraw': 'retirada',
           'on_panel': 'sobre panel', 'not_board': 'fuera', 'blank': 'tapada'}},
    'en': {'title': 'Gaze path over the board (049, searching green hexagon)',
           'target': 'target', 'leg': {'pre_start': 'anticipatory', 'search': 'search',
           'verification': 'verification', 'motor': 'motor', 'withdraw': 'withdraw',
           'on_panel': 'on panel', 'not_board': 'off board', 'blank': 'covered'}},
}
PHASE_COL = {'pre_start': '#e0a040', 'search': '#3b82f6', 'verification': '#8b5cf6',
             'motor': '#4a9933', 'withdraw': '#d33333', 'on_panel': '#d63bd6',
             'not_board': '#8c564b', 'blank': '#999999'}   # not_board=brown, distinto del azul de búsqueda
# Phase BANDS (timeline + bump): LIGHT TINTS of the same hues PHASE_COL uses for the gaze path, so a
# phase is the same colour family across every plot (search=blue, verification=purple, motor=green,
# withdraw=red). 'panel' (sample shown) and 'inter' (between trials) have no gaze equivalent.
BAND_COL = {'panel': '#f3d9b0', 'search': '#d6e4fb', 'verification': '#e2d6fa',
            'motor': '#d8efd0', 'withdraw': '#fbdada', 'inter': '#ececec'}


def trajectoryFigure(seq_csv, pkl_path, trial_name, lang, block_trial=None, out_name='trayectoria_mirada', title=None, show_cov=True):
    """Gaze path of one trial over the board image, coloured by phase, with the target marked.
    With block_trial=(block,trial) it isolates ONE specific instance of the piece (a participant can
    repeat a piece across blocks); otherwise it uses every sample of trial_name. With show_cov, the
    per-sample uncertainty ellipse (board-norm covariance) is drawn faintly around each gaze."""
    import csv as _csv
    import ast as _ast
    import pickle as _pickle
    from matplotlib.patches import Ellipse
    if not os.path.isfile(seq_csv):
        print(f'  WARNING: sequence CSV missing; skipping {out_name}')
        return
    # gaze over the panel / over a still-covered cell is NOT board-cell gaze (the person is
    # looking at the panel surface), so it is not projected onto the trajectory.
    exclude_phases = {'on_panel', 'blank'}
    pts, phases, frames = [], [], []
    for r in _csv.DictReader(open(seq_csv)):
        if r['trial_name'] != trial_name:
            continue
        if block_trial is not None and (int(r['block_index']), int(r['trial_index'])) != block_trial:
            continue
        if r['Phase'] in exclude_phases:
            continue
        try:
            nc = _ast.literal_eval(r['Board norm Coord'])
        except (ValueError, SyntaxError):
            continue
        if nc[0] is None or nc[1] is None:
            continue
        pts.append((float(nc[0]), float(nc[1]))); phases.append(r['Phase']); frames.append(int(r['Frame_N']))
    if not pts:
        print(f'  WARNING: no gaze for {out_name}')
        return
    target = None
    cov_by_frame = {}
    if os.path.isfile(pkl_path):
        d = _pickle.load(open(pkl_path, 'rb'))
        cands = [d['trials_data'][block_trial]] if (block_trial in d.get('trials_data', {})) \
            else [tm for tm in d['trials_data'].values() if list(tm.keys())[0] == trial_name]
        for tm in cands:
            bm = list(tm.values())[0]
            tn = bm.get('target_norm_coord')
            if tn:
                t = tn[0] if isinstance(tn[0], (list, tuple)) else tn   # flat or nested
                if t and t[0] is not None:
                    target = t
            for s in bm.get('sequence', []):
                if s.get('norm_board_cov') is not None:
                    cov_by_frame[s['frame']] = s['norm_board_cov']
            break
    s = STR_TRAJ[lang]
    _bgr = cv.imread(os.path.join(REPO_ROOT, 'docs', 'media', 'TableroSinBordes.png'))
    bg = cv.cvtColor(_bgr, cv.COLOR_BGR2RGB)
    # Align the board image with the normalized cell grid: the picture has a BLACK outer border, so
    # the actual board area (cells + their WHITE margins — the margin is part of the cell, it absorbs
    # eye-tracker error) fills only ~[0.025,0.974]x[0.039,0.960] of it. Map that NON-BLACK band to
    # data [0,1] so the markers land on the cells AND the white margins stay (crop only the black).
    _ys, _xs = np.where(_bgr.max(axis=2) > 50)
    _H, _W = bg.shape[:2]
    _fx0, _fx1 = _xs.min() / _W, _xs.max() / _W
    _fy0, _fy1 = _ys.min() / _H, _ys.max() / _H
    _ext = [(0 - _fx0) / (_fx1 - _fx0), (1 - _fx0) / (_fx1 - _fx0),
            (1 - _fy0) / (_fy1 - _fy0), (0 - _fy0) / (_fy1 - _fy0)]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.imshow(bg, extent=_ext, aspect='auto')
    # Keep the CELLS square: force the data box to the board's real width:height (the cell grid in
    # px), so a square cell renders square regardless of the figure size or how much room the legend
    # takes (aspect='auto' was stretching them to fill the axes).
    ax.set_aspect(((_fy1 - _fy0) * _H) / ((_fx1 - _fx0) * _W))
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    ax.plot(xs, ys, '-', color='#222', lw=0.8, alpha=0.45, zorder=2)
    # per-sample uncertainty ellipses (1-sigma, board-norm covariance): a faint cloud
    # that shows how far each gaze could actually be -> why a single cell is ambiguous.
    if show_cov:
        n_ell = 0
        for (mx, my), fr in zip(pts, frames):
            cov = cov_by_frame.get(fr)
            if cov is None:
                continue
            w, v = np.linalg.eigh(np.array(cov, float))
            ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
            ax.add_patch(Ellipse((mx, my), 2 * np.sqrt(max(w[1], 0)), 2 * np.sqrt(max(w[0], 0)),
                                  angle=ang, fill=False, edgecolor='#1f77b4', lw=0.6, alpha=0.16, zorder=2.4))
            n_ell += 1
    for ph in sorted(set(phases)):
        idx = [i for i, p in enumerate(phases) if p == ph]
        ax.scatter([xs[i] for i in idx], [ys[i] for i in idx], s=14, marker='x', zorder=3,
                   c=PHASE_COL.get(ph, '#888'), label=s['leg'].get(ph, ph), linewidths=0.8)
    if target:
        ax.scatter([target[0]], [target[1]], s=320, facecolors='none', edgecolors='k',
                   linewidths=2.2, zorder=4)
        ax.text(target[0], target[1] - 0.05, s['target'], ha='center', fontsize=8, fontweight='bold')
    ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title or s['title']); ax.legend(fontsize=7, loc='center left', bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout(); fig.savefig(os.path.join(MEDIA, f'{out_name}{_suffix(lang)}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {out_name}{_suffix(lang)}.png OK ({len(pts)} muestras)')


def panelFigure(debug_video):
    """Crop the sample panel with its detection outline (the cyan polygon) from a get_test_name
    frame: shows HOW the stimulus is identified, without gaze clutter."""
    if not os.path.isfile(debug_video):
        print('  WARNING: debug video missing; skipping panel detection figure')
        return
    cap = cv.VideoCapture(debug_video)
    best_n, best_fr, best_xy = 0, None, None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        mask = ((fr[:, :, 0].astype(int) < 30) & (np.abs(fr[:, :, 1].astype(int) - 255) < 25) &
                (np.abs(fr[:, :, 2].astype(int) - 255) < 25))   # panel polygon = (0,255,255)
        Hm, Wm = mask.shape
        mask[int(Hm * 0.58):, int(Wm * 0.62):] = False          # ignore the bottom-right cenital PiP
        n = int(mask.sum())
        if n > best_n:
            ys, xs = np.where(mask)
            best_n, best_fr, best_xy = n, fr.copy(), (xs, ys)
    cap.release()
    if best_fr is None or best_n < 50:
        print('  WARNING: no panel polygon found; skipping panel figure')
        return
    xs, ys = best_xy
    H, W = best_fr.shape[:2]
    x0, x1 = max(0, xs.min() - 30), min(W, xs.max() + 30)
    y0, y1 = max(0, ys.min() - 30), min(H, ys.max() + 30)
    cv.imwrite(os.path.join(MEDIA, 'deteccion_panel.png'), best_fr[y0:y1, x0:x1])
    print(f'  deteccion_panel.png  <- panel polygon crop ({x1-x0}x{y1-y0})')


def _suffix(lang):
    return '' if lang == 'es' else '_eng'


def runViz(participant, segment, data_root, tmp_out, dump=None, trace=False):
    cmd = [sys.executable, os.path.join(REPO_ROOT, 'src', 'process_video.py'),
           '-p', participant, '-t', 'gaze', '-v', '--no_window', '--slow_analysis',
           '--start_frame', str(segment[0]), '--end_frame', str(segment[1]),
           '--data_root', data_root, '--output_root', tmp_out]
    if dump:
        cmd += ['--dump_frames', ','.join(str(f) for f in sorted(set(dump)))]
    env = dict(os.environ, EEHA_TRACE_TOUCH='1') if trace else None
    return subprocess.run(cmd, env=env, capture_output=trace, text=True)


def copyDumps(dump_dir, mapping, prefix):
    for name, frame in mapping.items():
        src = os.path.join(dump_dir, f'{prefix}_{frame}.png')
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(MEDIA, f'{name}.png'))
            print(f'  {name}.png  <- frame {frame}')
        else:
            print(f'  WARNING: missing {prefix} for {name} (frame {frame})')


def gazeCrop(debug_video, name, bgr, strategy='most', avoid_panel=False):
    """Crop a legible native-resolution example of a gaze class from a debug video. The frame
    is chosen by `strategy`: 'most' (most marker pixels), 'earliest' (first frame with the
    marker — for the anticipatory case, where the panel is still being removed), or 'edge'
    (marker centroid farthest from the image centre — for a clearer off-board example). With
    `avoid_panel`, frames where a sample panel is being detected (its yellow polygon) are
    skipped, so a board / off-board example is not contaminated by a panel outline."""
    if not os.path.isfile(debug_video):
        print(f'  WARNING: debug video not found for {name}')
        return
    b, g, r = bgr
    cap = cv.VideoCapture(debug_video)
    cands = []  # (npx, frame, centroid, idx, panel_px)
    idx = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        mask = ((np.abs(fr[:, :, 0].astype(int) - b) < 25) &
                (np.abs(fr[:, :, 1].astype(int) - g) < 25) &
                (np.abs(fr[:, :, 2].astype(int) - r) < 25))
        n = int(mask.sum())
        if n >= 40:
            ys, xs = np.where(mask)
            # Panel polygon is drawn in (0,255,255); on real frames it scores ~1.5-2.5k px
            # while a clean board scores ~0 (the board's yellow pieces do NOT match), so a
            # simple count cleanly tells "a panel is shown" from "board only".
            pmask = ((fr[:, :, 0].astype(int) < 30) & (np.abs(fr[:, :, 1].astype(int) - 255) < 25) &
                     (np.abs(fr[:, :, 2].astype(int) - 255) < 25))
            cands.append((n, fr.copy(), (int(xs.mean()), int(ys.mean())), idx, int(pmask.sum())))
        idx += 1
    cap.release()
    if not cands:
        print(f'  WARNING: no "{name}" gaze in this segment')
        return
    pool = [c for c in cands if c[4] < 500] if avoid_panel else cands
    if not pool:                       # every candidate had a panel: do not drop the figure
        pool = cands
    if strategy == 'earliest':
        best = min(pool, key=lambda c: c[3])
    elif strategy == 'edge':
        H, W = pool[0][1].shape[:2]
        best = max(pool, key=lambda c: (c[2][0] - W / 2) ** 2 + (c[2][1] - H / 2) ** 2)
    else:  # 'most'
        best = max(pool, key=lambda c: c[0])
    best_fr, best_c = best[1], best[2]
    cx, cy = best_c
    H, W = best_fr.shape[:2]
    hw, hh = GAZE_CROP
    x0, x1 = max(0, cx - hw), min(W, cx + hw)
    y0, y1 = max(0, cy - hh), min(H, cy + hh)
    # General view (same viewpoint as the other figures) with the crop region boxed in RED, a
    # colour distinct from the YELLOW sample-panel polygon so the two are never confused, plus
    # the framed zoom. NO padding: the zoom keeps its native proportions; only the general view
    # is rescaled to the SAME pixel height as the zoom, so both render at one height in the doc.
    BOX = (0, 0, 255)  # red, BGR
    full = best_fr.copy()
    cv.rectangle(full, (x0, y0), (x1, y1), BOX, 3)
    zoom = cv.copyMakeBorder(best_fr[y0:y1, x0:x1], 3, 3, 3, 3, cv.BORDER_CONSTANT, value=BOX)
    th = zoom.shape[0]
    full = cv.resize(full, (max(1, int(round(W * th / float(H)))), th))
    cv.imwrite(os.path.join(MEDIA, f'{name}.png'), full)
    cv.imwrite(os.path.join(MEDIA, f'{name}_zoom.png'), zoom)
    print(f'  {name}.png (vista general) + {name}_zoom.png (alto={th}px, nativo, sin relleno)')


def cropDumpedGaze(dump_dir, world_frame, name, bgr):
    """Same crop as gazeCrop but on ONE verified dumped figframe (a specific WORLD frame whose
    classification was checked against the real reprocess), instead of a colour scan over the
    whole video. Used for the cases a scan gets wrong: the pre_start gaze that COUNTS (a cold
    scan mislabels it not_board) and a genuine off-board not_board (the reference trial has none)."""
    src = os.path.join(dump_dir, f'figframe_{world_frame}.png')
    if not os.path.isfile(src):
        print(f'  WARNING: dumped frame {world_frame} missing for {name}')
        return
    fr = cv.imread(src)
    b, g, r = bgr
    mask = ((np.abs(fr[:, :, 0].astype(int) - b) < 25) & (np.abs(fr[:, :, 1].astype(int) - g) < 25) &
            (np.abs(fr[:, :, 2].astype(int) - r) < 25))
    if int(mask.sum()) < 20:
        print(f'  WARNING: marker {bgr} not found in frame {world_frame} for {name}')
        return
    ys, xs = np.where(mask)
    cx, cy = int(xs.mean()), int(ys.mean())
    H, W = fr.shape[:2]
    hw, hh = GAZE_CROP
    x0, x1 = max(0, cx - hw), min(W, cx + hw)
    y0, y1 = max(0, cy - hh), min(H, cy + hh)
    BOX = (0, 0, 255)  # red, distinct from the yellow panel polygon
    full = fr.copy()
    cv.rectangle(full, (x0, y0), (x1, y1), BOX, 3)
    zoom = cv.copyMakeBorder(fr[y0:y1, x0:x1], 3, 3, 3, 3, cv.BORDER_CONSTANT, value=BOX)
    th = zoom.shape[0]
    full = cv.resize(full, (max(1, int(round(W * th / float(H)))), th))
    cv.imwrite(os.path.join(MEDIA, f'{name}.png'), full)
    cv.imwrite(os.path.join(MEDIA, f'{name}_zoom.png'), zoom)
    print(f'  {name}.png + {name}_zoom.png  <- frame verificado {world_frame} ({bgr})')


def readMarks(transitions_csv, trial_name):
    """Per-mark WORLD frame of the reference trial, read from the transitions CSV."""
    import csv
    marks = {}
    if not os.path.isfile(transitions_csv):
        return marks
    for row in csv.DictReader(open(transitions_csv)):
        if row['trial_name'] != trial_name:
            continue
        if row['event'] in ('target_found', 'motor_onset', 'target_touch', 'hand_exit', 'search_start'):
            marks[row['event']] = int(row['frame'])
    return marks


def readTrialCycle(transitions_csv, block, trial):
    """Full panel-to-panel timeline of ONE trial (block, trial), read from the transitions CSV:
    the sample-panel appearance/removal (state entries get_test_name / test_start_execution), the
    behavioural marks (search_start, target_found, motor_onset, target_touch, hand_exit) and the
    NEXT trial's panel appearance -- so a figure can span the whole cycle 'panel shown -> next
    panel shown' instead of just the reach. Returns a dict of WORLD frames (missing keys absent)."""
    import csv
    block, trial = str(block), str(trial)
    rows = list(csv.DictReader(open(transitions_csv)))
    c = {}
    last = None
    for r in rows:
        if r['block_index'] != block or r['trial_index'] != trial:
            continue
        f = int(r['frame'])
        ev, to = r['event'], r['to_state']
        if ev == 'state_change' and to == 'get_test_name':
            c.setdefault('panel_shown', f)
        elif ev == 'state_change' and to == 'test_start_execution':
            c.setdefault('panel_removed', f)
        elif ev in ('search_start', 'target_found', 'validation', 'motor_onset',
                    'target_touch', 'hand_exit', 'trial_end'):
            c.setdefault(ev, f)
        last = f
    # NEXT panel: first get_test_name entry strictly after this trial's last event
    if last is not None:
        nxt = [int(r['frame']) for r in rows
               if r['event'] == 'state_change' and r['to_state'] == 'get_test_name' and int(r['frame']) > last]
        if nxt:
            c['next_panel_shown'] = min(nxt)
    return c


def plotOcclusion(trace_text, marks, lang):
    frames, fT, fC = [], [], []
    for line in trace_text.splitlines():
        m = re.search(r'\[TRACE\s+\w+\s+(\d+)\].*fT=\s*([\d.\-]+)\s+fC=\s*([\d.\-]+)', line)
        if not m:
            continue
        try:
            t, c = float(m.group(2)), float(m.group(3))
        except ValueError:
            continue
        frames.append(int(m.group(1))); fT.append(t); fC.append(c)
    if not frames:
        print('  WARNING: empty occlusion trace; skipping oclusion_temporal')
        return
    s = STR[lang]
    mk = [('m_found', marks.get('target_found')), ('m_in', marks.get('motor_onset')),
          ('m_touch', marks.get('target_touch')), ('m_out', marks.get('hand_exit'))]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(frames, fT, '-o', ms=3, color='#1f77b4', label=s['fT_target'])
    ax.plot(frames, fC, '-o', ms=3, color='#999', label=s['fT_ctrl'])
    ax.axhline(0.13, ls='--', color='#d33333', lw=1, label=s['fT_thr'])
    top = max(0.3, max(fT) * 1.15)
    for key, fr in mk:
        if fr is None:
            continue
        ax.axvline(fr, color=MARK_COL[key], ls=':', lw=1.3)
        ax.text(fr, top * 0.96, s[key], rotation=90, va='top', ha='right', fontsize=7, color=MARK_COL[key])
    ax.set_xlabel(s['fT_x']); ax.set_ylabel(s['fT_y']); ax.set_ylim(0, top)
    ax.set_title(s['fT_title']); ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout(); fig.savefig(os.path.join(MEDIA, f'oclusion_temporal{_suffix(lang)}.png'), dpi=150)
    plt.close(fig)
    print(f'  oclusion_temporal{_suffix(lang)}.png OK')


def plotTimeline(cycle, lang, zoom=False, out_name='timeline_fases'):
    """Phase timeline of ONE trial. Two views (the inter-trial gap is ~3x the reach, so one window
    can't show both well): the FULL cycle 'sample panel -> next sample panel' for context, and a
    ZOOMed view to the trial proper (panel .. hand-exit) that drops the long inter-trial gap. The
    phases come from the state machine; the vertical marks are the behavioural events."""
    need = ['search_start', 'target_found', 'motor_onset', 'target_touch', 'hand_exit']
    if not all(k in cycle for k in need):
        print('  WARNING: missing marks; skipping timeline')
        return
    s = STR[lang]
    ss, tf, mo, tc, he = (cycle[k] for k in need)
    # panel appearance/removal frame the search; fall back to search_start if absent
    panel = cycle.get('panel_shown', ss)
    nxt = cycle.get('next_panel_shown')
    # ZOOM = the trial proper: drop BOTH the leading sample panel and the trailing inter-trial/next
    # panel (symmetry with the bump zoom). FULL = the whole cycle, panel -> next panel.
    phases = [(s['p_search'], ss, tf, BAND_COL['search']), (s['p_verif'], tf, mo, BAND_COL['verification']),
              (s['p_motor'], mo, tc, BAND_COL['motor']), (s['p_with'], tc, he, BAND_COL['withdraw'])]
    mk = [('m_found', tf), ('m_in', mo), ('m_touch', tc), ('m_out', he)]
    if not zoom:
        phases.insert(0, (s['p_panel'], panel, ss, BAND_COL['panel']))
        mk.insert(0, ('m_panel', panel))
        if nxt:
            phases.append((s['p_inter'], he, nxt, BAND_COL['inter']))
            mk.append(('m_next', nxt))
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=(8.5, 2.8) if zoom else (10, 2.8))
    for name, a, b, col in phases:
        ax.barh(0, max(b - a, 0.4), left=a, height=0.5, color=col, edgecolor='#555')
    for key, fr in mk:
        ax.axvline(fr, color=MARK_COL[key], ls=':', lw=1.3)
        ax.text(fr, 0.30, s[key], ha='center', va='bottom', rotation=90, fontsize=6.5, color=MARK_COL[key])
    ax.legend(handles=[mpatches.Patch(color=c, label=n) for n, _, _, c in phases],
              loc='upper center', bbox_to_anchor=(0.5, -0.32), ncol=len(phases), fontsize=7.5,
              frameon=False, handlelength=1.2, columnspacing=1.0)
    # explicit x-window with margins (avoids the auto-margin whitespace AND the clipped edge labels)
    lo0 = ss if zoom else panel
    hi0 = he if zoom else (nxt or he)
    pad = max(3, int((hi0 - lo0) * 0.06))
    ax.set_xlim(lo0 - pad, hi0 + pad)
    ax.set_yticks([]); ax.set_xlabel(s['tl_x']); ax.set_ylim(-0.35, 1.35)
    title = s['tl_title'] + (' — ' + ('detalle' if lang == 'es' else 'zoom') if zoom else '')
    ax.set_title(title, pad=10)
    fig.tight_layout(); fig.savefig(os.path.join(MEDIA, f'{out_name}{_suffix(lang)}.png'), dpi=150)
    plt.close(fig)
    print(f'  {out_name}{_suffix(lang)}.png OK ({"zoom" if zoom else "full"}, panel {panel} -> next {nxt})')


def bumpFigure(pkl_path, block_trial, lang, out_name='oclusion_bump', cycle=None, zoom=False):
    """Occlusion BUMP figure from the persisted `signal_trace`: the TARGET occlusion (fT) and the
    WHOLE-BOARD occlusion (board_occ), with the motor marks (entry / touch / hand-exit). Two views:
    FULL cycle (`zoom=False`) -- phase bands + this-trial and next-trial sample-panel marks for
    context; and ZOOM (`zoom=True`) -- tight on the reach bump. The occlusion is only recorded
    during the reach (execution/motor-recovery), so the curve is naturally confined to the bump;
    in the full view the flat stretches under the panel/inter-trial bands make that explicit, and
    the next-panel mark shows where the following sample comes up. Reads the PKL, no re-run."""
    import pickle as _pickle
    if not os.path.isfile(pkl_path):
        print(f'  WARNING: PKL missing; skipping {out_name}')
        return
    data = _pickle.load(open(pkl_path, 'rb'))['trials_data']
    if block_trial is None:
        # pick the confirmed-touch reach with the clearest whole-board occlusion (both curves show)
        bo = -1
        for k, t in data.items():
            if k == 'latest':
                continue
            mm = list(t.values())[0]
            if isinstance(mm, dict) and mm.get('target_touch_capture') is not None:
                v = (mm.get('touch_diag') or {}).get('board_occ_peak', 0) or 0
                if v > bo:
                    bo, block_trial = v, k
    tm = data.get(block_trial)
    m = list(tm.values())[0] if tm else None
    st = m.get('signal_trace', []) if m else []
    if not st:
        print(f'  WARNING: no signal_trace for {block_trial}; skipping {out_name}')
        return
    fr = [s['f'] for s in st]
    fT = [s['fT'] if s.get('fT') is not None else np.nan for s in st]
    bo = [s['bocc'] if s.get('bocc') is not None else np.nan for s in st]
    s = STR[lang]
    board_lbl = 'Oclusión de tablero' if lang == 'es' else 'Whole-board occlusion'
    title = 'Curvas de oclusión (modelo bump)' if lang == 'es' else 'Occlusion curves (bump model)'
    cycle = cycle or {}
    nxt = cycle.get('next_panel_shown')
    mo, he = m.get('motor_onset_capture'), m.get('hand_exit_capture')
    panel, ss = cycle.get('panel_shown'), cycle.get('search_start')
    top = 1.05
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    # FULL view: light phase bands behind the curve tie it to the trial cycle (the curve is flat
    # under the panel/search/inter-trial bands because occlusion is only measured during the reach).
    if not zoom and panel is not None and ss is not None:
        tf2 = cycle.get('target_found') or mo; tc = m.get('target_touch_capture')
        if mo is not None:           # keep target_found inside [search, motor] (it can land later)
            tf2 = min(max(tf2, ss), mo)
        for a, b, col in [(panel, ss, BAND_COL['panel']), (ss, tf2, BAND_COL['search']),
                          (tf2, mo, BAND_COL['verification']), (mo, tc, BAND_COL['motor']),
                          (tc, he, BAND_COL['withdraw']), (he, nxt, BAND_COL['inter'])]:
            if a is not None and b is not None and b > a:
                ax.axvspan(a, b, color=col, alpha=0.6, lw=0)
    ax.plot(fr, fT, '-', color='#d62728', lw=1.6, label=s['fT_target'])
    ax.plot(fr, bo, '-', color='#1f77b4', lw=1.6, label=board_lbl)
    if zoom:
        mk = [('m_in', mo), ('m_touch', m.get('target_touch_capture')), ('m_out', he)]
    else:
        mk = [('m_panel', panel), ('m_in', mo), ('m_touch', m.get('target_touch_capture')),
              ('m_out', he), ('m_next', nxt)]
    for key, f in mk:
        if f is None:
            continue
        ax.axvline(f, color=MARK_COL[key], ls=':', lw=1.3)
        ax.text(f, top * 0.96, s[key], rotation=90, va='top', ha='right', fontsize=7, color=MARK_COL[key])
    if zoom:   # tight on the reach bump (a few frames of baseline before the rise, then the bump)
        lo = (mo - 8) if mo is not None else (ss or fr[0]) - 2
        hi = (he + 6) if he is not None else fr[-1] + 6
    else:      # full cycle: this-trial panel .. next-trial panel
        lo = (panel if panel is not None else fr[0]) - 4
        hi = (nxt + 5) if nxt is not None else (he + 3 if he is not None else fr[-1] + 3)
    ax.set_xlim(lo, hi)
    ax.set_xlabel(s['fT_x']); ax.set_ylabel(s['fT_y']); ax.set_ylim(0, top)
    ax.set_title(title + (' — ' + ('detalle' if lang == 'es' else 'zoom') if zoom else ''))
    ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout(); fig.savefig(os.path.join(MEDIA, f'{out_name}{_suffix(lang)}.png'), dpi=150)
    plt.close(fig)
    print(f'  {out_name}{_suffix(lang)}.png OK ({"zoom" if zoom else "full"}, {len(st)} frames)')


def maskFigure(dump_dir, frame, lang, out_name='mascara_toque', patch_state=None):
    """Save the touch-detector intermediate masks for the TARGET cell (dumped at `frame` by
    process_video --dump_frames) as SEPARATE labelled images (one PNG per stage, NOT a mosaic),
    so the documentation can order/caption them freely: the live patch, the clean reference, the
    pixel diff, the edge/texture change, the SSIM map and the final change mask -- what the
    occlusion gates actually see when confirming a touch (technical doc 9). `patch_state` overrides
    the patch label (e.g. 'limpio'/'clean') so the SAME function can render a clean-frame baseline
    for the touch-vs-clean comparison. Files: `<out_name>_<stage>{_suffix}.png`."""
    es = lang == 'es'
    pstate = patch_state or ('toque' if es else 'touch')
    panels = [('patch', f'parche ({pstate})' if es else f'patch ({pstate})'),
              ('ref', 'referencia (limpio)' if es else 'reference (clean)'),
              ('diff', 'diferencia de píxel' if es else 'pixel diff'),
              ('edge', 'borde/textura' if es else 'edge/texture'),
              ('ssim', 'SSIM (cambio estructural)' if es else 'SSIM (structural change)'),
              ('changed', 'máscara final de cambio' if es else 'final change mask')]
    n = 0
    for k, lbl in panels:
        path = os.path.join(dump_dir, f'figmask_{frame}_{k}.png')
        if not os.path.isfile(path):
            continue
        im = plt.imread(path)
        fig, ax = plt.subplots(figsize=(3.0, 3.0))
        ax.imshow(im, cmap='gray' if im.ndim == 2 else None)
        ax.set_title(lbl, fontsize=10); ax.axis('off')
        fig.tight_layout(); fig.savefig(os.path.join(MEDIA, f'{out_name}_{k}{_suffix(lang)}.png'), dpi=140)
        plt.close(fig)
        n += 1
    if n == 0:
        print(f'  WARNING: no touch masks at frame {frame}; skipping {out_name}')
    else:
        print(f'  {out_name}_*{_suffix(lang)}.png OK ({n} separate masks)')


def boardMaskFigure(dump_dir, frame, lang, out_name='mascara_tablero'):
    """Whole-board occlusion masks (the board_occ equivalent of the target masks) dumped at `frame`
    by process_video --dump_frames, as SEPARATE images: the current board, the clean-board
    reference, the abs difference and the final change mask -- what the WHOLE-BOARD occlusion
    (board_occ, used for hand_exit) actually sees when the arm crosses the board (technical doc 9)."""
    es = lang == 'es'
    panels = [('patch', 'tablero (con la mano)' if es else 'board (with hand)'),
              ('ref', 'tablero limpio (referencia)' if es else 'clean board (reference)'),
              ('diff', 'diferencia' if es else 'difference'),
              ('changed', 'máscara de cambio (oclusión)' if es else 'change mask (occlusion)')]
    n = 0
    for k, lbl in panels:
        path = os.path.join(dump_dir, f'figboard_{frame}_{k}.png')
        if not os.path.isfile(path):
            continue
        im = plt.imread(path)
        fig, ax = plt.subplots(figsize=(3.4, 2.2))
        ax.imshow(im, cmap='gray' if im.ndim == 2 else None)
        ax.set_title(lbl, fontsize=10); ax.axis('off')
        fig.tight_layout(); fig.savefig(os.path.join(MEDIA, f'{out_name}_{k}{_suffix(lang)}.png'), dpi=140)
        plt.close(fig)
        n += 1
    if n == 0:
        print(f'  WARNING: no board masks at frame {frame}; skipping {out_name}')
    else:
        print(f'  {out_name}_*{_suffix(lang)}.png OK ({n} separate masks)')


ARUCO_FIX_PARTICIPANT = '042'   # participant with severe edge-marker loss, for the fix figure


def arucoFixFigure(data_root):
    """Why ArUcos are detected on the ORIGINAL image: a frame with markers drawn on the
    distorted (all detected) vs the undistorted (edge markers lost)."""
    vid = os.path.join(data_root, ARUCO_FIX_PARTICIPANT, 'world.mp4')
    if not os.path.isfile(vid):
        print('  WARNING: ArUco-fix participant video missing; skipping')
        return
    cap = cv.VideoCapture(vid)
    n = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    cap.set(cv.CAP_PROP_POS_FRAMES, int(n * 0.5))
    ok, raw = cap.read()
    cap.release()
    if not ok:
        return
    raw = ARUCOColorCorrection(raw)
    h, w = raw.shape[:2]
    dh = DistortionHandler(CALIB, w, h)
    co, io = detectAllArucos(raw)
    orig = raw.copy()
    if io is not None:
        cv.aruco.drawDetectedMarkers(orig, co, io, (0, 255, 0))
    cv.imwrite(os.path.join(MEDIA, 'aruco_original.png'), orig)
    und = dh.undistortImage(raw)
    cu, iu = detectAllArucos(und)
    undd = und.copy()
    if iu is not None:
        cv.aruco.drawDetectedMarkers(undd, cu, iu, (0, 255, 0))
    cv.imwrite(os.path.join(MEDIA, 'aruco_desdistorsionada.png'), undd)
    no = 0 if io is None else len(io)
    nu = 0 if iu is None else len(iu)
    print(f'  aruco_original.png ({no}) / aruco_desdistorsionada.png ({nu})  [{ARUCO_FIX_PARTICIPANT}]')


def panelPersistenceFigure(data_root, lang, participant='001', f0=13560, f1=13930):
    """Panel-detection PERSISTENCE over time (block 4 of 001): a real panel is a TALL run of
    consecutive frames above the latch threshold; a misread ArUco (e.g. 204) is a brief blip
    below it. Reproduced by detecting ArUcos on the real video (no pipeline needed)."""
    import matplotlib.patches as mp
    vid = os.path.join(data_root, participant, 'world.mp4')
    if not os.path.isfile(vid):
        print('  (panel_persistencia: missing video)'); return
    cap = cv.VideoCapture(vid)
    panels = {'green_hexagon': {408, 442}, 'red_triangle': {204, 238},
              'yellow_hexagon': {357, 374}, 'blue_circle': {34, 0}}
    det = []
    for f in range(f0, f1):
        cap.set(cv.CAP_PROP_POS_FRAMES, f); ok, img = cap.read()
        if not ok:
            det.append((f, None)); continue
        _, ids = detectAllArucos(img)
        s = set() if ids is None else set(int(i) for i in ids.flatten())
        det.append((f, next((pn for pn, mk in panels.items() if s & mk), None)))
    cap.release()
    counts = []; prev = None; c = 0
    for f, n in det:
        c = c + 1 if (n is not None and n == prev) else (1 if n is not None else 0)
        counts.append(c); prev = n
    cmap = {'green_hexagon': 'green', 'red_triangle': 'red', 'yellow_hexagon': 'goldenrod', 'blue_circle': 'blue'}
    T = {'es': dict(t='Persistencia del panel: el misread es un blip de 2 frames; los reales duran decenas',
                    xl='fotograma (world) — bloque 4 de 001', yl='frames consecutivos de panel',
                    latch='umbral=4 (latch)', old='umbral viejo=2', mis='misread 204\n(2 frames -> rechazado)'),
         'en': dict(t='Panel persistence: a misread is a 2-frame blip; real panels last tens of frames',
                    xl='frame (world) — block 4 of 001', yl='consecutive panel frames',
                    latch='threshold=4 (latch)', old='old threshold=2', mis='misread 204\n(2 frames -> rejected)')}[lang]
    fig, ax = plt.subplots(figsize=(11, 3.4))
    for i, (f, n) in enumerate(det):
        ax.bar(f, counts[i], width=1.0, color=cmap.get(n, 'lightgray'))
    ax.axhline(4, color='black', ls='--', lw=1.2); ax.text(f0 + 2, 4.2, T['latch'], fontsize=9)
    ax.axhline(2, color='dimgray', ls=':', lw=1); ax.text(f0 + 2, 2.15, T['old'], fontsize=8, color='dimgray')
    mis = [f for f, n in det if n == 'red_triangle' and f < 13700]
    if mis:
        mf = mis[len(mis) // 2]
        ax.annotate(T['mis'], xy=(mf, 2), xytext=(mf - 70, 9),
                    arrowprops=dict(arrowstyle='->', color='red'), color='red', fontsize=9)
    ax.legend(handles=[mp.Patch(color=c, label=l) for l, c in cmap.items()], loc='upper right', fontsize=8, ncol=2)
    ax.set_xlabel(T['xl']); ax.set_ylabel(T['yl']); ax.set_title(T['t']); ax.set_ylim(0, 16)
    fig.tight_layout(); fig.savefig(os.path.join(MEDIA, f'panel_persistencia{_suffix(lang)}.png'), dpi=110); plt.close(fig)
    print(f'  panel_persistencia{_suffix(lang)}.png')


def touchNoiseFigure(output_root, lang):
    """Touch fT noise vs real-touch peak, per target colour, across the cohort. Shows red has the
    lowest threshold AND the noisiest baseline -> early touches; and that magnitude does not cleanly
    separate (touch peak median ~0.45 overlaps the noise tails). Reads signal_trace + bump.peak_val."""
    import pickle, glob, collections
    import matplotlib.patches as mp
    noise = collections.defaultdict(list); peak = collections.defaultdict(list)
    for pkl in glob.glob(f"{output_root}/gaze/*/data_*.pkl"):
        try: d = pickle.load(open(pkl, 'rb'))['trials_data']
        except Exception: continue
        for k in d:
            o = d[k]; n = list(o.keys())[0]; m = o[n]
            if not isinstance(m, dict) or 'error' in n: continue
            col = n.split('_')[0]
            if col not in ('red', 'green', 'blue', 'yellow'): continue
            mo = m.get('motor_onset_capture')
            if mo:
                for r in m.get('signal_trace', []):
                    if r.get('ph') == 'exec' and r['f'] < mo - 5 and r.get('fT') is not None and r.get('act') == 1:
                        noise[col].append(r['fT'])
            tg = (m.get('bump') or {}).get('target')
            if tg and tg.get('peak_val') is not None and m.get('target_touch_capture') is not None:
                peak[col].append(tg['peak_val'])
    if not any(peak.values()):
        print('  (touch_ruido_color: no data in output_root)'); return
    thr = {'red': 0.13, 'yellow': 0.15, 'blue': 0.20, 'green': 0.20}
    cc = {'red': '#d62728', 'green': '#2ca02c', 'blue': '#1f77b4', 'yellow': '#e0b000'}
    T = {'es': dict(t='Ruido (búsqueda, gris) vs pico real de toque (color), por color del objetivo',
                    yl='fT (oclusión del target)', n='ruido en búsqueda (sin mano)', p='pico real de toque',
                    cap='verde: ruido bajo, separable | rojo/amarillo: ruido alto que solapa el toque y cruza un umbral bajo'),
         'en': dict(t='Noise (search, grey) vs real touch peak (colour), by target colour',
                    yl='fT (target occlusion)', n='search noise (no hand)', p='real touch peak',
                    cap='green: low noise, separable | red/yellow: high noise overlapping the touch and crossing a low threshold')}[lang]
    fig, ax = plt.subplots(figsize=(10, 4.2)); pos = 0; xt = []; xl = []
    for c in ['green', 'blue', 'yellow', 'red']:
        bx = ax.boxplot([noise[c], peak[c]], positions=[pos, pos + 0.6], widths=0.5, patch_artist=True, showfliers=False)
        for patch, fcc in zip(bx['boxes'], ['#bbbbbb', cc[c]]):
            patch.set_facecolor(fcc)
        ax.hlines(thr[c], pos - 0.3, pos + 0.9, color='black', ls='--', lw=1.3)
        ax.text(pos + 0.3, thr[c] + 0.02, f"umbral {thr[c]}" if lang == 'es' else f"thr {thr[c]}", ha='center', fontsize=8)
        xt.append(pos + 0.3); xl.append(c); pos += 2
    ax.set_xticks(xt); ax.set_xticklabels(xl); ax.set_ylabel(T['yl']); ax.set_ylim(-0.02, 1.0); ax.set_title(T['t'])
    ax.legend(handles=[mp.Patch(color='#bbbbbb', label=T['n']), mp.Patch(color='#888888', label=T['p'])], loc='upper left', fontsize=8)
    plt.figtext(0.5, 0.005, T['cap'], ha='center', fontsize=8, style='italic')
    fig.tight_layout(rect=[0, 0.04, 1, 1]); fig.savefig(os.path.join(MEDIA, f'touch_ruido_color{_suffix(lang)}.png'), dpi=110); plt.close(fig)
    print(f'  touch_ruido_color{_suffix(lang)}.png')


def borderFlickerFigure(old_root, new_root, lang, block_trial=(1, 2)):
    """Border-contour stability in search: the old (Canny + per-frame reference) flickers; the new
    (no-Canny + median/EMA reference) is flat. A BEFORE/AFTER figure -> needs both version outputs
    (old_root = a previous release, new_root = current). Skips quietly if either is missing."""
    import pickle, glob
    def cont_series(root, key):
        # use 008 by convention (the participant the doc text references)
        cand = [c for c in glob.glob(f"{root}/gaze/*/data_*.pkl") if '008' in os.path.basename(c)]
        if not cand: return None
        d = pickle.load(open(cand[0], 'rb'))['trials_data']
        o = d.get(key)
        if not o: return None
        m = o[list(o.keys())[0]]
        mo = m.get('motor_onset_capture')
        st = [r for r in m.get('signal_trace', []) if r.get('ph') == 'exec' and (mo is None or r['f'] < mo)]
        return [r['cont'] for r in st]
    old = cont_series(old_root, block_trial); new = cont_series(new_root, block_trial)
    if not old or not new:
        print('  (borde_flicker: needs old+new version outputs, skipped)'); return
    flick = lambda v: sum(1 for i in range(1, len(v)) if v[i] != v[i - 1])
    T = {'es': dict(t='Estabilidad del contorno del borde en búsqueda: menos parpadeo = arranque/motor más fiables',
                    old=f'v previa (Canny + ref por-frame) — {flick(old)} transiciones',
                    new=f'nuevo (sin-Canny + EMA) — {flick(new)} transiciones',
                    xl='fotograma de búsqueda (tablero visible, sin mano) — trial blue_circle de 008',
                    y0='no contorno', y1='contorno'),
         'en': dict(t='Border-contour stability in search: less flicker = more reliable start/motor marks',
                    old=f'previous v (Canny + per-frame ref) — {flick(old)} transitions',
                    new=f'new (no-Canny + EMA) — {flick(new)} transitions',
                    xl='search frame (board visible, no hand) — blue_circle trial of 008',
                    y0='no contour', y1='contour')}[lang]
    fig, axs = plt.subplots(2, 1, figsize=(10, 4))
    for ax, (v, lbl, col) in zip(axs, [(old, T['old'], '#d62728'), (new, T['new'], '#2ca02c')]):
        ax.step(range(len(v)), v, where='post', color=col, lw=1.6)
        ax.fill_between(range(len(v)), v, step='post', alpha=0.15, color=col)
        ax.set_yticks([0, 1]); ax.set_yticklabels([T['y0'], T['y1']]); ax.set_ylim(-0.2, 1.2)
        ax.set_title(lbl, fontsize=10, loc='left'); ax.set_xlim(0, len(v))
    axs[1].set_xlabel(T['xl']); fig.suptitle(T['t'], fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(os.path.join(MEDIA, f'borde_flicker{_suffix(lang)}.png'), dpi=110); plt.close(fig)
    print(f'  borde_flicker{_suffix(lang)}.png')


def pfoundFigure(output_root, lang):
    """Cohort distribution of target_found_confidence with the mass-based found cut. found is the
    first fixation whose mean ellipse mass on the target cell reaches the threshold (0.30): the
    histogram is split by that mark (green found / red not found) with the 0.30 cut drawn solid and
    the suggested analysis tiers (0.2 / 0.5) drawn thin. Reads the per-participant trials CSVs."""
    import csv, glob
    import numpy as np
    THR = 0.30
    seen = set(); found_v = []; notfound_v = []
    for f in glob.glob(f"{output_root}/gaze/*/trials_data_*.csv"):
        if '_sequence' in f or '_transitions' in f:
            continue
        pid = os.path.basename(os.path.dirname(f))
        with open(f) as fh:
            for row in csv.DictReader(fh):
                tn = row.get('trial_name', '')
                if tn.startswith(('missing_trial_error', 'transition_error')):
                    continue
                k = (pid, row.get('block_index'), row.get('trial_index'))
                if k in seen:
                    continue
                seen.add(k)
                c = (row.get('target_found_confidence') or '').strip()
                if c in ('', 'None'):
                    continue
                c = float(c)
                if (row.get('frame_target_found') or '').strip() not in ('', 'None'):
                    found_v.append(c)
                else:
                    notfound_v.append(c)
    if not found_v and not notfound_v:
        print('  (pfound_confianza: no data in output_root)'); return
    nf = len(found_v); nnf = len(notfound_v)
    T = {'es': dict(t='Distribución de target_found_confidence — found = masa ≥ 0,30',
                    xl='target_found_confidence (masa de la elipse sobre la casilla objetivo)',
                    yl='nº de trials', found=f'found ({nf})', nfound=f'no encontrado ({nnf})',
                    thr='umbral found 0,30',
                    cap='found = primera fijación con masa ≥ 0,30 (línea sólida). '
                        'Líneas finas: niveles sugeridos para análisis (0,2 / 0,5)'),
         'en': dict(t='Distribution of target_found_confidence — found = mass ≥ 0.30',
                    xl='target_found_confidence (ellipse mass on the target cell)',
                    yl='# trials', found=f'found ({nf})', nfound=f'not found ({nnf})',
                    thr='found threshold 0.30',
                    cap='found = first fixation with mass ≥ 0.30 (solid line). '
                        'Thin lines: suggested analysis tiers (0.2 / 0.5)')}[lang]
    bins = np.linspace(0, 1, 21)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.hist([found_v, notfound_v], bins=bins, stacked=True,
            color=['#2ca02c', '#d62728'], label=[T['found'], T['nfound']], edgecolor='white', lw=0.4)
    for x in (0.2, 0.5):
        ax.axvline(x, color='#888888', ls='--', lw=0.8)
    ax.axvline(THR, color='black', ls='-', lw=1.6, label=T['thr'])
    ax.set_xlabel(T['xl']); ax.set_ylabel(T['yl']); ax.set_title(T['t']); ax.set_xlim(0, 1)
    ax.legend(loc='upper center', fontsize=9)
    plt.figtext(0.5, 0.005, T['cap'], ha='center', fontsize=8, style='italic')
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(os.path.join(MEDIA, f'pfound_confianza{_suffix(lang)}.png'), dpi=110); plt.close(fig)
    print(f'  pfound_confianza{_suffix(lang)}.png')


def main():
    parser = argparse.ArgumentParser(description='Regenerate the documentation figures from a real run')
    parser.add_argument('--data_root', default=DEFAULT_DATA_ROOT)
    parser.add_argument('--tmp_out', default='/tmp/doc_figures')
    args = parser.parse_args()
    os.makedirs(MEDIA, exist_ok=True)

    # 1. Main reference trial: state frames + warp + occlusion ROIs
    dumps = set(STATE_FIGURES.values()) | set(WARP_FIGURES.values())
    print('Rendering reference trial (049)...')
    runViz(PARTICIPANT, SEGMENT, args.data_root, args.tmp_out, dump=dumps)
    dump_dir = os.path.join(args.tmp_out, 'gaze', PARTICIPANT)
    copyDumps(dump_dir, STATE_FIGURES, 'figframe')
    copyDumps(dump_dir, WARP_FIGURES, 'figwarp')

    # 2. Gaze-classification close-ups (counted / not-counted) from the rendered video
    debug_video = os.path.join(dump_dir, f'debug_{PARTICIPANT}.mp4')
    for name, (bgr, strat, avoid) in GAZE_CASES.items():
        gazeCrop(debug_video, name, bgr, strat, avoid_panel=avoid)

    # gaze_anticipada (pre_start, COUNTS): the red-hexagon gaze in the EARLY green_hexagon, rendered
    # from the warm block-0 start (its own output dir, not to clash with the 049 reference run). A
    # cold scan would mislabel this same gaze as not_board.
    ap, aseg, aframe = ANTICIPADA_SOURCE
    print(f'Rendering anticipatory-gaze trial ({ap}, warm block-0 start)...')
    runViz(ap, aseg, args.data_root, args.tmp_out + '_antic', dump=[aframe])
    antic_dir = os.path.join(args.tmp_out + '_antic', 'gaze', ap)
    cropDumpedGaze(antic_dir, aframe, 'gaze_anticipada', (0, 165, 255))
    # estado_2_retirada (panel removal) from the SAME warm frame: the gaze is correctly orange
    # (pre_start, counts), not the cold-start not_board (blue) artefact of the reference run.
    antic_src = os.path.join(antic_dir, f'figframe_{aframe}.png')
    if os.path.isfile(antic_src):
        shutil.copy(antic_src, os.path.join(MEDIA, 'estado_2_retirada.png'))
        print(f'  estado_2_retirada.png  <- warm frame {aframe}')

    # 3. on_panel gaze needs a trial that looks at the panel: separate short render
    pp, pseg = PANEL_SOURCE
    print(f'Rendering on_panel trial ({pp})...')
    runViz(pp, pseg, args.data_root, args.tmp_out)
    gazeCrop(os.path.join(args.tmp_out, 'gaze', pp, f'debug_{pp}.mp4'), 'gaze_panel', GAZE_PANEL_MAGENTA[0], GAZE_PANEL_MAGENTA[1])

    # 3c. Two "does NOT count" cases off the cells (both not_board, drawn blue):
    #   gaze_fuera    - gaze OUTSIDE the board during the search (the board is clean and visible).
    #   gaze_panel_no - gaze on the sample panel while it is being withdrawn (still off the cells).
    for src, fig in [(GAZE_FUERA_SOURCE, 'gaze_fuera'), (GAZE_PANELNO_SOURCE, 'gaze_panel_no')]:
        gp, gseg, gframe = src
        print(f'Rendering {fig} trial ({gp})...')
        runViz(gp, gseg, args.data_root, args.tmp_out, dump=[gframe])
        cropDumpedGaze(os.path.join(args.tmp_out, 'gaze', gp), gframe, fig, (255, 0, 0))

    # 3b. Sub-marks (entra/toca/sale) from a trial with a CENTRAL target / long reach (064),
    # so the three frames are clearly different (the 049 reference touches right at the onset).
    sp, sseg = SUBMARK_SOURCE
    print(f'Rendering sub-mark trial ({sp}, central target / long reach)...')
    runViz(sp, sseg, args.data_root, args.tmp_out, dump=SUBMARK_FIGURES.values())
    copyDumps(os.path.join(args.tmp_out, 'gaze', sp), SUBMARK_FIGURES, 'figframe')

    # 4. Colour correction and lens undistortion before/after (separate full-res)
    cap = cv.VideoCapture(os.path.join(args.data_root, PARTICIPANT, 'world.mp4'))
    cap.set(cv.CAP_PROP_POS_FRAMES, COLOR_FRAME)
    ok, raw = cap.read()
    cap.release()
    if ok:
        cv.imwrite(os.path.join(MEDIA, 'color_original.png'), raw)
        cv.imwrite(os.path.join(MEDIA, 'color_corregida.png'), ARUCOColorCorrection(raw.copy()))
        h, w = raw.shape[:2]
        dh = DistortionHandler(CALIB, w, h)
        cv.imwrite(os.path.join(MEDIA, 'undistort_original.png'), raw)
        cv.imwrite(os.path.join(MEDIA, 'undistort_corregida.png'), dh.undistortImage(raw))
        print('  color_* / undistort_*  <- frame', COLOR_FRAME)

    # 5. Data plots (bilingual): occlusion-over-time and the phase timeline.
    # The phase timeline spans the WHOLE cycle (sample panel -> next sample panel), so its marks are
    # read from the FULL reprocess transitions CSV (the next trial's panel lies outside this render's
    # segment). The reference trial (5,0) green_hexagon has the textbook phase ordering.
    ref_trans_full = os.path.join(DEFAULT_OUTPUT_ROOT, 'gaze', PARTICIPANT, f'trials_data_{PARTICIPANT}_transitions.csv')
    marks = readMarks(os.path.join(dump_dir, f'trials_data_{PARTICIPANT}_transitions.csv'), REF_TRIAL_NAME)
    cycle_tl = readTrialCycle(ref_trans_full, *TIMELINE_TRIAL)
    trace = runViz(PARTICIPANT, (9225, 9305), args.data_root, args.tmp_out + '_trace', trace=True)
    trace_text = (trace.stdout or '') + (trace.stderr or '')   # log() writes to stderr
    for lang in ('es', 'en'):
        plotOcclusion(trace_text, marks, lang)
        plotTimeline(cycle_tl, lang)                                    # full cycle (panel -> next panel)
        plotTimeline(cycle_tl, lang, zoom=True, out_name='timeline_fases_zoom')   # detail of the trial proper

    # 5b. Touch-detector intermediate masks (patch/ref/diff/edge/SSIM/changed) saved as SEPARATE
    # images at a CLEAN touch frame: trial (5,5) yellow_circle (strong target occlusion, fT~0.75,
    # non-triangle). A short re-run with --dump_frames captures the masks; maskFigure writes one PNG
    # per stage so the doc can order/caption them.
    cycle_clean = readTrialCycle(ref_trans_full, *CLEAN_TOUCH_TRIAL)
    touch_fr = cycle_clean.get('target_touch')
    # CLEAN baseline frame: mid-search, before the hand reaches the board (motor_onset), same cell.
    # Its change masks are ~empty -- the contrast that shows the detector only fires on the finger.
    # a few frames before the hand reaches the board (motor_onset), but late enough that the
    # tracking reference is established so the masks are dumped (board still clean here).
    mo_c, ss_c = cycle_clean.get('motor_onset'), cycle_clean.get('search_start')
    clean_fr = max(ss_c + 4, mo_c - 8) if (mo_c and ss_c) else (touch_fr - 18 if touch_fr else None)
    if touch_fr is not None:
        dframes = [f for f in (clean_fr, touch_fr) if f is not None]
        runViz(PARTICIPANT, CLEAN_TOUCH_SEGMENT, args.data_root, args.tmp_out + '_masks', dump=dframes)
        mdir = os.path.join(args.tmp_out + '_masks', 'gaze', PARTICIPANT)
        for lang in ('es', 'en'):
            maskFigure(mdir, touch_fr, lang)
            if clean_fr is not None:
                maskFigure(mdir, clean_fr, lang, out_name='mascara_limpio',
                           patch_state='limpio' if lang == 'es' else 'clean')
            # WHOLE-BOARD occlusion masks (board_occ): at the touch (arm over the board) and at the
            # clean baseline (board empty) -- the board-level counterpart of the target masks.
            boardMaskFigure(mdir, touch_fr, lang)
            if clean_fr is not None:
                boardMaskFigure(mdir, clean_fr, lang, out_name='mascara_tablero_limpio')

    # 6. Gaze trajectory over the board + sample-panel detection crop
    # Reference trajectory from the full reprocess (warm, complete) — the block-0 green_hexagon, same
    # board as the gallery (NOT block 3, not rotated) — so the early gaze is correctly phased.
    ref_seq = os.path.join(DEFAULT_OUTPUT_ROOT, 'gaze', PARTICIPANT, f'trials_data_{PARTICIPANT}_sequence.csv')
    ref_pkl = os.path.join(DEFAULT_OUTPUT_ROOT, 'gaze', PARTICIPANT, f'data_{PARTICIPANT}.pkl')
    for lang in ('es', 'en'):
        trajectoryFigure(ref_seq, ref_pkl, REF_TRIAL_NAME, lang, block_trial=(0, 7))
        # same path but emphasising the per-sample UNCERTAINTY ellipses (X = gaze, cloud = Sigma).
        ut = ('Mirada como elipse de incertidumbre (X = muestra, nube = Σ por muestra)' if lang == 'es'
              else 'Gaze as an uncertainty ellipse (X = sample, cloud = per-sample Σ)')
        trajectoryFigure(ref_seq, ref_pkl, REF_TRIAL_NAME, lang, block_trial=(0, 7),
                         out_name='trayectoria_incertidumbre', title=ut)
        # occlusion curves (fT + board_occ) for the clean reach (5,5): full cycle (phase bands +
        # this/next sample-panel marks) and a zoomed detail of the reach bump.
        bumpFigure(ref_pkl, CLEAN_TOUCH_TRIAL, lang, cycle=cycle_clean)
        bumpFigure(ref_pkl, CLEAN_TOUCH_TRIAL, lang, out_name='oclusion_bump_zoom', cycle=cycle_clean, zoom=True)
    panelFigure(debug_video)   # 049 reference: panel held frontally over the board (clean outline)

    # 6b. Trajectory GALLERY: one rich gaze path per varied target piece, from the reprocessed output
    # (Spanish only). Each isolates a single (block,trial) so paths from repeated pieces don't overlap.
    print('Trajectory gallery (varied pieces)...')
    for gp, gbt, gpiece in TRAJECTORY_GALLERY:
        gseq = os.path.join(DEFAULT_OUTPUT_ROOT, 'gaze', gp, f'trials_data_{gp}_sequence.csv')
        gpkl = os.path.join(DEFAULT_OUTPUT_ROOT, 'gaze', gp, f'data_{gp}.pkl')
        trajectoryFigure(gseq, gpkl, gpiece, 'es', block_trial=gbt,
                         out_name=f'trayectoria_{gpiece}', title=f'Recorrido de la mirada — {gpiece} ({gp})')

    # 7. ArUco fix illustration (independent of the run): markers on the original (all) vs
    # the undistorted (edge markers lost), on a participant with edge-marker loss.
    arucoFixFigure(args.data_root)

    # 8. Border / panel / touch figures (signal behaviour, bilingual).
    print('Border / panel / touch signal figures...')
    for lang in ('es', 'en'):
        # panel-detection persistence (real panels vs the misread blip) from the 001 video
        panelPersistenceFigure(args.data_root, lang)
        # touch fT noise vs real-touch peak per colour, from the cohort output
        touchNoiseFigure(DEFAULT_OUTPUT_ROOT, lang)
        # graded target_found_confidence vs binary frame_target_found, from the cohort output
        pfoundFigure(DEFAULT_OUTPUT_ROOT, lang)
        # border-contour flicker, OLD (pre-fix) vs NEW. before/after: needs a pre-fix output as
        # old_root. set EEHA_PREV_OUTPUT_ROOT to it (e.g. the archived OutputData_v1.3.0 captured
        # before the no-Canny+EMA change); skipped quietly if old==new or data missing.
        borderFlickerFigure(os.environ.get('EEHA_PREV_OUTPUT_ROOT', DEFAULT_OUTPUT_ROOT), DEFAULT_OUTPUT_ROOT, lang)

    print(f'Done. Figures in {MEDIA}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
