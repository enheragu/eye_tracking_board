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

DEFAULT_DATA_ROOT = os.environ.get('EEHA_DATA_ROOT', '/media/quique/EXTERNAL_USB1/BusquedaVisualAnalysis/InputData')
DEFAULT_OUTPUT_ROOT = os.environ.get('EEHA_OUTPUT_ROOT', '/media/quique/EXTERNAL_USB1/BusquedaVisualAnalysis/OutputData_v1.2.0')
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
           'p_search': 'búsqueda', 'p_verif': 'verificación', 'p_motor': 'motora (alcance)', 'p_with': 'retirada'},
    'en': {'fT_title': 'Target vs control occlusion over time (049, green hexagon)',
           'fT_target': 'target occlusion (fT)', 'fT_ctrl': 'control occlusion (fC)',
           'fT_thr': 'red threshold (0.13)', 'fT_x': 'frame (World)', 'fT_y': 'occluded fraction',
           'tl_title': 'Trial phases and their marks (049, green hexagon)', 'tl_x': 'frame (World)',
           'm_found': 'target seen', 'm_in': 'hand in', 'm_touch': 'touch', 'm_out': 'hand out',
           'p_search': 'search', 'p_verif': 'verification', 'p_motor': 'motor (reach)', 'p_with': 'withdraw'},
}
MARK_COL = {'m_found': '#3b82f6', 'm_in': '#e0a040', 'm_touch': '#d33333', 'm_out': '#4a9933'}
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


def trajectoryFigure(seq_csv, pkl_path, trial_name, lang, block_trial=None, out_name='trayectoria_mirada', title=None):
    """Gaze path of one trial over the board image, coloured by phase, with the target marked.
    With block_trial=(block,trial) it isolates ONE specific instance of the piece (a participant can
    repeat a piece across blocks); otherwise it uses every sample of trial_name."""
    import csv as _csv
    import ast as _ast
    import pickle as _pickle
    if not os.path.isfile(seq_csv):
        print(f'  WARNING: sequence CSV missing; skipping {out_name}')
        return
    pts, phases = [], []
    for r in _csv.DictReader(open(seq_csv)):
        if r['trial_name'] != trial_name:
            continue
        if block_trial is not None and (int(r['block_index']), int(r['trial_index'])) != block_trial:
            continue
        try:
            nc = _ast.literal_eval(r['Board norm Coord'])
        except (ValueError, SyntaxError):
            continue
        if nc[0] is None or nc[1] is None:
            continue
        pts.append((float(nc[0]), float(nc[1]))); phases.append(r['Phase'])
    if not pts:
        print(f'  WARNING: no gaze for {out_name}')
        return
    target = None
    if os.path.isfile(pkl_path):
        d = _pickle.load(open(pkl_path, 'rb'))
        cands = [d['trials_data'][block_trial]] if (block_trial in d.get('trials_data', {})) \
            else [tm for tm in d['trials_data'].values() if list(tm.keys())[0] == trial_name]
        for tm in cands:
            tn = list(tm.values())[0].get('target_norm_coord')
            if tn:
                t = tn[0] if isinstance(tn[0], (list, tuple)) else tn   # flat or nested
                if t and t[0] is not None:
                    target = t
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
    for ph in sorted(set(phases)):
        idx = [i for i, p in enumerate(phases) if p == ph]
        ax.scatter([xs[i] for i in idx], [ys[i] for i in idx], s=20, zorder=3,
                   c=PHASE_COL.get(ph, '#888'), label=s['leg'].get(ph, ph),
                   edgecolors='white', linewidths=0.3)
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


def plotTimeline(marks, lang):
    need = ['search_start', 'target_found', 'motor_onset', 'target_touch', 'hand_exit']
    if not all(k in marks for k in need):
        print('  WARNING: missing marks; skipping timeline')
        return
    s = STR[lang]
    ss, tf, mo, tc, he = (marks[k] for k in need)
    phases = [(s['p_search'], ss, tf, '#d6e4f5'), (s['p_verif'], tf, mo, '#cfe0f5'),
              (s['p_motor'], mo, tc, '#cdebc6'), (s['p_with'], tc, he, '#e3d4f5')]
    mk = [('m_found', tf), ('m_in', mo), ('m_touch', tc), ('m_out', he)]
    fig, ax = plt.subplots(figsize=(8, 2.2))
    for name, a, b, col in phases:
        ax.barh(0, max(b - a, 0.4), left=a, height=0.5, color=col, edgecolor='#555')
        ax.text((a + b) / 2, 0, name, ha='center', va='center', fontsize=8)
    for key, fr in mk:
        ax.axvline(fr, color=MARK_COL[key], ls=':', lw=1.3)
        ax.text(fr, 0.32, s[key], ha='center', fontsize=7, color=MARK_COL[key])
    ax.set_yticks([]); ax.set_xlabel(s['tl_x']); ax.set_ylim(-0.4, 0.5); ax.set_title(s['tl_title'])
    fig.tight_layout(); fig.savefig(os.path.join(MEDIA, f'timeline_fases{_suffix(lang)}.png'), dpi=150)
    plt.close(fig)
    print(f'  timeline_fases{_suffix(lang)}.png OK')


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

    # 5. Data plots (bilingual): occlusion-over-time and the phase timeline
    marks = readMarks(os.path.join(dump_dir, f'trials_data_{PARTICIPANT}_transitions.csv'), REF_TRIAL_NAME)
    trace = runViz(PARTICIPANT, (9225, 9305), args.data_root, args.tmp_out + '_trace', trace=True)
    trace_text = (trace.stdout or '') + (trace.stderr or '')   # log() writes to stderr
    for lang in ('es', 'en'):
        plotOcclusion(trace_text, marks, lang)
        plotTimeline(marks, lang)

    # 6. Gaze trajectory over the board + sample-panel detection crop
    # Reference trajectory from the full reprocess (warm, complete) — the block-0 green_hexagon, same
    # board as the gallery (NOT block 3, not rotated) — so the early gaze is correctly phased.
    ref_seq = os.path.join(DEFAULT_OUTPUT_ROOT, 'gaze', PARTICIPANT, f'trials_data_{PARTICIPANT}_sequence.csv')
    ref_pkl = os.path.join(DEFAULT_OUTPUT_ROOT, 'gaze', PARTICIPANT, f'data_{PARTICIPANT}.pkl')
    for lang in ('es', 'en'):
        trajectoryFigure(ref_seq, ref_pkl, REF_TRIAL_NAME, lang, block_trial=(0, 7))
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

    print(f'Done. Figures in {MEDIA}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
