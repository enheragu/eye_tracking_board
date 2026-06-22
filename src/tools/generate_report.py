#!/usr/bin/env python3
# encoding: utf-8

"""
    Builds a single self-contained HTML report (tabs + version checkboxes, inline SVG, no
    external dependencies -- CSS and JS inline, every chart an inline <svg>, no CDN / remote
    font / external image, so it opens offline from an email attachment) comparing the
    processing output of several versions, for the research team and for developer debug.
    Tabs: Detección, Tipo de fin, Cobertura marcas, Resultado del trial (reliable error_type per
    trial from v1.3 + the graded target_found_confidence from the v1.4 per-sample uncertainty model),
    Duración del trial, Toque, Frecuencias, and Comportamiento (an exploratory behaviour summary
    of the latest version: gaze-by-target colour/shape matrices, per-target found-vs-touched-vs-
    off-target, a time-to-X selector, and per-colour response time across blocks). Each tab
    carries its own legend.

    Example:
        python3 src/tools/generate_report.py \
            --roots /path/OutputData_v1.1.0 /path/OutputData_v1.2.0 /path/OutputData_v1.3.0 \
            --data_root /path/InputData --out /path/report.html
"""

import os
import sys
import csv
import glob
import html
import math
import pickle
import argparse
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from src.core.StateMachineHandler import _cell_mass, TARGET_FOUND_MASS_THR_DEFAULT

# The 'objetivo visto' criterion mirrors the writer EXACTLY: the SAME bivariate cell-mass
# (_cell_mass, imported -- not a marginal re-implementation) and the SAME threshold, resolved from
# the SAME env override AND the SAME default constant as StateMachine (TARGET_FOUND_MASS_THR_DEFAULT,
# imported -- so the report and the writer can never silently diverge). Earlier this file
# re-implemented the mass as an independent-axes product AND hardcoded 0.30, so the report
# disagreed with the published frame_target_found / target_found_confidence.
TARGET_FOUND_THR = float(os.environ.get('EEHA_TARGET_FOUND_MASS_THR', TARGET_FOUND_MASS_THR_DEFAULT))

ERROR_PREFIXES = ('missing_trial_error', 'transition_error', 'end_of_video_error')

# Human-readable labels + tint classes for the per-sample uncertainty model's error_type
# (pipeline >= 1.3): the trial-level anomaly derived from RELIABLE signals only. Empty when
# the reach was incomplete (no hand in/out) or there is no target cell.
ERROR_TYPE_LABEL = {
    'correct': 'correcto',
    'off_target': 'fuera de objetivo',
    'no_touch': 'sin toque',
    '': 'sin datos',
}
ERROR_TYPE_CLASS = {
    'correct': 'c-target',
    'off_target': 'c-error',
    'no_touch': 'c-panel',
    '': 'c-missing',
}


def versionLabel(root):
    # OutputData_v1.1.0 -> v1.1.0 ; tolerate the historical "OutpuData" typo
    base = os.path.basename(root.rstrip('/'))
    for tag in ('_v', 'Data_v'):
        if tag in base:
            return 'v' + base.split('_v')[-1]
    return base


def _toInt(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _toFloat(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _versionTuple(sw):
    try:
        return tuple(int(x) for x in str(sw).split('.')[:2])
    except (ValueError, AttributeError):
        return (0, 0)


def _errTrial(name):
    return {'name': name, 'status': 'error', 'duration': 0.0, 'search_start': None, 'init': None,
            'target_found': None, 'motor_onset': None, 'touch': None, 'exit': None, 'end': None,
            'reach_mm': None, 'anticip': None, 'time_to_target_s': None, 'search_duration_s': None,
            'reach_duration_s': None, 'withdraw_duration_s': None,
            'found_confidence': None, 'error_type': None, 'validated_piece': None, 'validation': None}


def _targetFound(metrics, board_size=(8, 5)):
    """Frame of the 'objetivo visto' mark = start of the first I-DT FIXATION whose mean ellipse
    MASS on the target cell reaches the mass threshold (TARGET_FOUND_THR) -- the SAME rule the writer uses
    (store_results): mass-based, falling back to the centroid majority vote when a fixation has no
    covariance (no uncertainty model). target_cord is [row,col] but sequence board_coord is
    [col,row], compared swapped. None when never found (or no sequence)."""
    seq = metrics.get('sequence')
    target = metrics.get('target_cord')
    if not seq or not target or target[0] is None:
        return None
    target_colrow = [int(target[1]), int(target[0])]
    pts = [s for s in seq if isinstance(s, dict) and s.get('norm_board_coord') and s['norm_board_coord'][0] is not None]
    ncols, nrows = board_size
    disp_max, min_n, thr = 0.06, 6, TARGET_FOUND_THR   # StateMachine defaults: _fixation_dispersion / _min_fixation_samples / _mass_threshold
    i = 0
    while i < len(pts):
        xs, ys = [pts[i]['norm_board_coord'][0]], [pts[i]['norm_board_coord'][1]]
        j = i + 1
        while j < len(pts):
            x, y = pts[j]['norm_board_coord']
            if (max(xs + [x]) - min(xs + [x])) + (max(ys + [y]) - min(ys + [y])) > disp_max:
                break
            xs.append(x); ys.append(y); j += 1
        if (j - i) >= min_n:
            masses = []
            for k in range(i, j):
                cov = pts[k].get('norm_board_cov')
                if cov is not None:
                    nb = pts[k]['norm_board_coord']
                    masses.append(_cell_mass(nb[0], nb[1], cov,
                                             target_colrow[0], target_colrow[1], ncols, nrows))
            if masses:
                if sum(masses) / len(masses) >= thr:
                    return int(pts[i]['frame'])
            else:
                on = sum(1 for k in range(i, j) if list(pts[k].get('board_coord', [])) == target_colrow)
                if on * 2 >= (j - i):
                    return int(pts[i]['frame'])
            i = j
        else:
            i += 1
    return None


def _targetFoundConfidence(metrics, board_size=(8, 5)):
    """Graded confidence [0,1] that the participant looked at the target piece, mirroring
    the writer (store_results / target_found_confidence): for each I-DT fixation, the mean
    per-sample Gaussian MASS on the target cell, taken as the max across the trial's
    fixations. Returns None when there is no per-sample uncertainty model (no norm_board_cov)
    -- i.e. the calibration has no uncertainty block -- so the report does not show a fake 0."""
    seq = metrics.get('sequence')
    target = metrics.get('target_cord')
    if not seq or not target or target[0] is None:
        return None
    ncols, nrows = board_size
    target_colrow = [int(target[1]), int(target[0])]
    pts = [s for s in seq if isinstance(s, dict) and s.get('norm_board_coord') and s['norm_board_coord'][0] is not None]
    if not any(s.get('norm_board_cov') for s in pts):
        return None
    disp_max, min_n = 0.06, 6
    i, tf_conf = 0, 0.0
    while i < len(pts):
        xs, ys = [pts[i]['norm_board_coord'][0]], [pts[i]['norm_board_coord'][1]]
        j = i + 1
        while j < len(pts):
            x, y = pts[j]['norm_board_coord']
            if (max(xs + [x]) - min(xs + [x])) + (max(ys + [y]) - min(ys + [y])) > disp_max:
                break
            xs.append(x); ys.append(y); j += 1
        if (j - i) >= min_n:
            masses = []
            for k in range(i, j):
                cov = pts[k].get('norm_board_cov')
                if cov is not None:
                    nb = pts[k]['norm_board_coord']
                    masses.append(_cell_mass(nb[0], nb[1], cov,
                                             target_colrow[0], target_colrow[1], ncols, nrows))
            if masses:
                tf_conf = max(tf_conf, sum(masses) / len(masses))
            i = j
        else:
            i += 1
    return round(tf_conf, 3)


def loadVersion(root, topic):
    """participant -> {(block,trial): {name,status,duration,marks...}} read from the PKL, the SOURCE
    OF TRUTH (the per-participant CSV is a user-facing view that, from 1.1.0 on, drops the
    missing/transition errors, so reading it hid losses). Frames come from the per-trial *_capture
    marks; target_found is derived from the gaze sequence and the durations from the frame diffs / fps
    using each version's start criterion (1.2 re-bases the start to the search onset, earlier ones
    use the board-confirm init), so the values match the CSV for real trials AND the errors appear."""
    topic_dir = os.path.join(root, topic)
    out = {}
    if not os.path.isdir(topic_dir):
        return out
    for participant in sorted(os.listdir(topic_dir)):
        pkl_path = os.path.join(topic_dir, participant, f'data_{participant}.pkl')
        if not os.path.isfile(pkl_path):
            continue
        try:
            with open(pkl_path, 'rb') as pf:
                d = pickle.load(pf)
        except (pickle.UnpicklingError, EOFError, OSError, AttributeError):
            continue
        fps = d.get('video_fps') or 30.0
        use_early = _versionTuple(d.get('sw_version')) >= (1, 2)   # 1.2 re-bases start to search onset
        trials = {}
        for (b, t), tm in d.get('trials_data', {}).items():
            if b is None or b == -1 or t == -1:
                continue
            name = next(iter(tm))
            if name.startswith(ERROR_PREFIXES):
                trials[(b, t)] = _errTrial(name)
                continue
            m = tm[name]

            def fr(key):
                val = m.get(key)
                return int(val) if isinstance(val, (int, float)) and val not in (None, -1) else None
            init, end = fr('init_capture'), fr('end_capture')
            early = fr('early_init_capture')
            motor, touch, exit_ = fr('motor_onset_capture'), fr('target_touch_capture'), fr('hand_exit_capture')
            start = early if (use_early and early is not None) else init
            tfound = _targetFound(m)
            # Per-sample uncertainty model (pipeline >= 1.3): read the trial-level outputs
            # straight from the PKL (source of truth). error_type / gaze_validated_* are
            # written by _deriveWrongPiece before serialisation; found_confidence is graded
            # and recomputed here the same way the writer does (it lives only in the CSV).
            err = m.get('error_type')
            fval = fr('frame_validation')

            def dur(a, c):
                return (c - a) / fps if (a is not None and c is not None) else None
            trials[(b, t)] = {'name': name, 'status': m.get('status'),
                              'duration': dur(start, end) or 0.0,
                              'search_start': early, 'init': init, 'target_found': tfound,
                              'motor_onset': motor, 'touch': touch, 'exit': exit_, 'end': end,
                              'reach_mm': None, 'anticip': None,
                              'time_to_target_s': dur(start, tfound),
                              'search_duration_s': dur(start, motor),
                              'reach_duration_s': dur(motor, touch),
                              'withdraw_duration_s': dur(touch, exit_),
                              'found_confidence': _targetFoundConfidence(m),
                              'error_type': (err if err else None),
                              'validated_piece': m.get('gaze_validated_piece') or None,
                              'validation': fval}
        out[participant] = trials
    return out


def gazeFrequencies(data_root):
    """participant -> (world_fps, gaze_rate_hz, continuity)"""
    freqs = {}
    if not data_root or not os.path.isdir(data_root):
        return freqs
    import cv2 as cv
    for participant in sorted(os.listdir(data_root)):
        pdir = os.path.join(data_root, participant)
        gt_path = os.path.join(pdir, 'gaze_timestamps.npy')
        wt_path = os.path.join(pdir, 'world_timestamps.npy')
        if not (os.path.isfile(gt_path) and os.path.isfile(wt_path)):
            continue
        try:
            gt = np.sort(np.load(gt_path)); wt = np.sort(np.load(wt_path))
        except (EOFError, ValueError, OSError) as e:
            # A truncated / 0-byte / still-copying .npy must not crash the whole report;
            # skip that participant's frequencies (it will appear once fully copied).
            print(f"  [freq] aviso: no se pudo leer {participant} ({e}); se omite.")
            continue
        world_fps = (len(wt) - 1) / (wt[-1] - wt[0]) if len(wt) > 1 and wt[-1] > wt[0] else float('nan')
        iv = np.diff(gt); iv = iv[iv > 0]
        if iv.size:
            med = np.median(iv)
            rate = 1.0 / med
            cont = float(np.mean(np.abs(iv - med) <= 0.2 * med))
        else:
            rate, cont = float('nan'), float('nan')
        freqs[participant] = (world_fps, rate, cont)
    return freqs


def isValid(trial):
    return trial is not None and not trial['name'].startswith(ERROR_PREFIXES)


# Colour per finish status / detection, used as CSS classes
STATUS_CLASS = {
    'test_finish_target_reached': 'c-target',
    'test_finish_execution': 'c-contour',
    'test_finish_by_next_panel': 'c-panel',
    'test_finish_by_end_of_video': 'c-eov',
}


def buildDetectionSummary(versions, labels):
    """Detection HEALTH per participant x version: how many real trials were segmented vs
    how many came out as detection errors. NOT a trial-by-trial matrix — the errors are a
    processing-quality figure, not trials of the participant, so they live only here."""
    participants = sorted(set().union(*[set(v.keys()) for v in versions]) if versions else set())
    head = ('<tr><th class="idx">#</th><th>Participante</th>'
            + ''.join(f'<th class="vcol v{i}">{html.escape(lab)}<br>reales / errores</th>'
                      for i, lab in enumerate(labels)) + '</tr>')
    rows = ''
    for n, p in enumerate(participants, 1):
        cells = ''
        for i, v in enumerate(versions):
            trials = v.get(p, {})
            valid = sum(1 for t in trials.values() if isValid(t))
            errs = sum(1 for t in trials.values() if not isValid(t))
            cls = 'c-ok' if errs == 0 else ('c-panel' if errs <= 3 else 'c-error')
            disp = f'{valid} / {errs}' if (valid or errs) else '&nbsp;'
            cells += f'<td class="vcol v{i} {cls}">{disp}</td>'
        rows += f'<tr><td class="idx">{n}</td><td class="name">{html.escape(p)}</td>{cells}</tr>'
    note = ('<p><b>Salud de la detección.</b> Trials <b>reales</b> segmentados frente a '
            '<b>errores de detección</b> por participante y versión. Un error '
            '(<code>missing_trial_error</code> = no se detectó el panel esperado; '
            '<code>transition_error</code> = apareció otro panel antes de recoger datos) '
            '<b>no es un trial del participante</b>: es un trial que el procesado no pudo '
            'segmentar (panel poco visible, pocos ArUcos, desincronización puntual). '
            'En el resto de pestañas solo aparecen los trials reales.</p>')
    return note + f'<table>{head}{rows}</table>'


# Human-readable finish-type labels (the raw status is an internal state name)
STATUS_LABEL = {
    'test_finish_execution': 'cruce del borde',
    'test_finish_by_next_panel': 'panel siguiente',
    'test_finish_by_end_of_video': 'fin de vídeo',
    'test_finish_target_reached': 'objetivo alcanzado',
}


def cellStatus(trial):
    if trial is None:
        return ('', 'c-missing')            # this version never reached this slot (not attempted)
    if not isValid(trial):
        return ('perdido', 'c-error')       # detected as missing/transition error -> a VISIBLE loss
    label = STATUS_LABEL.get(trial['status'], trial['status'].replace('test_finish_', '').replace('_', ' '))
    return (label, STATUS_CLASS.get(trial['status'], 'c-ok'))


def cellDuration(trial):
    if trial is None or not isValid(trial):
        return ('', 'c-missing')
    return (f"{trial['duration']:.2f}", STATUS_CLASS.get(trial['status'], 'c-ok'))


def buildTable(versions, labels, cell_fn):
    """Matrix layout: participants are COLUMNS, (block,trial) are ROWS, and the versions
    are stacked sub-cells inside each cell. Far more compact and scannable than one row
    per participant-trial. Version checkboxes toggle the sub-cells (class v{i})."""
    participants = sorted(set().union(*[set(v.keys()) for v in versions]) if versions else set())
    # Only slots that are a REAL trial in at least one version/participant (errors and the
    # -1 discards are not trials and must not create rows).
    keys = sorted({(b, t) for v in versions for p in participants
                   for (b, t), tr in v.get(p, {}).items() if isValid(tr)}) if participants else []
    head = ('<tr><th>Blk</th><th>Tr</th><th>Trial</th>'
            + ''.join(f'<th>{html.escape(p)}</th>' for p in participants) + '</tr>')
    rows = []
    for (block, trial) in keys:
        name = ''
        for p in participants:
            for v in versions:
                t = v.get(p, {}).get((block, trial))
                if t and isValid(t):
                    name = t['name']; break
            if name:
                break
        cells = ''
        for p in participants:
            subs = ''
            for i, v in enumerate(versions):
                txt, cls = cell_fn(v.get(p, {}).get((block, trial)))
                disp = html.escape(str(txt)) if str(txt) else '&nbsp;'
                subs += f'<span class="sub v{i} {cls}" title="{html.escape(labels[i])}">{disp}</span>'
            cells += f'<td class="mcell">{subs}</td>'
        rows.append(f'<tr><td>{block}</td><td>{trial}</td><td class="name">{html.escape(name)}</td>{cells}</tr>')
    return f'<table class="matrix">{head}{"".join(rows)}</table>'


def buildFreqTable(freqs):
    rows = ''
    for n, (participant, (wfps, rate, cont)) in enumerate(sorted(freqs.items()), 1):
        warn = ' c-error' if (cont == cont and cont < 0.95) else ''
        rows += (f'<tr><td class="idx">{n}</td><td>{html.escape(participant)}</td>'
                 f'<td>{wfps:.2f}</td><td class="{warn.strip()}">{rate:.1f}</td>'
                 f'<td class="{warn.strip()}">{cont*100:.1f}%</td></tr>')
    return ('<table><tr><th class="idx">#</th><th>Participante</th><th>Vídeo World (fps)</th>'
            '<th>Gaze (Hz)</th><th>Continuidad</th></tr>' + rows + '</table>')


def touchTime(trial):
    """Trial time (s) from the start to the target touch (the user-level trial end),
    derived from the frame marks: (touch-init)/(end-init) * trial_duration_s. The touch
    is observed in the motor phase, so it usually lands AFTER the border crossing
    (frame_end) and this time is a bit longer than trial_duration_s. None when the touch
    was not detected (or the version has no marks)."""
    if not trial:
        return None
    init, touch, end = trial.get('init'), trial.get('touch'), trial.get('end')
    if init is None or touch is None or end is None or end <= init:
        return None
    return (touch - init) / (end - init) * trial['duration']


def buildTouchTable(versions, labels):
    """Per trial: the border-crossing duration of every version (what was historically
    considered 'trial time') next to the touch-based time of the latest version, and
    their difference. Shows whether the finger/touch was detected at all (— if not)."""
    participants = sorted(set().union(*[set(v.keys()) for v in versions]) if versions else set())
    last = len(versions) - 1
    head = ('<tr><th>Part.</th><th>Blk</th><th>Tr</th><th>Trial</th>'
            + ''.join(f'<th class="vcol v{i}">{html.escape(lab)}<br>duración del trial (s)<br>'
                      f'<span style="font-weight:normal">inicio → cruce del borde</span></th>'
                      for i, lab in enumerate(labels))
            + f'<th class="vcol v{last}">{html.escape(labels[last])}<br>tiempo hasta tocar (s)<br>'
            f'<span style="font-weight:normal">inicio → toque de la pieza</span></th>'
            + f'<th class="vcol v{last}">alcance dentro del tablero (s)<br>'
            f'<span style="font-weight:normal">cruce del borde → toque</span></th></tr>')
    rows = []
    n_touch = n_trials = 0
    for participant in participants:
        keys = sorted({(b, t) for v in versions for (b, t), tr in v.get(participant, {}).items() if isValid(tr)})
        for (block, trial) in keys:
            name = ''
            for v in versions:
                t = v.get(participant, {}).get((block, trial))
                if t and isValid(t):
                    name = t['name']; break
            cells = ''
            for i, v in enumerate(versions):
                t = v.get(participant, {}).get((block, trial))
                if t is None:
                    cells += f'<td class="vcol v{i} c-missing"></td>'
                elif not isValid(t):
                    cells += f'<td class="vcol v{i} c-error">-</td>'
                else:
                    cells += f'<td class="vcol v{i} {STATUS_CLASS.get(t["status"], "c-ok")}">{t["duration"]:.2f}</td>'
            latest = versions[last].get(participant, {}).get((block, trial)) if versions else None
            if latest is not None and isValid(latest):
                n_trials += 1
            ts = touchTime(latest) if (latest is not None and isValid(latest)) else None
            if ts is None:
                touch_cell = f'<td class="vcol v{last} c-missing">&mdash;</td>'
                delta_cell = f'<td class="vcol v{last} c-missing">&mdash;</td>'
            else:
                n_touch += 1
                # Touch is observed in the motor phase, after the border: the reach time
                # inside the board (touch - border) is normally positive.
                reach = ts - latest['duration']
                touch_cell = f'<td class="vcol v{last} c-target">{ts:.2f}</td>'
                delta_cell = f'<td class="vcol v{last} c-target">{reach:+.2f}</td>'
            rows.append(f'<tr><td>{html.escape(participant)}</td><td>{block}</td><td>{trial}</td>'
                        f'<td class="name">{html.escape(name)}</td>{cells}{touch_cell}{delta_cell}</tr>')
    pct = (100.0 * n_touch / n_trials) if n_trials else 0.0
    summary = (f'<p>Toque detectado en <b>{n_touch}/{n_trials}</b> trials válidos de '
               f'<b>{html.escape(labels[last])}</b> ({pct:.0f}%). Es una marca de '
               f'<i>mejor esfuerzo</i>: &mdash; indica que no se detectó (no afecta al cierre del trial).</p>')
    # The per-trial table is long (participants × trials) and pushes the rest of the tab down, so
    # it is COLLAPSED by default behind the summary; expand it only to inspect a specific trial.
    return (summary + '<details><summary style="cursor:pointer;font-weight:bold;margin:6px 0;'
            'color:#1f77b4">▸ Ver el detalle por trial (tabla completa)</summary>'
            f'<table>{head}{"".join(rows)}</table></details>')


# --- Self-contained inline-SVG charts (no external library / CDN) -------------------

def _svgBars(pairs, width=760, bar_h=15, gap=5, unit='%', color='#1f77b4', label_w=64):
    """Horizontal bar chart from [(label, value 0..100), ...] as an inline <svg> string."""
    if not pairs:
        return '<p>Sin datos.</p>'
    height = gap + len(pairs) * (bar_h + gap)
    chart_w = width - label_w - 46
    parts = [f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
             f'font-family="Arial" role="img">']
    for i, (lab, val) in enumerate(pairs):
        y = gap + i * (bar_h + gap)
        val = max(0.0, min(100.0, float(val)))
        bw = val / 100.0 * chart_w
        parts.append(f'<text x="0" y="{y+bar_h-3}" font-size="11" fill="#333">{html.escape(str(lab))}</text>')
        parts.append(f'<rect x="{label_w}" y="{y}" width="{chart_w}" height="{bar_h}" fill="#eee"/>')
        parts.append(f'<rect x="{label_w}" y="{y}" width="{bw:.1f}" height="{bar_h}" fill="{color}"/>')
        parts.append(f'<text x="{label_w+chart_w+4}" y="{y+bar_h-3}" font-size="10" fill="#333">{val:.0f}{unit}</text>')
    parts.append('</svg>')
    return ''.join(parts)


def _svgHist(values, bins=20, lo=0.0, hi=1.0, width=760, height=200, color='#cdebc6',
             stroke='#4a9', marker=None, marker_label=''):
    """Histogram of values in [lo,hi] as inline <svg>, with an optional vertical marker."""
    if not values:
        return '<p>Sin datos.</p>'
    counts = [0] * bins
    span = (hi - lo) or 1.0
    for v in values:
        b = int((v - lo) / span * bins)
        b = max(0, min(bins - 1, b))
        counts[b] += 1
    mx = max(counts) or 1
    pad_l, pad_b, pad_t = 34, 22, 10
    plot_w = width - pad_l - 10
    plot_h = height - pad_b - pad_t
    parts = [f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
             f'font-family="Arial" role="img">']
    # axes
    parts.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+plot_h}" stroke="#999"/>')
    parts.append(f'<line x1="{pad_l}" y1="{pad_t+plot_h}" x2="{pad_l+plot_w}" y2="{pad_t+plot_h}" stroke="#999"/>')
    bw = plot_w / bins
    for i, c in enumerate(counts):
        bh = c / mx * plot_h
        x = pad_l + i * bw
        y = pad_t + plot_h - bh
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(0,bw-1):.1f}" height="{bh:.1f}" '
                     f'fill="{color}" stroke="{stroke}" stroke-width="0.5"/>')
    # x ticks at lo, mid, hi
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = pad_l + frac * plot_w
        parts.append(f'<text x="{x:.1f}" y="{height-6}" font-size="9" fill="#555" text-anchor="middle">{lo+frac*span:.2f}</text>')
    parts.append(f'<text x="2" y="{pad_t+8}" font-size="9" fill="#555">{mx}</text>')
    if marker is not None and lo <= marker <= hi:
        x = pad_l + (marker - lo) / span * plot_w
        parts.append(f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{pad_t+plot_h}" stroke="#d33" stroke-width="2" stroke-dasharray="4 3"/>')
        parts.append(f'<text x="{x+4:.1f}" y="{pad_t+12}" font-size="10" fill="#d33">{html.escape(marker_label)}</text>')
    parts.append('</svg>')
    return ''.join(parts)


def buildTouchDiagnostics(versions, labels):
    """Inline-SVG diagnostics for the target-touch mark of the latest version:
    per-participant coverage, and where the touch lands inside the motor window
    [border-in -> hand-exit] (justifies the interpolation fallback)."""
    if not versions:
        return '<p>Sin datos.</p>'
    latest = versions[-1]
    cov_pairs = []
    fracs = []
    n_touch = n_valid = n_interp = 0
    for participant in sorted(latest.keys()):
        trials = latest[participant]
        valid = [t for t in trials.values() if isValid(t)]
        if not valid:
            continue
        tt = sum(1 for t in valid if t.get('touch') is not None)
        cov_pairs.append((participant, 100.0 * tt / len(valid)))
        n_valid += len(valid); n_touch += tt
        for t in valid:
            mo, to, ex = t.get('end'), t.get('touch'), t.get('exit')
            if to is not None and ex is not None and mo is not None and ex > mo:
                fracs.append((to - mo) / (ex - mo))
            if to is None and ex is not None and mo is not None and ex > mo:
                n_interp += 1
    f_med = float(np.median(fracs)) if fracs else 0.24
    cov_pct = (100.0 * n_touch / n_valid) if n_valid else 0.0
    est_pct = (100.0 * (n_touch + n_interp) / n_valid) if n_valid else 0.0
    intro = (f'<p>Marca de <b>toque</b> de la versión <b>{html.escape(labels[-1])}</b>. '
             f'Detectado en <b>{n_touch}/{n_valid}</b> trials válidos (<b>{cov_pct:.0f}%</b>). '
             f'El toque es de <i>mejor esfuerzo</i>; abajo, dónde cae dentro de la ventana motora.</p>')
    cov = ('<h3>Cobertura del toque por participante</h3>'
           + _svgBars(cov_pairs, color='#2a8'))
    hist = ('<h3>¿En qué momento de la fase motora ocurre el toque?</h3>'
            + f'<p>La <b>fase motora</b> va desde que la mano <b>cruza el borde</b> del tablero '
            f'hasta que <b>sale</b>. Normalizamos ese intervalo a [0, 1]: <b>0</b> = justo al '
            f'cruzar el borde, <b>1</b> = justo al salir la mano. El histograma muestra, sobre '
            f'todos los trials, en qué punto de ese recorrido cae el toque. La mediana <b>{f_med:.2f}</b> '
            f'significa que el toque ocurre <b>pronto</b> (a ~{f_med*100:.0f}% del recorrido): la mano '
            f'cruza, toca casi enseguida, y luego tarda más en retirarse.</p>'
            + _svgHist(fracs, bins=20, lo=0.0, hi=1.0, marker=f_med, marker_label=f'mediana {f_med:.2f}'))
    interp = (f'<h3>Interpolación (capa de análisis)</h3>'
              f'<p>Para los trials sin toque pero con borde-entra y mano-sale, se puede '
              f'<b>estimar</b> el toque como <code>borde + {f_med:.2f}·(salida − borde)</code>. '
              f'Eso cubriría <b>{n_interp}</b> trials adicionales → cobertura estimada '
              f'<b>{est_pct:.0f}%</b>. Es una estimación (sesgo de selección: se calibra con los '
              f'toques detectados), no un valor medido; las marcas crudas se publican sin tocar.</p>')
    return intro + cov + hist + interp


def buildMarkCoverage(versions, labels):
    """Per version, the % of VALID trials that have each event mark detected. Answers
    'what fraction of trials do we have motion / touch / hand-exit / search-start for'."""
    marks = [('search_start', 'Inicio de búsqueda (retirada del panel)'),
             ('target_found', 'Objetivo visto (gaze sobre la casilla)'),
             ('motor_onset', 'Entrada de la mano (motor_onset)'),
             ('touch', 'Toque de la pieza (dedo)'),
             ('exit', 'Salida de la mano (hand_exit)')]
    head = ('<tr><th>Marca / fase</th>'
            + ''.join(f'<th class="vcol v{i}">{html.escape(lab)}</th>' for i, lab in enumerate(labels))
            + '</tr>')
    # Per version: does it have these marks at all? The frame_* marks are 1.1.0+, so older
    # versions must show "n/a", not a misleading 0%.
    valid_by_v = [[t for p in v.values() for t in p.values() if isValid(t)] for v in versions]
    has_marks = [any(t.get('motor_onset') is not None or t.get('touch') is not None for t in valid)
                 for valid in valid_by_v]
    rows = ''
    for key, label in marks:
        cells = ''
        for i, v in enumerate(versions):
            valid = valid_by_v[i]
            if not has_marks[i]:
                cells += f'<td class="vcol v{i} c-missing">n/a</td>'
                continue
            n = len(valid)
            cov = (100.0 * sum(1 for t in valid if t.get(key) is not None) / n) if n else 0.0
            cls = 'c-ok' if cov >= 85 else ('c-panel' if cov >= 60 else 'c-error')
            cells += f'<td class="vcol v{i} {cls}">{cov:.0f}%</td>'
        rows += f'<tr><td class="name">{html.escape(label)}</td>{cells}</tr>'
    note = ('<p>Porcentaje de <b>trials válidos</b> con cada marca detectada, por versión. Las '
            'marcas son <b>aditivas</b>: que falte una no afecta a las demás ni a la segmentación '
            '(la columna de esa marca queda en blanco en el CSV). El <b>inicio de búsqueda</b>, el '
            '<b>movement_onset</b> por oclusión y esta cobertura por marca son de <b>v1.2</b>; '
            '<b>motor_onset</b> (contorno) y <b>toque</b> existen desde 1.1.0.</p>')
    per_part = buildStageCoveragePerParticipant(versions, labels)
    return note + f'<table>{head}{rows}</table>' + per_part


def buildStageCoveragePerParticipant(versions, labels):
    """Per PARTICIPANT (latest version), % of valid trials that have each mark — the same idea
    as 'touch coverage per participant', but for every stage. Reveals uneven participants."""
    if not versions:
        return ''
    latest = versions[-1]
    marks = [('target_found', 'Objetivo visto'), ('motor_onset', 'Mano entra'),
             ('touch', 'Toque'), ('exit', 'Mano sale')]
    head = ('<tr><th class="idx">#</th><th>Participante</th>'
            + ''.join(f'<th>{lab}</th>' for _, lab in marks) + '<th>Trials</th></tr>')
    rows = ''
    n = 0
    for p in sorted(latest.keys()):
        valid = [t for t in latest[p].values() if isValid(t)]
        if not valid:
            continue
        n += 1
        cells = ''
        for key, _ in marks:
            cov = 100.0 * sum(1 for t in valid if t.get(key) is not None) / len(valid)
            cls = 'c-ok' if cov >= 85 else ('c-panel' if cov >= 60 else 'c-error')
            barcol = '#4a9933' if cov >= 85 else ('#e0a040' if cov >= 60 else '#d33333')
            # Inline bar so the coverage reads as a per-participant histogram at a glance.
            bar = ('<span style="display:inline-block;width:54px;height:9px;background:#e6e6e6;'
                   'border-radius:2px;vertical-align:middle;overflow:hidden">'
                   f'<span style="display:block;height:9px;width:{cov:.0f}%;background:{barcol}"></span></span>')
            cells += f'<td class="{cls}" style="white-space:nowrap">{cov:.0f}% {bar}</td>'
        rows += f'<tr><td class="idx">{n}</td><td class="name">{html.escape(p)}</td>{cells}<td>{len(valid)}</td></tr>'
    note = (f'<h3>Cobertura de detección por participante — {html.escape(labels[-1])}</h3>'
            '<p>% de trials válidos con cada marca detectada (<b>objetivo visto</b>, <b>mano entra</b>, '
            '<b>toque</b>, <b>mano sale</b>), <b>por participante</b>, con barra (verde &ge;85%, '
            'naranja 60–85%, rojo &lt;60%). Revela qué participantes tienen cobertura desigual de cada '
            'etapa, no solo la media de la versión.</p>')
    return note + f'<table>{head}{rows}</table>'


def buildPhaseHistograms(versions, labels):
    """Distribution (histogram) of each per-phase duration, one row per phase and one column
    per version: shows the SPREAD, not just an average, and lets versions be compared. The
    durations are 1.2.0+ (older versions show n/a). Median marked with a vertical line."""
    import statistics as _st
    phases = [('time_to_target_s', 'Tiempo hasta ver el objetivo', 3.0),
              ('search_duration_s', 'Búsqueda (inicio → mano entra)', 4.0),
              ('reach_duration_s', 'Alcance (mano entra → toque)', 1.5),
              ('withdraw_duration_s', 'Retirada (toque → mano sale)', 2.0)]
    note = ('<p>Distribución de la duración (s) de cada <b>etapa</b> del trial, por versión. '
            'Cada histograma muestra cuántos trials caen en cada rango de duración; la línea '
            'vertical es la <b>mediana</b>. Es más informativo que una media: revela colas y '
            'asimetrías (p. ej. el alcance se acumula cerca de 0 en objetivos de borde).</p>')
    head = ('<tr><th>Etapa</th>'
            + ''.join(f'<th class="vcol v{i}">{html.escape(lab)}</th>' for i, lab in enumerate(labels))
            + '</tr>')
    rows = ''
    for key, label, hi in phases:
        cells = ''
        for i, v in enumerate(versions):
            vals = [t[key] for p in v.values() for t in p.values()
                    if isValid(t) and t.get(key) is not None]
            if not vals:
                cells += f'<td class="vcol v{i} c-missing">n/a</td>'
                continue
            med = _st.median(vals)
            svg = _svgHist(vals, bins=24, lo=0.0, hi=hi, width=300, height=120,
                           marker=med, marker_label=f'mediana {med:.2f}s')
            cells += f'<td class="vcol v{i}">{svg}<div style="font-size:11px;color:#555">n={len(vals)}</div></td>'
        rows += f'<tr><td class="name">{html.escape(label)}</td>{cells}</tr>'
    return note + f'<table>{head}{rows}</table>'


def buildOutcomeSummary(versions, labels):
    """Trial OUTCOME from the per-sample uncertainty model (pipeline >= 1.3): the reliable
    error_type per trial -- correcto / fuera de objetivo / sin toque / sin datos (incomplete
    reach). Per version (how many trials of each type) and, for the latest version, per
    participant (who has more off-target / no-touch). Older versions show n/a (no error_type)."""
    order = ['correct', 'no_touch', 'off_target', '']
    has_et = [any(t.get('error_type') for p in v.values() for t in p.values() if isValid(t))
              for v in versions]
    head = ('<tr><th>Resultado del trial</th>'
            + ''.join(f'<th class="vcol v{i}">{html.escape(lab)}</th>' for i, lab in enumerate(labels))
            + '</tr>')
    # one row per error_type, value = count (and % of valid trials of that version)
    rows = ''
    totals = [sum(1 for p in v.values() for t in p.values() if isValid(t)) for v in versions]
    for et in order:
        cells = ''
        for i, v in enumerate(versions):
            if not has_et[i]:
                cells += f'<td class="vcol v{i} c-missing">n/a</td>'
                continue
            n = sum(1 for p in v.values() for t in p.values()
                    if isValid(t) and (t.get('error_type') or '') == et)
            tot = totals[i] or 1
            cls = ERROR_TYPE_CLASS[et]
            cells += f'<td class="vcol v{i} {cls}">{n} <span style="color:#777">({100.0*n/tot:.0f}%)</span></td>'
        rows += f'<tr><td class="name">{html.escape(ERROR_TYPE_LABEL[et])}</td>{cells}</tr>'
    note = ('<p><b>Resultado del trial</b> (modelo de incertidumbre por muestra, '
            '<code>error_type</code>), derivado solo de señales <b>fiables</b> (toque del objetivo, '
            'entrada/salida de la mano, última fijación de tablero — no usa la pieza tocada, poco '
            'fiable): <span class="c-target">correcto</span> = se confirmó el toque del objetivo; '
            '<span class="c-error">fuera de objetivo</span> = alcance completo pero ni se tocó el '
            'objetivo ni la última fijación se comprometió con él (la respuesta fue a otro sitio); '
            '<span class="c-panel">sin toque</span> = alcance completo y la última fijación sí cayó en '
            'el objetivo, pero no se confirmó el toque (probablemente toque demasiado sutil); '
            '<span class="c-missing">sin datos</span> = el alcance no fue completo (falta entrada o '
            'salida de la mano) o no hay casilla objetivo. Es un dato de v1.3+: versiones anteriores '
            'salen como <b>n/a</b>.</p>')
    return note + f'<table>{head}{rows}</table>' + buildOutcomePerParticipant(versions, labels)


def buildOutcomePerParticipant(versions, labels):
    """Per participant (latest version): count of each reliable outcome. Surfaces who has more
    off-target / no-touch trials -- a data-quality and a behaviour signal at once."""
    latest = versions[-1] if versions else {}
    if not any(t.get('error_type') for p in latest.values() for t in p.values() if isValid(t)):
        return ''
    head = ('<tr><th class="idx">#</th><th>Participante</th><th class="c-target">correcto</th>'
            '<th class="c-error">fuera de objetivo</th><th class="c-panel">sin toque</th>'
            '<th class="c-missing">sin datos</th><th>Trials</th></tr>')
    rows = ''
    n = 0
    for p in sorted(latest.keys()):
        valid = [t for t in latest[p].values() if isValid(t)]
        if not valid:
            continue
        n += 1
        cnt = {et: sum(1 for t in valid if (t.get('error_type') or '') == et) for et in ('correct', 'off_target', 'no_touch', '')}
        off_cls = ' class="c-error"' if cnt['off_target'] else ''
        not_cls = ' class="c-panel"' if cnt['no_touch'] else ''
        rows += (f'<tr><td class="idx">{n}</td><td class="name">{html.escape(p)}</td>'
                 f'<td>{cnt["correct"]}</td><td{off_cls}>{cnt["off_target"]}</td>'
                 f'<td{not_cls}>{cnt["no_touch"]}</td>'
                 f'<td class="c-missing">{cnt[""]}</td><td>{len(valid)}</td></tr>')
    note = (f'<h3>Resultado por participante — {html.escape(labels[-1])}</h3>'
            '<p>Recuento de cada resultado fiable por participante. Resalta quién acumula más '
            '<b>fuera de objetivo</b> / <b>sin toque</b> (señal de comportamiento y de calidad).</p>')
    return note + f'<table>{head}{rows}</table>'


def buildConfidence(versions, labels):
    """Histogram of the GRADED target_found_confidence (mass of the gaze ellipse on the target
    cell, [0,1]) for the latest version -- the uncertainty-aware 'did they look at the target'.
    Replaces the binary found/not-found with a soft measure: a near-miss (adjacent cell) is no
    longer a hard zero. Empty when the calibration has no uncertainty model."""
    if not versions:
        return ''
    latest = versions[-1]
    vals = [t['found_confidence'] for p in latest.values() for t in p.values()
            if isValid(t) and t.get('found_confidence') is not None]
    if not vals:
        return ('<h3>Confianza graduada de objetivo visto</h3>'
                '<p>Sin modelo de incertidumbre en las calibraciones de esta versión '
                '(columna <code>target_found_confidence</code> vacía): no se puede graficar.</p>')
    med = float(np.median(vals))
    n = len(vals)
    thr = TARGET_FOUND_THR
    thr_str = f'{thr:.2f}'.replace('.', ',')
    # frame_target_found fires at mass >= target_found_mass_threshold (the shared TARGET_FOUND_THR)
    found = sum(1 for v in vals if v >= thr)
    # finer suggested reading (orientativo, doc §7.5), NOT applied by the pipeline
    high = sum(1 for v in vals if v >= 0.5)
    near = sum(1 for v in vals if 0.2 <= v < 0.5)
    miss = sum(1 for v in vals if v < 0.2)
    svg = _svgHist(vals, bins=20, lo=0.0, hi=1.0, width=620, height=200,
                   marker=thr, marker_label=f'found ≥ {thr_str}')
    note = (f'<h3>Confianza graduada de objetivo visto — {html.escape(labels[-1])}</h3>'
            '<p>Distribución de <code>target_found_confidence</code> ∈ [0,1]: la <b>masa de la '
            'elipse de incertidumbre de la mirada</b> sobre la casilla objetivo, en la mejor '
            'fijación del trial (modelo de incertidumbre por muestra, v1.4+). La marca '
            f'<code>frame_target_found</code> se dispara con <b>masa ≥ {thr_str}</b> (re-ajustado en '
            'v1.4.1 con la masa bivariante correcta; en vez de exigir la mayoría de centroides en la '
            'celda exacta), así que un <i>near-miss</i> en la frontera del objetivo, dentro del error '
            'del aparato, cuenta como visto.</p>'
            f'<p>Sobre {n} trials con modelo: <b>found</b> (masa ≥ {thr_str}) {found} '
            f'({100.0*found/n:.0f}%). Mediana {med:.2f}. Lectura más fina (orientativa, doc §7.5, '
            f'<b>no la aplica el procesado</b>): ≥0,5 (casilla más probable) {high} '
            f'({100.0*high/n:.0f}%) · 0,2–0,5 (frontera/contigua) {near} ({100.0*near/n:.0f}%) · '
            f'&lt;0,2 {miss} ({100.0*miss/n:.0f}%).</p>')
    return note + svg


# Shown once at the top (above the tabs): how to read + the timing caveat + version marking.
GLOBAL_INTRO = """
<p>Compara el procesamiento entre <b>versiones</b> del software para ver las <b>mejorías</b> de una
a otra. Por defecto se muestran solo la <b>primera y la última</b> (las casillas activan/ocultan
cada una; se aplica a todas las pestañas). Cada versión tiene un <b>color de borde</b> propio (el
cuadrito junto a su casilla); en las matrices, el borde izquierdo de cada valor apilado indica de
qué versión es, y el relleno es el tipo de fin.</p>
<p style="background:#fff6d6;border:1px solid #e0a040;padding:6px 10px;border-radius:5px">
<b>Tiempos (v1.2).</b> En v1.2 el trial arranca en el <b>inicio de la búsqueda</b> (primera mirada
al tablero con el panel aún retirándose), no en "tablero completamente visible"; el <b>final</b>
sigue siendo el <b>cruce del borde</b>. Por eso la <b>duración en v1.2 es algo MAYOR</b> y <b>no es
comparable</b> con 1.0/1.1 — compara duraciones solo <i>dentro</i> de v1.2. El <b>toque</b> es una
marca aparte, no el final del trial.</p>
"""

# Each tab carries its OWN short legend at the top, so the table is read WITH its explanation.
TAB_INTRO = {
    'Detección': '<p><b>Salud de la detección.</b> Trials <b>reales</b> segmentados frente a '
        '<b>errores de detección</b> por participante y versión, leídos del <code>.pkl</code> (la '
        'fuente de verdad) para que el conteo sea <b>consistente entre versiones</b>. Un error '
        '(<code>missing_trial_error</code> = panel esperado no detectado; <code>transition_error</code> '
        '= otro panel antes de recoger datos) <b>no es un trial del participante</b>: es uno que el '
        'procesado no pudo segmentar (panel poco visible, pocos ArUcos, desincronización).</p>',
    'Tipo de fin': '<p>Cómo se cerró cada trial. <span class="c-contour">cruce del borde</span> = '
        'criterio por defecto; <span class="c-panel">panel siguiente</span> = se cerró al aparecer el '
        'panel del trial siguiente (la duración es una cota superior); <span class="c-eov">fin de '
        'vídeo</span> = terminó la grabación; <span class="c-error">perdido</span> = el trial no se '
        'detectó en esa versión. Mostrar los <b>perdidos</b> evita que una versión con muchas pérdidas '
        '(p. ej. 0.8.0) parezca "completa".</p>',
    'Cobertura marcas': '<p>% de trials con cada marca temporal, <b>por versión</b> (tabla superior) y '
        '<b>por participante</b> (inferior, para ver cobertura desigual). Una marca puede faltar sin '
        'afectar a las demás. Versiones anteriores a 1.1.0 salen como <b>n/a</b> (no tenían marcas).</p>',
    'Resultado del trial': '<p>Resultado <b>fiable</b> de cada trial (<code>error_type</code>, de v1.3) y, '
        'debajo, la <b>confianza graduada</b> de objetivo visto (<code>target_found_confidence</code>, '
        'del modelo de incertidumbre por muestra de v1.4). Sustituyen los binarios duros por medidas '
        'conscientes del error del aparato (un near-miss deja de ser un cero). Versiones previas salen como n/a.</p>',
    'Duración del trial (s)': '<p>Duración de cada trial (inicio → cruce del borde), por versión y '
        'coloreada por tipo de fin. Es el tiempo que normalmente se analiza (el inicio cambió en v1.2, '
        'ver arriba).</p>',
    'Toque': '<p>Tres columnas: (1) <b>duración del trial</b> (inicio → borde) por versión; (2) <b>tiempo '
        'hasta tocar</b> (inicio → toque) de la última versión; (3) <b>alcance</b> (borde → toque). Como '
        'la mano entra y <i>luego</i> alcanza, "hasta tocar" &gt; "hasta el borde". Un <b>&mdash;</b> = el '
        'toque no se detectó (marca de <i>mejor esfuerzo</i>, no cierra el trial).</p>',
    'Frecuencias': '<p>Frecuencia real medida por participante. La del <b>gaze</b> convierte conteos de '
        'muestras a tiempo (no es 200 Hz para todos). La <b>continuidad</b> = fracción de intervalos entre '
        'muestras dentro de ±20% del intervalo mediano: ~100% = muestreo regular; un valor bajo '
        '(resaltado) avisa de muestreo irregular en ese participante.</p>',
}

CSS = """
body{font-family:Arial,sans-serif;margin:16px;color:#222}
h1{font-size:20px} h3{margin-top:18px}
.tabs{margin:12px 0;border-bottom:2px solid #ccc}
.tabs button{border:none;background:#eee;padding:8px 14px;cursor:pointer;font-size:14px;margin-right:3px;border-radius:6px 6px 0 0}
.tabs button.active{background:#1f77b4;color:#fff}
.vsel{margin:8px 0;font-size:13px}
.tab{display:none} .tab.active{display:block}
table{border-collapse:collapse;font-size:12px;margin-top:8px}
th,td{border:1px solid #bbb;padding:3px 7px;text-align:center}
th{background:#f2f2f2;position:sticky;top:0}
td.name{text-align:left;color:#555}
td.idx,th.idx{color:#999;text-align:right}
.c-ok{background:#eef} .c-target{background:#cdebc6} .c-contour{background:#d6e4f5}
.c-panel{background:#ffe2b8} .c-eov{background:#e3d4f5} .c-error{background:#f6c6c6}
.c-missing{background:#f7f7f7;color:#bbb}
table.matrix{font-size:11px}
table.matrix th{font-size:11px;padding:2px 4px}
table.matrix td.name{text-align:left;max-width:120px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
table.matrix td.mcell{padding:1px}
.sub{display:block;min-height:11px;line-height:13px;padding:0 3px;margin:1px 0;border-radius:2px;font-size:10px;white-space:nowrap;text-align:center}
"""

JS = """
function showTab(n){document.querySelectorAll('.tab').forEach((t,i)=>t.classList.toggle('active',i==n));
document.querySelectorAll('.tabs button').forEach((b,i)=>b.classList.toggle('active',i==n));}
// Toggling a version shows/hides ITS columns in every tab AND keeps every tab's checkbox
// for that version in sync (so the same versions are checked on all tabs, not just here).
function toggleV(i,on){
  document.querySelectorAll('.v'+i).forEach(e=>e.style.display=on?'':'none');
  document.querySelectorAll('.vchk'+i).forEach(c=>c.checked=on);
}
// On load: apply each version's default checked state (first+last) to the columns.
document.querySelectorAll('.vsel:first-of-type input[type=checkbox]').forEach((c,i)=>toggleV(i,c.checked));
showTab(0);
"""


# ----------------------------------------------------------------------------- behaviour summary
# NOTE: the PKL `sequence` stores the RAW record-time phase ('pre_start'/'execution'/'motor'), not
# the CSV's re-split ('search'/'verification'/'withdraw'). 'execution' = search+verification (board
# gaze before the hand enters); 'motor' = reach + withdrawal. The behaviour summary counts the gaze
# that belongs to SEARCH: anticipatory ('pre_start') + 'execution' + the reach part of 'motor' (up
# to the touch). Withdrawal ('motor' after the touch), on_panel, blank and not_board are excluded.
BEHAV_COLORS = ['red', 'yellow', 'blue', 'green']
BEHAV_SHAPES = ['circle', 'triangle', 'hexagon', 'square', 'trapezoid']


def _blend(hexc, frac):
    """White -> hexc by frac in [0,1] (a light, readable tint for heat cells)."""
    frac = max(0.0, min(1.0, frac))
    r, g, b = int(hexc[1:3], 16), int(hexc[3:5], 16), int(hexc[5:7], 16)
    return '#%02x%02x%02x' % (int(255 + (r - 255) * frac), int(255 + (g - 255) * frac), int(255 + (b - 255) * frac))


def _svgMatrix(M, rows, cols, base, vmax=50.0, cw=58, ch=30, lw=72, hh=24):
    """Heat matrix [rows x cols] of % values (None = blank) as inline SVG. Diagonal cells (target
    == looked) get a heavy border so the 'guidance' reads at a glance."""
    W, H = lw + len(cols) * cw + 8, hh + len(rows) * ch + 8
    p = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Arial" role="img">']
    for j, c in enumerate(cols):
        p.append(f'<text x="{lw+j*cw+cw/2:.0f}" y="{hh-7}" font-size="11" text-anchor="middle" fill="#333">{html.escape(c)}</text>')
    for i, rname in enumerate(rows):
        p.append(f'<text x="{lw-6}" y="{hh+i*ch+ch/2+4:.0f}" font-size="11" text-anchor="end" fill="#333">{html.escape(rname)}</text>')
        for j, c in enumerate(cols):
            v = M[i][j]
            x, y = lw + j * cw, hh + i * ch
            diag = rname == c
            fill = _blend(base, (v or 0) / vmax) if v is not None else '#ffffff'
            p.append(f'<rect x="{x}" y="{y}" width="{cw-1}" height="{ch-1}" fill="{fill}" '
                     f'stroke="{"#222" if diag else "#ddd"}" stroke-width="{2 if diag else 0.5}"/>')
            if v is not None:
                tc = '#fff' if (v or 0) > vmax * 0.62 else '#333'
                p.append(f'<text x="{x+cw/2:.0f}" y="{y+ch/2+4:.0f}" font-size="11" text-anchor="middle" '
                         f'fill="{tc}" font-weight="{"bold" if diag else "normal"}">{v:.0f}</text>')
    p.append('</svg>')
    return ''.join(p)


def _svgSearchTime(keys, means, block_of, width=860, height=240):
    """Mean search time (s) per trial in presentation order, with block shading + per-block mean."""
    if not keys:
        return '<p>Sin datos.</p>'
    pad_l, pad_r, pad_t, pad_b = 42, 12, 14, 40
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    ymax = (max(means) * 1.1) or 1.0   # guard all-zero means (no div-by-zero in Y)
    n = len(keys)
    def X(i): return pad_l + (i + 0.5) / n * pw
    def Y(v): return pad_t + ph * (1 - v / ymax)
    p = [f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="Arial" role="img">']
    # block bands (alternating tints), a vertical separator at each boundary, a labelled header strip,
    # and the per-block mean line -- so where each block starts/ends reads at a glance.
    blocks = sorted(set(block_of))
    band = ['#eef3fb', '#f7f7f7']
    for bi, b in enumerate(blocks):
        idx = [i for i in range(n) if block_of[i] == b]
        x0, x1 = X(idx[0]) - pw / n / 2, X(idx[-1]) + pw / n / 2
        rot = (b == 3)                           # block 3 is presented with the board rotated 180
        p.append(f'<rect x="{x0:.1f}" y="{pad_t}" width="{x1-x0:.1f}" height="{ph}" fill="{"#f3e6e6" if rot else band[bi%2]}"/>')
        if bi:                                   # separator between blocks
            p.append(f'<line x1="{x0:.1f}" y1="{pad_t}" x2="{x0:.1f}" y2="{pad_t+ph}" stroke="#bbb" stroke-width="1" stroke-dasharray="3 2"/>')
        # header strip with the block label
        p.append(f'<rect x="{x0:.1f}" y="{pad_t}" width="{x1-x0:.1f}" height="15" fill="{"#a55" if rot else "#5a7fb5"}" opacity="0.9"/>')
        p.append(f'<text x="{(x0+x1)/2:.1f}" y="{pad_t+11}" font-size="9" fill="#fff" text-anchor="middle" font-weight="bold">Bloque {b}{" (girado)" if rot else ""}</text>')
        bm = sum(means[i] for i in idx) / len(idx)
        p.append(f'<line x1="{x0:.1f}" y1="{Y(bm):.1f}" x2="{x1:.1f}" y2="{Y(bm):.1f}" stroke="#d33333" stroke-width="1.6"/>')
    # axes + 0.5 s grid ticks
    p.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+ph}" stroke="#999"/>')
    p.append(f'<line x1="{pad_l}" y1="{pad_t+ph}" x2="{pad_l+pw}" y2="{pad_t+ph}" stroke="#999"/>')
    gv = 0.5
    while gv < ymax:
        p.append(f'<text x="{pad_l-5}" y="{Y(gv)+3:.1f}" font-size="9" fill="#777" text-anchor="end">{gv:.1f}</text>')
        gv += 0.5
    # polyline + points
    pts = ' '.join(f'{X(i):.1f},{Y(means[i]):.1f}' for i in range(n))
    p.append(f'<polyline points="{pts}" fill="none" stroke="#1f77b4" stroke-width="1.3"/>')
    for i in range(n):
        p.append(f'<circle cx="{X(i):.1f}" cy="{Y(means[i]):.1f}" r="2.2" fill="#1f77b4"/>')
    p.append(f'<text x="{pad_l}" y="{height-6}" font-size="10" fill="#555">trial en orden de presentación (bloque · trial) →</text>')
    p.append(f'<text x="6" y="{pad_t+8}" font-size="10" fill="#555">s</text>')
    p.append('</svg>')
    return ''.join(p)


def _svgColorBlocks(by_cb_measure, width=520, height=240):
    """Mean response time per TARGET COLOUR across blocks (= repetition number of each piece): a line
    per colour over blocks 0..5. Block 3 (board rotated 180 by design) is shaded as a reminder."""
    import statistics as _st
    blocks = sorted({b for (_c, b) in by_cb_measure})
    if not blocks:
        return '<p>Sin datos.</p>'
    cols = {'red': '#d33333', 'yellow': '#e0a040', 'blue': '#1f77b4', 'green': '#4a9933'}
    series = {c: [(_st.mean(by_cb_measure[(c, b)]) if by_cb_measure.get((c, b)) else None) for b in blocks]
              for c in cols}
    allv = [v for s in series.values() for v in s if v is not None]
    if not allv:
        return '<p>Sin datos.</p>'
    # ZOOM the y-axis to the data band (the per-colour differences are small ~0.9-1.6s; a 0-axis
    # would squash them into a flat-looking strip). Tick every 0.2 s.
    ymin, ymax = min(allv) * 0.92, max(allv) * 1.06
    if ymax - ymin < 1e-6:             # all values equal -> avoid a zero-height band
        ymin, ymax = ymin - 0.1, ymax + 0.1
    pad_l, pad_r, pad_t, pad_b = 38, 70, 12, 34
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    def X(i): return pad_l + (i / max(1, len(blocks) - 1)) * pw
    def Y(v): return pad_t + ph * (1 - (v - ymin) / (ymax - ymin))
    p = [f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="Arial" role="img">']
    if 3 in blocks:
        bx = X(blocks.index(3))
        p.append(f'<rect x="{bx-pw/(2*max(1,len(blocks)-1)):.1f}" y="{pad_t}" width="{pw/max(1,len(blocks)-1):.1f}" height="{ph}" fill="#f3e6e6"/>')
        p.append(f'<text x="{bx:.1f}" y="{pad_t+10}" font-size="8" fill="#a55" text-anchor="middle">B3 girado</text>')
    p.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+ph}" stroke="#999"/>')
    p.append(f'<line x1="{pad_l}" y1="{pad_t+ph}" x2="{pad_l+pw}" y2="{pad_t+ph}" stroke="#999"/>')
    gt = (int(ymin / 0.2) + 1) * 0.2
    while gt < ymax:
        p.append(f'<line x1="{pad_l}" y1="{Y(gt):.1f}" x2="{pad_l+pw}" y2="{Y(gt):.1f}" stroke="#eee"/>')
        p.append(f'<text x="{pad_l-5}" y="{Y(gt)+3:.1f}" font-size="9" fill="#777" text-anchor="end">{gt:.1f}</text>')
        gt += 0.2
    p.append(f'<text x="{pad_l-26}" y="{pad_t+ph/2:.0f}" font-size="9" fill="#555" transform="rotate(-90 {pad_l-26} {pad_t+ph/2:.0f})" text-anchor="middle">t. respuesta (s)</text>')
    for b in blocks:
        p.append(f'<text x="{X(blocks.index(b)):.1f}" y="{pad_t+ph+14}" font-size="9" fill="#555" text-anchor="middle">B{b}</text>')
    for ci, (c, col) in enumerate(cols.items()):
        pts = [(X(i), Y(v)) for i, v in enumerate(series[c]) if v is not None]
        if pts:
            p.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in pts)}" fill="none" stroke="{col}" stroke-width="1.6"/>')
            for x, y in pts:
                p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.4" fill="{col}"/>')
            p.append(f'<text x="{pad_l+pw+6}" y="{pad_t+12+ci*14}" font-size="10" fill="{col}">{c}</text>')
    p.append(f'<text x="{pad_l}" y="{height-4}" font-size="9" fill="#555">bloque (= repetición de cada pieza) →</text>')
    p.append('</svg>')
    return ''.join(p)


def buildBehaviorSummary(root, topic):
    """Exploratory behaviour summary from the LATEST run (not a version comparison): WHAT people
    look at as a function of the target (colour / shape confusion matrices over the search-relevant
    phases) and HOW search time evolves across the session. Reads the per-participant PKL sequences."""
    import collections
    col_t, shp_t = collections.Counter(), collections.Counter()
    col_n, shp_n = collections.Counter(), collections.Counter()
    # time-to-X measures (endpoint, label, % of trials it is detected in): the search/decision time
    # from the trial start to each landmark. 'verifica' (last board fixation before the touch) is the
    # cleanest cognitive measure (excludes the motor reach, confounded by cell distance) AND the
    # best detected, so it is the default; the rest let the team explore the search-vs-reach split.
    TIME_MEASURES = [('valid', 'hasta verificar el objetivo (cognitivo)', 'frame_validation'),
                     ('found', 'hasta encontrar el objetivo', None),
                     ('motor', 'hasta que entra la mano (búsqueda+decisión)', 'motor_onset_capture'),
                     ('touch', 'hasta tocar el objetivo (búsqueda+alcance)', 'target_touch_capture')]
    by_pos = {mk: collections.defaultdict(list) for mk, _, _ in TIME_MEASURES}
    by_cb = {mk: collections.defaultdict(list) for mk, _, _ in TIME_MEASURES}   # (colour, block) -> times
    tgt_n, tgt_found, tgt_touch = collections.Counter(), collections.Counter(), collections.Counter()
    # off_target: complete reach where the response went somewhere ELSE (reliable error_type).
    # Tallied per target so a piece that is often mis-selected (not just mis-looked) shows up.
    tgt_off, tgt_reach = collections.Counter(), collections.Counter()
    topic_dir = os.path.join(root, topic)
    if not os.path.isdir(topic_dir):
        return '<p>Sin datos.</p>'
    np_ = 0
    for participant in sorted(os.listdir(topic_dir)):
        pkl = os.path.join(topic_dir, participant, f'data_{participant}.pkl')
        if not os.path.isfile(pkl):
            continue
        try:
            d = pickle.load(open(pkl, 'rb'))
        except Exception:
            continue
        np_ += 1
        fps = d.get('video_fps') or 30.0
        for (b, t), tm in d.get('trials_data', {}).items():
            if not isinstance(b, int) or b == -1 or t == -1:
                continue
            name = next(iter(tm))
            if name.startswith(ERROR_PREFIXES) or '_' not in name:
                continue
            m = tm[name]
            tcol, tshp = name.split('_', 1)
            touch = m.get('target_touch_capture')
            for s in m.get('sequence', []):
                ph = s.get('phase')
                counts = (ph in ('pre_start', 'execution')
                          or (ph == 'motor' and isinstance(touch, (int, float)) and s.get('frame', 0) <= touch))
                if not counts:
                    continue
                lc, ls = s.get('color'), s.get('shape')
                if lc in BEHAV_COLORS:
                    col_t[(tcol, lc)] += 1; col_n[tcol] += 1
                if ls in BEHAV_SHAPES:
                    shp_t[(tshp, ls)] += 1; shp_n[tshp] += 1
            # time-to-X: from the trial start (the first board gaze, even while the panel is still
            # being removed -> early_init_capture) up to each landmark.
            st = m.get('early_init_capture', m.get('init_capture'))
            tf = _targetFound(m)
            ends = {'valid': m.get('frame_validation'), 'found': tf,
                    'motor': m.get('motor_onset_capture'), 'touch': touch}
            if isinstance(st, (int, float)):
                for mk, ev in ends.items():
                    if isinstance(ev, (int, float)) and ev > st:
                        by_pos[mk][(b, t)].append((ev - st) / fps)
                        by_cb[mk][(tcol, b)].append((ev - st) / fps)
            # per-target reliability: how often each piece is FOUND (gaze), TOUCHED (occlusion)
            # and, among complete reaches, OFF-TARGET (the reliable error_type).
            et = m.get('error_type') or ''
            for key in (name, tcol, tshp):
                tgt_n[key] += 1
                if tf is not None:
                    tgt_found[key] += 1
                if touch is not None:
                    tgt_touch[key] += 1
                if et in ('correct', 'off_target', 'no_touch'):   # a complete reach with a verdict
                    tgt_reach[key] += 1
                    if et == 'off_target':
                        tgt_off[key] += 1

    def matrix(pairs, tot, rows, cols):
        return [[(100.0 * pairs[(r, c)] / tot[r] if tot[r] else None) for c in cols] for r in rows]
    trc = [c for c in BEHAV_COLORS if col_n[c]]
    trs = [s for s in BEHAV_SHAPES if shp_n[s]]
    color_svg = _svgMatrix(matrix(col_t, col_n, trc, BEHAV_COLORS), trc, BEHAV_COLORS, '#d33333')
    shape_svg = _svgMatrix(matrix(shp_t, shp_n, trs, BEHAV_SHAPES), trs, BEHAV_SHAPES, '#1f77b4')

    # time-to-X charts (one per measure), shown one at a time via a <select>
    charts, opts = '', ''
    for mk, label, _ in TIME_MEASURES:
        keys = sorted(by_pos[mk])
        means = [sum(by_pos[mk][k]) / len(by_pos[mk][k]) for k in keys]
        svg = _svgSearchTime(keys, means, [k[0] for k in keys])
        cb_svg = _svgColorBlocks(by_cb[mk])
        body = (f'<div style="display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start">'
                f'<div>{svg}</div>'
                f'<div><b>Por color a lo largo de los bloques</b> (= repetición de cada pieza)<br>{cb_svg}</div></div>')
        charts += f'<div class="timechart" id="tc_{mk}" style="{"" if mk == "valid" else "display:none"}">{body}</div>'
        opts += f'<option value="{mk}" {"selected" if mk == "valid" else ""}>{html.escape(label)}</option>'
    select = ('<select onchange="document.querySelectorAll(\'.timechart\').forEach(e=>e.style.display=\'none\');'
              'document.getElementById(\'tc_\'+this.value).style.display=\'\'">' + opts + '</select>')

    def ft_bar(v, color):
        return (f'<div style="display:inline-block;width:90px;height:13px;background:#eee;vertical-align:middle">'
                f'<div style="width:{v*0.9:.0f}px;height:13px;background:{color}"></div></div> {v:.0f}%')

    any_off = bool(sum(tgt_reach.values()))
    def ft_table(keys, header):
        rows = ''
        for k in sorted(keys, key=lambda x: tgt_found[x] / max(1, tgt_n[x])):
            n = tgt_n[k]
            # off-target rate is over COMPLETE reaches only (the denominator where it is defined),
            # not over all trials, so it is not diluted by trials without a full reach.
            off_cell = (f'<td>{ft_bar(100*tgt_off[k]/tgt_reach[k], "#d33333")}'
                        f' <span style="color:#888">(n={tgt_reach[k]})</span></td>') if any_off else ''
            rows += (f'<tr><td>{html.escape(k)}</td><td>{n}</td>'
                     f'<td>{ft_bar(100*tgt_found[k]/n, "#5a7fb5")}</td>'
                     f'<td>{ft_bar(100*tgt_touch[k]/n, "#4a9933")}</td>{off_cell}</tr>')
        off_h = '<th>fuera de objetivo<br><span style="font-weight:normal">(de alcances completos)</span></th>' if any_off else ''
        return (f'<table class="cmp"><tr><th>{header}</th><th>n</th>'
                f'<th>encontrado (mirada)</th><th>tocado (oclusión)</th>{off_h}</tr>{rows}</table>')
    color_ft = ft_table([c for c in BEHAV_COLORS if tgt_n[c]], 'Color')
    shape_ft = ft_table([s for s in BEHAV_SHAPES if tgt_n[s]], 'Forma')

    return (
        f'<p><b>Resumen exploratorio del comportamiento</b> (última versión, {np_} participantes; '
        f'no es una comparación de versiones).</p>'
        '<h3>¿Qué mira la gente según el objetivo?</h3>'
        '<p>Matriz objetivo (fila) × figura mirada (columna): <b>% de la mirada al tablero</b> que '
        'cae en cada categoría, durante las fases que cuentan como búsqueda '
        '(<i>anticipada, búsqueda, verificación, motora hasta el toque</i>; se excluyen retirada, '
        'fuera y tapada). El tablero está <b>equilibrado</b> (25&nbsp;% por color, 20&nbsp;% por '
        'forma), así que el <b>nivel de azar</b> es 25&nbsp;% (color) y 20&nbsp;% (forma): la '
        'diagonal por encima de ese nivel indica <b>búsqueda guiada</b> por esa característica.</p>'
        f'<div style="display:flex;gap:34px;flex-wrap:wrap;align-items:flex-start">'
        f'<div><b>Color</b> (azar 25&nbsp;%)<br>{color_svg}</div>'
        f'<div><b>Forma</b> (azar 20&nbsp;%)<br>{shape_svg}</div></div>'
        '<h3>¿Hay piezas más difíciles? Encontrado (mirada) vs tocado (oclusión) vs fuera de objetivo</h3>'
        '<p>Por color y por forma del objetivo: fracción de trials en que el objetivo se '
        '<b>encuentra</b> con la mirada (fijación sobre la casilla), en que se detecta el '
        '<b>toque</b>, y &mdash;sobre los alcances completos&mdash; en que la respuesta fue '
        '<b>fuera de objetivo</b> (<code>error_type=off_target</code>: se alcanzó y retiró la mano '
        'sin tocar ni mirar el objetivo). Un «encontrado» bajo señala una pieza más difícil de '
        'discriminar visualmente (p.&nbsp;ej. el <b>rojo</b>); «encontrado» es la fijación I-DT cuya '
        'masa de mirada sobre la casilla alcanza el umbral (igual que la marca <code>frame_target_found</code>) y depende de '
        'detectar esa fijación (≈80&nbsp;% de cobertura global), así que es un indicador '
        '<i>exploratorio</i>. «Fuera de objetivo» usa solo señales fiables (v1.3+).</p>'
        f'<div style="display:flex;gap:34px;flex-wrap:wrap;align-items:flex-start">{color_ft}{shape_ft}</div>'
        '<h3>Tiempo hasta el objetivo a lo largo del experimento</h3>'
        f'<p>Cada punto es la <b>media entre los {np_} participantes</b> del tiempo desde el '
        '<b>inicio</b> del trial (primer gaze sobre el tablero, aún con el panel retirándose) hasta '
        'el hito elegido; las bandas y la cabecera marcan cada <b>bloque</b> y la línea roja es la '
        'media del bloque. Por defecto <b>hasta verificar</b> (la última fijación en el tablero '
        'antes del toque): mide la <b>búsqueda + decisión</b> sin la ejecución motora (confundida '
        'por la distancia de la casilla) y es el hito mejor detectado (98&nbsp;%). Revela la '
        '<b>familiarización</b>: el primer bloque arranca más lento. (Se excluyen los trials '
        '<code>-1</code>: demos/re-presentaciones.)</p>'
        f'<p><b>Hito:</b> {select}</p>{charts}')


def main():
    parser = argparse.ArgumentParser(description='Generate an HTML comparison report')
    parser.add_argument('--roots', nargs='+', required=True, help='Output roots to compare (oldest first).')
    parser.add_argument('--data_root', default=None, help='Input data root (for the gaze/video frequencies).')
    parser.add_argument('-t', dest='topic', default='gaze')
    parser.add_argument('--out', default='report.html')
    args = parser.parse_args()

    labels = [versionLabel(r) for r in args.roots]
    versions = [loadVersion(r, args.topic) for r in args.roots]
    freqs = gazeFrequencies(args.data_root)

    # Each version gets a distinct BORDER colour (the cell fill already encodes the finish
    # status, so version uses a separate channel): the matrix sub-cells carry a left border in
    # this colour, and the checkbox shows the same swatch, so you can tell which stacked value
    # belongs to which version without relying on the fill colour.
    VCOLORS = ['#444444', '#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e']
    vcolor = [VCOLORS[i % len(VCOLORS)] for i in range(len(labels))]
    version_css = ''.join(f'.sub.v{i}{{border-left:4px solid {vcolor[i]}}}' for i in range(len(labels)))

    # By default only the FIRST and LAST version are shown (the clearest "antes vs ahora"
    # for the analysis team); the intermediate ones can be enabled with their checkbox.
    last_i = len(labels) - 1
    vsel = '<div class="vsel">Versiones: ' + ''.join(
        f'<label><input type="checkbox" class="vchk{i}" {"checked" if (i == 0 or i == last_i) else ""} '
        f'onchange="toggleV({i},this.checked)"> '
        f'<span style="display:inline-block;width:10px;height:10px;background:{vcolor[i]};'
        f'border-radius:2px;vertical-align:middle"></span> {html.escape(l)}</label> '
        for i, l in enumerate(labels)) + '</div>'

    tabs_html = [
        ('Detección', buildDetectionSummary(versions, labels)),
        ('Tipo de fin', buildTable(versions, labels, cellStatus)),
        ('Cobertura marcas', buildMarkCoverage(versions, labels)),
        ('Resultado del trial', buildOutcomeSummary(versions, labels)
                  + '<hr style="margin:18px 0">' + buildConfidence(versions, labels)),
        ('Duración del trial (s)', buildTable(versions, labels, cellDuration)),
        ('Toque', buildTouchTable(versions, labels) + '<hr style="margin:18px 0">'
                  + buildTouchDiagnostics(versions, labels)),
        ('Frecuencias', buildFreqTable(freqs) if freqs else '<p>Sin datos de entrada para frecuencias.</p>'),
        ('Comportamiento', buildBehaviorSummary(args.roots[-1], args.topic)),
    ]
    tab_buttons = ''.join(f'<button onclick="showTab({i})">{html.escape(t)}</button>' for i, (t, _) in enumerate(tabs_html))
    # Each tab opens with its OWN legend (TAB_INTRO) so the table is read with its explanation,
    # instead of one Leyenda tab at the end that you reach after already seeing the tables.
    tab_divs = ''.join(f'<div class="tab">{("" if t in ("Frecuencias", "Comportamiento") else vsel)}'
                       f'{TAB_INTRO.get(t, "")}{body}</div>'
                       for (t, body) in tabs_html)

    doc = (f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>Informe procesamiento</title>'
           f'<style>{CSS}{version_css}</style></head><body>'
           f'<h1>Informe de procesamiento — comparación de versiones</h1>'
           f'<p>Versiones: {", ".join(html.escape(l) for l in labels)}</p>'
           f'{GLOBAL_INTRO}'
           f'<div class="tabs">{tab_buttons}</div>{tab_divs}'
           f'<script>{JS}</script></body></html>')

    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(doc)
    print(f"Report written to {args.out}  ({len(labels)} versions, {len(freqs)} participants with frequencies)")

    # Companion CSV with the per-participant frequencies, easy to merge in the analysis
    if freqs:
        freq_csv = os.path.splitext(args.out)[0] + '_frequencies.csv'
        with open(freq_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['participant', 'world_fps', 'gaze_sampling_rate_hz', 'gaze_continuity'])
            for participant, (wfps, rate, cont) in sorted(freqs.items()):
                w.writerow([participant, f"{wfps:.4f}", f"{rate:.4f}", f"{cont:.4f}"])
        print(f"Frequencies CSV written to {freq_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
