#!/usr/bin/env python3
# encoding: utf-8

"""
    Builds a single self-contained HTML report (tabs + version checkboxes, no external
    dependencies) comparing the processing output of several versions, for the research
    team. Tabs: trial detection, finish type, durations, sampling frequencies, legend.

    Example:
        python3 src/tools/generate_report.py \
            --roots /path/OutpuData_v0.8.0 /path/OutputData_v1.0.0 /path/OutputData_v1.1.0 \
            --data_root /path/InputData --out /path/report.html
"""

import os
import sys
import csv
import glob
import html
import pickle
import argparse
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

ERROR_PREFIXES = ('missing_trial_error', 'transition_error', 'end_of_video_error')


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
            'reach_duration_s': None, 'withdraw_duration_s': None}


def _targetFound(metrics):
    """Frame of the 'objetivo visto' mark = first gaze that lands on the target cell, derived from
    the per-trial gaze sequence with the SAME rule as the writer (store_results): target_cord is
    [row,col] but the sequence board_coord is [col,row], so it is compared swapped. None when the
    target was never looked at (or there is no sequence)."""
    seq = metrics.get('sequence')
    target = metrics.get('target_cord')
    if not seq or not target or target[0] is None:
        return None
    target_colrow = [int(target[1]), int(target[0])]
    for s in seq:
        if isinstance(s, dict) and list(s.get('board_coord', [])) == target_colrow:
            return int(s['frame'])
    return None


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
                              'withdraw_duration_s': dur(touch, exit_)}
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
    head = ('<tr><th>Participante</th>'
            + ''.join(f'<th class="vcol v{i}">{html.escape(lab)}<br>reales / errores</th>'
                      for i, lab in enumerate(labels)) + '</tr>')
    rows = ''
    for p in participants:
        cells = ''
        for i, v in enumerate(versions):
            trials = v.get(p, {})
            valid = sum(1 for t in trials.values() if isValid(t))
            errs = sum(1 for t in trials.values() if not isValid(t))
            cls = 'c-ok' if errs == 0 else ('c-panel' if errs <= 3 else 'c-error')
            disp = f'{valid} / {errs}' if (valid or errs) else '&nbsp;'
            cells += f'<td class="vcol v{i} {cls}">{disp}</td>'
        rows += f'<tr><td class="name">{html.escape(p)}</td>{cells}</tr>'
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
    for participant, (wfps, rate, cont) in sorted(freqs.items()):
        warn = ' c-error' if (cont == cont and cont < 0.95) else ''
        rows += (f'<tr><td>{html.escape(participant)}</td>'
                 f'<td>{wfps:.2f}</td><td class="{warn.strip()}">{rate:.1f}</td>'
                 f'<td class="{warn.strip()}">{cont*100:.1f}%</td></tr>')
    return ('<table><tr><th>Participante</th><th>Vídeo World (fps)</th>'
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
    head = ('<tr><th>Participante</th>'
            + ''.join(f'<th>{lab}</th>' for _, lab in marks) + '<th>Trials</th></tr>')
    rows = ''
    for p in sorted(latest.keys()):
        valid = [t for t in latest[p].values() if isValid(t)]
        if not valid:
            continue
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
        rows += f'<tr><td class="name">{html.escape(p)}</td>{cells}<td>{len(valid)}</td></tr>'
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
        ('Duración del trial (s)', buildTable(versions, labels, cellDuration)),
        ('Toque', buildTouchTable(versions, labels) + '<hr style="margin:18px 0">'
                  + buildTouchDiagnostics(versions, labels)),
        ('Frecuencias', buildFreqTable(freqs) if freqs else '<p>Sin datos de entrada para frecuencias.</p>'),
    ]
    tab_buttons = ''.join(f'<button onclick="showTab({i})">{html.escape(t)}</button>' for i, (t, _) in enumerate(tabs_html))
    # Each tab opens with its OWN legend (TAB_INTRO) so the table is read with its explanation,
    # instead of one Leyenda tab at the end that you reach after already seeing the tables.
    tab_divs = ''.join(f'<div class="tab">{("" if t == "Frecuencias" else vsel)}'
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
