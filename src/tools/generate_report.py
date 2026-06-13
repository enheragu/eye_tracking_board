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


def loadVersion(root, topic):
    """participant -> {(block,trial): {name,status,duration,init,touch,end}}.
    The frame_* marks only exist from 1.1.0 on; older CSVs leave them as None."""
    topic_dir = os.path.join(root, topic)
    out = {}
    if not os.path.isdir(topic_dir):
        return out
    for participant in sorted(os.listdir(topic_dir)):
        csv_path = os.path.join(topic_dir, participant, f'trials_data_{participant}.csv')
        if not os.path.isfile(csv_path):
            continue
        trials = {}
        for row in csv.DictReader(open(csv_path)):
            key = (int(row['block_index']), int(row['trial_index']))
            trials.setdefault(key, {'name': row['trial_name'], 'status': row['Finish Status'],
                                    'duration': float(row['trial_duration_s']),
                                    'init': _toInt(row.get('frame_init')),
                                    'touch': _toInt(row.get('frame_target_touch')),
                                    'exit': _toInt(row.get('frame_hand_exit')),
                                    'end': _toInt(row.get('frame_end'))})
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
        gt = np.sort(np.load(gt_path)); wt = np.sort(np.load(wt_path))
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


def cellDetection(trial):
    if trial is None:
        return ('', 'c-missing')
    if not isValid(trial):
        return ('error', 'c-error')
    return ('OK', 'c-ok')


def cellStatus(trial):
    if trial is None:
        return ('', 'c-missing')
    if not isValid(trial):
        return (trial['name'][:18], 'c-error')
    short = trial['status'].replace('test_finish_', '').replace('_', ' ')
    return (short, STATUS_CLASS.get(trial['status'], 'c-ok'))


def cellDuration(trial):
    if trial is None:
        return ('', 'c-missing')
    if not isValid(trial):
        return ('-', 'c-error')
    return (f"{trial['duration']:.2f}", STATUS_CLASS.get(trial['status'], 'c-ok'))


def buildTable(versions, labels, cell_fn):
    """Matrix layout: participants are COLUMNS, (block,trial) are ROWS, and the versions
    are stacked sub-cells inside each cell. Far more compact and scannable than one row
    per participant-trial. Version checkboxes toggle the sub-cells (class v{i})."""
    participants = sorted(set().union(*[set(v.keys()) for v in versions]) if versions else set())
    keys = sorted(set().union(*[set(v.get(p, {}).keys()) for v in versions for p in participants])) if participants else []
    head = ('<tr><th>Blk</th><th>Tr</th><th>Trial</th>'
            + ''.join(f'<th>{html.escape(p)}</th>' for p in participants) + '</tr>')
    rows = []
    for (block, trial) in keys:
        name = ''
        for p in participants:
            for v in versions:
                t = v.get(p, {}).get((block, trial))
                if t:
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
            + ''.join(f'<th class="vcol v{i}">{html.escape(lab)}<br>dur. borde (s)</th>'
                      for i, lab in enumerate(labels))
            + f'<th class="vcol v{last}">{html.escape(labels[last])}<br>t→toque (s)</th>'
            + f'<th class="vcol v{last}">alcance toque&minus;borde (s)</th></tr>')
    rows = []
    n_touch = n_trials = 0
    for participant in participants:
        keys = sorted(set().union(*[set(v.get(participant, {}).keys()) for v in versions]))
        for (block, trial) in keys:
            name = ''
            for v in versions:
                t = v.get(participant, {}).get((block, trial))
                if t:
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
    return summary + f'<table>{head}{"".join(rows)}</table>'


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
    hist = ('<h3>Posición del toque en la ventana motora [borde-entra → mano-sale]</h3>'
            + f'<p>0 = cruce del borde (entra la mano) · 1 = salida de la mano. Mediana en '
            f'<b>{f_med:.2f}</b> (el toque cae cerca del primer cuarto, no en el centro).</p>'
            + _svgHist(fracs, bins=20, lo=0.0, hi=1.0, marker=f_med, marker_label=f'mediana {f_med:.2f}'))
    interp = (f'<h3>Interpolación (capa de análisis)</h3>'
              f'<p>Para los trials sin toque pero con borde-entra y mano-sale, se puede '
              f'<b>estimar</b> el toque como <code>borde + {f_med:.2f}·(salida − borde)</code>. '
              f'Eso cubriría <b>{n_interp}</b> trials adicionales → cobertura estimada '
              f'<b>{est_pct:.0f}%</b>. Es una estimación (sesgo de selección: se calibra con los '
              f'toques detectados), no un valor medido; las marcas crudas se publican sin tocar.</p>')
    return intro + cov + hist + interp


LEGEND = """
<h3>Cómo leer este informe</h3>
<p>La comparación entre versiones está para <b>ver las mejorías</b> del procesamiento de una a otra.</p>
<ul>
<li><b>Detección</b> (nivel <i>máquina de procesamiento</i>): si el trial se segmentó (OK) o quedó como error en cada versión — informa de si el procesamiento mantuvo la sincronía, no de si el trial "salió bien".</li>
<li><b>Tipo de fin</b> (nivel <i>trial</i>): cómo se cerró el trial. <span class="c-contour">execution</span> = la mano cruzó el borde del tablero (criterio robusto y por defecto, igual que en 1.0.0); <span class="c-panel">by next panel</span> = se cerró al aparecer el panel siguiente (la duración es una cota superior); <span class="c-eov">by end of video</span> = fin de grabación. El <b>toque</b> de la pieza objetivo NO cierra el trial: se publica aparte como marca (<code>frame_target_touch</code>, best-effort).</li>
<li><b>Tiempos</b>: duración del trial (s) en cada versión, coloreada por tipo de fin. El criterio de inicio/fin es el mismo que en 1.0.0, así que la duración es comparable; 1.1.0 solo añade marcas.</li>
<li><b>Tiempo a dedo</b>: para la última versión, el tiempo de trial medido <b>hasta el toque</b> de la pieza (fin <i>a nivel usuario</i>), junto a la duración por <b>cruce del borde</b> de cada versión (lo que antes considerábamos "tiempo de trial"). El toque ocurre <b>después</b> del cruce del borde (la mano entra y luego alcanza la pieza), así que <code>t→toque</code> suele ser algo mayor; la columna <b>alcance toque&minus;borde</b> es ese tiempo de alcance dentro del tablero. Un <b>&mdash;</b> indica que el toque no se detectó en ese trial.</li>
<li><b>Frecuencias</b>: frecuencia real medida por participante. La del <b>gaze</b> se usa para convertir conteos de muestras a tiempo (tiempo = nº muestras / frecuencia gaze); <b>no</b> es 200 Hz para todos. La <b>continuidad</b> cercana al 100% indica muestreo regular; un valor bajo (resaltado) avisa de muestreo irregular en ese participante.</li>
<li>Marca/desmarca versiones con las casillas de arriba para mostrar u ocultar columnas.</li>
</ul>
"""

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
function toggleV(i,on){document.querySelectorAll('.v'+i).forEach(e=>e.style.display=on?'':'none');}
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

    vsel = '<div class="vsel">Versiones: ' + ''.join(
        f'<label><input type="checkbox" checked onchange="toggleV({i},this.checked)"> {html.escape(l)}</label> '
        for i, l in enumerate(labels)) + '</div>'

    tabs_html = [
        ('Detección', buildTable(versions, labels, cellDetection)),
        ('Tipo de fin', buildTable(versions, labels, cellStatus)),
        ('Tiempos (s)', buildTable(versions, labels, cellDuration)),
        ('Tiempo a dedo', buildTouchTable(versions, labels)),
        ('Diagnóstico toque', buildTouchDiagnostics(versions, labels)),
        ('Frecuencias', buildFreqTable(freqs) if freqs else '<p>Sin datos de entrada para frecuencias.</p>'),
        ('Leyenda', LEGEND),
    ]
    tab_buttons = ''.join(f'<button onclick="showTab({i})">{html.escape(t)}</button>' for i, (t, _) in enumerate(tabs_html))
    tab_divs = ''.join(f'<div class="tab">{("" if t in ("Frecuencias","Leyenda","Diagnóstico toque") else vsel)}{body}</div>'
                       for (t, body) in tabs_html)

    doc = (f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>Informe procesamiento</title>'
           f'<style>{CSS}</style></head><body>'
           f'<h1>Informe de procesamiento — comparación de versiones</h1>'
           f'<p>Versiones: {", ".join(html.escape(l) for l in labels)}</p>'
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
