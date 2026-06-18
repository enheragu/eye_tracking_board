# Changelog

All notable changes to this project are documented in this file. Versions prior to
1.0.0 were reconstructed from the git history. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.4.0](https://github.com/enheragu/eye_tracking_board/releases/tag/v1.4.0) - 2026-06-18

A per-sample gaze **uncertainty model** that treats each gaze as a distribution rather than a
point, the `target_found` mark re-based on that uncertainty, and reliability hardening of two
detections (board frame and sample panel).

### Added
- **Per-sample gaze uncertainty model.** Each gaze now carries a measured 2×2 covariance
  (jitter + accuracy/drift, scaled by confidence and eccentricity), calibrated per participant
  from the in-video panels and propagated by the now inverse-variance smoother. It projects onto
  the board as a probabilistic cell distribution (`cell_dist` / `onboard_mass`) per sample and a
  graded `target_found_confidence` per trial; covered gaze (`on_panel` / `blank`) reads 0% board.
  A cell is ~⅛×⅕ of the board and the device resolves to ~½ a cell, so the gaze is treated as a
  distribution, not a point. Docs [§7.4–7.5](docs/documentacion_tecnica.md), [guide §6](docs/guia_procesamiento.md).

### Changed
- **`target_found` by ellipse mass, not a discrete cell vote.** `frame_target_found` (and with it
  `time_to_target_s` and the search/verification boundary) now fires on the first fixation whose
  mean uncertainty mass on the target reaches `target_found_mass_threshold` (0.30), instead of
  requiring a majority of sample centroids in the exact cell. A boundary-hugging fixation — inside
  the device error the model already accounts for — counts as found instead of being forced to a
  hard yes/no (cohort: found 1017 → 1068). Falls back to the centroid vote when a fixation has no
  covariance. Docs [§7.5](docs/documentacion_tecnica.md), [§10](docs/documentacion_tecnica.md).
- **Robust board-frame detection (`detectContour`).** The frame colour threshold is now median + MAD
  with a temporal EMA, so a hand entering a margin no longer corrupts it, and the rectangle is found
  directly on the mask (no Canny, no close). This cuts contour flicker without losing trial starts.
  Doc §4.
- **Panel robustness against ArUco false positives.** Panel confirmation now requires 4 consecutive
  frames (was 2): a misread marker is a one-marker, few-frame blip, while a real card persists for
  dozens of frames, which removes the phantom-panel cascades that desynced the sequence. Doc §5.

## [1.3.0] - 2026-06-16 <!-- developed but never separately tagged; ships within v1.4.0 -->

Gaze-quality correction, a signal-processing model for the motor marks, and reliability
hardening of the pipeline. The gaze is cleaned before mapping (per-participant **drift
correction** from the in-video calibration panels — CV-gated so it never worsens — plus
**blink exclusion** and velocity-gated smoothing, all preserving the sampling rate). The motor
marks (`motor_onset` / `target_touch` / `hand_exit`) are re-derived post-hoc from the per-frame
occlusion profile (the **bump model**: rise → peak → valley on the target `fT` and whole-board
`board_occ` occlusion curves), which is persisted as `signal_trace`, so a change to the post-hoc
stage reprocesses in **seconds without re-decoding video**. A reliable **`off_target`** anomaly
(a completed reach — hand in *and* out of the board — with neither a target touch nor the gaze
committed to the target) plus the **gaze-validated piece** capture "went elsewhere" errors. The
key lesson of this release: **result quality must not depend on remembering a flag** — slow/
precise is now the default, every output records its `run_config`, and incomplete/partial runs
warn loudly. Validated on the **22 participants**: on the common trials, hand-exit coverage
**77 → 86%** (the bump model), target-touch unchanged; the absolute drops seen in a first run
were traced to fast-mode subsampling and a mis-tuned exception config, both fixed below.

### Added
- **Gaze drift correction** ([`GazeCorrectionHandler`](src/core/GazeCorrectionHandler.py),
  offline [`gaze_calibration.py`](src/tools/gaze_calibration.py)): per-participant, segment-aware
  offset from the recoverable in-video 9-dot panels, adopted only when a leave-one-panel-out +
  bootstrap gate proves it does not worsen (8/22 apply). Plus **blink exclusion**
  (`blinks.pldata`) and a confidence²-weighted, saccade-segmented **gaze smoother** in
  [`EyeDataHandler`](src/core/EyeDataHandler.py).
- **Bump model for the motor marks** (post-hoc, from `signal_trace`): `target_touch` = target
  occlusion peak, `hand_exit` = whole-board occlusion valley (adaptive to the local target for
  finger-only touches), `motor_onset` = the contour entry **validated by the occlusion** (a
  contour lost without any occlusion rise is an artifact — homography flicker or an edge hand —
  and is moved to the real rise; cause recorded in `motor_onset_source`). Relaxed temporal
  **congruence** + `reach_style`. Re-applied without video by
  [`reprocess_landmarks.py`](src/tools/reprocess_landmarks.py).
- **`target_found`** as an I-DT **dwell** (windowed dispersion on the corrected gaze), and a
  new **`validation`** phase/mark (last board fixation before the touch).
- **Off-target anomaly detection**: `error_type` = `correct` / `off_target` / `no_touch` from
  reliable signals (reach completed + target touch + gaze on target), plus `gaze_validated_piece`
  (which piece the eyes committed to). `touched_piece`/`touched_cell` are included as
  **experimental** (the cheap occlusion cannot yet separate the fingertip from the arm) and do
  not drive `error_type`. Per-cell occlusion ([`getCellOcclusionMap`](src/core/BoardHandler.py))
  and **panel presence** are recorded in `signal_trace`.
- **Provenance + guards**: every output records `run_config` (mode, topic, offline, frame range);
  `store_results` warns when a run is **incomplete** (likely subsampled) or a **partial** segment.
- **Config-safety check** ([`check_correct_trials.py`](src/tools/check_correct_trials.py)): flags
  per-participant exception sequences that turn a **real trial into a discarded `-1`** without a
  documented reason (caught 001/035/055).
- **Debug figures for flagged trials** ([`debug_flagged_trials.py`](src/tools/debug_flagged_trials.py)):
  gaze path + target/touched/validated markers + a zoomed board thumbnail at the press, into each
  participant's `debug_figures/`.
- **HTML report — "Comportamiento" tab** ([`generate_report.py`](src/tools/generate_report.py)):
  an exploratory behaviour summary for the analysis team (and developer debug), from the latest
  run: gaze-by-target **colour/shape confusion matrices** (search guided by colour > shape vs the
  balanced-board 25%/20% chance), per-target **found (gaze) vs touched (occlusion)** rates (red the
  hardest to discriminate), a **time-to-X selector** (verifica / encuentra / mano entra / toque;
  default *verifica* — the cleanest cognitive measure and best detected), and **per-colour response
  time across blocks** (familiarisation + the block-3 rotation bump).
- **Whole-board occlusion masks** for the documentation: the `board_occ` counterpart of the
  target-cell touch masks (current board / clean reference / diff / change mask), captured via
  `_dbg_board_masks` and dumped by `process_video --dump_frames`.
- **Documentation figures reworked** ([`generate_doc_figures.py`](src/tools/generate_doc_figures.py)):
  two-view occlusion **bump** and **phase-timeline** figures (full panel-to-panel cycle + zoomed
  detail), touch masks as separate clean-vs-touch comparisons, a varied-piece **trajectory gallery**,
  cross-plot colour coherence, and bilingual (`_eng`) variants; the default output root tracks
  `__version__`. Gaze-drift diagnostics in [`gaze_drift_figures.py`](src/tools/gaze_drift_figures.py).

### Changed
- **Slow/precise is the DEFAULT** in both entry points (`process_video.py`, `run_all.py`); the
  fast ~6.5× subsampling is an explicit `--fast_analysis` opt-in (iteration only). The `-t topic`
  default is `gaze` in both (was `fixations` in `process_video.py`).
- `data_<id>.pkl` is the source of truth (it now carries `signal_trace`, `bump`, the wrong-piece
  fields, `run_config`); the CSVs are a projection derived from it.

### Fixed
- **Fast-mode frame subsampling could miss a marginally-detected panel** and lose the whole
  trial (measured: a first relaunch lost ~half the trials of 2 participants); the slow re-run
  recovers them — it only affects *which trials are detected*, not the marks.
- **001 block-4 exception config** wrongly turned the real trial 4 (red_hexagon) into a discarded
  `-1` (present since the file was created → wrong in every version), desyncing the block;
  reverted to the default. 035/055 `-1` (deliberate, the participant did not understand the task)
  documented.

## [1.2.0](https://github.com/enheragu/eye_tracking_board/releases/tag/v1.2.0) - 2026-06-15

Re-based trial model (homography + occlusion) and a complete per-trial timeline. The trial
now **starts at the search onset** (panel removal / first board gaze) instead of the
full-board contour confirmation, so `trial_duration_s` **increases** and is **not
comparable** with 0.x/1.0/1.1 — by design; reprocess the whole cohort with 1.2 and compare
only within 1.2. All gaze in the trial window is now recorded (tagged by phase), and state
transitions plus behavioural marks are published as one timeline. Validated on the **22
participants** (**1296 real trials**): vs 1.1.0 on the 18 common participants, target-touch
coverage **82.3% → 94.7%**, hand-exit **65.9% → 78.5%**, **+31 net real trials**, and **0
implausible touches** (none before board appearance or after the hand leaves). Per-session config
fixes recovered desynced sequences (a re-presented panel left one participant at 10 detected
trials → **62**); the ArUco fix below removes `few_arucos` failures at the cost of a small
contour-timing shift (one participant **−3** trials). Detail in the
[guide](docs/guia_procesamiento.md) and [technical doc](docs/documentacion_tecnica.md).

### Added
- **Unified per-trial timeline** (`trials_data_<id>_transitions.csv` + the stacked
  `combined_transitions_<topic>.csv`): every **state change** *and* every **behavioural
  mark** (`search_start`, `target_found`, `motor_onset`, `target_touch`, `hand_exit`,
  `trial_end`) interleaved by frame, with `time_s` and the block/trial. The raw state
  transitions are also stored in `data_<id>.pkl/.yaml` (`state_transitions`).
- **All gaze recorded, tagged by phase.** The panel-removal gaze that used to be discarded
  is now kept: `on_panel` (sample panel), `blank` (cell still covered), `not_board` (off
  board); plus `withdraw` (after the touch, through the motor-recovery window). Analysts
  filter the phase they want; only `search`/`verification`/`motor`-up-to-touch feed the
  summary counters.
- **Re-based trial start** (`frame_search_start`); `frame_init` (full-board confirm) kept as
  an internal reference. New covariates `anticipatory_gaze`, `anticipation_lead_s` and
  per-phase durations `search_duration_s` / `reach_duration_s` / `withdraw_duration_s`.
- **Board geometry reference** (`target_geometry.csv`): per-cell grid/metric position (mm)
  and Fitts-style `reach_distance_mm`, published once (a board property) instead of repeated
  on every trial row.
- Touch/hand-exit robustness: per-colour touch thresholds with warm-target (red/yellow)
  handling, session clean-board template, edge/Sobel + SSIM change components, and
  panel-as-occluder masking. `hand_exit` decoupled from the touch (whole-board + local-target
  occlusion baselines, contour fallback). `scikit-image` added to requirements.
- **ArUco detection robustness (homography).** Markers are now detected on the **original
  (distorted)** image and their corners undistorted, instead of detecting on the *undistorted*
  image. The undistort (`alpha=0`) pushed the edge markers — especially the top row — out of
  frame and lost them (measured: participant 042 lost markers in **16/16** frames, ~5/frame),
  weakening the homography and causing `few_arucos` failures. Resolution and gaze projection
  are unchanged (`newK == K`; recovered corners match to 0.15 px). Recovers precision for
  participants with marginal ArUco visibility.

### Changed
- **`trial_duration_s` is re-based and NOT comparable** with previous versions (starts at the
  search onset, ends at the border crossing).
- Summary per-colour counts now run **up to the touch** (`search`/`verification`/`motor`
  before `frame_target_touch`); `withdraw` and the pre-trial gaze are in the sequence CSV but
  do **not** count.
- Removed `frame_movement_onset` (redundant with the contour-based `frame_motor_onset`);
  `reach_distance_mm` moved from the per-trial CSV to `target_geometry.csv`.

### Known limitation
- Target-touch stays best-effort and does **not** close the trial; a missing touch can be
  estimated from the motor marks (see the technical doc). The robust trial end is the border
  crossing (`frame_end`), unaffected by touch misses.

## [1.1.0](https://github.com/enheragu/eye_tracking_board/releases/tag/v1.1.0) - 2026-06-13

Per-trial event marks (search / motor phases) and a best-effort target-touch detector,
plus state-machine robustness. Trial start/end keep the 1.0.0 criteria, so
`trial_duration_s` is unchanged and comparable; 1.1.0 only **adds** marks. Validated on
the 20 participants: **0 valid trials lost vs 1.0.0, +21 rescued.** Detail in the
[guide](docs/guia_procesamiento.md) and [technical doc](docs/documentacion_tecnica.md).

### Added
- **Per-trial event marks** (frame columns + per-phase durations): `frame_early_init`,
  `frame_init`, `frame_target_found`, `frame_motor_onset`, `frame_target_touch`,
  `frame_hand_exit`, `frame_end`. Independent — a missing one only blanks its column.
- **Target-touch detector** (`frame_target_touch`, best-effort, does **not** close the
  trial): change-detection on the target cell, watched through the motor phase. Coverage
  ~11% → **~83%**.
- **Early gaze during panel removal** (`pre_start` phase).
- **Per-participant gaze sampling rate** (`gaze_sampling_rate`, measured — not assumed
  200 Hz) and `gaze_continuity`.
- **HTML report**: matrix comparison (participants × trials, versions as sub-cells), tabs
  incl. time-to-touch and touch diagnostics (inline-SVG); frequencies + combined CSVs.
- Spurious ArUco filtering, stabilised cell grid, configurable IO roots, headless debug video.

### Changed (state-machine robustness)
- Unexpected panels handled uniformly: an out-of-sequence panel no longer terminates the
  run, a panel swap no longer hangs, and `test_motor_recovery` yields to any confirmed
  panel (recovers re-presented trials — the +21 rescued).

### Known limitation
- Target-touch is best-effort (~83%); residual misses are the occlusion signal or very few
  ArUcos, **not** the homography (the warped view is stable, measured). A missing touch can
  be estimated as `frame_motor_onset + 0.24·(frame_hand_exit − frame_motor_onset)`.

### Fixed
- Log messages no longer overwritten by the progress bar (`print` → `log`).

## [1.0.0](https://github.com/enheragu/eye_tracking_board/releases/tag/v1.0.0) - 2026-06-12

First stable, validated release. Outputs are **not directly comparable** with 0.x:
trial start/end criteria are more precise, so all participants should be reprocessed.
The software version is now stored in every output (`sw_version`) and logged on each
run. `src/tools/compare_outputs.py` compares two output roots (e.g. 0.x vs 1.x) at a
glance.

### Fixed
- Board border mask accepted any dark pixel (min/max bug + hue scale mismatch): black
  clothing could break or fake the board contour. Detection is now far more selective.
- 180-degree board rotation could be triggered by markers not belonging to the board;
  now only board markers vote, with hysteresis.
- Gaze on the right/bottom board edge could be misclassified as `not_board` (integer
  cell sizes); `Board norm Coord` is now normalized against the board area itself.
- Robustness fixes: corrupt-frame check, color-correction overflow clipping, threaded
  video writer/reader bugs.

### Changed
- `init_capture`/`end_capture` are backdated to the actual board appearance/last
  sighting: durations no longer include detection-confirmation overhead (~0.2 s), and
  trial start requires ~0.2 s of sustained visibility (no more degenerate trials).
- New finish statuses `test_finish_by_next_panel` and `test_finish_by_end_of_video`:
  trials interrupted by the next panel or by the end of the recording are closed as
  valid (with their end backdated) instead of stored as errors.
- Compact single-frame debug view (camera + overlays + board PiP) replaces the 2x3 mosaic.
- Configurable IO roots (`--data_root`/`--output_root`, `EEHA_DATA_ROOT`/`EEHA_OUTPUT_ROOT`);
  data and outputs live on the external drive by default.
- New layout: `src/core/` (library), `src/tools/` (auxiliary tools), `src/process_video.py`
  and `src/run_all.py` (entry points), `scripts/` (shell wrappers), `calibration/`, `docs/media/`.

### Performance
- ~6.5x faster end to end (participant 002: 48:45 -> 7:23): sequential decoding instead
  of per-frame seeks, single undistort+ArUco detection per frame, panel handler
  short-circuit by marker ids, no per-marker deep copies, gated per-sample logging.
- `src/run_all.py` processes participants in parallel with bounded jobs/threads.

## [0.8.0](https://github.com/enheragu/eye_tracking_board/releases/tag/v0.8.0) - 2026-06-02
- Complete processing guide (`docs/guia_procesamiento.md`): pipeline, CSV formats,
  methodological limitations and processing magnitude of the 20 processed participants.
- README with experiment description, trial tables and dissemination (SEPEX 2024,
  RECA14, CIP 2026, VSS 2026, doctoral congresses).
- Postprocessing/plotting scripts updated; print material and media uploaded.
- This is the version whose output supported the 2025-2026 conference contributions.

## [0.7.0](https://github.com/enheragu/eye_tracking_board/releases/tag/v0.7.0) - 2025-05-07
- Expected trial-sequence checking: missing trials and panel-transition jumps are
  stored as errored trials (`missing_trial_error_*`, `transition_error_*`) instead of
  silently lost; checking scripts for output completeness and trial sequences.
- Visualization and slow-analysis CLI flags; per-state frame skip tied to video FPS.
- ArUco detection centralised in the state machine step (first detection reuse).
- Board configuration updated for the new physical board; per-participant trial
  config exceptions; logs tagged per participant for parallel runs.

## [0.6.0](https://github.com/enheragu/eye_tracking_board/releases/tag/v0.6.0) - 2024-10-15
- Gaze topic processing with the Pupil Labs confidence threshold (> 0.6): the version
  used for the SEPEX 2024 conference poster.
- Run-all batch script; gaze projection over the original video (`project_data.py`).
- Logging through tqdm without disturbing the progress bar.

## [0.5.0](https://github.com/enheragu/eye_tracking_board/releases/tag/v0.5.0) - 2024-10-13
- CSV export: per-trial summary and chronological gaze sequence.
- Fixation duration propagated across frames; several gaze samples matched per frame
  (gaze sampled faster than world video).
- Board rotation handling (180-degree flipped board) and contour inertia.
- Fixed gaze-frame to image-frame coordinate translation (vertical flip).

## [0.4.0](https://github.com/enheragu/eye_tracking_board/releases/tag/v0.4.0) - 2024-10-07
- Adaptive board-border detection: border color sampled from image reference edges.
- Five-state machine (init, get_test_name, test_start/execution/finish).
- Speed multiplier to skip frames in states that allow it.
- Fixed image point projection to board coordinates; faster eye-data loading.

## [0.3.0](https://github.com/enheragu/eye_tracking_board/releases/tag/v0.3.0) - 2024-10-01
- Pupil Labs `.pldata` ingestion (gaze/fixations) matched to world timestamps.
- First state machine handling the experiment execution.
- Color correction of the camera image.

## [0.2.0](https://github.com/enheragu/eye_tracking_board/releases/tag/v0.2.0) - 2024-06-27
- Object-oriented refactor: board, panel, ArUco-board and distortion handlers.
- Custom ArUco board projection (homography from configured marker layout) for both
  the game board and the sample panels, replacing contour-only localization.

## [0.1.0](https://github.com/enheragu/eye_tracking_board/releases/tag/v0.1.0) - 2024-06-12
- First prototype: color square detection over video, board grid from detected
  contour, camera calibration and distortion correction, debug mosaic.
