# Changelog

All notable changes to this project are documented in this file. Versions prior to
1.0.0 were reconstructed from the git history. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
