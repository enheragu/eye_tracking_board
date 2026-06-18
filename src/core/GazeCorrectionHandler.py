
import os
import json

import numpy as np

from src.core.utils import log


class GazeCorrectionHandler:
    """Applies the per-participant gaze drift correction measured offline by
    src/tools/gaze_calibration.py.

    The artifact (calibration/gaze/<id>.json) stores, per calibration panel, the
    measured offset (recorded gaze - true dot) in NORMALISED image coordinates
    with origin at the TOP-LEFT - the same frame EyeDataHandler emits after its
    1-Y flip. At runtime, for a gaze sample at time t (seconds relative to the
    world recording start), the offset is the linear interpolation of the panels
    that belong to the SAME calibration segment as t (Pupil resets the bias at
    each registered recalibration, so we never interpolate across one). The
    offset is clamped at the segment ends (no extrapolation).

    If the artifact is missing, malformed, or its cross-validated gain did not
    clear the adoption gate (apply=false), this handler is the identity - it
    never worsens a participant's gaze.
    """

    def __init__(self, participant_id, calib_dir=None, world_t0=None):
        self.participant_id = participant_id
        self.world_t0 = world_t0
        self.apply = False
        self.calib_times = []
        self._seg_panels = {}   # segment -> (times[], off_x[], off_y[])

        if calib_dir is None:
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            calib_dir = os.path.join(repo_root, "calibration", "gaze")
        path = os.path.join(calib_dir, f"{participant_id}.json")

        if not os.path.isfile(path):
            log(f"[GazeCorrectionHandler] {participant_id}: no artifact at {path} -> identity")
            return
        try:
            with open(path) as f:
                art = json.load(f)
        except (OSError, ValueError) as exc:
            log(f"[GazeCorrectionHandler] {participant_id}: unreadable artifact ({exc}) -> identity")
            return

        if not art.get("apply", False):
            reason = art.get("stats", {}).get("reason", "gate not passed")
            log(f"[GazeCorrectionHandler] {participant_id}: apply=false ({reason}) -> identity")
            return

        self.calib_times = art.get("calib_segment_times", [])
        panels = art.get("panel_offsets", [])
        for p in sorted(panels, key=lambda d: d["t"]):
            seg = self._seg_panels.setdefault(p["seg"], ([], [], []))
            seg[0].append(p["t"]); seg[1].append(p["off_x"]); seg[2].append(p["off_y"])
        if self._seg_panels:
            self.apply = True
            g = art.get("stats", {}).get("gain")
            log(f"[GazeCorrectionHandler] {participant_id}: drift correction active "
                f"({len(panels)} panels, gain={g})")

    def _segment_of(self, rel_t):
        seg = 0
        for i, c in enumerate(self.calib_times):
            if rel_t >= c:
                seg = i
        return seg

    def offset(self, rel_t):
        """Offset (off_x, off_y) in normalised top-left coords to SUBTRACT at rel_t."""
        if not self.apply:
            return 0.0, 0.0
        seg = self._segment_of(rel_t)
        panels = self._seg_panels.get(seg)
        if panels is None:                 # no panel in this segment -> no correction
            return 0.0, 0.0
        times, ox, oy = panels
        return float(np.interp(rel_t, times, ox)), float(np.interp(rel_t, times, oy))

    def correct_topleft(self, rel_t, x, y):
        """x, y in normalised top-left image coords (EyeDataHandler's [X, 1-Y])."""
        if not self.apply:
            return x, y
        ox, oy = self.offset(rel_t)
        return x - ox, y - oy

    def correct_bottomleft(self, abs_ts, nx, ny):
        """Correct a sample stored as Pupil norm_pos (bottom-left). Uses absolute
        timestamp and world_t0 to get rel_t. Returns corrected (nx, ny)."""
        if not self.apply or self.world_t0 is None:
            return nx, ny
        ox, oy = self.offset(abs_ts - self.world_t0)
        # top-left y is (1-ny); subtracting oy there maps back to +oy in bottom-left
        return nx - ox, ny + oy


class GazeUncertaintyModel:
    """Per-sample gaze measurement-noise model (Phase 0 'uncertainty' block of
    calibration/gaze/<id>.json, written by src/tools/gaze_calibration.py).

    Everything is exposed in NORMALISED image coordinates (the frame EyeDataHandler
    works in), so the smoother can weight and propagate directly:

      - sigma_factor(conf, ecc_px): scalar VARIANCE multiplier = conf_var_factor(conf)
        * spatial_var_factor(ecc). Used as the inverse-variance smoother weight
        (1/factor, estimation-optimal) and to scale the base covariance.
      - base_cov: the base 2x2 jitter covariance (anisotropic/tilted), normalised.
      - bias_cov: the accuracy/drift covariance (corr_px if drift was applied else
        base_px, the residual radial magnitude split isotropically per axis),
        normalised. This is the floor that integrating samples cannot beat.

    .available is False when the block is absent (older artifact / insufficient
    panels); the caller then keeps the conf^2 weighting and attaches no covariance.
    """

    def __init__(self, participant_id, calib_dir=None):
        self.available = False
        self.base_cov = None
        self.bias_cov = None
        self._cf = None             # (conf[], var_factor[])
        self._sf = None             # (ecc_px[], var_factor[])

        if calib_dir is None:
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            calib_dir = os.path.join(repo_root, "calibration", "gaze")
        path = os.path.join(calib_dir, f"{participant_id}.json")
        if not os.path.isfile(path):
            return
        try:
            with open(path) as f:
                art = json.load(f)
        except (OSError, ValueError):
            return

        # Warn (don't fail) if the artifact was produced by an OLDER calibration method:
        # the stored fingerprint no longer matches the current code -> it should be re-run.
        stored_fp = art.get("fingerprint")
        if stored_fp:
            try:
                from src.tools.gaze_calibration import calibration_fingerprint
                cur_fp = calibration_fingerprint()
                if cur_fp != stored_fp:
                    log(f"[GazeUncertaintyModel] {participant_id}: calibration fingerprint "
                        f"{stored_fp} != current {cur_fp}; artifact is STALE -> re-run gaze_calibration.py")
            except Exception:
                pass

        u = art.get("uncertainty")
        if not u:
            return

        # Parse the noise model. A truthy but malformed 'uncertainty' block (hand-edited, or
        # written by an older method missing a key) leaves the model unavailable instead of
        # crashing the run -- same fail-safe as a missing/unreadable artifact.
        try:
            W, H = 1280.0, 720.0
            jc = u["jitter_cov_px"]
            ang = np.radians(jc["angle_deg"])
            R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
            cov_px = R @ np.diag([jc["sigma_major"] ** 2, jc["sigma_minor"] ** 2]) @ R.T
            D = np.diag([1.0 / W, 1.0 / H])
            self.base_cov = D @ cov_px @ D

            cf = u.get("conf_var_factor", {})
            if cf.get("conf"):
                self._cf = (np.asarray(cf["conf"], float), np.asarray(cf["factor"], float))
            else:
                self._cf = (np.array([0.5, 1.0]), np.array([1.0, 1.0]))
            sf = u.get("spatial_var_factor", {})
            if sf.get("ecc"):
                self._sf = (np.asarray(sf["ecc"], float), np.asarray(sf["factor"], float))
            else:
                self._sf = (np.array([0.0, 1000.0]), np.array([1.0, 1.0]))

            # accuracy/drift covariance. Prefer the ANISOTROPIC 2x2 (second moment of the
            # residual vectors from calibration); fall back to the isotropic radial split of
            # base_px for older artifacts.
            bc = u.get("bias_cov_px")
            if bc is not None:
                self.bias_cov = D @ np.asarray(bc, float) @ D
            else:
                stats = art.get("stats", {})
                bias_px = stats.get("corr_px") if art.get("apply", False) else stats.get("base_px")
                bias_px = float(bias_px) if bias_px else 0.0
                var_axis = (bias_px ** 2) / 2.0
                self.bias_cov = np.diag([var_axis / (W * W), var_axis / (H * H)])

            # Calibration measures these in TOP-LEFT image coords, but the smoother attaches
            # the covariance to Pupil's BOTTOM-LEFT norm_pos (y up). The y-flip (y -> 1-y)
            # negates the xy term, so convert here; step() then flips position AND covariance
            # back to top-left together, keeping every stage in one consistent frame.
            flip = np.array([[1.0, -1.0], [-1.0, 1.0]])
            self.base_cov = self.base_cov * flip
            self.bias_cov = self.bias_cov * flip
            self.available = True
        except (KeyError, TypeError, ValueError):
            self.available = False
            return

    def sigma_factor(self, conf, ecc_px):
        """Scalar variance multiplier for a sample (conf x spatial), clamped > 0."""
        cf = float(np.interp(conf, self._cf[0], self._cf[1]))
        sf = float(np.interp(ecc_px, self._sf[0], self._sf[1]))
        return max(cf * sf, 1e-3)
