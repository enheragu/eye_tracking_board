
import os

import time
import bisect

import numpy as np
import pandas as pd

from src.core.deps.file_methods import load_pldata_file
from src.core.utils import log, print_named_dict
from src.core.GazeCorrectionHandler import GazeCorrectionHandler, GazeUncertaintyModel

GAZE_CONFIDENCE_THRESHOLD = 0.6
BLINK_MARGIN_S = 0.05
SMOOTH_WINDOW = 4
SMOOTH_VEL_THRESHOLD_PX = 15.0
WORLD_W, WORLD_H = 1280, 720


def load_blink_intervals(root_path):
    """onset->offset blink windows (+-margin) from blinks.pldata; empty if absent.

    Pupil keeps emitting gaze during a blink (at low confidence), and the
    eyelid-droop frames around it can survive the confidence filter with a
    displaced position. We drop those windows explicitly on top of the
    confidence threshold."""
    try:
        bl = load_pldata_file(directory=root_path, topic='blinks', track_progress_in_console=False)
    except (FileNotFoundError, OSError, ValueError):
        return np.zeros((0, 2))
    events = sorted((d['timestamp'], d['type']) for d in bl.data)
    intervals, open_ts = [], None
    for ts, kind in events:
        if kind == 'onset':
            open_ts = ts
        elif kind == 'offset' and open_ts is not None:
            intervals.append((open_ts - BLINK_MARGIN_S, ts + BLINK_MARGIN_S))
            open_ts = None
    return np.array(intervals) if intervals else np.zeros((0, 2))


def in_blink(timestamp, intervals):
    if len(intervals) == 0:
        return False
    return bool(np.any((timestamp >= intervals[:, 0]) & (timestamp <= intervals[:, 1])))


def velocity_gated_smooth(data, win=SMOOTH_WINDOW, vel_threshold_px=SMOOTH_VEL_THRESHOLD_PX, unc_model=None):
    """In-place velocity-gated smoother over a time-ordered list of gaze samples
    (each a dict with 'norm_pos' and 'confidence').

    Reduces per-sample jitter WITHIN fixations using a centered (bidirectional)
    weighted average, but resets at each saccade (a sample-to-sample jump >
    vel_threshold_px) so saccades and pursuit are NOT blurred. One output per input
    at the same timestamp: the sampling frequency is unchanged.

    With a `unc_model` (GazeUncertaintyModel, available), the weights are
    INVERSE-VARIANCE (1/sigma_factor, the estimation-optimal weighting) and each
    sample gets a 'cov' = per-sample covariance in normalised image coords:
        Sigma_jitter = base_cov / sum(weights in window)   [floored by the observed
        window scatter] + bias_cov (the accuracy/drift floor).
    Without a model it falls back to the conf^2 weighting and attaches no 'cov'."""
    if len(data) < 2 * win + 1:
        return
    pos = np.array([d['norm_pos'] for d in data], float)
    conf = np.array([d.get('confidence', 1.0) for d in data], float)
    vel = np.zeros(len(pos))
    vel[1:] = np.hypot(np.diff(pos[:, 0]) * WORLD_W, np.diff(pos[:, 1]) * WORLD_H)
    seg = np.cumsum(vel > vel_threshold_px)

    use_unc = unc_model is not None and getattr(unc_model, 'available', False)
    if use_unc:
        ecc = np.hypot((pos[:, 0] - 0.5) * WORLD_W, (pos[:, 1] - 0.5) * WORLD_H)
        factor = np.array([unc_model.sigma_factor(conf[i], ecc[i]) for i in range(len(pos))])
        weight = 1.0 / factor
        base_cov, bias_cov = unc_model.base_cov, unc_model.bias_cov
    else:
        weight = conf ** 2

    out = pos.copy()
    cov = [None] * len(pos)
    for s in np.unique(seg):
        idx = np.where(seg == s)[0]
        for j, i in enumerate(idx):
            sel = idx[max(0, j - win):min(len(idx), j + win + 1)]
            ww = weight[sel]
            sw = float(ww.sum())
            if sw > 0:
                out[i] = [(pos[sel, 0] * ww).sum() / sw, (pos[sel, 1] * ww).sum() / sw]
            if use_unc:
                sj = base_cov / sw if sw > 0 else base_cov
                # floor: when the window actually scatters more than the formal estimate,
                # adopt the OBSERVED 2x2 covariance (its SHAPE, not just its size), so the
                # ellipse follows the real per-axis dispersion of this fixation.
                if len(sel) >= 3 and sw > 0:
                    dx = pos[sel, 0] - out[i, 0]; dy = pos[sel, 1] - out[i, 1]
                    neff = sw * sw / float((ww * ww).sum())
                    wxx = float((ww * dx * dx).sum() / sw)
                    wyy = float((ww * dy * dy).sum() / sw)
                    wxy = float((ww * dx * dy).sum() / sw)
                    emp = np.array([[wxx, wxy], [wxy, wyy]]) / max(neff, 1.0)
                    # robust floor: raise sj so it dominates the observed scatter in EVERY
                    # direction (whiten by sj, cap eigenvalues at >=1, unwhiten). A trace
                    # test would miss a purely horizontal/vertical fusion (bigger on one
                    # axis, smaller on the other -> same trace). This way the ellipse
                    # follows the real per-axis dispersion without dropping below the
                    # calibration jitter.
                    try:
                        L = np.linalg.cholesky(sj + 1e-15 * np.eye(2))
                        Minv = np.linalg.solve(L, np.linalg.solve(L, emp).T).T
                        wv, Vv = np.linalg.eigh(Minv)
                        if wv[-1] > 1.0:
                            sj = L @ (Vv @ np.diag(np.maximum(wv, 1.0)) @ Vv.T) @ L.T
                    except np.linalg.LinAlgError:
                        # sj near-singular: the sum dominates BOTH sj and emp in every
                        # direction (both PSD) and is conservative -- avoids the trace test,
                        # which would miss a purely horizontal/vertical fusion.
                        sj = sj + emp
                cov[i] = sj + bias_cov

    for i, d in enumerate(data):
        d['norm_pos'] = [float(out[i, 0]), float(out[i, 1])]
        if use_unc and cov[i] is not None:
            c = cov[i]
            d['cov'] = [[float(c[0, 0]), float(c[0, 1])], [float(c[1, 0]), float(c[1, 1])]]

def check_duplicated_timestamps(data):
    timestamps_checked = set()
    timestamps_duplicated = set()

    for dic in data:
        timestamp = dic['timestamp']
        if timestamp in timestamps_checked:
            timestamps_duplicated.add(timestamp)
        else:
            timestamps_checked.add(timestamp)

    return timestamps_duplicated


"""
    When projecting fixation data take into account that theres a big discrepancy
    in how the matching occurs, theres a lot more thata in fixations that in frames, 
    so some frames can end up with more than one fixation associated:
        FPS of eye1.mp4 is 123.88
        FPS of world.mp4 is 29.81
"""
class EyeDataHandlerPLDATA:
    def __init__(self, root_path, data_path, video_fps, topic_data='fixations', participant_id=None):
        start_time = time.time()

        self.topic_data = topic_data
        self.participant_id = participant_id
        blink_intervals = load_blink_intervals(root_path)

        pldata = load_pldata_file(directory=data_path, topic=topic_data, track_progress_in_console=True)
        # log(f'{type(data.data)} - {data.data}')

        world_timestamps_path = os.path.join(root_path,'world_timestamps.npy')
        self.world_timestamps = np.load(world_timestamps_path)
        # log(f'{self.world_timestamps = }')
        # log(f'{len(self.world_timestamps) = }')

        log(f"[EyeDataHandlerPLDATA(::__init__] Process all {topic_data} topic data to match world timestamps, fixations data from:")
        log(f"\t\t· {world_timestamps_path}")
        log(f"\t\t· {data_path}/{topic_data}.pldata")
        log(f"\t\t· {data_path}/{topic_data}_timestamps.npy")


        self.video_fps = video_fps
        self.fixation_start_world_frame = {}
        self.data = []

        last_world_index = 0
        all_timestamps = []
        blink_excluded = 0
        for index, item in enumerate(pldata.data):
            dict_obj = dict(item)
            current_timestamp = dict_obj['timestamp']
            all_timestamps.append(current_timestamp)
            duration = 0 if not 'duration' in dict_obj else dict_obj['duration']
            if dict_obj['confidence'] > GAZE_CONFIDENCE_THRESHOLD:
                if in_blink(current_timestamp, blink_intervals):
                    blink_excluded += 1
                    continue
                self.data.append({'norm_pos': list(dict_obj['norm_pos']),
                                'timestamp': current_timestamp,
                                'duration': duration,
                                'confidence': dict_obj['confidence']})

        # print_named_dict('[EyeDataHandlerPLDATA(::__init__] dict_obj', dict_obj)
        self.data.sort(key=lambda x: x['timestamp'])
        self.world_timestamps = sorted(self.world_timestamps)

        # --- gaze corrections applied at the vuelo in the dataloader ---
        # (1) blinks already excluded above; (2) per-participant drift correction
        # from the offline artifact (identity if absent or gated off); (3)
        # velocity-gated scatter smoother (dense gaze only). All three preserve
        # the input sampling rate and timestamps.
        world_t0 = self.world_timestamps[0] if len(self.world_timestamps) else None
        gaze_corr = GazeCorrectionHandler(participant_id, world_t0=world_t0) if participant_id is not None else None
        unc_model = GazeUncertaintyModel(participant_id) if participant_id is not None else None
        if gaze_corr is not None and gaze_corr.apply and self.data:
            for d in self.data:
                nx, ny = d['norm_pos']
                d['norm_pos'] = list(gaze_corr.correct_bottomleft(d['timestamp'], nx, ny))
        if topic_data == 'gaze' and self.data:
            velocity_gated_smooth(self.data, unc_model=unc_model)
        log(f"[EyeDataHandlerPLDATA(::__init__] Blink-excluded samples: {blink_excluded}; "
            f"drift correction: {'ON' if (gaze_corr is not None and gaze_corr.apply) else 'identity'}; "
            f"scatter smoothing: {'ON' if topic_data == 'gaze' and self.data else 'off'}; "
            f"uncertainty model: {'ON' if (unc_model is not None and unc_model.available) else 'off'}")

        # Real gaze sampling rate over ALL samples (valid AND invalid): each sample
        # represents one sampling interval (1/rate), so a valid sample landing on a
        # cell counts 1/rate seconds. Invalid (low-confidence) samples in between are
        # "no data" gaps, they do NOT stretch the previous valid sample; using the
        # valid-only rate would inflate dwell times. The rate is taken from the MEDIAN
        # inter-sample interval (robust to isolated pauses). gaze_continuity is the
        # fraction of intervals within +-20% of that median: ~1.0 means a regular,
        # continuous stream (no hidden dropped samples); a low value would mean the
        # stream has gaps and the rate cannot be trusted as a uniform clock. The Pupil
        # Core nominal rate is 200 Hz but the exported gaze runs lower (~124 Hz), so
        # this must be stored and used to convert counts to time, not assumed 200 Hz.
        self.gaze_sampling_rate = None
        self.gaze_continuity = None
        if len(all_timestamps) > 2:
            all_timestamps.sort()
            intervals = np.diff(all_timestamps)
            intervals = intervals[intervals > 0]
            if intervals.size:
                median_dt = float(np.median(intervals))
                self.gaze_sampling_rate = 1.0 / median_dt
                self.gaze_continuity = float(np.mean(np.abs(intervals - median_dt) <= 0.2*median_dt))
        rate_str = f"{self.gaze_sampling_rate:.1f} Hz (continuity {self.gaze_continuity:.2%})" if self.gaze_sampling_rate else "N/A"
        log(f"[EyeDataHandlerPLDATA(::__init__] Real gaze sampling rate (all samples, median interval): {rate_str}")
        log(f"[EyeDataHandlerPLDATA(::__init__] Total samples: {len(all_timestamps)}; valid (conf>{GAZE_CONFIDENCE_THRESHOLD}): {len(self.data)}")
        log(f"[EyeDataHandlerPLDATA(::__init__] Duplicated timestamps in {topic_data} file: {len(check_duplicated_timestamps(self.data))}")

        duplicated = 0

        for index, item in enumerate(self.data):
            timestamp = item['timestamp']
            duration = item['duration']
            duration_frames = int(self.video_fps*(duration/1000.0))
            duration_frames = 1 if duration_frames < 1 else duration_frames
            
            # Find index in which this new timestamp would 'fit'
            video_frame = bisect.bisect_right(self.world_timestamps, timestamp)

            def check_video_frame(video_frame, world_timestamps, fixation_start_world_frame, recursion = 0, max_recursion = 2):
                if video_frame >= len(world_timestamps):
                    # Should not be bigger than world_timestamps
                    video_frame = len(world_timestamps) - 1
                else:
                    video_frame = video_frame - 1
                
                video_frame = max(0, video_frame)
                
                # okey, its repeated, just assign it to next frame...
                # if recursion < max_recursion and video_frame in fixation_start_world_frame:
                #     video_frame += 1
                #     check_video_frame(video_frame, world_timestamps, fixation_start_world_frame, recursion+1)
                    
                return video_frame
            
            video_frame = check_video_frame(video_frame=video_frame, world_timestamps=self.world_timestamps, 
                                            fixation_start_world_frame=self.fixation_start_world_frame,
                                            max_recursion=1)
            # if video_frame in self.fixation_start_world_frame:
            #     duplicated += 1
            
            # Propagate duration :)
            for frame in range(video_frame, video_frame + duration_frames):
                if frame not in self.fixation_start_world_frame:
                    self.fixation_start_world_frame[frame] = []    
                
                self.fixation_start_world_frame[frame].append(index)
            # self.fixation_start_world_frame[video_frame] = index

        # log(f"[EyeDataHandlerPLDATA(::__init__] Duplicated timestamps when matching {topic_data} to world video frames: {duplicated}")
                 
        execution_time = time.time() - start_time
        
        max_index = max(list(self.fixation_start_world_frame.keys()))
        min_index = min(list(self.fixation_start_world_frame.keys()))
        
        frames_with_fixations = sum([len(fixation_list) for fixation_list in self.fixation_start_world_frame.values()])
        frame_with_max_fixations = max([len(fixation_list) for fixation_list in self.fixation_start_world_frame.values()])
        log(f"[EyeDataHandlerPLDATA(::__init__] Total number of {topic_data} data, once propagated and filtered: {frames_with_fixations}; frame with max {topic_data} data has: {frame_with_max_fixations} items") #; {self.fixation_start_world_frame[min_index] = }; {self.fixation_start_world_frame[max_index] = }"")
        log(f"[EyeDataHandlerPLDATA(::__init__] Number of frames: {len(self.world_timestamps)}; Pupil timestamp item[0] = {self.world_timestamps[0]}; Pupil timestamp item[-1] ={self.world_timestamps[-1]}")
        log(f"[EyeDataHandlerPLDATA(::__init__] Finished process, took {execution_time:.2f} seconds")

    def step(self, frame_index):
        coord_list = []
        cov_list = []   # per-coord 2x2 covariance in normalised TOP-LEFT image coords (or None)
        if frame_index in self.fixation_start_world_frame:
            timestamp_idx_list = self.fixation_start_world_frame[frame_index]
            # log(f"[EyeDataHandlerPLDATA(::step] Frame {frame_index} contains {len(timestamp_idx_list)} fixation data")
            for fixation_timestamp_idx in timestamp_idx_list:
                d = self.data[fixation_timestamp_idx]
                X, Y = d['norm_pos'][0], d['norm_pos'][1]
                ## Flip fixation points from original coord system (bottom-left) to image
                ## coordinate system, origint at top-left
                coord_list.append([X, 1-Y])
                c = d.get('cov')
                if c is not None:
                    # bottom-left -> top-left (y -> 1-y): variances keep, xy sign flips
                    cov_list.append([[c[0][0], -c[0][1]], [-c[1][0], c[1][1]]])
                else:
                    cov_list.append(None)
                # log(f"[EyeDataHandlerPLDATA(::step] Data in frame {frame_index} is: {self.data[fixation_timestamp_idx]['norm_pos']}")

        self.last_cov_list = cov_list
        return coord_list
    


class EyeDataHandlerCSV:
    
    def __init__(self, root_path, data_path, video_fps, topic_data='fixations'):
        start_time = time.time()

        self.video_fps = video_fps
        csv_path = os.path.join(data_path, f"{topic_data}.csv")
        self.data = pd.read_csv(csv_path)
        # log(self.data)
        
        log(f"[EyeDataHandlerCSV::__init__] Process all {topic_data} topic data to match world timestamps, fixations data from:")
        log(f"\t\t· {csv_path}")
        
        
        log(f"[EyeDataHandlerCSV::__init__] Total number of {topic_data}: {self.data.shape[0]}")
           
        execution_time = time.time() - start_time        
        
        log(f"[EyeDataHandlerCSV::__init__] Finished process, took {execution_time:.2f} seconds")

    def step(self, frame_index):
        coordinate_list = []
        gaze_points = self.data[(self.data["start_frame_index"] <= frame_index) & (self.data["end_frame_index"] >= frame_index)]
        if not gaze_points.empty:
            gaze_points = gaze_points.sort_values(by="id")
            gaze_points = gaze_points[["norm_pos_x", "norm_pos_y"]]
            gaze_points = gaze_points.to_numpy()
            X, Y = gaze_points[:, 0], gaze_points[:, 1]

            # Flip the fixation points from the original coordinate system,
            # where the origin is at botton left, to the image coordinate system,
            # where the origin is at top left
            Y = 1 - Y
            coordinate_list = gaze_points.tolist()
        
        return coordinate_list