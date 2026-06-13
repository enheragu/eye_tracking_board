#!/usr/bin/env python3
# encoding: utf-8

import os
import sys
import argparse

from tqdm import tqdm

import cv2 as cv

# Entry point lives in src/, make the repo root importable regardless of CWD
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.core.utils import log, setVerboseLog
from src.core.version import __version__

from src.core.BoardHandler import BoardHandler
from src.core.DistortionHandler import DistortionHandler
from src.core.PanelHandler import PanelHandler
# from src.core.EyeDataHandler import EyeDataHandlerCSV as EyeDataHandler
from src.core.EyeDataHandler import EyeDataHandlerPLDATA as EyeDataHandler
from src.core.ArucoBoardHandler import ARUCOColorCorrection
from src.core.StateMachineHandler import StateMachine, ExceptionNoMoreBlocks
from src.core.ThreadVideoStream import ThreadVideoWriter


## Default IO roots. Input data lives in the external drive and all the output is
# stored next to it so the (limited) local disk is not filled with video data.
# Both can be overridden through CLI arguments or these environment variables
DEFAULT_DATA_ROOT = os.environ.get('EEHA_DATA_ROOT', '/media/quique/EXTERNAL_USB1/BusquedaVisualAnalysis/InputData')
DEFAULT_OUTPUT_ROOT = os.environ.get('EEHA_OUTPUT_ROOT', f'/media/quique/EXTERNAL_USB1/BusquedaVisualAnalysis/OutputData_v{__version__}')

# Hue in degrees (0-360); epsilon in degrees too
colors_dict = {'red':    {'h': 350, 'eps': 29},
               'green':  {'h': 125, 'eps': 35},
               'blue':   {'h': 220, 'eps': 35},
               'yellow': {'h': 50,  'eps': 28}}

colors_list = {'red': (0,0,255), 'green': (0,255,0), 'blue': (255,0,0), 'yellow': (0,255,255), 'board': (255,255,0)}

parser = argparse.ArgumentParser(description='Process PupilLabs data with board')
parser.add_argument('-p', dest='participant', type=str, default='00001', metavar='id', help='Participant folder id')
parser.add_argument('-t', dest='topic', type=str, default='fixations', metavar='topic', help='Eye data topic to process (gaze/fixations)')
parser.add_argument('-o', '--use_offline_data', action='store_true', default=False, help='Makes use of offline data instead of online data. Note that offline data includes fixations but not gaze data.')
parser.add_argument('-v', '--visualization', action='store_true', default=False, help='Enable visualization for process.')
parser.add_argument('-s', '--slow_analysis', action='store_true', default=False, help='Enable slow analysis for a better transition detection.')
parser.add_argument('--data_root', type=str, default=DEFAULT_DATA_ROOT, help='Root folder containing one folder per participant.')
parser.add_argument('--output_root', type=str, default=DEFAULT_OUTPUT_ROOT, help='Root folder in which output data is stored.')
parser.add_argument('--debug_log', action='store_true', default=False, help='Enable high frequency (per gaze sample) terminal logging.')
parser.add_argument('--start_frame', type=int, default=0, help='First frame to process (for debugging a specific segment).')
parser.add_argument('--end_frame', type=int, default=None, help='Last frame to process (for debugging a specific segment).')
parser.add_argument('--no_window', action='store_true', default=False, help='With -v, record the debug video without opening a window (headless).')
args = parser.parse_args()

participant_id = args.participant
eye_data_topic = args.topic
slow_analysis = args.slow_analysis
enable_visualization = args.visualization
setVerboseLog(args.debug_log)

# Avoid oversubscription when several participants run in parallel (see run_all.py)
if 'EEHA_CV_THREADS' in os.environ:
    cv.setNumThreads(int(os.environ['EEHA_CV_THREADS']))

WINDOW_STREAM_BOARD = f'Board View {participant_id}'

participant_path = os.path.join(args.data_root, participant_id)
data_path = participant_path if not args.use_offline_data else os.path.join(participant_path,'offline_data')
video_path = os.path.join(participant_path,'world.mp4')

if not os.path.isdir(participant_path):
    raise FileNotFoundError(f"Participant data folder not found: {participant_path}. "
                            f"Check that the data drive is mounted or provide --data_root.")

log(f"[processVideo::{participant_id}] eyes_board_color version: {__version__}")
log(f"[processVideo::{participant_id}] Repo root folder: {REPO_ROOT}")
log(f"[processVideo::{participant_id}] Data folder: {participant_path}")
output_path = os.path.join(args.output_root, eye_data_topic, participant_id)
os.makedirs(output_path, exist_ok=True)

camera_calibration_path = os.path.join(REPO_ROOT, 'calibration/camera_calib.json')
game_configuration= os.path.join(REPO_ROOT,'cfg/game_config.yaml')
game_aruco_board_cfg= os.path.join(REPO_ROOT,'cfg/game_aruco_board.yaml')
trial_blocks_cfg= os.path.join(REPO_ROOT,f'cfg/trials_config_exceptions/{participant_id}_trials_config.yaml')
default_trial_blocks_cfg= os.path.join(REPO_ROOT,f'cfg/default_trials_config.yaml')
if not os.path.exists(trial_blocks_cfg):
    log(f"[processVideo::{participant_id}] Using default file for trials configuration: {trial_blocks_cfg}.")
    trial_blocks_cfg = default_trial_blocks_cfg
else:
    log(f"[processVideo::{participant_id}] Using trials configuration from exception file: {trial_blocks_cfg}.")

samples_configuration= os.path.join(REPO_ROOT,'cfg/sample_shape_cfg')
frame_speed_multiplier = 1 # process one frame each N to go faster
init_capture_idx = 0

mouse_callback_image = None
def mouse_callback(event, x, y, flags, param):
    global mouse_callback_image
    if event == cv.EVENT_LBUTTONDOWN and mouse_callback_image is not None:
        image = mouse_callback_image
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV_FULL)
        h, s, v = hsv_image[y, x]
        log(f'Hue: {int(h/255.0*360)}, Saturation: {s}, Value: {v}')
    if event == cv.EVENT_LBUTTONDOWN and mouse_callback_image is None:
        log(f'No image provided to detect HSV components.')

if enable_visualization and not args.no_window:
    cv.namedWindow(WINDOW_STREAM_BOARD, cv.WINDOW_AUTOSIZE)
    cv.setMouseCallback(WINDOW_STREAM_BOARD, mouse_callback)


def processVideo(video_path):
    global init_capture_idx, mouse_callback_image

    stream = cv.VideoCapture(video_path)
    if not stream.isOpened():
        log(f"[processVideo::{participant_id}] Could not open video {video_path}")
        exit()

    total_frames = int(stream.get(cv.CAP_PROP_FRAME_COUNT))
    log(f'[processVideo::{participant_id}] Processing participant: {participant_id}')
    log(f'[processVideo::{participant_id}] Processing: {video_path}')
    log(f'[processVideo::{participant_id}] Video with: {total_frames} frames')

    fps = stream.get(cv.CAP_PROP_FPS)
    frame_width = int(stream.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(stream.get(cv.CAP_PROP_FRAME_HEIGHT))
    log(f"[processVideo::{participant_id}] Processing video of {fps} FPS and {frame_width = }; {frame_height = }")
    log(f"[processVideo::{participant_id}] Processing video duration {total_frames/fps} seconds")

    if enable_visualization:
        # Threaded writer: encoding/disk IO does not block the processing loop
        writer = ThreadVideoWriter(os.path.join(output_path,f'debug_{participant_id}.mp4'),
                                   cv.VideoWriter_fourcc(*'mp4v'), int(fps/frame_speed_multiplier),
                                   (frame_width, frame_height)).start()

    distortion_handler = DistortionHandler(calibration_json_path=camera_calibration_path,
                                           frame_width=frame_width, frame_height=frame_height)

    board_handler = BoardHandler(aruco_board_cfg_path = game_aruco_board_cfg,
                                 game_cfg_path=game_configuration, colors_dict=colors_dict,
                                 colors_list=colors_list, distortion_handler=distortion_handler)

    panel_handler = PanelHandler(panel_configuration_path=samples_configuration, colors_dict=colors_dict,
                                 colors_list=colors_list, distortion_handler=distortion_handler,
                                 enable_visualization=enable_visualization)

    eye_data_handler = EyeDataHandler(root_path=participant_path, data_path=data_path, video_fps=fps, topic_data=eye_data_topic)

    state_machine_handler = StateMachine(board_handler, panel_handler, eye_data_handler, distortion_handler,
                                         sequence_cfg_path = trial_blocks_cfg, video_fps=fps, slow_analysis=slow_analysis)

    capture_idx = args.start_frame
    end_limit = total_frames if args.end_frame is None else min(total_frames, args.end_frame + 1)
    if capture_idx > 0:
        # Single seek to the start point; from here on decoding is sequential
        stream.set(cv.CAP_PROP_POS_FRAMES, capture_idx)

    with tqdm(total=end_limit, desc=f"Frames from {participant_id}", initial=capture_idx) as pbar:
        while capture_idx < end_limit:
            ret, original_image = stream.read()
            if not ret or original_image is None:
                log(f"[processVideo::{participant_id}] Can't receive frame (stream end?). Exiting ...")
                break
            original_image = ARUCOColorCorrection(original_image)

            try:
                state_machine_handler.step(original_image, capture_idx)
            except ExceptionNoMoreBlocks:
                log(f"[processVideo::{participant_id}] End of execution (no more blocks in sequence).")
                break

            if enable_visualization:
                mosaic, log_frame = state_machine_handler.visualization(original_image=original_image, capture_idx=capture_idx,
                                                                        last_capture_idx=total_frames, frame_width=frame_width,
                                                                        frame_height=frame_height, participan_id=participant_id)
                mouse_callback_image = mosaic
                writer.write(log_frame)
                # --no_window only records the debug video (works headless)
                if not args.no_window:
                    cv.imshow(WINDOW_STREAM_BOARD, mosaic)
                    key = cv.pollKey()
                    if key == ord('q') or key == ord('Q') or key == 27:
                        break

            capture_increment = frame_speed_multiplier*state_machine_handler.getFrameMultiplier()
            # Skip intermediate frames with grab(): decodes but skips the costly
            # retrieve/convert. Seeking per frame (CAP_PROP_POS_FRAMES) forced a
            # keyframe seek + decode forward on every iteration
            for _ in range(capture_increment - 1):
                if not stream.grab():
                    break
            capture_idx += capture_increment

            pbar.set_postfix(FPS=f"{state_machine_handler.tm.getFPS():.2f}")
            pbar.update(capture_increment)

    if stream.isOpened():  stream.release()
    if enable_visualization:  writer.release()
    cv.destroyAllWindows()

    state_machine_handler.store_results(output_path=output_path, participant_id=participant_id, video_fps=fps)


if __name__ == "__main__":
    processVideo(video_path)
