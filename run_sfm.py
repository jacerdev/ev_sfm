#%% Prepare Runtime Environment ###
%load_ext autoreload
%autoreload 2
%matplotlib inline
# Force all libraries to use a single thread
import os
os.environ["OMP_NUM_THREADS"] = "1"       # OpenMP (used by many OpenCV builds and Scikit-learn)
os.environ["MKL_NUM_THREADS"] = "1"       # Intel MKL (used by NumPy/SciPy on Intel machines)
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS (used by NumPy/SciPy on standard Linux/Docker)
# Force OpenCV to use a single thread (to make sure cv2.SVDecomp doesnt deadlock)
import cv2
cv2.setNumThreads(0)

# Path setup
import sys
try: base_path = os.path.dirname(__file__)
except NameError: base_path = os.getcwd()
repo_path = os.path.join(base_path, "event_based", "SuperEvent")
if repo_path not in sys.path: sys.path.append(repo_path)

# Make PyTorch’s CUDA allocator allow GPU memory segments to grow dynamically
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Fix seed
import numpy as np
import torch
import random
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)


from itertools import count
import logging
import sys
import time
# Logging Setup
class ColorFormatter(logging.Formatter):
    COLORS = {logging.DEBUG: "\033[37m",
              logging.INFO: "",
              logging.WARNING: "\033[93m",
              logging.ERROR: "\033[31m",}
    RESET = "\033[0m"
    def format(self, record):
        msg = super().format(record)
        color = self.COLORS.get(record.levelno, "")
        return f"{color}{msg}{self.RESET}"
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(ColorFormatter('%(funcName)s(): %(message)s'))
logging.getLogger().handlers.clear()
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING) # silence matplotlib logging

# custom profiler
class Timer:
    def __init__(self, name="Block"): self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        self.interval = self.end - self.start
        logging.debug(f">>> {self.name} duration: {self.interval:.4f} sec <<<")

# Color Generator
colors = [
    [255,   0,   0],   # red
    [  0, 128,   0],   # green
    [  0,   0, 255],   # blue
    [  0, 255, 255],   # cyan
    [255,   0, 255],   # magenta
    [255, 255,   0],   # yellow
    [255, 165,   0],   # orange
    [128,   0, 128],   # purple
    [165,  42,  42],   # brown
    [255, 192, 203],   # pink
    [128, 128, 128],   # gray
    [  0, 255,   0],   # lime
    [  0, 128, 128],   # teal
    [128,   0,   0],   # maroon
    [  0,   0, 128],   # navy
    [128, 128,   0],   # olive
    [ 64, 224, 208],   # turquoise
    [255, 215,   0],   # gold
    [148,   0, 211],   # darkviolet
]
color_gen = (colors[1:][i % len(colors[1:])] for i in count())


from event_based.uzh_dataset import EventDataset
from frame_based.eth3d_dataset import FrameDataset
from event_based.feature_matcher import EventFeatureMatcher
from frame_based.feature_matcher import FeatureMatcher
from frame_based.pairwise_matcher import PairwiseMatcher
from event_based.SuperEvent.data_preparation.util.data_io import load_ts_sparse
from event_based.SuperEvent.util.visualization import ts2image
from utils.preprocessing import load_image
from sfm.objects import PinholeCamera, extract_sfm_scene
from sfm.pipeline import IncrementalSfM
from utils.optimization import compute_residual_stats
from utils.visualization import draw_matches, plot_images, plot_sfm, make_border, draw_points


#%% Configuration ###
dataset_name = "electro_rig_undistorted" # electro_rig_undistorted, delivery_area_undistorted
                                         # slider_depth, urban, office_zigzag, office_spiral
USE_CV2 = True
SKIP_FRAMES = False

FEATURE_BASED = True
USE_PRECOMPUTED_MATCHES = False

# Load baselines from JSON
baselines_path = os.path.join(base_path, "baselines.json")
import json
with open(baselines_path, "r") as f:
    baselines_config = json.load(f)

# Pull values based on dataset_name
baseline = baselines_config.get(dataset_name)

INDEX_0 = baseline["INDEX_0"]
INDEX_1 = baseline["INDEX_1"]
INDEX_END = baseline["INDEX_END"]
FLIP_ORDER = baseline["FLIP_ORDER"]
#%% Instantiate Dataset
DATA = "eth3d_frames"  if dataset_name in ["electro_rig_undistorted", "delivery_area_undistorted"] else "uzh_events" # "eth3d_frames", "uzh_events", or "uzh_frames"
if DATA == "eth3d_frames":
    ds_dir = "/home/jacer/Documents/TU-Berlin/13.WS2526/EventPJ/datasets/ETH3D/" + dataset_name
    out_dir = "/home/jacer/Documents/TU-Berlin/13.WS2526/EventPJ/workspace/outputs/ETH3D/" + dataset_name
    dataset = FrameDataset(ds_dir)
    image_paths, t_windows = dataset.image_paths, None
    def open_image(path): return load_image(path)

    if FEATURE_BASED:
        matcher = FeatureMatcher((dataset.intrinsics['height'], dataset.intrinsics['width']),
                                 matcher="flann", robust_matching=False)
    elif USE_PRECOMPUTED_MATCHES: matcher = PairwiseMatcher(extract_matches_func=dataset.match_keypoints)
    else: matcher = PairwiseMatcher(matches_dir=out_dir+"/matches_loftr")
    
    config = {
        # Pruning matches
        'MIN_ASSOCIAT_DIST': 5.0, 'PNP_RANSAC_TH': 10.0, 'EPI_RANSAC_TH': 3.0,
        # Filtering Points
        'CELL_SIZE': 20, 'OCCUPANCY_CELL_SIZE': 20, 'MIN_ANGLE': 1.0,
        # Keyframe Selection
        'MIN_BASELINE': 0.5, 'MIN_OVERLAP': 0.7, 'MIN_VISIBILITY': 0.6, 
        # Bundle Adjustment
        'BA_EVERY_N': 4, 'BA_WINDOW': 4,
    }
    THICKNESS = 6

else: #if DATA in ["uzh_events", "uzh_frames"]:
    ds_dir = "/home/jacer/Documents/TU-Berlin/13.WS2526/EventPJ/datasets/UZH/" + dataset_name
    out_dir = "/home/jacer/Documents/TU-Berlin/13.WS2526/EventPJ/workspace/outputs/UZH/" + dataset_name
    dataset = EventDataset(ds_dir, out_dir+"/mcts")
    
    config = {
        # Pruning matches
        'PNP_RANSAC_TH': 10.0, 'EPI_RANSAC_TH': 10.0,
        # Filtering Points
        'CELL_SIZE': 1, 'OCCUPANCY_CELL_SIZE': 1, 'MIN_ANGLE': 1., 
        # Keyframe Selection
        'MIN_BASELINE': .5, 'MIN_OVERLAP': .99, 'MIN_VISIBILITY': .99,
        # Bundle Adjustment
        'BA_EVERY_N': 2, 'BA_WINDOW': 4,
    }
    THICKNESS = 2

    if DATA == "uzh_events":
        image_paths, t_windows = dataset.mcts_paths, dataset.mcts_t_windows
        def open_image(path): return ts2image(load_ts_sparse(path))
        
        root_dir = "/home/jacer/Documents/TU-Berlin/13.WS2526/EventPJ/workspace/evis_geoba"
        matcher = EventFeatureMatcher(root_dir, (dataset.intrinsics['height'], dataset.intrinsics['width']),
                                      matcher="bf", robust_matching=False, windowed_matching=False, window_size=None)
    elif DATA == "uzh_frames":
        INDEX_1 *= 2
        image_paths, t_windows = dataset.image_paths, None
        def open_image(path): return load_image(path)
        
        if FEATURE_BASED:
            matcher = FeatureMatcher((dataset.intrinsics['height'], dataset.intrinsics['width']),
                                     matcher="flann", robust_matching=False)
        else:
            config['MIN_ASSOCIAT_DIST'] = 5.0
            matcher = PairwiseMatcher(matches_dir=out_dir+"/matches_loftr")

if FLIP_ORDER: image_paths = image_paths[::-1]

#%% Initialize Scene ###
%matplotlib inline

params = dataset.intrinsics['params'] # fx, fy, cx, cy, k1, k2, p1, p2, k3
K = np.array([[params[0], 0, params[2]], [0, params[1], params[3]], [0, 0, 1]])
distortion_coeffs = np.array(params[4:]) if len(params)>4 else None

camera_model = PinholeCamera(K, distortion_coeffs, dataset.intrinsics['height'], dataset.intrinsics['width'])


# Instantiate SfM pipeline
sfm = IncrementalSfM(camera_model, matcher, config=config, feature_based=FEATURE_BASED, use_cv2=USE_CV2)

# Consistency check
if INDEX_1 >= len(image_paths):
    logging.warning(f"INDEX_1 ({INDEX_1}) is out of bounds for dataset length {len(image_paths)}. Adjusting indices.")
    INDEX_1 = len(image_paths) - 1
    INDEX_0 = max(0, INDEX_1 - 5)

# Initialize scene
logging.info(f"Initializing scene with baseline frames {INDEX_0} and {INDEX_1}")
frame0_path, frame1_path = image_paths[INDEX_0], image_paths[INDEX_1]
t_window0, t_window1 = (t_windows[INDEX_0], t_windows[INDEX_1]) if t_windows is not None else (None, None)
frame0, frame1, x0, x1 = sfm.bootstrap(frame0_path, frame1_path, t_window0, t_window1, color0=colors[0], color1=next(color_gen))

# Plot
img0, img1 = open_image(frame0.path), open_image(frame1.path)
picture = draw_matches(make_border(img0), make_border(img1), x0, x1, thickness=THICKNESS)
plot_images(picture, title=f"{frame0.path.stem} - {frame1.path.stem}")

#%% Incremental SfM ###
%%prun -s cumulative -l 50
%matplotlib inline

i, stride = INDEX_1, 1
i += stride
while 0 <= i < (INDEX_END if INDEX_END != -1 else len(image_paths)):
    frame2_path, t_window = image_paths[i], t_windows[i] if t_windows is not None else None
    logging.warning(f"{i+1}/{len(image_paths)}: stride= {stride}"); i += stride

    # Process frame
    with Timer("Frame processing"): results = sfm.process_frame(frame2_path, t_window, color=next(color_gen))
    if results is None or results['baseline'] is None:
        logging.error(f"Too few matches extracted: {sfm.last_keyframe.path.name} - {frame2_path.name}")
        results = {'baseline':np.inf, 'tracking': {'kps1':[], 'kps2':[]}, 'mapping': {'kps1':[], 'kps2':[]}}

    # Skipping strategy
    if SKIP_FRAMES:
        velocity = results['baseline'] / stride
        ideal_stride = int(config['MIN_BASELINE'] / (velocity + 1e-9))
        stride = np.clip(int(0.5 * stride + 0.5 * ideal_stride), 1, 10) # alpha-blending to prevent wild oscillating

    # Evaluate
    if sfm.n_skipped == 0:
        median_err, mad, rmse= compute_residual_stats(sfm.keyframes_id_map, sfm.keyframes_list, sfm.points_list,K)
        logging.warning(f"Global residual stats ({len(sfm.keyframes_list)} keyframes, {len(sfm.points_list)} points): Median Error= {median_err:.2f}, Robust SD= {1.4826 * mad:.2f}, RMSE= {rmse:.2f}")

    # Plot
    FEW_2D3D, FEW_2D2D = False, False
    if len(results['tracking']['kps1']) < 30: logging.error(f"Too few 2D-3D matches extracted: {len(results['tracking']['kps1'])}"); FEW_2D3D=True
    elif len(results['mapping']['kps1']) < 20: logging.error(f"Too few 2D-2D matches extracted: {len(results['mapping']['kps1'])}"); FEW_2D2D=True
    if FEW_2D3D or FEW_2D2D:
        frame1 = sfm.last_keyframe if sfm.n_skipped > 0 else sfm.keyframes_list[-2]
        img1, img2 = open_image(frame1.path), open_image(frame2_path)

        x1_track, x2_track = results['tracking']['kps1'], results['tracking']['kps2']
        img1 = draw_points(img1, x1_track, thickness=THICKNESS)
        img2 = draw_points(img2, x2_track, thickness=THICKNESS)

        x1_map, x2_map = results['mapping']['kps1'], results['mapping']['kps2']
        picture = draw_matches(make_border(img1), make_border(img2), x1_map, x2_map, thickness=THICKNESS)
        plot_images(picture, title=f"{frame1.path.stem} - {frame2_path.stem}")


#%% Plot Scene ###
%matplotlib tk

DEPTH_PERCENTILE_RANGE = baseline.get("DEPTH_PERCENTILE_RANGE", (1, 99))
sfm_scene = extract_sfm_scene(sfm.keyframes_list, sfm.points_list,
                              pt_stride=1, colored_pts=False)
depths = sfm_scene['points_xyz'][:, 2]
depth_mask = (depths >= np.percentile(depths, DEPTH_PERCENTILE_RANGE[0])) & (depths <= np.percentile(depths, DEPTH_PERCENTILE_RANGE[1]))
sfm_scene['points_xyz'] = sfm_scene['points_xyz'][depth_mask]

plot_sfm(**sfm_scene, camera_size=1, point_size=10, color_range=(1, 99))


#%% Plot all computed positions before BA
%matplotlib tk

frames_list_np = np.array(sfm.frames_list, dtype=np.object_)
frames_rotations = np.stack(frames_list_np[:, 0]).astype(np.float32)
frames_positions = (-frames_rotations.swapaxes(-2, -1) @ np.stack(frames_list_np[:, 1]).reshape(-1, 3, 1)).squeeze()
frames_directions = frames_rotations[:, -1]

all_frames_colors = np.zeros((frames_list_np.shape[0], 3), dtype=sfm_scene['frames_rgb'].dtype)
keyframes_mask = frames_list_np[:, 2] != -1
all_frames_colors[keyframes_mask] = sfm_scene['frames_rgb']

plot_sfm(frames_positions, frames_directions, sfm_scene['points_xyz'], all_frames_colors,camera_size=1,point_size=4)

# %%
