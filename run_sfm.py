#%% Prepare Runtime Environment ###
%load_ext autoreload
%autoreload 2
%matplotlib inline
from utils.utils import setup_runtime_environment, Timer, colors, color_gen
setup_runtime_environment()

import logging
import numpy as np

from sfm.objects import PinholeCamera, extract_sfm_scene
from sfm.pipeline import IncrementalSfM
from utils.optimization import compute_residual_stats
from utils.visualization import draw_matches, plot_images, plot_sfm, make_border, draw_points


#%% Configuration ###
dataset_name = "slider_depth" # electro_rig_undistorted, delivery_area_undistorted
                                         # slider_depth, urban, office_zigzag, office_spiral
USE_CV2 = True
SKIP_FRAMES = False
FEATURE_BASED = True

# Load baselines from JSON
from config import setup_dataset_and_matcher
setup_vars = setup_dataset_and_matcher(dataset_name)
INDEX_0, INDEX_1, INDEX_END = setup_vars['INDEX_0'], setup_vars['INDEX_1'], setup_vars['INDEX_END']
dataset, image_paths, t_windows = setup_vars['dataset'], setup_vars['image_paths'], setup_vars['t_windows']
open_image, matcher, config, THICKNESS = setup_vars['open_image'], setup_vars['matcher'], setup_vars['config'], setup_vars['THICKNESS']

#%% Initialize Scene ###
%matplotlib inline

params = dataset.intrinsics['params'] # fx, fy, cx, cy, k1, k2, p1, p2, k3
K = np.array([[params[0], 0, params[2]], [0, params[1], params[3]], [0, 0, 1]])
distortion_coeffs = np.array(params[4:]) if len(params)>4 else None

camera_model = PinholeCamera(K, distortion_coeffs, dataset.intrinsics['height'], dataset.intrinsics['width'])

# Instantiate SfM pipeline
sfm = IncrementalSfM(camera_model, matcher, config=config, feature_based=FEATURE_BASED, use_cv2=USE_CV2)

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

DEPTH_PERCENTILE_RANGE = (5, 95)
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

