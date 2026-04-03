import numpy as np


class PinholeCamera:
    def __init__(self, K, distortion_coeffs=None, height=None, width=None):
        self.K = K
        self.distortion_coeffs = distortion_coeffs
        self.height = height
        self.width = width

class Frame:
    _next_id = 0
    def __init__(self, model, R, t, frame_id=None, t_window=None, path=None, rgb=None, feature_based=False):
        if frame_id is None: self.id, Frame._next_id = Frame._next_id, Frame._next_id + 1
        else: self.id = frame_id
        self.model = model
        self.R = R
        self.t = t
        self.t_window = t_window # temporal window
        self.path = path
        self.rgb = np.array(rgb) if rgb is not None else np.zeros(3) # 8 bit
        self.keypoints = [] # observations
        self.descriptors = [] if feature_based else None
        self.scores = [] if feature_based else None
        self.points = [] # Point_ids (used just for pnp correspondences)

    def add_observations(self, kps, pts, descs=None, scores=None):
        """Registers a batch of new observations to the frame and links them to 3D points"""
        assert len(kps) == len(pts)
        start_idx = len(self.keypoints)

        self.keypoints.extend(kps)
        if descs is not None: self.descriptors.extend(descs)
        if scores is not None: self.scores.extend(scores)
        for i in range(len(kps)):
            self.points.append(pts[i].id)
            pts[i].observations.append((self.id, start_idx + i))  # feat_idx = start_idx + i

class Point:
    _next_id = 0
    def __init__(self, xyz, id=None, rgb=None):
        if id is None: self.id, Point._next_id = Point._next_id, Point._next_id + 1
        else:self.id = id
        self.xyz = xyz[:3] # 3D coordinates
        self.rgb = np.array(rgb) if rgb is not None else np.zeros(3) # 8 bit
        self.observations = [] # [(frame_id, keypoint_idx),..]

def get_visible_points(frames_list, points_list):
    frame_ids = {frame.id for frame in frames_list}
    return [
        pt for pt in points_list
        if any(frame_id in frame_ids for frame_id, _ in pt.observations)
    ]

def get_nonvisible_points(frames_list, points_list):
    frame_ids = {frame.id for frame in frames_list}
    return [
        pt for pt in points_list
        if not any(frame_id in frame_ids for frame_id, _ in pt.observations)
    ]

def extract_ba_table(frames_id_map, active_frames, points_list):
    """
    turns the bipartite graph of relationships between frames and points into
    a flat table [ Observation | Frame idx | 3D Point idx | 2D Coordinate ]
    the outputs are basically the columns of the table.
    observations are sorted as active_observations then fixed_observations
    frames (rvecs and tvecs) are sorted as active_frames then fixed_frames
    Args:
        active_frames (list of Frame objects)
        points_list (list of Point objects)
    Returns:
        rvecs (N_frames x 3): Rodrigues rotation vectors
        tvecs (N_frames x 3): frame translation vectors (as used in projection X_cam = R*X + t)
        points (N_points x 3): 3D points
        obs_frames_idxs (array): frame indices for each observation (index in rvecs and tvecs)
        obs_points_idxs (array): 3D point indices for each observation (in ´points´)
        obs_keypoints (array): 2D coordinates of the observation
        n_active_frames: frames (rvecs and tvecs) are sorted as active_frames + fixed_frames
        n_active_obs: observations are sorted as active_observations + fixed_observations
    """
    n_active = len(active_frames)
    ba_frames = list(active_frames)
    frame_id_to_idx = {f.id: i for i, f in enumerate(ba_frames)}

    active_obs, fixed_obs = [], []

    for pt_idx, pt in enumerate(points_list):
        for frame_id, kp_idx in pt.observations:
            if frame_id not in frame_id_to_idx:
                ba_frames.append(frames_id_map[frame_id])
                frame_id_to_idx[frame_id] = len(ba_frames)-1

            frame_idx = frame_id_to_idx[frame_id]
            kp = ba_frames[frame_idx].keypoints[kp_idx]

            row = (frame_idx, pt_idx, kp[0], kp[1])
            if frame_idx < n_active: active_obs.append(row)
            else: fixed_obs.append(row)

    obs = np.asarray(active_obs + fixed_obs)
    if obs.size == 0: return None

    return {
        "Rs": np.asarray([f.R for f in ba_frames]),
        "tvecs": np.asarray([f.t.reshape(3) for f in ba_frames]),
        "points": np.asarray([p.xyz for p in points_list]),
        "obs_frames_idxs": obs[:, 0].astype(np.int32),
        "obs_points_idxs": obs[:, 1].astype(np.int32),
        "obs_keypoints": obs[:, 2:4],
        "n_active_frames": n_active,
        "n_active_obs": len(active_obs),
    }

##  Post-processing     #########################

def extract_sfm_scene(frames_list, points_list, pt_stride=1, colored_pts=False):
    """Return positions (N,3), directions (N,3), points (M,3) for visualization"""
    frames_positions, frames_directions, frames_rgb = [], [], []
    for frame in frames_list:
        if frame.t is None or frame.R is None:
            #continue
            raise ValueError(f"Camera {frame.id} has undefined extrinsics")
        center = -frame.R.T @ frame.t
        frames_positions.append(center)
        frames_directions.append(frame.R[-1]) # 3rd column of R^-1 (= R^T) is camera's Z axis expressed in world coords
        frames_rgb.append(frame.rgb)

    points_xyz = np.array([p.xyz for i, p in enumerate(points_list) if i % pt_stride == 0])
    points_rgb = np.array([p.rgb for i, p in enumerate(points_list) if i % pt_stride == 0], dtype=np.uint8)

    return {
        "frames_positions": np.array(frames_positions),
        "frames_directions": np.array(frames_directions),
        "frames_rgb": np.array(frames_rgb, dtype=np.uint8),
        "points_xyz": points_xyz,
        "points_rgb": points_rgb if colored_pts else None,
    }
