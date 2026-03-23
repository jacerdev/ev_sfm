# https://www.eth3d.net/datasets

import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as scipy_Rotation

from utils.geometry import compute_rotation_error, compute_translation_error


class FrameDataset:
    def __init__(self, root_dir, camera_number=5):
        self.root = Path(root_dir)
        self.camera_number = camera_number # 4..7

        self.images_txt_path = self.root / "rig_calibration_undistorted" / "images.txt"
        self.cameras_txt_path = self.root / "rig_calibration_undistorted" / "cameras.txt"
        self.points3D_txt_path = self.root / "rig_calibration_undistorted" / "points3D.txt"

        self.intrinsics = {}
        self.image_paths = []

        self.cam_id = -1
        self.frame_ids = []
        self.extrinsics = [] # list of (R, t)
        self.all_points2d = [] # list of lists [(x,y), ...] per image
        self.all_point3d_refs = [] # list of dicts [{point2d_idx, point3d_id}, ...] per image
        self.points3d = {} # {point3d_id: {"xyz": _, "rgb": _, "error": _, "observations": {frame_id: point2d_idx}}

        self._load_images()
        self._load_camera()
        self._load_points3D()
        print(f" {len(self.frame_ids)} images, {sum(len(sublist) for sublist in self.all_points2d)} observations, {len(self.points3d)} 3D points.")


        # sort images by file name (matches their natural)
        z = zip(self.image_paths, self.frame_ids, self.extrinsics, self.all_points2d, self.all_point3d_refs)
        z = sorted(z, key=lambda x: x[0])  # sort by image_paths
        (self.image_paths, self.frame_ids, self.extrinsics, self.all_points2d, self.all_point3d_refs) = map(list, zip(*z))
        self.path_to_idx = {p: i for i, p in enumerate(self.image_paths)} # to simulate incremental processing

        # Make ground truth relative to frame 0
        self.R_ref, self.t_ref = self.extrinsics[0]
        # get translation scale
        R1_gt, t1_gt = self.extrinsics[1]
        R1_rel = R1_gt @ self.R_ref.T
        t1_rel = t1_gt - R1_rel @ self.t_ref
        self.translation_scale = np.linalg.norm(t1_rel)

    def _load_camera(self):
        """Load intrinsics for the chosen camera"""
        assert self.cam_id != -1, "No images found for given camera number"
        with open(self.cameras_txt_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                parts = line.split()
                camera_id = int(parts[0])
                if camera_id != self.cam_id:
                    continue
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                params = [float(p) for p in parts[4:]]
                self.intrinsics = {
                    "model": model,
                    "height": height,
                    "width": width,
                    "params": params # fx, fy, cx, cy
                }
        # Later we will select the intrinsics matching CAMERA_ID in images.txt

    def _load_images(self):
        """
        Load image paths, extrinsics, 2D points, and 3D point references for the selected camera
        populates self.image_paths, self.extrinsics, self.all_points2d, self.all_point3d_refs
        """
        with open(self.images_txt_path) as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

        # images.txt has two lines per image
        for i in range(0, len(lines), 2):
            header = lines[i].split()
            frame_id = int(header[0])
            qw, qx, qy, qz = map(float, header[1:5])
            tx, ty, tz = map(float, header[5:8])
            cam_id = int(header[8])
            image_path = header[9]

            # Map camera_number to CAMERA_ID
            if f"cam{self.camera_number}_" not in image_path:
                continue  # skip images not belonging to the chosen camera

            # Rotation and translation
            R = scipy_Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            t = np.array([tx, ty, tz])

            # 2D points (only include triangulated points)
            point2d_line = lines[i+1].split()
            points2d = []
            point3d_refs = {}
            for j in range(0, len(point2d_line), 3):
                x, y, pt3d_id = float(point2d_line[j]), float(point2d_line[j+1]), int(point2d_line[j+2])
                points2d.append((x, y)) # always append to keep indices aligned with points3D.txt tracks
                if pt3d_id != -1:
                    point3d_refs[len(points2d)-1] = pt3d_id

            if self.cam_id == -1: self.cam_id = cam_id
            self.frame_ids.append(frame_id)
            self.image_paths.append(self.root / "images" / image_path)
            self.extrinsics.append((R, t))
            self.all_points2d.append(points2d)
            self.all_point3d_refs.append(point3d_refs)

    def _load_points3D(self):
        """
        Load 3D points, and corresponding observations.
        populates self.points3d: {point3d_id: {"xyz": _, rgb": _, "error": _,
                                              "observations": {frame_id: point2d_idx}}}
        """
        self.points3d = {}

        with open(self.points3D_txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                point3d_id = int(parts[0])

                # Geometry and color
                xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                rgb = np.array([int(parts[4]), int(parts[5]), int(parts[6])])
                error = float(parts[7])

                # Extract tracks: pairs of (IMAGE_ID, POINT2D_IDX)
                track_data = parts[8:]
                observations = {}

                for k in range(0, len(track_data), 2):
                    frame_id = int(track_data[k])
                    point2d_idx = int(track_data[k + 1])

                    observations[frame_id] = point2d_idx

                self.points3d[point3d_id] = {
                    "xyz": xyz,
                    "rgb": rgb,
                    "error": error,
                    "observations": observations
                }

    def match_keypoints(self, path1, path2):
        """
        simulates a matche extractor.
        Returns (N, 2) arrays for keypoints in image 1 and image 2 that correspond.
        """
        idx1 = self.path_to_idx[path1]
        idx2 = self.path_to_idx[path2]
        refs1 = self.all_point3d_refs[idx1]
        refs2 = self.all_point3d_refs[idx2]
        pt3d_to_pt2d_idx1 = {pt3d: pt2d for pt2d, pt3d in refs1.items()} # inverse lookup for image1: {point3d_id: point2d_idx}

        # Iterate through image2 points and check if they exist in Image 1
        matches_img1, matches_img2 = [], []
        for pt2d_idx2, pt3d_id in refs2.items():
            if pt3d_id in pt3d_to_pt2d_idx1: # pt2d_idx2 -> pt3d_id -> pt2d_idx1
                pt2d_idx1 = pt3d_to_pt2d_idx1[pt3d_id]
                matches_img1.append(self.all_points2d[idx1][pt2d_idx1])
                matches_img2.append(self.all_points2d[idx2][pt2d_idx2])

        return np.array(matches_img1), np.array(matches_img2)

    def _to_frame0(self, R1, t1):
        # Relative pose w.r.t reference
        R_rel = R1 @ self.R_ref.T
        t_rel = t1 - R_rel @ self.t_ref
        t_rel = t_rel / self.translation_scale # Normalize to unit norm (for comparison with essential matrix)
        return R_rel, t_rel

    def sanity_check(self, frame_obj):
        frame_id = frame_obj.id
        R, t = frame_obj.R, frame_obj.t
        R_gt, t_gt = self._to_frame0(self.extrinsics[frame_id][0], self.extrinsics[frame_id][1])
        rot_angle_err = compute_rotation_error(R, R_gt)
        t_angle_err, t_norm_err = compute_translation_error(t, t_gt)
        return rot_angle_err, t_angle_err, t_norm_err


# broken implementation cuz objects changed
from sfm.objects import PinholeCamera, Point, Frame

def build_scene_graph(dataset):
    """
    takes populated ETH3DDataset instance and returns:
    - camera_model: Shared Camera object
    - frames: List of Frame objects (fully linked)
    - points3d: Dict {point3d_id: Point3D_Object}
    """

    # Create Shared Camera Model
    intrinsics = dataset.intrinsics
    K = np.eye(3)
    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = intrinsics["params"]
    camera_model = PinholeCamera(K, intrinsics["width"], intrinsics["height"])

    # Create All 3D Points
    points3d = {}
    for pt3d_id, p_data in dataset.points3d.items():
        pt3d_obj = Point(pt3d_id, p_data["xyz"], p_data["rgb"])
        points3d[pt3d_id] = pt3d_obj

    # Create Frames and Link to 3D Points
    frames = []
    for i in range(len(dataset.frame_ids)):
        img_id = dataset.frame_ids[i]
        R, t = dataset.extrinsics[i]
        frame = Frame(img_id, camera_model, R, t)

        # Get 2D points and their 3D references for this image
        points2d = dataset.all_points2d[i]
        point3d_refs = dataset.all_point3d_refs[i]
        for idx, pt2d in enumerate(points2d):
            # iterate by index to match the order in dataset lists
            # idx is to be discarded and observations will be assigned new indices in frame.add_keypoint
            if idx in point3d_refs:
                pt3d_id = point3d_refs[idx]
                if pt3d_id in points3d:
                    pt3d_obj = points3d[pt3d_id]
                    frame.add_keypoint(pt2d, pt3d_obj)

        frames.append(frame)

    return camera_model, frames, points3d