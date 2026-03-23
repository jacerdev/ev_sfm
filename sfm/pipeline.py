import cv2
import logging
import numpy as np

from utils.matching import pair_matches
from sfm.bundle_adj import run_bundle_adjustment
from sfm.objects import Frame, Point, extract_ba_table, get_visible_points, get_nonvisible_points
from sfm.routines import initialize_scene, track_frame, map_observations, prune_triangulated_kps
from utils.geometry import R_to_rodrigues, rodrigues_to_R_vectorized
from utils.keypoints import associate_keypoints_mutual
from utils.optimization import compute_residual_stats


class IncrementalSfM:
    def __init__(self, camera_model, matcher, config=None, feature_based=True, use_cv2=True):
        Frame._next_id, Point._next_id = 0, 0 # reset objects IDs

        # invariant parameters
        self.camera_model = camera_model
        self.matcher = matcher
        self.cfg = config if config is not None else self._get_config()
        self.feature_based = feature_based
        self.use_cv2 = use_cv2

        self.K = camera_model.K
        self.dist_coeffs = camera_model.distortion_coeffs
        self.image_shape = (camera_model.height, camera_model.width)

        # state variables
        self.keyframes_list = []
        self.points_list = []
        self.keyframes_id_map = {}
        self.points_id_map = {}
        self.last_keyframe = None
        self.last_feats = None # last keyframe's non-triangulated features

        self.frames_list = [] # stores (R, t, id, path) initialization values (without bundle adjustments)
        self.n_skipped = 0 # just for logging

    def bootstrap(self, frame0_path, frame1_path, t_window0=None, t_window1=None, color0=(255, 0, 0), color1=(0, 255, 0)):
        # Get matches
        data = self._extract_matches(frame0_path, frame1_path)
        x0, x1 = data['kps1'], data['kps2']
        if self.feature_based: descs0, scores0, descs1, scores1, unused_feats = data['descs1'], data['scores1'], data['descs2'], data['scores2'], data['unused_feats']
        else: descs0, scores0, descs1, scores1, unused_feats = (None,)*5

        # Initialize scene
        init_data = initialize_scene(self.K, x0, x1, ransac_thresh=self.cfg['EPI_RANSAC_TH'], use_cv2=self.use_cv2)
        x0, x1, X01 = x0[init_data['indices']], x1[init_data['indices']], init_data['points']
        if self.feature_based: descs0, scores0, descs1, scores1 = descs0[init_data['indices']], scores0[init_data['indices']], descs1[init_data['indices']], scores1[init_data['indices']]
        frame0 = Frame(self.camera_model, np.eye(3), np.zeros(3), t_window=t_window0, path=frame0_path, rgb=color0, feature_based=self.feature_based)  # World origin
        frame1 = Frame(self.camera_model, init_data['R'], init_data['t'], t_window=t_window1, path=frame1_path, rgb=color1, feature_based=self.feature_based)  # Relative pose

        # Filter matches
        indices = prune_triangulated_kps(frame0.R, frame0.t, frame1.R, frame1.t, x1, X01,
                                             self.image_shape, min_angle=self.cfg['MIN_ANGLE'], cell_size=self.cfg['CELL_SIZE'])
        x0, x1, X01 = x0[indices], x1[indices], X01[indices]
        if self.feature_based: descs0, scores0, descs1, scores1 = descs0[indices], scores0[indices], descs1[indices], scores1[indices]

        # Register observations and corresponding triangulations
        X01_obj = [Point(xyz) for xyz in X01]
        frame0.add_observations(x0, X01_obj, descs=descs0, scores=scores0)
        frame1.add_observations(x1, X01_obj, descs=descs1, scores=scores1)

        # Update global state
        self.keyframes_list.append(frame0)
        self.frames_list.append((frame0.R, frame0.t, frame0.id, frame0.path))
        self.keyframes_id_map[frame0.id] = frame0
        self._add_keyframe(frame1, X01_obj, unused_feats)
        logging.info(f"New 3D points: {len(X01_obj)}")

        return frame0, frame1, x0, x1 # return just for debugging and visualization

    def process_frame(self, frame2_path, t_window=None, color=(0, 255, 0)):
        # Get correspondences for tracking (PnP) and mapping (triangulation)
        frame1 = self.last_keyframe
        data = self._extend_matches(frame2_path)
        if data is None: self.n_skipped+=1; return None

        x, X_idxs, x1, x2 = data['kps'], data['pts_idxs'], data['kps1'], data['kps2']
        if self.feature_based: descs, scores, descs1, scores1, descs2, scores2, unused_feats = data['descs'], data['scores'], data['descs1'], data['scores1'], data['descs2'], data['scores2'], data['unused_feats']
        else: descs, scores, descs1, scores1, descs2, scores2, unused_feats = (None,)*7
        if len(data['kps']) < (4 if self.use_cv2 else 6): self.n_skipped+=1; return {"baseline": None,
                                                            "tracking": {"kps1": np.asarray(frame1.keypoints)[X_idxs], "kps2": x},
                                                            "mapping": {"kps1": x1, "kps2": x2}}
        # Track Frame
        logging.info(f"Tracking Frame {self.n_skipped + 1} since last KeyFrame (id={frame1.id})")
        X_obj = np.array([self.points_id_map[pt_id] for pt_id in np.asarray(frame1.points)[X_idxs]])
        tracking_data = track_frame(frame1, x, X_obj, ransac_thresh=self.cfg['PNP_RANSAC_TH'], use_cv2=self.use_cv2)#, pnp_init_error=self.cfg['PNP_INIT_ERR'])
        baseline = np.linalg.norm(tracking_data['t'] - frame1.t)
        if baseline >= self.cfg['MIN_BASELINE']:
            x, X_obj,X_idxs  = x[tracking_data['indices']], X_obj[tracking_data['indices']], X_idxs[tracking_data['indices']]
            if self.feature_based: descs, scores = descs[tracking_data['indices']], scores[tracking_data['indices']]

            # Triangulate Points and filter matches
            mapping_data = map_observations(self.K, frame1.R, frame1.t, tracking_data['R'], tracking_data['t'],
                                            x1, x2, ransac_thresh=self.cfg['EPI_RANSAC_TH'], use_cv2=self.use_cv2)
            x1, x2, X12 = x1[mapping_data['indices']], x2[mapping_data['indices']], mapping_data['points']
            if self.feature_based: descs1, scores1, descs2, scores2 = descs1[mapping_data['indices']], scores1[mapping_data['indices']], descs2[mapping_data['indices']], scores2[mapping_data['indices']]
            indices = prune_triangulated_kps(frame1.R, frame1.t, tracking_data['R'], tracking_data['t'], x2, X12,
                                             self.image_shape, x=x, min_angle=self.cfg['MIN_ANGLE'],
                                             cell_size=self.cfg['CELL_SIZE'], occ_cell_size=self.cfg['OCCUPANCY_CELL_SIZE'])
            x1, x2, X12 = x1[indices], x2[indices], X12[indices]
            if self.feature_based: descs1, scores1, descs2, scores2 = descs1[indices], scores1[indices], descs2[indices], scores2[indices]

            # Decide whether to create new keyframe
            n_new, n_old_visible, n_old_invisible = len(X12), len(X_obj), len(frame1.keypoints)
            overlap = n_old_visible / (n_old_visible + n_new)
            visibility = n_old_visible / (n_old_visible + n_old_invisible)
            if overlap < self.cfg['MIN_OVERLAP'] or visibility < self.cfg['MIN_VISIBILITY']:
                frame2 = Frame(self.camera_model, tracking_data['R'], tracking_data['t'], t_window=t_window, path=frame2_path, rgb=color, feature_based=self.feature_based)
                logging.info(f"New Keyframe (id={frame2.id}). Baseline: {baseline:.2f}, Visible points: {n_old_visible}, Overlap: {overlap:.2f}, Visibility: {visibility:.2f}")

                # Add observations and corresponding triangulations
                frame2.add_observations(x, X_obj, descs=descs, scores=scores)
                X12_obj = [Point(xyz, rgb=color) for xyz in X12]
                frame1.add_observations(x1, X12_obj, descs=descs1, scores=scores1)
                frame2.add_observations(x2, X12_obj, descs=descs2, scores=scores2)

                # Update global state
                self._add_keyframe(frame2, X12_obj, unused_feats)
                logging.info(f"New 3D points: {len(X12_obj)}")

                # Bundle Adjustment on active frames
                if len(self.keyframes_list) % self.cfg['BA_EVERY_N'] and len(self.keyframes_list) > 0 == 0:
                    self._run_bundle_adjustment()

                self.n_skipped = 0
            else:
                logging.info(f"Skipped Frame. High Overlap: {overlap:.2f} and Visibility: {visibility:.2f}")
                self.n_skipped += 1
                self.frames_list.append((tracking_data['R'], tracking_data['t'], -1, frame2_path))
        else:
            logging.info(f"Skipped Frame. Low Baseline: {baseline:.2f}")
            self.n_skipped += 1
            self.frames_list.append((tracking_data['R'], tracking_data['t'], -1, frame2_path))

        return {"baseline": baseline,
                # just for debugging/visualization
                "tracking": { # matches linked to existing 3D points (PnP)
                    "kps1": np.asarray(frame1.keypoints)[X_idxs],
                    "kps2": x},
                "mapping": { # matches for creating new 3D points (triangulation) (raw if low baseline, else filtered)
                    "kps1": x1,
                    "kps2": x2}}

    def _run_bundle_adjustment(self):
        # Run BA on active frames and points
        active_keyframes = self.keyframes_list[-self.cfg['BA_WINDOW']:]
        visible_points = get_visible_points(active_keyframes, self.points_list)
        ba_table = extract_ba_table(self.keyframes_id_map, active_keyframes, visible_points)
        ba_table['rvecs'] = np.asarray([R_to_rodrigues(R) for R in ba_table.pop('Rs')])
        logging.info(f"Bundle adjustment: {len(ba_table['rvecs'])} frames ({ba_table['n_active_frames']} active), {len(ba_table['points'])} points, {ba_table['obs_frames_idxs'].size} observations ({ba_table['n_active_obs']} active)")
        rvecs_opt, tvecs_opt, points_opt = run_bundle_adjustment(**ba_table, K=self.K, verbose=False)

        # Update reconstruction
        Rs_opt = rodrigues_to_R_vectorized(rvecs_opt[:len(active_keyframes)])
        for j, f in enumerate(active_keyframes): f.R, f.t = Rs_opt[j], tvecs_opt[j]
        for j, pt in enumerate(visible_points): pt.xyz = points_opt[j]

        # Local residuals statistics
        exiting_keyframes, remaining_keyframes = active_keyframes[: self.cfg['BA_EVERY_N']], active_keyframes[self.cfg['BA_EVERY_N'] :]
        visible_points = get_visible_points(exiting_keyframes, self.points_list)
        exiting_points = get_nonvisible_points(remaining_keyframes, visible_points)
        median_err, mad, rmse = compute_residual_stats(self.keyframes_id_map, exiting_keyframes, exiting_points, self.K)
        logging.info(f"Local residual stats ({len(exiting_keyframes)} exiting frames, {len(exiting_points)} exiting points): Median Error= {median_err:.2f}, Robust SD= {1.4826 * mad:.2f}, RMSE= {rmse:.2f}")


#   Methods just to improve readability
    def _add_keyframe(self, new_frame, new_points, new_last_feats=None):
        """Adds a new keyframe to the reconstruction and updates state variables."""
        self.keyframes_list.append(new_frame)
        self.keyframes_id_map[new_frame.id] = new_frame
        self.points_list.extend(new_points)
        self.points_id_map.update({pt.id: pt for pt in new_points})
        self.frames_list.append((new_frame.R, new_frame.t, new_frame.id, new_frame.path))
        self.last_keyframe = new_frame
        self.last_feats = new_last_feats

    def _get_unused_feats(self, feats, matches):
        """Returns unmatched features of current frame (the others were matched with last keyframe and used for triangulation and PnP)"""
        if self.feature_based:
            unused_feats = np.ones(len(feats['keypoints']), dtype=bool)
            unused_feats[matches[matches != -1]] = False
            return {'keypoints': feats['keypoints'][unused_feats],
                    'descriptors': feats['descriptors'][unused_feats],
                    'scores': feats['scores'][unused_feats]}
        else: return None

    def _extract_matches(self, frame0_path, frame1_path):
        """Returns matches between 2 frames (for bootstrapping)"""
        if self.feature_based:
            feats0 = self.matcher.extract_features(frame0_path)
            feats1 = self.matcher.extract_features(frame1_path)

            if self.dist_coeffs is not None:
                feats0['keypoints'] = cv2.undistortPoints(feats0['keypoints'].reshape(-1, 1, 2), self.K, self.dist_coeffs, P=self.K).reshape(-1, 2)
                feats1['keypoints'] = cv2.undistortPoints(feats1['keypoints'].reshape(-1, 1, 2), self.K, self.dist_coeffs, P=self.K).reshape(-1, 2)

            matches, matching_scores = self.matcher.match_features(feats0, feats1)
            x0, descs0, scores0, x1, descs1, scores1 = pair_matches(feats0, feats1, matches, matching_scores).values()
            unused_feats = self._get_unused_feats(feats1, matches)
            return {'kps1': x0, 'descs1': descs0, 'scores1': scores0,
                    'kps2': x1, 'descs2': descs1, 'scores2': scores1,
                    'unused_feats': unused_feats}
        else:
            x0, x1 = self.matcher.extract_matches(frame0_path, frame1_path)
            return {'kps1': x0, 'kps2': x1}

    def _extend_matches(self, frame2_path):
        """Returns matches between last keyframe and current frame for PnP and for triangulation"""
        frame1 = self.last_keyframe
        if self.feature_based:
            # extract matches between last keyframe and current frame (inter frames matching)
            # first len(frame1.keypoints) features in feats1 are already triangulated, the rest are to be triangulated
            feats1 = {'keypoints': np.concatenate((frame1.keypoints, self.last_feats['keypoints']), axis=0),
                      'descriptors': np.concatenate((frame1.descriptors, self.last_feats['descriptors']), axis=0),
                      'scores': np.concatenate((frame1.scores, self.last_feats['scores']), axis=0)}
            feats2 = self.matcher.extract_features(frame2_path)

            if feats1['descriptors'] is None or len(feats1['descriptors']) == 0 or feats2['descriptors'] is None or len(feats2['descriptors']) == 0:
                logging.error("Skipping matching. One image has 0 descriptors.")
                return None

            if self.dist_coeffs is not None:
                feats1['keypoints'] = cv2.undistortPoints(feats1['keypoints'].reshape(-1, 1, 2), self.K, self.dist_coeffs, P=self.K).reshape(-1, 2)
                feats2['keypoints'] = cv2.undistortPoints(feats2['keypoints'].reshape(-1, 1, 2), self.K, self.dist_coeffs, P=self.K).reshape(-1, 2)

            matches, matching_scores = self.matcher.match_features(feats1, feats2)
            N_pnp = len(frame1.keypoints)

            # correspondences for tracking
            matches_pnp = matches[:N_pnp]
            mask_matches_pnp = matches_pnp != -1
            x = feats2['keypoints'][matches_pnp[mask_matches_pnp]]
            descs, scores = feats2['descriptors'][matches_pnp[mask_matches_pnp]], feats2['scores'][matches_pnp[mask_matches_pnp]]
            X_idxs = np.where(mask_matches_pnp)[0]

            # correspondences for mapping
            x1, descs1, scores1, x2, descs2, scores2 = pair_matches(self.last_feats, feats2, matches[N_pnp:], matching_scores[N_pnp:]).values()

            # unused features saved for potential triangulation with future keyframe's features
            unused_feats = self._get_unused_feats(feats2, matches)
            
            return {'kps': x, 'pts_idxs': X_idxs, 'descs': descs, 'scores': scores,
                    'kps1': x1, 'descs1': descs1, 'scores1': scores1,
                    'kps2': x2, 'descs2': descs2, 'scores2': scores2,
                    'unused_feats': unused_feats}
        else:
            # extract matches between last keyframe and current frame (inter frames matching)
            kps1, kps2 = self.matcher.extract_matches(frame1.path, frame2_path)

            # associate keypoints in last keyframe (intra frame matching)
            indices_triang = associate_keypoints_mutual(kps1, frame1.keypoints, min_dist=self.cfg['MIN_ASSOCIAT_DIST'])
            indices_triang = np.asarray(indices_triang, dtype=np.int32)
            mask_triang = indices_triang != -1
            
            return {'kps': kps2[mask_triang], 'pts_idxs': indices_triang[mask_triang],
                    'kps1': kps1[~mask_triang], 'kps2': kps2[~mask_triang]}

    """    def _get_config(self):
        return {
            # Matching:
            'FEATURE_BASED': True,
            'USE_CV2': False,
            'CELL_SIZE': 20,
            'OCCUPANCY_CELL_SIZE': 5,
            # non feature based matching:
            'MIN_ASSOCIAT_DIST': 5.0,
            'CHEAT': True,

            # Tracking:
            'PNP_RANSAC_TH': 5.0,
            'PNP_INIT_ERR': 5.0,

            # Triangulation:
            'EPI_RANSAC_TH': 3.0,
            'MIN_BASELINE': 0.5,
            'MIN_ANGLE': 1,

            # keyframe:
            'MIN_OVERLAP': 0.7,
            'MIN_VISIBILITY': 0.6,

            # Bundle adjustment:
            'BA_EVERY_N': 4,
            'BA_WINDOW': 4,
        }

     eth3d delivery area
    {
    # Matching:
    'FEATURE_BASED': False,
    'CELL_SIZE': 30,
    'OCCUPANCY_CELL_SIZE': 20,
    # non feature based matching:
    'MIN_ASSOCIAT_DIST': 5.0,
    'CHEAT': False,

    # Tracking:
    'PNP_RANSAC_TH': 5.0,
    'PNP_INIT_ERR': 5.0,

    # Triangulation:
    'EPI_RANSAC_TH': 3.0,
    'MIN_BASELINE': 0.5,
    'MIN_ANGLE': 1,

    # keyframe:
    'MIN_OVERLAP': 0.7,
    'MIN_VISIBILITY': 0.6,

    # Bundle adjustment:
    'BA_EVERY_N': 4,
    'BA_WINDOW': 4,
}
    """