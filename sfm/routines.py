import logging
import numpy as np

from multiview.projection import project_points
from utils.keypoints import get_parallax_mask, get_free_cell_mask, get_spatial_nms_mask
from multiview.epipolar import E_from_F, RANSAC_epipolar, SfM_from_E
from multiview.pnp import RANSAC_PnP, PnP_nonlinear
from multiview.triangulation import triangulation_nonlinear, triangulation_linear
from utils.geometry import R_to_rodrigues, rodrigues_to_R


def initialize_scene(K, x0, x1, ransac_thresh=3.0, use_cv2=False):
    """
    initializes the scene by estimating pose and 3d points between two frames with enough baseline
    Returns: dict with  'R', 't': pose of frame1 relative to frame0 which is world coordinate system
                        'indices': indices into x0 and x1 of accepted matches for triangulation
                        'points': triangulated points
    """
    logging.debug(f"2D-2D matches:  {len(x1)}")
    assert len(x1) >= 8, "Not enough matches for epipolar geometry estimation"

    # Filter matches with fundamental matrix RANSAC
    inliers, F = RANSAC_epipolar(x0, x1, thresh=ransac_thresh, min_iters=200, use_iterative_refinement=True, use_cv2=use_cv2)
    logging.debug(f"Epipolar RANSAC: {len(x1)} -> {len(inliers)}")
    x0, x1 = x0[inliers], x1[inliers]

    # Estimate epipolar geometry
    assert F is not None, "Epipolar RANSAC failed to estimate Fundamental Matrix."
    E = E_from_F(F, K)
    x0_h = np.c_[x0, np.ones(len(x0))]
    x1_h = np.c_[x1, np.ones(len(x1))]
    K_inv = np.linalg.inv(K)
    x0_norm_h = (K_inv @ x0_h.T).T
    x1_norm_h = (K_inv @ x1_h.T).T
    x0_norm = x0_norm_h[:, :2] / x0_norm_h[:, 2:3]
    x1_norm = x1_norm_h[:, :2] / x1_norm_h[:, 2:3]

    #  Estimate initial pose and triangulate initial 3d points
    R_rel, t_rel, X_init = SfM_from_E(E, x0_norm, x1_norm, use_cv2=use_cv2)

    # Refine 3d points with nonlinear triangulation
    X_opt = triangulation_nonlinear(K, np.eye(3), np.zeros(3),  # frame 0
                                    K, R_rel, t_rel,  # frame 1
                                    x0, x1, X_init)  # observations and initial 3d points

    # Remove points not in front of camera
    mask_cheirality = X_opt[:,2] > 0
    logging.debug(f"Points in front: {len(x1)} -> {mask_cheirality.sum()}")
    logging.debug(f"Initial Pose:    {(-R_rel.T @ t_rel[:, None]).flatten()}")

    return {"R": R_rel, "t": t_rel,
            'indices': inliers[mask_cheirality],
            "points": X_opt[mask_cheirality]}

def track_frame(frame1, x, X_obj, ransac_thresh=5.0, use_cv2=False):
    """
    Estimates the pose of frame2 given previous frame1
    Returns: dict with  'R', 't': pose of frame2 (relative to world)
                        'indices': indices into x, X_obj of accepted for PnP
    """
    logging.debug(f"2D-3D matches: {len(x)}")

    K = frame1.model.K
    X = np.array([pt.xyz for pt in X_obj])

    # init with previous Frame pose if linear PnP in PnP RANSAC fails (warm start)
    R_init, t_init = frame1.R, frame1.t
    err_prev = np.mean(np.abs(project_points(K, R_init, t_init, X) - x)) # Mean Absolute Error

    # Filter 2D-3D correspondences with PnP RANSAC and get corresponding fitting pose
    try:
        inliers, R__t = RANSAC_PnP(K, x, X, p=.9999, thresh=ransac_thresh, min_iters=200, use_cv2=use_cv2)
        logging.debug(f"PnP RANSAC:    {len(x)} -> {len(inliers)}")
        assert R__t is not None and len(inliers) > 0
        x, X, X_obj = x[inliers], X[inliers], X_obj[inliers]

        err_lin = np.mean(np.abs(project_points(K, R__t[0], R__t[1], X) - x))
        if err_lin < err_prev: R_init, t_init = R__t
        else: logging.error(f"Higher residual error with linear PnP initialization ({err_lin:.2f} > {err_prev:.2f}). Using prev pose.")
    except AssertionError:
        inliers = np.arange(len(x))
        logging.error("PnP RANSAC failed. Initializing with previous frame's pose.")

    # Refine pose with nonlinear PnP
    rvec_init = R_to_rodrigues(R_init)
    rvec_opt, t_opt = PnP_nonlinear(K, rvec_init, t_init, x, X, max_iters=20)
    R_opt = rodrigues_to_R(rvec_opt)

    return {"R": R_opt, "t": t_opt, "indices": inliers}

def map_observations(K, R1, t1, R2, t2, x1, x2, ransac_thresh=3.0, use_cv2=False):
    """
    triangulates matches
    Returns: dict with  'indices': indices into x1, x2 of accepted matches for triangulation
                        'pts': triangulated points
    """
    logging.debug(f"2D-2D matches:   {len(x2)}")
    if len(x2) == 0:
        logging.error("No matches to triangulate")
        return {"indices": np.array([]), "points": np.array([])}

    # Filter matches with epipolar matrix RANSAC (essential if use_cv2, else fundamental)
    try:
        inliers, _ = RANSAC_epipolar(x1, x2, thresh=ransac_thresh, min_iters=100, use_iterative_refinement=False, use_cv2=use_cv2, K=K)
        logging.debug(f"Epipolar RANSAC: {len(x2)} -> {len(inliers)}")
        x1, x2 = x1[inliers], x2[inliers]
    except AssertionError:
        inliers = np.arange(len(x2))
        logging.error("Epipolar RANSAC failed. skipped filtering.")

    # Triangulate 3D points
    P1 = K @ np.hstack((R1, t1.reshape(3, 1)))
    P2 = K @ np.hstack((R2, t2.reshape(3, 1)))
    X_init = triangulation_linear(P1, P2, x1, x2, use_cv2=use_cv2)

    # Refine 3d points with nonlinear triangulation
    X_opt = triangulation_nonlinear(K, R1, t1, K, R2, t2, x1, x2, X_init)

    # Remove points not in front of both cameras
    x_cam1 = X_opt @ R1.T + t1
    x_cam2 = X_opt @ R2.T + t2
    mask_cheirality = np.logical_and(x_cam1[:,2] > 0, x_cam2[:,2] > 0)
    logging.debug(f"Points in front: {len(x2)} -> {mask_cheirality.sum()}")

    return {"indices": inliers[mask_cheirality],
            "points": X_opt[mask_cheirality]}

def prune_triangulated_kps(R1, t1, R2, t2, x2, X12, image_shape, x=None, min_angle=1, cell_size=10, occ_cell_size=None):
    """returns indices into x1, x2, X12 of accepted triangulations"""
    if len(x2) == 0: return np.arange(len(x2))

    indices = np.arange(len(X12))
    mask_parallax = get_parallax_mask(R1, t1, R2, t2, X12, min_angle=min_angle, x2=x2, image_shape=image_shape, cell_size=cell_size)### TODO
    x2 = x2[mask_parallax]
    logging.debug(f"Parallax check:    {len(mask_parallax)} -> {mask_parallax.sum()}")

    occupancy_cell_size = occ_cell_size if occ_cell_size is not None else cell_size
    mask_free = get_free_cell_mask(x2, image_shape, x, cell_size=occupancy_cell_size)
    x2 = x2[mask_free]
    logging.debug(f"Occupancy check:   {len(mask_free)} -> {mask_free.sum()}")

    mask_nms_local = get_spatial_nms_mask(x2, image_shape, cell_size)
    logging.debug(f"Spatial NMS check: {len(mask_nms_local)} -> {mask_nms_local.sum()}")

    return indices[mask_parallax][mask_free][mask_nms_local]
