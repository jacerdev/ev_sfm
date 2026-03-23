import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from multiview.projection import project_points
from utils.geometry import rodrigues_to_R_vectorized


def run_bundle_adjustment(rvecs, tvecs, points, obs_frames_idxs, obs_points_idxs, obs_keypoints,
                          n_active_frames, n_active_obs, K, verbose=False):
    """
    rvecs (N_cams, 3), tvecs (N_cams, 3), points (N_pts, 3)
    obs_frame_idxs (N_obs,): Frame index for each observation
    obs_points_idxs (N_obs,): 3D point index for each observation
    obs_keypoints (N_obs, 2): observations (2d points)

    Returns: rvecs_opt, tvecs_opt, points_opt
    """
    n_points = points.shape[0]

    # Split active and fixed parameters
    active_rvecs, active_tvecs = rvecs[:n_active_frames], tvecs[:n_active_frames]
    fixed_rvecs, fixed_tvecs = rvecs[n_active_frames:], tvecs[n_active_frames:]

    frame_params = np.hstack((active_rvecs, active_tvecs))  # interleave !
    params = np.hstack((frame_params.reshape(-1), points.reshape(-1)))  # [R_1, t_1, ... R_n, t_n, pt_1, ... pt_m]

    A = ba_jacobian_structure(n_active_frames, n_points, n_active_obs, obs_frames_idxs, obs_points_idxs)

    params_opt = least_squares(fun_residuals, params, jac_sparsity=A,
                                     verbose=2 if verbose else 1, x_scale='jac', ftol=1e-4, method='trf', loss='soft_l1', f_scale=2.0, # loss='linear', ftol=xtol=gtol=1e-6
                                     args=(n_active_frames, n_points, obs_frames_idxs, obs_points_idxs, obs_keypoints,
                                           K, fixed_rvecs, fixed_tvecs))
    frame_params = params_opt.x[:n_active_frames * 6].reshape((n_active_frames, 6))
    rvecs_opt, tvecs_opt = frame_params[:, :3], frame_params[:, 3:]
    rvecs_opt = np.vstack((rvecs_opt, fixed_rvecs))
    tvecs_opt = np.vstack((tvecs_opt, fixed_tvecs))

    points_opt = params_opt.x[n_active_frames * 6:].reshape((n_points, 3))

    return rvecs_opt, tvecs_opt, points_opt

def fun_residuals(params, n_active_frames, n_points, obs_frame_idxs, obs_points_idxs, obs_keypoints,
                  K, fixed_rvecs, fixed_tvecs):
    frame_params = params[:n_active_frames * 6].reshape((n_active_frames, 6))

    # Reconstruct full list of parameters (Active + Fixed)
    rvecs = np.vstack((frame_params[:, :3], fixed_rvecs))
    tvecs = np.vstack((frame_params[:, 3:], fixed_tvecs))
    Rs = rodrigues_to_R_vectorized(rvecs)
    points = params[n_active_frames * 6:].reshape((n_points, 3))

    obs_Rs = Rs[obs_frame_idxs]
    obs_tvecs = tvecs[obs_frame_idxs]
    obs_points = points[obs_points_idxs]
    projections = project_points(K, obs_Rs, obs_tvecs, obs_points)

    return (projections - obs_keypoints).ravel()


def ba_jacobian_structure(n_active_frames, n_points, n_active_obs, obs_frame_idxs, obs_points_idxs):
    """
    Rows[0: n_active_obs * 2] -> depend on active frames + points
    Rows[n_active_obs * 2 :] -> depend ONLY on points
    Cols = n_frames*6 + n_points*3. [active Frames | fixed Frames | Points]
    """
    n_observations = len(obs_frame_idxs)
    n_residuals = n_observations * 2
    n_params = n_active_frames * 6 + n_points * 3  # Note: Fixed Frames are NOT in params

    A = lil_matrix((n_residuals, n_params), dtype=int)

    # camera block (Only for the first N_active_obs rows)
    frame_idxs = obs_frame_idxs[:n_active_obs]
    i = np.arange(n_active_obs)
    for s in range(6):
        A[2 * i,     frame_idxs * 6 + s] = 1
        A[2 * i + 1, frame_idxs * 6 + s] = 1

    # point Block (all rows)
    all_i = np.arange(n_observations)
    point_col_offset = n_active_frames * 6
    for s in range(3):
        A[2 * all_i,     point_col_offset + obs_points_idxs * 3 + s] = 1
        A[2 * all_i + 1, point_col_offset + obs_points_idxs * 3 + s] = 1

    return A
