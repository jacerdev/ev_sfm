import cv2
import numpy as np

from multiview.projection import dlt_projection_matrix, decompose_P_normalized, project_points
from utils.geometry import rodrigues_to_R
from utils.optimization import RANSAC_fitting, LM_optimizer, jacobian_numeric_dense


def PnP_linear(K, x, X):
    """
    x, X: (N, 2) , X: (N, 3)
    Returns: R (3,3), t (3,)
    """
    """if x.shape[1] == 2: x_h = np.c_[x, np.ones(x.shape[0])]
    else: x_h = x
    x_norm = (np.linalg.inv(K) @ x_h.T).T

    # solve DLT
    P_norm = dlt_projection_matrix(x_norm, X)
    R, t = decompose_P_normalized(P_norm) # Decompose

    # Ensure points are in front of camera
    X_cam = (R @ X.T).T + t.T
    if np.mean(X_cam[:, 2] > 0) < 0.5: # if majority of z are negative, flip
        R = -R
        t = -t

    return R, t"""

    success, rvec_cv, t_cv = cv2.solvePnP(X, x, K, None,  # No distortion coefficients
                                          flags=cv2.SOLVEPNP_SQPNP)  # SOLVEPNP_EPNP)
    if not success: raise ValueError("cv2.solvePnP failed")
    rvec_init, t_init = rvec_cv.flatten(), t_cv.flatten()
    return rodrigues_to_R(rvec_init), t_init

def RANSAC_PnP(K, x, X, p=.999, e= .35, thresh=3.0, min_iters=100, max_iters=1000, use_iterative_refinement=True, use_cv2=False):
    """
    x: (N, 2), X: (N, 3)
    Returns: inlier indices
    """
    if use_cv2:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(X.astype(np.float32), x.astype(np.float32), K, None,
                                                          flags=cv2.SOLVEPNP_SQPNP,#SOLVEPNP_ITERATIVE,
                                                          reprojectionError=thresh,
                                                          confidence=p,
                                                          iterationsCount=max_iters)
        if success:
            R, _ = cv2.Rodrigues(rvec)
            return inliers.flatten(), (R, tvec.flatten())
        else: return np.array([], dtype=int), None

    sample_size = 6

    """def estimator(x, X):
        return dlt_projection_matrix(x, X)

    def error_fn(P, x, X):
        if X.shape[1] == 3: X = np.c_[X, np.ones(X.shape[0])]
        PX = P @ X.T
        PX = PX[:2, :] / PX[2, :]
        return np.linalg.norm(x[:,:2] - PX.T, axis=1)
    """
    def estimator(x_sample, X_sample):
        return PnP_linear(K, x_sample, X_sample)

    def error_fn(R_t, x_sample, X_sample):
        R, t = R_t
        return np.linalg.norm(x_sample - project_points(K, R, t, X_sample), axis=1)

    inliers, R_t = RANSAC_fitting(x, X, sample_size, estimator, error_fn,
                                    p, e, thresh, min_iters=min_iters, max_iters=max_iters, use_iterative_refinement=use_iterative_refinement)
    return inliers, R_t

def PnP_nonlinear(K, rvec, tvec, x, X, max_iters=20, tol=1e-6, lambda_init=1e-3):
    """
    x: (N, 2), X: (N, 3)
    Returns: rvec_optimized (3,), tvec_optimized (3,)
    """
    assert len(X) > 0, "No points to refine pose"
    params = np.concatenate([rvec, tvec])

    def residual_func(params):
        rvec_optimized, tvec_optimized = params[:3], params[3:]
        return PnP_residuals(K, rvec_optimized, tvec_optimized, x, X)

    def jacobian_func(params):
        rvec_optimized, tvec_optimized = params[:3], params[3:]
        return PnP_jacobian(K, rvec_optimized, tvec_optimized, X)

    params_optimized = LM_optimizer(params, residual_func, jacobian_func,
                                    max_iters=max_iters, tol=tol, lambda_init=lambda_init)

    return params_optimized[:3], params_optimized[3:]

def PnP_residuals(K, rvec, tvec, x, X):
    """
    residual = projection - observation
    x: (N, 2), X: (N, 3)
    Returns: (2*N,) flattened residuals
    """
    #R = rodrigues_to_R(rvec)
    R, _ = cv2.Rodrigues(rvec)
    proj_pixels = project_points(K, R, tvec, X)

    e = proj_pixels - x
    return e.ravel()

def PnP_jacobian(K, rvec, tvec, X):
    """
    Fixed Jacobian implementation handling OpenCV's (3,9) output shape.
    """
    N = X.shape[0]
    fx, fy = K[0, 0], K[1, 1]

    # 1. Get Rotation and Jacobian
    R, dR_drvec = cv2.Rodrigues(rvec)

    # --- CRITICAL FIX ---
    # cv2.Rodrigues returns (3, 9) [rows=params, cols=R_elements]
    # We need (9, 3) [rows=R_elements, cols=params] for our tensor logic
    if dR_drvec.shape == (3, 9):
        dR_drvec = dR_drvec.T

        # Now reshape to (3, 3, 3) -> (row_R, col_R, param_rvec)
    dR_d_rvec_tensor = dR_drvec.reshape(3, 3, 3)

    # 2. Project points
    p = (X @ R.T) + tvec.reshape(1, 3)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    inv_z = 1.0 / z
    inv_z2 = inv_z ** 2

    # 3. Jacobian of Projection
    du_dp = np.stack([fx * inv_z, np.zeros(N), -fx * x * inv_z2], axis=1)
    dv_dp = np.stack([np.zeros(N), fy * inv_z, -fy * y * inv_z2], axis=1)

    # 4. Jacobian wrt Translation
    J_t = np.zeros((2 * N, 3))
    J_t[0::2, :] = du_dp
    J_t[1::2, :] = dv_dp

    # 5. Jacobian wrt Rotation
    # d(R*X)/drvec: Sum over cols of R (dim 1) and cols of X (dim 1)
    # Tensor: (row_R, col_R, param) @ (point, col_X)
    dp_dr = np.einsum('ijk,nj->nik', dR_d_rvec_tensor, X)

    J_ru = np.einsum('ni,nij->nj', du_dp, dp_dr)
    J_rv = np.einsum('ni,nij->nj', dv_dp, dp_dr)

    J_r = np.zeros((2 * N, 3))
    J_r[0::2, :] = J_ru
    J_r[1::2, :] = J_rv

    J = np.hstack([J_r, J_t])
    return J

def PnP_jacobian_old(K, rvec, tvec, X):
    """
    X: (N, 3)
    Returns: (2*N, 6) Jacobian
    """
    R = rodrigues_to_R(rvec)
    N = X.shape[0]

    # project points
    p = (R @ X.T).T + tvec.reshape(1, 3)
    px, py, pz = p[:, 0], p[:, 1], p[:, 2]
    pz = np.where(np.abs(pz) < 1e-9, 1e-9, pz) # Avoid division by zero
    inv_z = 1.0 / pz
    inv_z2 = inv_z ** 2

    fx, fy = K[0, 0], K[1, 1]

    # precompute du_dp and dv_dp
    # d(u)/dp = [fx/z,   0,  -fx*x/z^2]
    # d(v)/dp = [   0, fy/z, -fy*y/z^2]
    du_dp = np.stack([fx * inv_z, np.zeros(N), -fx * px * inv_z2], axis=1)  # (N, 3)
    dv_dp = np.stack([np.zeros(N), fy * inv_z, -fy * py * inv_z2], axis=1)  # (N, 3)

    # jacobian wrt Translation is just du_dp and dv_dp because dp/dt = Identity
    J_t = np.zeros((2 * N, 3))
    J_t[0::2, :] = du_dp
    J_t[1::2, :] = dv_dp

    # jacobian wrt Rotation d(proj)/d(rvec) = d(proj)/d(p) * d(p)/d(rvec)
    # where dp/drvec ~ -R * [X]_skew (rodrigues formula)
    J_r = np.zeros((2 * N, 3))
    for i in range(N):
        Xi = X[i, :]
        dp_dr = -R @ np.array([[0, -Xi[2], Xi[1]],
                               [Xi[2], 0, -Xi[0]],
                               [-Xi[1], Xi[0], 0]])
        J_point = np.vstack([du_dp[i], dv_dp[i]])
        J_r[2 * i: 2 * i + 2, :] = J_point @ dp_dr # Combine: (2,3) = (2,3) @ (3,3)

    J = np.hstack([J_r, J_t])
    return J

if __name__ == "__main__":
    np.random.seed(42)

    # Setup Synthetic Data
    K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=float)
    X = np.random.rand(20, 3) * 5 + np.array([0, 0, 10])  # Points in front of camera
    # Pose
    rvec_true = np.array([0.5, -0.2, 0.1])
    tvec_true = np.array([0.0, 0.0, 0.0])
    # Generate observations 'x'
    R_true = rodrigues_to_R(rvec_true)
    x_obs = project_points(K, R_true, tvec_true, X)

    # Compute Ground Truth (Numeric)
    def residual_wrapper(params): # Wrapper for Numeric Jacobian
        r = params[:3]
        t = params[3:]
        return PnP_residuals(K, r, t, x_obs, X)
    params_0 = np.hstack([rvec_true, tvec_true])
    J_numeric = jacobian_numeric_dense(params_0, residual_wrapper, eps=1e-6)

    # Compute Analytic Implementations
    J_analytical = PnP_jacobian(K, rvec_true, tvec_true, X)
    J_analytical_old = PnP_jacobian_old(K, rvec_true, tvec_true, X)

    # Compare Results
    def evaluate(name, J_analytic, J_num):
        diff = np.abs(J_analytic - J_num)
        max_err = np.max(diff)
        mean_err = np.mean(diff)
        is_close = np.allclose(J_analytic, J_num, atol=1e-4)

        status = "CORRECT" if is_close else "INCORRECT"
        print(f"\n{name}:")
        print(f"  Status: {status}")
        print(f"  Max Diff: {max_err:.8f}")
        print(f"  Mean Diff: {mean_err:.8f}")

    evaluate("Analytical vs numeric)", J_analytical, J_numeric)
    evaluate("Analytical_old vs numeric)", J_analytical_old, J_numeric)
