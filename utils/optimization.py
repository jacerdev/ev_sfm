import logging

import numpy as np
from scipy._lib._sparse import issparse
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve, lsqr

from sfm.objects import extract_ba_table
from multiview.projection import project_points


def RANSAC_fitting(x1, x2, sample_size, estimator, error_fn,
                   p=.99, e=.25, thresh=1.0, min_iters=100, max_iters=10000,
                   use_iterative_refinement=False):
    """
    Generic RANSAC wrapper.
    estimator: callable(x1_subset, x2_subset) -> Model
    error_fn: callable(Model, x1, x2) -> errors
    Returns: indices of best inliers
    """
    assert sample_size <= x1.shape[0], "Not enough samples for RANSAC to fit a model"
    rng = np.random.default_rng(42)
    best_inliers, best_model = np.array([], dtype=int), None

    if e >= 1.0: e = 0.99
    iter_count = int(np.ceil(np.log(1 - p) / np.log(1 - (1 - e) ** sample_size)))
    iter_count = min(max(iter_count, min_iters), max_iters)
    for _ in range(iter_count):
        sample_indices = rng.choice(x1.shape[0], sample_size, replace=False)
        try:
            model = estimator(x1[sample_indices], x2[sample_indices])
        except Exception: continue

        errors = error_fn(model, x1, x2)
        inliers = np.where(errors < thresh)[0]
        if len(inliers) > len(best_inliers):
            best_inliers, best_model = inliers, model

    if use_iterative_refinement:
        for i in range(5):
            try:
                model = estimator(x1[best_inliers], x2[best_inliers])
            except Exception: break

            errors = error_fn(model, x1, x2)
            refined_inliers = np.where(errors < thresh)[0]
            if len(refined_inliers) < sample_size or np.array_equal(refined_inliers, best_inliers):
                break
            best_inliers, best_model = refined_inliers, model

    return best_inliers, best_model

def LM_optimizer(params0, residual_func, jacobian_func=None,
                 max_iters=20, tol=1e-6, lambda_init=1e-3, sparse_threshold=0.5):
    """
    General Levenberg-Marquardt optimizer for nonlinear least squares. Supports Dense (numpy) and Sparse (scipy.sparse) Jacobians
    ----------
    params0 : 1D array of initial guess for the parameters
    residual_func: callable(params) -> residual vector (1D array)
    jacobian_func: callable(params) -> Jacobian matrix
        - Can be a dense (numpy) or a sparse (scipy.sparse)
        - If sparse, the optimizer will decide whether to use sparse solvers based on sparsity fraction
    sparse_threshold: Minimum fraction of zeros in sparse Jacobian to trigger sparse solvers
    -------
    Returns: Optimized params (1D array)
    """
    if jacobian_func is None: jacobian_func = lambda params: jacobian_numeric_dense(params, residual_func)
    params = params0.copy()
    lambda_ = lambda_init # damping parameter
    step_count = 0

    for it in range(max_iters):
        e = residual_func(params)
        J = jacobian_func(params)

        # Decide whether to use sparse or dense solver
        use_sparse = False
        if issparse(J):
            sparsity = 1.0 - J.nnz / (J.shape[0] * J.shape[1])
            use_sparse = sparsity >= sparse_threshold
        if use_sparse: J = J.tocsr()

        # LM step
        H = J.T @ J
        H += lambda_ * eye(len(params), format='csr') if use_sparse else lambda_ * np.eye(len(params))
        g = J.T @ e

        # Solve for update
        try:
            if use_sparse: delta = spsolve(H, -g)
            else: delta = np.linalg.solve(H, -g)
        except:
            if use_sparse: delta = lsqr(H, -g, atol=1e-12, btol=1e-12)[0]
            else: delta = np.linalg.lstsq(H, -g, rcond=None)[0]

        # Trial update
        params_new = params + delta
        e_new = residual_func(params_new)

        # Adaptive damping
        if np.linalg.norm(e_new) < np.linalg.norm(e):
            params, lambda_, step_count = params_new, lambda_ * 0.1, step_count + 1
        else:
            lambda_ *= 10

        if np.linalg.norm(delta) < tol:
            break

    """if step_count == 0: print("already optimized")
    elif step_count >= max_iters: print("max iterations reached")
    else: print(f"converged after {step_count} steps")"""
    return params

def compute_residual_stats(frames_id_map, frames_list, points_list, K):
    ba_table = extract_ba_table(frames_id_map, frames_list, points_list)
    residuals = project_points(K, ba_table['Rs'][ba_table['obs_frames_idxs']], ba_table['tvecs'][ba_table['obs_frames_idxs']], ba_table['points'][ba_table['obs_points_idxs']]) - ba_table['obs_keypoints']
    dists = np.linalg.norm(residuals.reshape(-1, 2), axis=1)
    median_error = np.median(dists)
    mad = np.median(np.abs(dists - median_error))  # median absolute deviation
    rmse = np.sqrt(np.mean(dists ** 2))
    return median_error, mad, rmse

def jacobian_numeric_dense(params, residual_func, eps=1e-8):
    """
    Computes numerical Jacobian via finite differences.
    Returns: J (N_residuals, N_params)
    """
    f0 = residual_func(params)
    J = np.zeros((len(f0), len(params)))

    for k in range(len(params)):
        dp = np.zeros_like(params)
        dp[k] = eps
        f1 = residual_func(params + dp)
        J[:, k] = (f1 - f0) / eps

    return J
