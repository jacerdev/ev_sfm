import numpy as np
from scipy.sparse import block_diag

from multiview.projection import project_points
from utils.geometry import rodrigues_to_R
from utils.optimization import LM_optimizer, jacobian_numeric_dense


def triangulation_linear(P1, P2, x1, x2, use_cv2=False):
    """
    x1, x2: (N, 2)
    Returns: 3D points (N, 3)
    """
    if use_cv2:
        import cv2
        points4D = cv2.triangulatePoints(P1, P2, x1.T, x2.T)
        X = points4D[:3, :] / (points4D[3, :] + 1e-10)
        return X.T

    N = x1.shape[0]

    A = np.zeros((N, 4, 4))
    A[:, 0, :] = x1[:, 0:1] * P1[2, :] - P1[0, :]
    A[:, 1, :] = x1[:, 1:2] * P1[2, :] - P1[1, :]
    A[:, 2, :] = x2[:, 0:1] * P2[2, :] - P2[0, :]
    A[:, 3, :] = x2[:, 1:2] * P2[2, :] - P2[1, :]

    _, _, Vt = np.linalg.svd(A) # Vectorized SVD (np >= v1.8.0)
    X_hom = Vt[:, 3, :]  # Shape (N, 4)
    w = X_hom[:, 3:4]
    X = X_hom[:, :3] / (w + 1e-10)

    return X

def triangulation_nonlinear(K1, R1, t1, K2, R2, t2, x1, x2, X,
                            max_iters=20, tol=1e-6, lambda_init=1e-3):
    """
    x1, x2: (N, 2), X0: (N,3)
    Returns: (N, 3) 3D optimized points
    """
    params = X.flatten() # (3*N,)

    def residual_func(params):
        X_optimized = params.reshape(-1, 3)
        return triangulation_residuals(K1, R1, t1, K2, R2, t2, x1, x2, X_optimized)

    def jacobian_func(params):
        X_optimized = params.reshape(-1, 3)
        return triangulation_jacobian(K1, R1, t1, K2, R2, t2, X_optimized)

    params_optimized = LM_optimizer(params, residual_func, jacobian_func,
                                    max_iters=max_iters, tol=tol, lambda_init=lambda_init)

    return params_optimized.reshape(-1, 3)

def triangulation_residuals(K1, R1, t1, K2, R2, t2, x1, x2, X):
    """
    residual = projection - observation
    x1, x2: (N, 2), X: (N,3)
    Returns: (4*N,) flattened interleaved residuals [u1, v1, u2, v2, ...]
    """
    proj1 = project_points(K1, R1, t1, X)
    proj2 = project_points(K2, R2, t2, X)

    e1 = proj1 - x1
    e2 = proj2 - x2

    return np.hstack([e1, e2]).ravel()

def triangulation_jacobian(K1, R1, t1, K2, R2, t2, X):
    """
    X: (N, 3)
    Returns: (4*N, 3*N) Block diagonal matrix
    """
    def get_cam_derivs(K, R, t):
        # Derivatives w.r.t 3D Point X
        # d(u)/dX = (fx/z) * (R0 - (x/z)*R2)
        # d(v)/dX = (fy/z) * (R1 - (y/z)*R2)

        # project points
        X_cam = X @ R.T + t
        x, y, z = X_cam[:, 0], X_cam[:, 1], X_cam[:, 2]
        z = np.where(np.abs(z) < 1e-6, 1e-6, z)
        inv_z = 1.0 / z

        fx, fy = K[0, 0], K[1, 1]
        fx_z = (fx * inv_z)[:, None]
        fy_z = (fy * inv_z)[:, None]
        x_z = (x * inv_z)[:, None]
        y_z = (y * inv_z)[:, None]
        r0, r1, r2 = R[0], R[1], R[2]

        du_dX = fx_z * (r0 - x_z * r2)
        dv_dX = fy_z * (r1 - y_z * r2)

        return du_dX, dv_dX

    # Compute for both views
    du1, dv1 = get_cam_derivs(K1, R1, t1)  # (N, 3)
    du2, dv2 = get_cam_derivs(K2, R2, t2)
    blocks = np.stack([du1, dv1, du2, dv2], axis=1) # (N, 4, 3)

    # (4N, 3N) block diagonal matrix
    return block_diag(blocks, format="csr")


if __name__ == "__main__":
    np.random.seed(42)

    # Setup Data: 2 Cameras + Points
    N = 5
    K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=float)
    R1, t1 = np.eye(3), np.zeros(3)
    R2 = rodrigues_to_R(np.array([0.1, 0.1, 0.0]))
    t2 = np.array([1.0, 0.0, 0.0])
    # Points in front of both cameras
    X_true = np.random.rand(N, 3) + np.array([0, 0, 5])
    # Generate observations
    x1_obs = project_points(K, R1, t1, X_true)
    x2_obs = project_points(K, R2, t2, X_true)

    #  Compute Ground Truth (Numeric)
    def residual_wrapper(params_flat): #  Wrapper for Numeric Jacobian
        X_reshaped = params_flat.reshape(5, 3)
        return triangulation_residuals(K, R1, t1, K, R2, t2, x1_obs, x2_obs, X_reshaped)
    params_0 = X_true.ravel()
    J_numeric = jacobian_numeric_dense(params_0, residual_wrapper, eps=1e-6)

    # Compute Analytic Jacobian
    J_sparse = triangulation_jacobian(K, R1, t1, K, R2, t2, X_true)
    J_analytic = J_sparse.toarray()  # Convert to dense for comparison

    # Compare
    diff = np.abs(J_analytic - J_numeric)
    max_diff = np.max(diff)
    print(f"Jacobian Shape: {J_analytic.shape}")  # should be (4N, 3N)
    print(f"Max Difference: {max_diff:.8f}")

    if max_diff < 1e-4: print("SUCCESS: Triangulation Jacobian is correct.")
    else: print("FAILURE: Jacobians do not match.")

    # Check Sparsity Structure (the matrix should be block diagonal with blocks of 4 rows and 3 cols)
    block_mask = np.kron(np.eye(N, dtype=bool), np.ones((4, 3), dtype=bool))
    off_diagonal_values = J_analytic[~block_mask]
    total_off_diagonal_error = np.sum(np.abs(off_diagonal_values))
    if total_off_diagonal_error < 1e-9: print("Sparsity structure is correct (Only diagonal 4x3 blocks contain data).")
    else: print("FAILURE: Non-zero elements found outside diagonal blocks.")