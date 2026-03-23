import numpy as np
import scipy

from utils.geometry import condition_2d, condition_3d, solve_dlt


def project_points(K, R, t, X, dist_coeffs=None):
    """
    X: (N, 3)
    K: (3, 3)
    R: (3, 3), t: (3,) or R: (N, 3, 3), t: (N, 3)
    Returns: (N, 2) pixel coordinates
    """
    if R.ndim == 3:
        X = X[:, None, :]
        t = t[:, None, :]

    R_T = R.swapaxes(-2, -1)
    x = X @ R_T + t
    x = x.reshape(-1, 3) # squeeze in the vectorized case
    x_norm = x[:, :2] / (x[:, 2:3] + 1e-9)
    x, y = x_norm[:, 0], x_norm[:, 1]

    if dist_coeffs is not None:
        k1, k2, p1, p2, k3 = dist_coeffs
        xy = x * y
        r_sq = x ** 2 + y ** 2
        radial_scaling = 1 + k1*r_sq + k2*r_sq**2 + k3*r_sq**3
        delta_x = 2*p1*xy + p2*(r_sq + 2*x** 2)
        delta_y = p1 * (r_sq + 2*y**2) + 2*p2*xy
        x = x * radial_scaling + delta_x
        y = y * radial_scaling + delta_y

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = x * fx + cx
    v = y * fy + cy

    return np.stack([u, v], axis=1)

def dlt_projection_matrix(x, X):
    """
    Compute projection matrix P (3x4) using robust DLT.
    x: (N, 2), X: (N, 3)
    """
    assert x.shape[0] == X.shape[0] >= 6
    if x.shape[1] == 2: x = np.c_[x, np.ones(x.shape[0])]
    if X.shape[1] == 3: X = np.c_[X, np.ones(X.shape[0])]

    # condition
    T2d = condition_2d(x)
    x_c = (T2d @ x.T).T
    T3d = condition_3d(X)
    X_c = (T3d @ X.T).T

    # Build design matrix A (2N x 12)
    N = x.shape[0]
    A = np.zeros((2 * N, 12))
    u, v = x_c[:, 0], x_c[:, 1]
    A[0::2, 0:4] = X_c
    A[0::2, 8:12] = -u[:, None] * X_c
    A[1::2, 4:8] = X_c
    A[1::2, 8:12] = -v[:, None] * X_c

    # solve linear system
    h = solve_dlt(A)
    P_cond = h.reshape(3, 4)

    # decondition
    P = np.linalg.inv(T2d) @ P_cond @ T3d
    return P / P[-1, -1]

def decompose_P(P):
    """
    Returns: K (3,3), R (3,3), t (3,)
    """
    M = P[:, :3] # P = [M | u]
    K, R = scipy.linalg.rq(M) # P = [K.R | u]

    # Ensure K has positive diagonal (and compensate with R to preserve M)
    D = np.diag(np.sign(np.diag(K)))
    K = K @ D
    R = D @ R

    t = np.linalg.inv(K) @ P[:, 3] # P = K.[R | t]

    # normalize
    s = K[2,2]
    K /= s
    #scale_matrix = (K @ np.c_[R, t]) / P
    #if not np.allclose(scale_matrix, np.mean(scale_matrix), rtol=1e-6): print("K[R|t] doesnt recover P")
    return K, R, t

def decompose_P_normalized(P):
    """
    Decomposes normalized Projection Matrix P ~ [R|t]
    Returns: R (3,3), t (3,)
    """
    M = P[:, :3]  # ~ lambda * R
    p4 = P[:, 3]  # ~ lambda * t

    # SVD of M to find Rotation and Scale
    U, S, Vt = np.linalg.svd(M)

    # Recover Rotation
    R = U @ Vt
    if np.linalg.det(R) < 0: # Ensure determinant is +1 (proper rotation)
        R = -R
        S = -S  # Flip scale sign to match

    scale = np.mean(S) # ideally S = [s, s, s]
    if abs(scale) < 1e-9:
        raise ValueError("P is rank deficient / scale is zero")

    t = p4 / scale # Recover Translation

    return R, t
