import cv2
import numpy as np

from multiview.triangulation import triangulation_linear
from utils.geometry import condition_2d, solve_dlt
from utils.optimization import RANSAC_fitting


def E_from_F(F, K1, K2=None):
    if K2 is None: K2 = K1
    E = K2.T @ F @ K1
    u, s, vt = np.linalg.svd(E)
    E_rank2 = u @ np.diag([1.0, 1.0, 0.0]) @ vt
    return E_rank2

def dlt_epipolar_matrix(x1, x2, normalized=False):
    """
    Computes Fundamental (normalized=False) or Essential (normalized=True) Matrix.
    Uses QR decomposition to handle large N robustly.
    x1, x2: (N, 2)
    """
    if x1.shape[1] == 2: x1 = np.c_[x1, np.ones(x1.shape[0])]
    if x2.shape[1] == 2: x2 = np.c_[x2, np.ones(x2.shape[0])]

    # condition (only if not already normalized)
    if not normalized:
        T1 = condition_2d(x1)
        x1_c = (T1 @ x1.T).T
        T2 = condition_2d(x2)
        x2_c = (T2 @ x2.T).T
    else:
        x1_c, x2_c = x1, x2

    # build design matrix A (N x 9)
    #############A = x2_c[:, 0:1] * x1_c
#    A = np.hstack([A, x2_c[:, 1:2] * x1_c])
 #   A = np.hstack([A, x2_c[:, 2:3] * x1_c])    
    A1 = x2_c[:, 0:1] * x1_c
    A2 = x2_c[:, 1:2] * x1_c
    A3 = x2_c[:, 2:3] * x1_c
    A = np.hstack((A1, A2, A3))

    # Solve linear system
    h = solve_dlt(A)
    M = h.reshape(3, 3)

    # enforce Rank-2 constraint
    U, S, Vt = np.linalg.svd(M)
    if normalized:
        # Essential Matrix E: two equal non-zero singular values (1, 1, 0)
        s_avg = (S[0] + S[1]) / 2.0
        E = U @ np.diag([s_avg, s_avg, 0.0]) @ Vt
        return E
    else:
        # Fundamental Matrix F: (s1, s2, 0)
        S[-1] = 0.0
        F_rank2 = U @ np.diag(S) @ Vt
        F = T2.T @ F_rank2 @ T1 # decondition
        return F / F[-1, -1]

def compute_sampson_distance(M, x1, x2):
    """
    calculates the Sampson distance between two point sets
    x1, x2: (N, 2)
    """
    if x1.shape[1] == 2: x1 = np.c_[x1, np.ones(x1.shape[0])]
    if x2.shape[1] == 2: x2 = np.c_[x2, np.ones(x2.shape[0])]

    Mx1 = (M @ x1.T).T
    MTx2 = (M.T @ x2.T).T

    num = np.sum(x2 * Mx1, axis=1) ** 2
    den = Mx1[:, 0] ** 2 + Mx1[:, 1] ** 2 + MTx2[:, 0] ** 2 + MTx2[:, 1] ** 2
    return num / (den + 1e-15)

def RANSAC_epipolar(x1, x2, p=.999, e=.35, thresh=3.0, min_iters=100, max_iters=1000, use_iterative_refinement=True, use_cv2=False, K=None):
    """
    x1, x2: (N, 2)
    returns inliers' indices and fundamental matrix
    """
    if use_cv2:
        assert len(x2) >= (5 if K is not None else 8), "Not enough points for geometric verification"

        if K is not None: # use 5-point algorithm (Essential Matrix)
            E, mask = cv2.findEssentialMat(x1, x2, K, method=cv2.RANSAC, prob=p, threshold=thresh)
            inliers = np.where(mask.ravel() == 1)[0]
            return inliers, E
        else: # use 8-point algorithm (Fundamental Matrix)
            F, mask = cv2.findFundamentalMat(x1, x2, cv2.FM_RANSAC, thresh, p)
            inliers = np.where(mask.ravel() == 1)[0]
            return inliers, F

    sample_size = 8

    def estimator(x1_sample, x2_sample):
        return dlt_epipolar_matrix(x1_sample, x2_sample)

    def error_fn(M, x1_sample, x2_sample):
        return compute_sampson_distance(M, x1_sample, x2_sample)

    inliers, epipolar_matrix =  RANSAC_fitting(x1, x2, sample_size, estimator, error_fn,
                                               p, e, thresh, min_iters=min_iters, max_iters=max_iters, use_iterative_refinement=use_iterative_refinement)
    return inliers, epipolar_matrix

def SfM_from_E(E, x1_norm, x2_norm, use_cv2=False):
    """
    Decomposes Essential Matrix E into R, t candidates and triangulates points.
    Then uses chirality checks to disambiguate
    x1, x2: (N, 2) Normalized coordinates (!)
    Returns: R (3,3), t (3,), X (N,3)
    """
    if use_cv2:
        _, R, t, mask = cv2.recoverPose(E, x1_norm, x2_norm, np.eye(3))
        t = t.flatten()
        P1 = np.hstack([np.eye(3), np.zeros((3,1))])
        P2 = np.hstack([R, t.reshape(3,1)])
        X_best = triangulation_linear(P1, P2, x1_norm, x2_norm)
        return R, t, X_best

    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0: U[:, -1] *= -1
    if np.linalg.det(Vt) < 0: Vt[-1, :] *= -1    
    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t  = U[:, 2]
    ####################if np.linalg.det(R1) < 0: R1 *= -1
#    if np.linalg.det(R2) < 0: R2 *= -1
    candidates = [(R1,  t), (R1, -t),
                  (R2,  t), (R2, -t)]
    best_idx = None
    best_positive = -1
    X_best = None
    P1 = np.hstack([np.eye(3), np.zeros((3,1))])
    for i, (R, tvec) in enumerate(candidates):
        P2 = np.hstack([R, tvec.reshape(3,1)])
        X = triangulation_linear(P1, P2, x1_norm, x2_norm)
        C = -R.T @ tvec
        positive = cheirality_check(X, R, C)#(X[:1], R, C.reshape(3,1))
        if positive > best_positive:
            best_positive, best_idx, X_best = positive, i, X

    R_best, t_best = candidates[best_idx]
    return R_best, t_best, X_best

def cheirality_check(points3D, R, C, R0=None, C0=None):
    """
    counts points with positive depth in both views
    points3D: (N, 3)"""
    if R0 is None:
        R0 = np.eye(3)
        C0 = np.zeros((3,))
    C = C.reshape(3)

    # check camera 1 (reference)
    view_dir_1 = R0[2, :]
    depths1 = (points3D - C0) @ view_dir_1
    # check cam 2
    view_dir_2 = R[2, :]
    depths2 = (points3D - C) @ view_dir_2

    # BOTH are positive
    n_positive = np.sum((depths1 > 0) & (depths2 > 0))

    return n_positive

def compute_epipolar_lines(F: np.ndarray, x1: np.ndarray, x2: np.ndarray):
    """x1, x2: (N, 2)"""
    if x1.shape[1] == 2: x1 = np.c_[x1, np.ones(x1.shape[0])]
    if x2.shape[1] == 2: x2 = np.c_[x2, np.ones(x2.shape[0])]
    x1, x2 = x1.T, x2.T
    lines1 = F.T @ x2
    lines2 = F @ x1
    return lines1.T, lines2.T

def compute_epipoles(F):
    # right null space of F -> epipole in image 1
    _, _, Vt = np.linalg.svd(F)
    e1 = Vt[-1]
    e1 /= e1[2]
    # right null space of F.T is epipole in image 2
    _, _, Vt = np.linalg.svd(F.T)
    e2 = Vt[-1]
    e2 /= e2[2]
    return e1, e2
