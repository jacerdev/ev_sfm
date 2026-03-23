import cv2
import numpy as np
import scipy


def condition_2d(points):
    t = np.mean(points[:, :2], axis=0)
    s = np.sqrt(2) / np.mean(np.linalg.norm(points[:, :2] - t, axis=1))
    T = np.array([[s, 0, -s*t[0]], [0, s, -s*t[1]], [0, 0, 1]])
    return T

def condition_3d(points):
    t = np.mean(points[:, :3], axis=0)
    s = np.sqrt(3) / np.mean(np.linalg.norm(points[:, :3] - t, axis=1))
    T = np.array([[s, 0, 0, -s*t[0]],
                  [0, s, 0, -s*t[1]],
                  [0, 0, s, -s*t[2]],
                  [0, 0, 0, 1]])
    return T

def solve_dlt(A):
    """
    solves the homogeneous linear system A h = 0.
    Returns the unit vector h corresponding to the smallest singular value.

    (!) This can hang with multi-threaded numpy/scipy/OpenCV.
        Either force single-threaded
        (set os.environ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"] = "1" before imports
        or if using OpenCV method, set cv2.setNumThreads(0) which affects all OpenCV functions but not numpy/scipy)
        or use A.T @ A method (fastest and robust, but less accurate).
    """
    # method 1: direct SVD of A (DEADLOCKS for tall matrices)
    # _, _, Vt = np.linalg.svd(A) # (!)
    # h = Vt[-1]

    # method 2: QR decomposition (fast but DEADLOCKS for tall matrices)
    R = scipy.linalg.qr(A, mode='r') # (!)
    _, _, Vt = np.linalg.svd(R)
    h = np.squeeze(Vt)[-1]

    # method 3: openCV SVD (faster but DEADLOCKS for tall matrices)
    # _, _, Vt = cv2.SVDecomp(A.astype(np.float32)) # (!)
    # h = Vt[-1]

    # method 4: normal equations A.T @ A (fastest AND robust but inaccurate. squaring A, squares the condition number)
    # ATA = A.T @ A
    # ATA += 1e-12 * np.trace(ATA) * np.eye(ATA.shape[0]) # can reduce conditioning damage
    # _, V = np.linalg.eigh(ATA)
    # h = V[:, 0] # np.linalg.eigh sorts eigenvalues from smallest to largest
                  # np.linalg.svd sorts singular values from largest to smallest

    return h

def compute_rotation_error(R, R_gt):
    """Rotation error: estimate − ground truth (geodesic distance in deg)"""
    R_err = R @ R_gt.T
    rot_angle = np.arccos(np.clip((np.trace(R_err) - 1)/2, -1, 1))
    return np.degrees(rot_angle)

def compute_translation_error(t, t_gt):
    """Translation direction error (in deg) and translation magnitude error"""
    ta = t / np.linalg.norm(t)
    tb = t_gt / np.linalg.norm(t_gt)
    t_angle_err = np.arccos(np.clip(np.dot(ta, tb), -1, 1))
    t_norm_err = np.linalg.norm(t - t_gt)
    return np.degrees(t_angle_err), t_norm_err

def compute_parallax_angle(R1, t1, R2, t2, X):
    C1 = (-R1.T @ t1).reshape(1, 3)
    C2 = (-R2.T @ t2).reshape(1, 3)

    # unit vectors from cameras to 3D points
    v1 = (X - C1) / (np.linalg.norm(X - C1, axis=1, keepdims=True) + 1e-8)
    v2 = (X - C2) / (np.linalg.norm(X - C2, axis=1, keepdims=True) + 1e-8)

    cos_theta = np.clip(np.sum(v1 * v2, axis=1), -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    return np.degrees(angle_rad)

def R_to_rodrigues(R):
    rvec, _ = cv2.Rodrigues(R) # cv2 to robustly handle the pi-singularity and numerical stability during setup
    return rvec.flatten()

def rodrigues_to_R(rvec):
    theta = np.linalg.norm(rvec)
    if theta < 1e-10: return np.eye(3)
    k = rvec / theta
    K = np.array([[0, -k[2], k[1]], # skew-symmetric matrix
                 [k[2], 0, -k[0]],
                 [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K) # R = I + sin(theta) K + (1-cos(theta)) K^2
    return R

def rodrigues_to_R_vectorized(rvecs):
    theta = np.linalg.norm(rvecs, axis=1, keepdims=True)
    k = rvecs / np.where(theta > 0, theta, 1.0)

    N = rvecs.shape[0]
    K = np.zeros((N, 3, 3)) # skew-symmetric matrices
    K[:, 0, 1] = -k[:, 2]
    K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2]
    K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]
    K[:, 2, 1] = k[:, 0]

    Rs = np.eye(3) + np.sin(theta).reshape(N, 1, 1) * K + (1 - np.cos(theta).reshape(N, 1, 1)) * (K @ K)
    return Rs
