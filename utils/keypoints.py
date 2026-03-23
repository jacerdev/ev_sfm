import numpy as np
from scipy.spatial import cKDTree

from utils.geometry import compute_parallax_angle


def get_parallax_mask(R1, t1, R2, t2, X12, x2=None, image_shape=None, min_angle=1.0, cell_size=50):
    """
    Spatially adaptive parallax check.
    Enforces strict geometry where possible, but lowers standards in regions that would otherwise be empty
    """
    angles = compute_parallax_angle(R1, t1, R2, t2, X12)

    if x2 is None or image_shape is None: return angles > min_angle

    flat_indices, _ = _get_flat_indices(x2, image_shape, cell_size)
    final_mask = np.zeros(len(X12), dtype=bool)
    for cell_idx in np.unique(flat_indices): # only iterate over cells that actually have points
        # isolate points in this cell
        cell_mask = (flat_indices == cell_idx)
        cell_angles = angles[cell_mask]

        # try Strict Check
        strict_pass = cell_angles > min_angle

        if np.sum(strict_pass) > 0:
            final_mask[cell_mask] = strict_pass
        else:
            # this region (likely center) has no points meeting the standard
            # we accept weaker geometry to maintain tracking in this area.
            rescue_pass = cell_angles > (min_angle / 2) # TODO: make it a parameter
            final_mask[cell_mask] = rescue_pass

    return final_mask

def get_free_cell_mask(kps_new, image_shape, kps_old=None, cell_size=40):
    """Returns mask: True if point is in an empty cell (not blocked by kps_old)
    (prevents triangulation of points that are too close to already-triangulated points)
    """
    if len(kps_new) == 0: return np.array([], dtype=bool)

    new_indices, n_cells = _get_flat_indices(kps_new, image_shape, cell_size)

    # mark cells occupied by old points
    occupied_grid = np.zeros(n_cells, dtype=bool)
    if kps_old is not None and len(kps_old) > 0:
        old_indices, _ = _get_flat_indices(kps_old, image_shape, cell_size)
        occupied_grid[old_indices] = True

    return ~occupied_grid[new_indices]

def get_spatial_nms_mask(kps, image_shape, cell_size=40):
    """Returns mask: True for the first point found in every grid cell
    (prevents triangulation of points that are too close to each other)"""
    if len(kps) == 0: return np.array([], dtype=bool)

    new_indices, _ = _get_flat_indices(kps, image_shape, cell_size)
    _, unique_idx = np.unique(new_indices, return_index=True)

    mask = np.zeros(len(kps), dtype=bool)
    mask[unique_idx] = True
    return mask

def _get_flat_indices(kps, image_shape, cell_size):
    h, w = image_shape
    grid_w = int(np.ceil(w / cell_size))
    grid_h = int(np.ceil(h / cell_size))

    x = (kps[:, 0] // cell_size).astype(int).clip(0, grid_w - 1)
    y = (kps[:, 1] // cell_size).astype(int).clip(0, grid_h - 1)
    return y * grid_w + x, grid_h * grid_w

def associate_keypoints_mutual(points2d_A, points2d_B, min_dist=5):
    """
    points2d_A (N, 2), points2d_B (M, 2)
    Returns: list of length N with indices of points in points2d_B matching
             points in points2d_A. (-1 if no match)
    """
    if len(points2d_B) == 0:
        return [-1] * len(points2d_A)

    A = np.asarray(points2d_A)   # Nx2
    B = np.asarray(points2d_B)   # Mx2
    tree_B = cKDTree(B)
    tree_A = cKDTree(A)

    d_ab, idx_ab = tree_B.query(A, distance_upper_bound=min_dist) # A -> B
    d_ba, idx_ba = tree_A.query(B, distance_upper_bound=min_dist) # B -> A

    matches = []
    for i, (j, d) in enumerate(zip(idx_ab, d_ab)):
        if j == len(B):
            matches.append(-1)
        elif idx_ba[j] == i:
            matches.append(j)
        else:
            matches.append(-1)

    return matches

def associate_keypoints_ratio(points2d_A, points2d_B, threshold=5, ratio=0.75):
    """
    points2d_A (N, 2), points2d_B (M, 2)
    Returns: list of length N with indices of points in points2d_B matching
             points in points2d_A. (-1 if no match)
    """
    if len(points2d_B) == 0:
        return [-1] * len(points2d_A)

    B = np.asarray(points2d_B)
    tree = cKDTree(B)

    dists, idxs = tree.query(points2d_A, k=2, distance_upper_bound=threshold)

    matches = []
    for d, idx in zip(dists, idxs):
        if idx[0] == len(B) or idx[1] == len(B):
            matches.append(-1)
        elif d[0] < ratio * d[1]:
            matches.append(idx[0])
        else:
            matches.append(-1)

    return matches
