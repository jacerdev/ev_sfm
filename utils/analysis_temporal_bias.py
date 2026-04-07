import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import binned_statistic


def get_events_in_tube(events, center, t_window, radius):
    """
    Extracts events within a spatiotemporal 'tube' (cylinder).
    Args:
        events: full event stream (N, 4). assumes events are sorted by time
        center (tuple): Spatial center (x, y) of the feature
        t_window (tuple): (t_start, t_end) the temporal depth
        radius: Spatial radius r
    Returns:
        np.ndarray: Subset of events inside the tube.
    """
    t_start, t_end = t_window
    cx, cy = center

    idx_start = np.searchsorted(events[:, 0], t_start)
    idx_end = np.searchsorted(events[:, 0], t_end)
    if idx_start == idx_end: return np.empty((0, 4))

    # temporal filtering
    time_slice = events[idx_start:idx_end]
    # spatial filtering (circular)
    dist_sq = (time_slice[:, 1] - cx) ** 2 + (time_slice[:, 2] - cy) ** 2
    spatial_mask = dist_sq <= (radius ** 2)

    return time_slice[spatial_mask]

def collect_temporal_statistics(points_list, frames_map, events_stream, radius=5):
    """
    Iterates through all 3D points and collects:
    1. Raw relative timestamps for every observation.
    2. Pairwise differences in relative time for consecutive matches in a track.

    Args:
        points_list: List of Point objects from SfM.
        frames_map: Dict mapping frame_id -> Frame object.
        events_stream: The full event stream (N, 4).
        radius: Radius for event tube extraction.

    Returns:
        all_rel_times: List of float (0.0 to 1.0).
        pairwise_diffs: List of float (diff between consecutive frames in a track).
    """
    all_rel_times = []
    pairwise_diffs = []

    print(f"Collecting statistics for {len(points_list)} points...")

    for pt in points_list:
        sorted_obs = sorted(pt.observations, key=lambda x: x[0])

        # Temporary list to store rel_times for this specific track
        track_rel_times = []

        for frame_id, kp_idx in sorted_obs:
            frame = frames_map[frame_id]
            u, v = frame.keypoints[kp_idx]

            # Retrieve pre-calculated window for this frame
            t_start, t_end = frame.t_window
            if t_start is None or t_end is None: continue

            window_duration = t_end - t_start

            # Extract Events
            tube = get_events_in_tube(
                events_stream,
                center=(u, v),
                t_window=(t_start, t_end),
                radius=radius
            )

            if len(tube) == 0:
                continue

            # Compute Relative Time
            # (0.0 = Start of Window/Old, 1.0 = End of Window/New)
            t_med = np.median(tube[:, 0])
            rel_t = (t_med - t_start) / window_duration

            # Clip for safety (floating point errors)
            rel_t = np.clip(rel_t, 0.0, 1.0)

            all_rel_times.append(rel_t)
            track_rel_times.append(rel_t)

        # Compute Pairwise Differences for this track
        # If we have [t0, t1, t2], we compute (t1-t0) and (t2-t1)
        if len(track_rel_times) >= 2:
            for i in range(len(track_rel_times) - 1):
                diff = track_rel_times[i + 1] - track_rel_times[i]
                pairwise_diffs.append(diff)

    return all_rel_times, pairwise_diffs

def plot_temporal_diagnostics(all_rel_times, pairwise_diffs):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # --- Plot 1: Global Distribution of Relative Times ---
    ax0 = axes[0]
    ax0.hist(all_rel_times, bins=50, range=(0, 1), color='teal', alpha=0.7, edgecolor='black')
    # ax0.set_title("Distribution of Features' Relative Timestamps in the MCTS")
    ax0.set_xlabel("Relative Time")
    ax0.set_ylabel("Count of Observations")
    ax0.grid(True, alpha=0.3)
    
    mu = np.mean(all_rel_times)
    sigma = np.std(all_rel_times)
    print(f"Relative Timestamps: Mean: {mu:.3f}, Std: {sigma:.3f}")
    # ax0.text(0.05, 0.9, f"Mean: {mu:.3f}\nStd: {sigma:.3f}", transform=ax0.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    # ax0.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Center')
    # ax0.legend()

    ax1 = axes[1]
    ax1.hist(pairwise_diffs, bins=50, range=(-1, 1), color='orange', alpha=0.7, edgecolor='black')
    
    ax1.set_xlabel("Difference in Relative Time")
    ax1.set_ylabel("Count of Pairs")
    
    mu = np.mean(pairwise_diffs)
    sigma = np.std(pairwise_diffs)
    print(f"Pairwise Differences: Mean: {mu:.3f}, Std: {sigma:.3f}")
    # ax1.text(0.05, 0.9, f"Mean: {mu:.3f}\nStd: {sigma:.3f}", transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.8))

    # ax1.axvline(0.0, color='black', linestyle='-', linewidth=1)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

