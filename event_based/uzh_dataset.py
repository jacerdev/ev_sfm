# https://rpg.ifi.uzh.ch/davis_data.html

import numpy as np
from pathlib import Path
import re

from event_based.feature_matcher import extract_mcts


class EventDataset:
    def __init__(self, root_dir, mcts_dir, delta_t=None, interval_s=None):
        self.root = Path(root_dir)
        self.mcts_dir = Path(mcts_dir)
        if delta_t is None: self.delta_t = [0.001, 0.003, 0.01, 0.03, 0.1]#[0.01, 0.02, 0.03, 0.04, 0.05]
        self.interval_s = interval_s if interval_s is not None else self.delta_t[-1]

        # load calibration
        self.intrinsics = {"model": "?", "params": np.loadtxt(self.root / 'calib.txt')}

        # load groundtruth
        gt_path = self.root / 'groundtruth.txt'
        self.poses = np.loadtxt(gt_path) if gt_path.exists() else np.empty((0, 8)) # [t, tx, ty, tz, qx, qy, qz, qw]

        # load images
        img_path = self.root / 'images.txt'
        self.image_paths, self.image_timestamps = [], []  # List of (timestamp, Path)
        if img_path.exists():
            with img_path.open('r') as f:
                for line in f:
                    parts = line.strip().split()
                    self.image_paths.append(self.root / Path(parts[1]))
                    self.image_timestamps.append(float(parts[0]))

        # Load Events
        events_path = self.root / 'events.txt'
        events = np.loadtxt(events_path) # [t, x, y, p]
        self.events_stream = events
        self.events_stream[:,0] -= events[0,0]
        self.intrinsics["height"], self.intrinsics["width"] = 180, 240
        self.mcts_dir.mkdir(parents=True, exist_ok=True)
        self.mcts_paths = sorted(self.mcts_dir.glob("*.npz"))
        if not any(self.mcts_paths):
            n_mcts = extract_mcts(self.events_stream, self.mcts_dir, (self.intrinsics["height"], self.intrinsics["width"]),
                                    self.delta_t, self.interval_s)
            self.mcts_paths = sorted(self.mcts_dir.glob("*.npz"))
            print(f"Saved {n_mcts} MCTSs to {self.mcts_dir}")
        else:
            print(f"Found {len(self.mcts_paths)} MCTSs. (delete them to regenerate with given delta_t and interval_s)")

        # Extract timestamps
        self.mcts_t_windows = []
        t_pattern = re.compile(r"t=(\d+)us")
        for path in self.mcts_paths:
            match = t_pattern.search(path.name)
            if match:
                t_end = int(match.group(1))
                self.mcts_t_windows.append(((t_end - self.delta_t[-1]*1e6)/1e6, t_end/1e6))
            else:
                print(f"Warning: Could not parse timestamp from {path.name}")
                self.mcts_t_windows.append((None, None))



