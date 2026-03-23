# https://rpg.ifi.uzh.ch/davis_data.html

import numpy as np
from pathlib import Path
import re
import torch
from tqdm import tqdm

from event_based.SuperEvent.ts_generation.generate_ts import TsGenerator
from event_based.SuperEvent.data_preparation.util.data_io import save_ts_sparse, load_ts_sparse


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

    def __len__(self):
        return len(self.mcts_paths)

    def __getitem__(self, idx):
        return load_ts_sparse(self.mcts_paths[idx])


def extract_mcts(events_file, out_dir, shape, delta_t, interval_s):
    """Converts events to MCTS snapshots. (set interval_s to max(delta_t) to not skip any event)"""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(events_file, np.ndarray): events = events_file
    else: events = np.loadtxt(events_file)
    events_torch = torch.from_numpy(events).to(device).float()

    TsGen = TsGenerator(settings={"shape": shape, "delta_t": delta_t}, device=device)

    # define snapshot timestamps and find event indices at snapshot times
    event_timestamps = events_torch[:, 0].contiguous()
    stamps = torch.arange(event_timestamps[0], event_timestamps[-1], interval_s, device=device) [1:]
    event_indices = torch.searchsorted(event_timestamps, stamps) # can contain duplicates (interval_s with no events)

    # processing
    prev_event_idx, i = 0, 0
    for curr_event_idx in tqdm(event_indices):
        if curr_event_idx > prev_event_idx:
            TsGen.batch_update(events_torch[prev_event_idx:curr_event_idx])
            ts = TsGen.get_ts() # take snapshot [H, W, Channels]

            t_us = int(round(stamps[i].item() * 1e6))
            filename = f"{i:05d}_t={t_us:010d}us.npz"
            save_ts_sparse(str(out_dir / filename), ts.cpu().numpy())

            prev_event_idx, i = curr_event_idx, i+1

    return len(event_indices)

if __name__ == "__main__":
    delta_t = [0.001, 0.003, 0.01, 0.03, 0.1]
    interval_s = delta_t[-1]
    root_dir = Path("/home/jacer/Documents/TU-Berlin/13.WS2526/EventPJ/datasets/outdoors_walking")
    mcts_dir = Path("/home/jacer/Documents/TU-Berlin/13.WS2526/EventPJ/workspace/outputs/outdoors_walking/mcts")
    mcts_dir.mkdir(parents=True, exist_ok=True)
    events_path = root_dir / 'events.txt'
    events = np.loadtxt(events_path)  # [t, x, y, p]

    events[:, 0] -= events[0, 0]
    height, width = int(np.max(events[:, 2])) + 1, int(np.max(events[:, 1])) + 1

    n_mcts = extract_mcts(events, mcts_dir, (height, width), delta_t, interval_s)
    mcts_paths = sorted(mcts_dir.glob("*.npz"))
    print(f"Saved {n_mcts} MCTSs to {mcts_dir}")


    quit()


    from utils.visualization import plot_images, make_border
    from event_based.SuperEvent.util.visualization import ts2image

    mcts_dir = Path("/home/jacer/Documents/TU-Berlin/13.WS2526/EventPJ/workspace/outputs/shapes_translation/mcts")

    mcts_paths = sorted(mcts_dir.glob("*.npz"))

    N = 10
    for i in range(0, len(mcts_paths), N):
        imgs = []
        path1 = mcts_paths[i]
        for path2 in mcts_paths[i:i+N if i+N < len(mcts_paths) else len(mcts_paths)]:
            img = ts2image(load_ts_sparse(path2))
            imgs.append(make_border(img))
        plot_images(imgs, ncols=5, title=f"{path1.stem} - {path2.stem}")
