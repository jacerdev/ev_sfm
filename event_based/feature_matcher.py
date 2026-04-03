import cv2
from functools import partial
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
import yaml

from utils.matching import match_features_BF, match_features_windowed
from utils.utils import ensure_batch

def submodule_path_setup():
    from pathlib import Path
    import sys
    feature_matcher_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd() / "event_based"
    repo_path = str(feature_matcher_dir / "SuperEvent")
    if repo_path not in sys.path: sys.path.append(repo_path)

submodule_path_setup()
from event_based.SuperEvent.data_preparation.util.data_io import load_ts_sparse, save_ts_sparse
from event_based.SuperEvent.models.super_event import SuperEvent, SuperEventFullRes
from event_based.SuperEvent.ts_generation.generate_ts import TsGenerator


class EventFeatureMatcher:
    def __init__(self, root_dir, image_shape, robust_matching=False, windowed_matching=False, window_size=None):
        config_path = root_dir+"/event_based/SuperEvent/config/super_event.yaml"
        backbone_config_path = root_dir+"/event_based/SuperEvent/config/backbones/maxvit.yaml"
        model_path = root_dir+"/event_based/SuperEvent/saved_models/super_event_weights.pth"

        # extractor
        pad_h, pad_w = (40 - image_shape[0] % 40) % 40, (40 - image_shape[1] % 40) % 40 # Pad to multiple of 40
        self.image_shape = np.asarray((image_shape[1] + pad_w, image_shape[0] + pad_h))
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with open(config_path, 'r') as f: config = yaml.safe_load(f)
        with open(backbone_config_path, 'r') as f: backbone_config = yaml.safe_load(f)
        config = config | backbone_config
        config["backbone_config"]["input_channels"] = config["input_channels"]
        self.config = config
        if config.get("pixel_wise_predictions", False): model = SuperEventFullRes(config)
        else: model = SuperEvent(config)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        self.extractor = model.to(device).eval()

        # matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        self.match_features_func = partial(match_features_BF, bf_instance=self.matcher, robust=robust_matching)
        if windowed_matching and window_size is not None:
            self.match_features_func = partial(match_features_windowed, bin_size=window_size, match_features_func=self.match_features_func)

    def extract_features(self, mcts_path):
        mcsts = load_ts_sparse(mcts_path)
        return extract_features_superevent(mcsts, self.extractor)[0]

    def match_features(self, feats1, feats2):
        return self.match_features_func(feats1, feats2)


def extract_features_superevent(MCTSs, superevent_model, detection_threshold=0.01, nms_box_size=3):
    assert nms_box_size % 2 == 1 and nms_box_size >= 3, "nms_box_size must be odd and >= 3"
    MCTSs = ensure_batch(MCTSs)
    B, H, W, _ = MCTSs.shape

    device = next(superevent_model.parameters()).device
    input_ts = torch.from_numpy(MCTSs).permute(0, 3, 1, 2).float().to(device)
    pad_h, pad_w = (40 - H % 40) % 40, (40 - W % 40) % 40 # Pad to multiple of 40
    if pad_h > 0 or pad_w > 0: input_ts = F.pad(input_ts, (0, pad_w, 0, pad_h))
    with torch.inference_mode(): out = superevent_model(input_ts)

    nms_mask = out["prob"] == torch.nn.functional.max_pool2d(out["prob"], kernel_size=nms_box_size, stride=1, padding=int(nms_box_size/2))

    det_mask = (out["prob"] > detection_threshold) & nms_mask
    pts_candidates = det_mask.nonzero()
    row_col_batch = [pts_candidates[pts_candidates[:, 0] == i][:, 1:] for i in range(B)]

    desc_map_batch = out["descriptors"].cpu().detach().numpy().transpose(0, 2, 3, 1)
    prob_map_batch = out["prob"].cpu().detach().numpy()

    results = []
    for row_col, desc_map, prob_map in zip(row_col_batch, desc_map_batch, prob_map_batch):
        row_col = row_col.cpu().detach().numpy()
        kps = np.stack([row_col[:, 1], row_col[:, 0]], axis=1).astype(np.float32) # float (!)
        descs = desc_map[row_col[:, 0], row_col[:, 1]]
        scores = prob_map[row_col[:, 0], row_col[:, 1]]

        sorted_indices = np.argsort(-scores)  # sort by score so first occurrences are the best
        results.append({
            'keypoints': kps[sorted_indices],
            'descriptors': descs[sorted_indices],
            'scores': scores[sorted_indices], })

    return results

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
    root_dir = Path("data/UZH/slider_depth")
    mcts_dir = Path("data/outputs/UZH/slider_depth/mcts")

    delta_t = [0.001, 0.003, 0.01, 0.03, 0.1]
    interval_s = delta_t[-1]
    mcts_dir.mkdir(parents=True, exist_ok=True)
    events_path = root_dir / 'events.txt'
    events = np.loadtxt(events_path)  # [t, x, y, p]

    events[:, 0] -= events[0, 0]
    height, width = int(np.max(events[:, 2])) + 1, int(np.max(events[:, 1])) + 1

    n_mcts = extract_mcts(events, mcts_dir, (height, width), delta_t, interval_s)
    mcts_paths = sorted(mcts_dir.glob("*.npz"))
    print(f"Saved {n_mcts} MCTSs to {mcts_dir}")
    #quit()

    ## Visualize MCTSs ##

    from utils.visualization import plot_images, make_border
    from event_based.SuperEvent.util.visualization import ts2image
    #mcts_dir = Path("data/outputs/UZH/slider_depth/mcts")
    #mcts_paths = sorted(mcts_dir.glob("*.npz"))

    N = 40
    for i in range(0, len(mcts_paths), N):
        imgs = []
        path1 = mcts_paths[i]
        for path2 in mcts_paths[i:i+N if i+N < len(mcts_paths) else len(mcts_paths)]:
            img = ts2image(load_ts_sparse(path2))
            imgs.append(make_border(img))
        plot_images(imgs, ncols=5)#, title=f"{path1.stem} - {path2.stem}")