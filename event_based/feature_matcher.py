import cv2
from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from event_based.SuperEvent.data_preparation.util.data_io import load_ts_sparse
from event_based.SuperEvent.models.super_event import SuperEvent, SuperEventFullRes

from utils.matching import match_features_flann, match_features_BF, match_features_windowed
from utils.preprocessing import ensure_batch


class EventFeatureMatcher:
    def __init__(self, root_dir, image_shape, matcher="bf", robust_matching=False, windowed_matching=False, window_size=None):
        config_path = root_dir+"/event_based/SuperEvent/config/super_event.yaml"
        backbone_config_path = root_dir+"/event_based/SuperEvent/config/backbones/maxvit.yaml"
        model_path = root_dir+"/event_based/SuperEvent/saved_models/super_event_weights.pth"

        pad_h, pad_w = (40 - image_shape[0] % 40) % 40, (40 - image_shape[1] % 40) % 40 # Pad to multiple of 40
        self.image_shape = np.asarray((image_shape[1] + pad_w, image_shape[0] + pad_h))
        self.matcher_type = matcher
        self.robust = robust_matching
        self.windowed = windowed_matching
        self.window_size = window_size if window_size is not None else max(self.image_shape) // 1

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

        if self.matcher_type == "bf":
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        elif self.matcher_type == "flann":
            index_params = dict(algorithm=1, trees=8)
            search_params = dict(checks=64)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            raise ValueError(f"Unknown feature matcher {self.matcher_type}")

    def extract_features(self, mcts_path):
        mcsts = load_ts_sparse(mcts_path)
        return extract_features_superevent(mcsts, self.extractor)[0]

    def match_features(self, feats1, feats2):
        if self.matcher_type == "bf":
            match_features_func = partial(match_features_BF, bf_instance=self.matcher, robust=self.robust)
        elif self.matcher_type == "flann":
            match_features_func = partial(match_features_flann, flann_instance=self.matcher, robust=self.robust)
        else:
            raise ValueError(f"Unknown feature matcher {self.matcher_type}")

        if self.windowed:
            return match_features_windowed(feats1, feats2, bin_size=self.window_size, match_features_func=match_features_func)
        else:
            return match_features_func(feats1, feats2)

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

        #results.append({"keypoints": kps, "descriptors": descs, "scores": scores})

    return results