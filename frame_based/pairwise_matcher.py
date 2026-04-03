import kornia.feature as KF
import logging
import numpy as np
from pathlib import Path
import torch

from utils.utils import load_image, ensure_batch


class PairwiseMatcher:
    def __init__(self, min_conf=None, matches_dir=None, extract_matches_func=None):
        """
        if extract_matches_func is provided; other parameters are ignored
        elif matches_dir is provided; cached matches if present are used (min_conf ignored)
        else; LoFTR is used (matches are cached in matches_dir if provided)
        """
        if extract_matches_func is not None:
            self.extract_matches_func = extract_matches_func
            logging.info("Using custom match extraction function")
        else:
            self.loftr = KF.LoFTR(pretrained='indoor').to("cuda" if torch.cuda.is_available() else "cpu").eval()
            self.min_conf, self.matches_dir = min_conf, matches_dir
            self.extract_matches_func = lambda img1_path, img2_path: fetch_loftr_matches(img1_path, img2_path, loftr_model=self.loftr, min_conf=min_conf,
                                                                                         matches_dir=matches_dir)
            logging.info("Using LoFTR for match extraction")

    def extract_matches(self, img1_path, img2_path):
        with torch.random.fork_rng(devices=['cuda']):
            torch.manual_seed(0)
            return self.extract_matches_func(img1_path, img2_path)

def fetch_loftr_matches(img1_path, img2_path, loftr_model, min_conf=.2, matches_dir=None):
    """Returns LoFTR matches for an image pair. Loads cached matches if available."""
    if matches_dir is not None:
        matches_dir = Path(matches_dir)
        matches_dir.mkdir(parents=True, exist_ok=True)
        stem1, stem2 = sorted([img1_path.stem, img2_path.stem])
        matches_path = matches_dir / (f"{stem1}_{stem2}" + ".npz")
        if matches_path.exists():
            logging.debug(f"cached ({matches_path.name})")
            data = np.load(matches_path)
            if img1_path.stem == stem1: return data["kps1"], data["kps2"]
            else: return data["kps2"], data["kps1"]

    img1, img2 = load_image(img1_path), load_image(img2_path)
    results = extract_matches_loftr(img1, img2, loftr_model, min_conf=min_conf)[0]
    kps1, kps2 = results[0], results[1]

    if matches_dir is not None:
        if img1_path.stem == stem1: np.savez(matches_path, kps1=kps1, kps2=kps2)
        else: np.savez(matches_path, kps1=kps2, kps2=kps1)
    return kps1, kps2

def extract_matches_loftr(imgs1, imgs2, loftr_model, min_conf=.2):
    """
    imgs1, imgs2 (np.ndarray) image or batches of images
    model (torch.nn.Module): Loaded LoFTR model.

    Returns: list: [(kps0, kps1),...]
    """
    imgs1, imgs2 = ensure_batch(imgs1, to_gray=True), ensure_batch(imgs2, to_gray=True)

    device = next(loftr_model.parameters()).device
    imgs1_tensor = torch.from_numpy(imgs1).permute(0, 3, 1, 2).float().to(device) / 255.
    imgs2_tensor = torch.from_numpy(imgs2).permute(0, 3, 1, 2).float().to(device) / 255.
    input_dict = {"image0": imgs1_tensor, "image1": imgs2_tensor}
    with torch.inference_mode(): out = loftr_model(input_dict)

    batch_ids = out['batch_indexes'].cpu().numpy()
    kpts1_flat = out['keypoints0'].cpu().numpy()
    kpts2_flat = out['keypoints1'].cpu().numpy()
    conf_flat = out['confidence'].cpu().numpy()
    del input_dict, out, imgs1_tensor, imgs2_tensor # free GPU memory from intermediate tensors (loftr is too memory hungry)
    # torch.cuda.empty_cache()

    results = []
    for b in range(imgs1.shape[0]):
        # Mask out points belonging to this batch index
        mask_b = batch_ids == b
        kpts1 = kpts1_flat[mask_b]
        kpts2 = kpts2_flat[mask_b]
        conf = conf_flat[mask_b]

        # filter by Confidence
        if min_conf is not None:
            mask_conf = conf > min_conf
            kpts1 = kpts1[mask_conf]
            kpts2 = kpts2[mask_conf]
            conf = conf[mask_conf]
            if len(kpts1) == 0:
                results.append((np.array([]), np.array([])))
                continue

        sort_idx = np.argsort(-conf) # sort by confidence so that first occurrences are the best
        kpts1 = kpts1[sort_idx]
        kpts2 = kpts2[sort_idx]

        results.append((kpts1, kpts2))

    return results

"""
def clean_gpu():
    Force garbage collection and clear CUDA cache.
    import gc
    gc.collect()
    torch.cuda.empty_cache()"""
