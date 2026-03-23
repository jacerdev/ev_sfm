import cv2
import kornia.feature as KF
import numpy as np
import torch
import torch.nn.functional as F

from utils.preprocessing import load_image, ensure_batch
from utils.matching import match_features_lightglue, match_features_flann, match_features_BF


class FeatureMatcher:
    def __init__(self, image_shape, matcher="flann", robust_matching=False):
        pad_h, pad_w = (16 - image_shape[0] % 16) % 16, (16 - image_shape[1] % 16) % 16  # Pad to multiple of 16
        self.image_shape = np.asarray((image_shape[1] + pad_w, image_shape[0] + pad_h)) # for normalization inside lightglue
        self.matcher_type = matcher
        self.robust = robust_matching

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.extractor = KF.DISK.from_pretrained('depth').to(device).eval()

        if self.matcher_type == "lightglue":
            self.matcher = KF.LightGlue(features='disk').to(device).eval()
        elif self.matcher_type == "flann":
            index_params = dict(algorithm=1, trees=8)  # KD-tree
            search_params = dict(checks=64)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        elif self.matcher_type == "bf":
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        else:
            raise ValueError(f"Unknown feature matcher {self.matcher_type}")

    def extract_features(self, img_path):
        img = load_image(img_path)
        features = extract_features_disk(img, self.extractor)[0]
        return features

    def match_features(self, feats1, feats2):
        if self.matcher_type == "lightglue":
            if 'image_size' not in feats1: feats1['image_size'] = self.image_shape # for normalization inside lightglue
            if 'image_size' not in feats2: feats2['image_size'] = self.image_shape
            matches, matching_scores = match_features_lightglue(feats1, feats2, self.matcher)
            return matches[0], matching_scores[0]
        elif self.matcher_type == "flann":
            return match_features_flann(feats1, feats2, self.matcher, robust=self.robust)
        else: # BF
            return match_features_BF(feats1, feats2, self.matcher, robust=self.robust)


def extract_features_disk(images, disk_model):
    """
    images (np.ndarray): image or batch of images
    superpoint_model: Loaded Kornia DISK
    """
    images = ensure_batch(images)  # (B, H, W, C)

    device = next(disk_model.parameters()).device
    input_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float().to(device) / 255.0
    pad_h, pad_w = (16 - images.shape[1] % 16) % 16, (16 - images.shape[2] % 16) % 16 # Pad to multiple of 16
    if pad_h > 0 or pad_w > 0: input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h))
    with torch.inference_mode(): out = disk_model(input_tensor)

    results = []
    for features in out:
        kpts = features.keypoints.cpu().numpy()
        descs = features.descriptors.cpu().numpy()
        scores = features.detection_scores.cpu().numpy()

        sorted_indices = np.argsort(-scores) # sort by score so first occurrences are the best

        results.append({
            'keypoints': kpts[sorted_indices],
            'descriptors': descs[sorted_indices],
            'scores': scores[sorted_indices],})
            #'image_size': np.asarray((input_tensor.shape[3], input_tensor.shape[2]))})
    return results



