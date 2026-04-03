import numpy as np
import torch


def pair_matches(feats1, feats2, matches, matching_scores):
    mask_matches = matches != -1
    sorted_indices = np.argsort(-matching_scores[mask_matches])  # sort matches so that first occurrences are the best

    x1 = feats1['keypoints'][mask_matches][sorted_indices]
    descs1 = feats1['descriptors'][mask_matches][sorted_indices]
    scores1 = feats1['scores'][mask_matches][sorted_indices]

    valid_matches = matches[mask_matches][sorted_indices]  # matches[mask_matches] gives indices into feats2
    x2 = feats2['keypoints'][valid_matches]
    descs2 = feats2['descriptors'][valid_matches]
    scores2 = feats2['scores'][valid_matches]

    return {'x1': x1, 'descs1': descs1, 'scores1': scores1,
            'x2': x2, 'descs2': descs2, 'scores2': scores2}
    return x1, descs1, scores1, x2, descs2, scores2


def match_features_lightglue(feats0, feats1, lightglue_model):
    """
    feats0, feats1: Lists of dicts that includes 'image_size':(width,height)
    lightglue_model: Loaded Kornia LightGlue
    """
    device = next(lightglue_model.parameters()).device
    def _to_torch(arr): return torch.from_numpy(arr).float().to(device).unsqueeze(0)
    if isinstance(feats0, dict): feats0 = [feats0]
    if isinstance(feats1, dict): feats1 = [feats1]

    matches_all = []
    matching_scores_all = []
    for f0, f1 in zip(feats0, feats1):
        f0['keypoints'] = np.clip(f0['keypoints'], 0, f0['image_size'])
        f1['keypoints'] = np.clip(f1['keypoints'], 0, f1['image_size'])################### because of undistortion

        input_data = {'image0': {'keypoints': _to_torch(f0['keypoints']),
                                 'descriptors': _to_torch(f0['descriptors']),
                                 'scores': _to_torch(f0['scores']),
                                 'image_size': _to_torch(f0['image_size'])},
                      'image1': {'keypoints': _to_torch(f1['keypoints']),
                                 'descriptors': _to_torch(f1['descriptors']),
                                 'scores': _to_torch(f1['scores']),
                                 'image_size': _to_torch(f1['image_size'])}}
        with torch.inference_mode(): out = lightglue_model(input_data)

        matches_all.append(out['matches0'][0].cpu().numpy())
        matching_scores_all.append(out['matching_scores0'][0].cpu().numpy())

    return matches_all, matching_scores_all

def match_features_BF(feats1, feats2, bf_instance, ratio=0.75, robust=False):
    descs1 = feats1["descriptors"]
    descs2 = feats2["descriptors"]

    if robust:
        matches, distances = _robust_match(descs1, descs2, bf_instance, ratio)
    else:
        match_objs = bf_instance.match(descs1, descs2)
        matches = np.full(len(descs1), -1, dtype=int)
        distances = np.full(len(descs1), np.inf, dtype=float)
        for m in match_objs:
            matches[m.queryIdx] = m.trainIdx
            distances[m.queryIdx] = m.distance

    scores = np.exp(-distances) * feats1.get("scores", 1.0)
    return matches, scores

def match_features_flann(feats1, feats2, flann_instance, ratio=0.75, robust=False):
    descs1 = feats1["descriptors"]
    descs2 = feats2["descriptors"]

    if robust:
        matches, distances = _robust_match(descs1, descs2, flann_instance, ratio)
    else:
        knn = flann_instance.knnMatch(descs1, descs2, k=2) # k=2 for ratio test (can afford it because of its speed)
        matches = np.full(len(descs1), -1, dtype=int)
        distances = np.full(len(descs1), np.inf, dtype=float)
        for qi, (m, n) in enumerate(knn):
            if m.distance < ratio * n.distance:
                matches[qi] = m.trainIdx
                distances[qi] = m.distance

    scores = np.exp(-distances) * feats1.get("scores", 1.0)
    return matches, scores


def _robust_match(descs1, descs2, matcher_instance, ratio):
    knn12 = matcher_instance.knnMatch(descs1, descs2, k=2)
    knn21 = matcher_instance.knnMatch(descs2, descs1, k=2)

    fwd = {}
    for i, (m, n) in enumerate(knn12):
        if m.distance < ratio * n.distance: fwd[i] = m
    bwd = {}
    for j, (m, n) in enumerate(knn21):
        if m.distance < ratio * n.distance: bwd[j] = m

    matches = np.full(len(descs1), -1, dtype=int)
    distances = np.full(len(descs1), np.inf, dtype=float)
    for i, m in fwd.items():
        j = m.trainIdx
        if j in bwd and bwd[j].trainIdx == i:
            matches[i] = j
            distances[i] = m.distance

    return matches, distances


def match_features_windowed(feats1, feats2, bin_size, match_features_func):
    kp1 = feats1['keypoints']
    kp2 = feats2['keypoints']
    num_p1 = len(kp1)

    bucket_pairs = get_buckets(kp1, kp2, bin_size)

    p2_resolved = {}
    final_matches, final_scores = np.full(num_p1, -1, dtype=int), np.full(num_p1, -1, dtype=int)
    for idx1_list, idx2_list in bucket_pairs:
        if len(idx1_list) < 2 or len(idx2_list) < 2:
            continue
        # Slice subsets of feats
        sub1 = {k: v[idx1_list] for k, v in feats1.items()}
        sub2 = {k: v[idx2_list] for k, v in feats2.items()}

        # Match within this bucket
        local_matches, local_scores = match_features_func(sub1, sub2)
        for i, local_idx2 in enumerate(local_matches):
            if local_idx2 == -1:
                continue

            g1 = idx1_list[i]
            g2 = idx2_list[local_idx2]
            score = local_scores[i] # score check for conflict resolution

            # If point in Image 2 is new or this match is better than previous one
            if g2 not in p2_resolved or score > p2_resolved[g2][1]:
                # Remove previous owner of this point if it exists
                if g2 in p2_resolved:
                    prev_g1 = p2_resolved[g2][0]
                    final_matches[prev_g1] = -1
                    final_scores[prev_g1] = 0

                # Update match
                p2_resolved[g2] = (g1, score)
                final_matches[g1] = g2
                final_scores[g1] = score

    return final_matches, final_scores

def get_buckets(kp1, kp2, bin_size):
    buckets1, buckets2 = {}, {}
    for i, p in enumerate(kp1):
        x, y = p[:2]
        buckets1.setdefault((int(y // bin_size), int(x // bin_size)), []).append(i)
    for i, p in enumerate(kp2):
        x, y = p[:2]
        buckets2.setdefault((int(y // bin_size), int(x // bin_size)), []).append(i)

    pairs = []
    for (r, c), idx1 in buckets1.items():
        idx2_window = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if (r + dr, c + dc) in buckets2:
                    idx2_window.extend(buckets2[(r + dr, c + dc)])

        if idx2_window:
            pairs.append((idx1, idx2_window))
    return pairs


