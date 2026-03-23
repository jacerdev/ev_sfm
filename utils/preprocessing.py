import cv2
import numpy as np


def load_image(path):
    img = cv2.imread(str(path))
    if img is None: raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def rgb2gray_batch(imgs):
    """Convert RGB images to grayscale"""
    return np.dot(imgs[..., :3], [0.299, 0.587, 0.114])[..., np.newaxis]

def ensure_batch(img, to_gray=False):
    """
    Ensure batch of images.
    img: np.ndarray of shape (H, W), (H, W, 1), (H, W, 3), (B, H, W, 1), (B, H, W, 3)
    Returns: np.ndarray of shape (B, H, W, 1) if to_gray else (B, H, W, C)
    """
    img = np.asarray(img)

    # ensure batch dimension
    if img.ndim == 2:  # (H, W)
        img = img[None, ..., None]
    elif img.ndim == 3: # (H, W, C)
        img = img[None, ...]
    elif img.ndim == 4: # (B, H, W, C)
        pass
    else:
        raise ValueError(f"Unsupported shape: {img.shape}")

    if to_gray and img.shape[-1] == 3: img = rgb2gray_batch(img)
    return img