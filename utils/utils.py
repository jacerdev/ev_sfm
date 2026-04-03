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

import os
import torch
import random
import logging
import sys
import time
from event_based.feature_matcher import submodule_path_setup

def setup_runtime_environment():
    # Force all libraries to use a single thread
    os.environ["OMP_NUM_THREADS"] = "1"       # OpenMP (used by many OpenCV builds and Scikit-learn)
    os.environ["MKL_NUM_THREADS"] = "1"       # Intel MKL (used by NumPy/SciPy on Intel machines)
    os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS (used by NumPy/SciPy on standard Linux/Docker)
    # Force OpenCV to use a single thread
    cv2.setNumThreads(0)

    # Make PyTorch’s CUDA allocator allow GPU memory segments to grow dynamically
    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Path setup
    submodule_path_setup()

    # Fix seed
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    # Logging Setup
    class ColorFormatter(logging.Formatter):
        COLORS = {logging.DEBUG: "\033[37m",
                  logging.INFO: "",
                  logging.WARNING: "\033[93m",
                  logging.ERROR: "\033[31m",}
        RESET = "\033[0m"
        def format(self, record):
            msg = super().format(record)
            color = self.COLORS.get(record.levelno, "")
            return f"{color}{msg}{self.RESET}"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColorFormatter('%(funcName)s(): %(message)s'))
    logging.getLogger().handlers.clear()
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger('matplotlib').setLevel(logging.WARNING) # silence matplotlib logging

# custom profiler
class Timer:
    def __init__(self, name="Block"): self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        self.interval = self.end - self.start
        logging.debug(f">>> {self.name} duration: {self.interval:.4f} sec <<<")

# Color Generator
colors = [
    [255,   0,   0],   # red
    [  0, 128,   0],   # green
    [  0,   0, 255],   # blue
    [  0, 255, 255],   # cyan
    [255,   0, 255],   # magenta
    [255, 255,   0],   # yellow
    [255, 165,   0],   # orange
    [128,   0, 128],   # purple
    [165,  42,  42],   # brown
    [255, 192, 203],   # pink
    [128, 128, 128],   # gray
    [  0, 255,   0],   # lime
    [  0, 128, 128],   # teal
    [128,   0,   0],   # maroon
    [  0,   0, 128],   # navy
    [128, 128,   0],   # olive
    [ 64, 224, 208],   # turquoise
    [255, 215,   0],   # gold
    [148,   0, 211],   # darkviolet
]
def get_color_gen():
    while True:
        for c in colors[1:]: yield c
color_gen = get_color_gen()