# Frame-Based SfM Components

This directory contains modules for handling traditional frame-based camera data and feature matching.

## ETH3D Dataset Loader (`eth3d_dataset.py`)
A standalone utility to read and process the [ETH3D SfM datasets](https://www.eth3d.net/datasets). It parses COLMAP-style text files (`images.txt`, `cameras.txt`, `points3D.txt`) to extract Cameras intrinsics & extrinsics, as well as 3D Points & Observations
- **Scene Graph Construction**: Can build a fully linked SfM scene graph using the project's internal `objects.py` structures.

### Usage
To visualize a dataset:
```bash
python -m frame_based.eth3d_dataset
```
(update `data_path` in the file)


## Other Components
- `feature_matcher.py`: Wrappers for DISK features and LightGlue/FLANN matching.
- `pairwise_matcher.py`: High-level interface for LoFTR pairwise matching.
