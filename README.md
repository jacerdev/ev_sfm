# Event-Based & Frame-Based Structure from Motion (SfM)

This repository contains a complete incremental Structure from Motion (SfM) pipeline, designed to process both standard frame-based inputs and asynchronous event-stream data. 

**The core defining feature of this project is the from-scratch implementation of the entire incremental SfM pipeline.** OpenCV can be toggled, but the default pipeline is implemented from scratch relying solely on NumPy and SciPy
---

## Setup & Execution

### 1. Prepare Environment
```bash
mamba env create -f environment.yml
mamba activate sfm_env
```
### 2. Run Pipeline
```bash
python run_sfm.py
```

---

## The Core Pipeline

### 1. `run_sfm.py`
The overarching script that initializes configurations, instantiates the required dataset/matcher objects (event-based or frame-based), and bootstraps the `IncrementalSfM` pipeline. It processes each frame consecutively, handling map initialization, skipping heuristics, tracking, bundle-adjustment scheduling, and 3D visualization.

### 2. `sfm/pipeline.py` & `sfm/routines.py`
These files drive the logical SfM loop:
- **`pipeline.py` (IncrementalSfM)**: Maintains the state of the 3D map. Maps 2D tracks across keyframes to global 3D objects, and orchestrates the transition between mapping and tracking.
- **`routines.py`**: The bridge between state-management and raw math. Handles scene initialization, tracking (`track_frame`), and triangulating new observations (`map_observations`).

### 3. `multiview/`
- **Epipolar Geometry (`multiview/epipolar.py`)**
- **Perspective-n-Point (`multiview/pnp.py`)**
- **Triangulation (`multiview/triangulation.py`)**



