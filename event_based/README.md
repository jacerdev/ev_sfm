# Event-Based SfM Components

handles data processing and feature matching pipelines for event-based sensors using learned representations.

## UZH Dataset (`uzh_dataset.py`)
Parses data configurations associated with the [UZH DAVIS datasets](https://rpg.ifi.uzh.ch/davis_data.html), loading ground-truth poses, timestamps, and images. It calls the `extract_mcts` functionality to prepare event data for downstream tasks.

## Feature Matcher (`feature_matcher.py`)
Abstracts the `SuperEvent` framework:
- **MCTS Extraction**: The `extract_mcts` function batches raw events and abstracts them into multi-channel temporal surfaces (MCTS). You can execute this file directly to batch-process a raw `events.txt` file into dense `.npz` representations
- **Matching & Feature Maps**: Wrapper for SuperEvent's feature extraction and matching