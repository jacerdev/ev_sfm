from event_based.uzh_dataset import EventDataset
from frame_based.eth3d_dataset import FrameDataset
from event_based.feature_matcher import EventFeatureMatcher
from frame_based.feature_matcher import FeatureMatcher
from frame_based.pairwise_matcher import PairwiseMatcher
from event_based.SuperEvent.data_preparation.util.data_io import load_ts_sparse
from event_based.SuperEvent.util.visualization import ts2image
from utils.utils import load_image

def setup_dataset_and_matcher(dataset_name, feature_based=True):
    DATA = "eth3d_frames"  if dataset_name in ["electro_rig_undistorted", "delivery_area_undistorted"] else "uzh_events" # "eth3d_frames", "uzh_events", or "uzh_frames"
    
    if DATA == "eth3d_frames":
        INDEX_0 = 0
        INDEX_1 = 10
        INDEX_END = -1
        FLIP_ORDER = False

        ds_dir = "/home/jacer/Documents/TU-Berlin/13.WS2526/EventPJ/datasets/ETH3D/" + dataset_name
        out_dir = "/home/jacer/Documents/TU-Berlin/13.WS2526/EventPJ/workspace/outputs/ETH3D/" + dataset_name
        dataset = FrameDataset(ds_dir)
        image_paths, t_windows = dataset.image_paths, None
        def open_image(path): return load_image(path)

        if feature_based:
            matcher = FeatureMatcher((dataset.intrinsics['height'], dataset.intrinsics['width']),
                                     matcher="flann", robust_matching=False)
        else:
            #matcher = PairwiseMatcher(extract_matches_func=dataset.match_keypoints)
            matcher = PairwiseMatcher(matches_dir=out_dir+"/matches_loftr")
        
        config = {
            # Pruning matches
            'MIN_ASSOCIAT_DIST': 5.0, 'PNP_RANSAC_TH': 10.0, 'EPI_RANSAC_TH': 3.0,
            # Filtering Points
            'CELL_SIZE': 20, 'OCCUPANCY_CELL_SIZE': 20, 'MIN_ANGLE': 1.0,
            # Keyframe Selection
            'MIN_BASELINE': 0.5, 'MIN_OVERLAP': 0.7, 'MIN_VISIBILITY': 0.6, 
            # Bundle Adjustment
            'BA_EVERY_N': 4, 'BA_WINDOW': 4,
        }
        THICKNESS = 6

    else: #if DATA in ["uzh_events", "uzh_frames"]:
        INDEX_0 = 0
        INDEX_1 = 3
        INDEX_END = -1
        FLIP_ORDER = False # (try it for urban dataset)
        
        ds_dir = "/home/jacer/Documents/TU-Berlin/13.WS2526/EventPJ/datasets/UZH/" + dataset_name
        out_dir = "/home/jacer/Documents/TU-Berlin/13.WS2526/EventPJ/workspace/outputs/UZH/" + dataset_name
        dataset = EventDataset(ds_dir, out_dir+"/mcts")
        
        config = {
            # Pruning matches
            'PNP_RANSAC_TH': 10.0, 'EPI_RANSAC_TH': 10.0,
            # Filtering Points
            'CELL_SIZE': 1, 'OCCUPANCY_CELL_SIZE': 1, 'MIN_ANGLE': 1., 
            # Keyframe Selection
            'MIN_BASELINE': .5, 'MIN_OVERLAP': .99, 'MIN_VISIBILITY': .99,
            # Bundle Adjustment
            'BA_EVERY_N': 2, 'BA_WINDOW': 4,
        }
        THICKNESS = 2

        if DATA == "uzh_events":
            image_paths, t_windows = dataset.mcts_paths, dataset.mcts_t_windows
            def open_image(path): return ts2image(load_ts_sparse(path))
            
            root_dir = "/home/jacer/Documents/TU-Berlin/13.WS2526/EventPJ/workspace/evis_geoba"
            matcher = EventFeatureMatcher(root_dir, (dataset.intrinsics['height'], dataset.intrinsics['width']),
                                          robust_matching=False, windowed_matching=False, window_size=None)
        elif DATA == "uzh_frames":
            INDEX_1 *= 2
            image_paths, t_windows = dataset.image_paths, None
            def open_image(path): return load_image(path)
            
            if feature_based:
                matcher = FeatureMatcher((dataset.intrinsics['height'], dataset.intrinsics['width']),
                                         matcher="flann", robust_matching=False)
            else:
                config['MIN_ASSOCIAT_DIST'] = 5.0
                matcher = PairwiseMatcher(matches_dir=out_dir+"/matches_loftr")

    if FLIP_ORDER: image_paths = image_paths[::-1]
    
    return {
        'INDEX_0': INDEX_0, 'INDEX_1': INDEX_1, 'INDEX_END': INDEX_END,
        'dataset': dataset, 'image_paths': image_paths, 't_windows': t_windows,
        'open_image': open_image, 'matcher': matcher, 'config': config, 'THICKNESS': THICKNESS
    }
