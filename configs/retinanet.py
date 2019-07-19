net_cfg = {
    'net_name': 'RetinaNet',
    'num_classes': 2
}

anchor_cfg = {
    'dense_anchor': False,
    'feature_maps': [[80, 80], [40, 40], [20, 20], [10,10], [5,5]],
    'min_dim': 640,
    'steps': [8, 16, 32, 64, 128],
    'min_sizes': [[16, 20.2, 25.4], [32, 40.3, 50.8], [64, 80.6, 101.6], [128, 161.3, 203.2], [256, 322.54, 406.37]],
    'anchors': [16, 20.2, 25.4, 32, 40.3, 50.8, 64, 80.6, 101.6, 128, 161.3, 203.2, 256, 322.54, 406.37],
    'aspect_ratios': [[2/3, 3/2], [2/3, 3/2], [2/3, 3/2], [2/3, 3/2],[2/3, 3/2]],
    'variance': [0.1, 0.2],
    'clip': False
}

train_cfg = {
    'input_size': 640,
    'loss_type': 'focal',
    'loc_weight': 2.0,
    'cls_weight': 2.0,
    'landmark_weight': 0.1,
    'distillation_weight': 3.0,
    'use_landmark': True
}

test_cfg = {
    'save_folder': 'RetinaNet',
    'is_anchor_base': True,
    'is_ctr': False
    
}