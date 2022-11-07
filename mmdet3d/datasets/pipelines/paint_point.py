from ..builder import PIPELINES
import numpy as np
from torch.nn import functional as F
import torch
import copy
import random

from mmdet3d.core.points import LiDARPoints

camera_names = ['front_left_camera', 'front_right_camera',
                'side_left_camera', 'side_right_camera',
                'rear_left_camera', 'rear_right_camera']

@PIPELINES.register_module()
class PaintPointsWithImageFeature:
    
    def __init__(self, used_cameras=4, avg_flag=True, drop_camera_prob=0):
        self.used_cameras = used_cameras
        self.avg_flag = avg_flag
        self.drop_camera_prob = drop_camera_prob
     
    def get_image_features(self, images_path):
        full_image_features = []
            
        for camera_idx in range(self.used_cameras):
            cur_image_file = images_path[camera_idx]
            cur_feature_file = cur_image_file.replace(camera_names[camera_idx], camera_names[camera_idx]+'_feature')
            cur_feature_file = cur_feature_file.replace('.jpg', '_0.npy')
            feature = np.load(cur_feature_file)
            feature = torch.from_numpy(feature)
            full_image_features.append(feature)
        
        return full_image_features
    

    def __call__(self, results):
        image_features = self.get_image_features(results['filename'])
        lidar_raw = results['points'].tensor.numpy()
        
        camera_features_list = []
        for camera_idx in range(self.used_cameras):
            img_feature = image_features[camera_idx]
            cam_calib = results['lidar2img'][camera_idx]
            lidar_pts = copy.deepcopy(lidar_raw)
            lidar_pts[:, 3] = 1
            lidar_pts = np.matmul(cam_calib, lidar_pts.T).T
            coor_x = lidar_pts[:, 0] / lidar_pts[:, 2]
            coor_y = lidar_pts[:, 1] / lidar_pts[:, 2]
            
            image_shape = results['img_shape'][camera_idx]
            h, w = image_shape[0], image_shape[1]
            coor_y = coor_y / (h - 1) * 2 - 1
            coor_x = coor_x / (w - 1) * 2 - 1
            coor_y = torch.from_numpy(coor_y).unsqueeze(1)
            coor_x = torch.from_numpy(coor_x).unsqueeze(1)

            grid = torch.cat([coor_x, coor_y], dim=1).unsqueeze(0).unsqueeze(0) # Nx2 -> 1x1xNx2
            grid = grid.float()
            # align_corner=True provides higher performance
            mode = 'bilinear'
            camera_features = F.grid_sample(
            img_feature,
            grid,
            mode=mode,
            padding_mode='zeros',
            align_corners=True)  # 1xCx1xN feats
            camera_features = camera_features.squeeze().t().numpy().astype(np.float64)  # (N, c)
            camera_features_list.append(camera_features)
            
        all_camera_features = np.stack(camera_features_list, axis=0)
        if self.avg_flag:
            all_camera_features = np.mean(all_camera_features, axis=0)  # TODO(swc): just avg now
        else:
            front_camera_feature = np.max(all_camera_features[:2], axis=0)
            side_camera_feature = np.max(all_camera_features[2:], axis=0)
            all_camera_features = np.concatenate([front_camera_feature, side_camera_feature], axis=-1)
            
        augmented_lidar = np.concatenate((lidar_raw, all_camera_features), axis=1)
        if self.drop_camera_prob > 0: 
            if random.randint(1, self.drop_camera_prob) == 1:
                augmented_lidar[:,4:] = 0
        augmented_lidar = LiDARPoints(augmented_lidar, points_dim=augmented_lidar.shape[-1])
        results['points'] = augmented_lidar
        return results