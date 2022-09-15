from ..builder import PIPELINES
import numpy as np
from torch.nn import functional as F
import torch
import copy
\
from mmdet3d.core.points import LiDARPoints

camera_names = ['front_left_camera', 'front_right_camera',
                'side_left_camera', 'side_right_camera',
                'rear_left_camera', 'rear_right_camera']

@PIPELINES.register_module()
class PaintPointsWithImageFeature:
    
    def __init__(self, used_cameras=4):
        self.used_cameras = used_cameras
    
     
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
        
        point_features_list = []
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
            point_features = F.grid_sample(
            img_feature,
            grid,
            mode=mode,
            padding_mode='zeros',
            align_corners=True)  # 1xCx1xN feats
            point_features = point_features.squeeze().t().numpy().astype(np.float64)  # (N, c)
            point_features_list.append(point_features)
            
        multi_point_features = np.stack(point_features_list, axis=0)
        multi_point_features = np.mean(multi_point_features, axis=0)  # TODO(swc): just avg now
        augmented_lidar = np.concatenate((lidar_raw, multi_point_features), axis=1)
        augmented_lidar = LiDARPoints(augmented_lidar, points_dim=augmented_lidar.shape[-1])
        results['points'] = augmented_lidar
        return results