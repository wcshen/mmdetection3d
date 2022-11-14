import argparse
import os
import warnings
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

import mmdet
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor

from mmcv.ops import Voxelization
from mmdet3d.models.middle_encoders import PointPillarsScatter
from mmdet3d.models.dense_heads.anchor3d_head import Anchor3DHead

import onnxruntime
import onnx

from torch.nn import functional as F
from mmdet3d.core import bbox3d2result, voxel
import cv2

if mmdet.__version__ > '2.23.0':
    # If mmdet version > 2.23.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg
    
    
def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num (torch.Tensor): Actual number of points in each voxel.
        max_num (int): Max number of points in each voxel

    Returns:
        torch.Tensor: Mask indicates which points are valid inside a voxel.
    """
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(
        max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator


def point_augment(features, num_points, coors):
    vx = 0.25
    vy = 0.25
    vz = 8.0
    x_offset = vx / 2 + 0
    y_offset = vy / 2 -10
    z_offset = vz / 2 -2
    features_ls = [features]
    # Find distance of x, y, and z from cluster center
    points_mean = features[:, :, :3].sum(
        dim=1, keepdim=True) / num_points.type_as(features).view(
            -1, 1, 1)
    f_cluster = features[:, :, :3] - points_mean
    features_ls.append(f_cluster)

    # Find distance of x, y, and z from pillar center
    f_center = features[:, :, :3]
    f_center[:, :, 0] = f_center[:, :, 0] - (
        coors[:, 3].type_as(features).unsqueeze(1) * vx +
        x_offset)
    f_center[:, :, 1] = f_center[:, :, 1] - (
        coors[:, 2].type_as(features).unsqueeze(1) * vy +
        y_offset)
    f_center[:, :, 2] = f_center[:, :, 2] - (
        coors[:, 1].type_as(features).unsqueeze(1) * vz +
        z_offset)
    features_ls.append(f_center)

    # Combine together feature decorations
    features = torch.cat(features_ls, dim=-1)
    voxel_count = features.shape[1]
    mask = get_paddings_indicator(num_points, voxel_count, axis=0)
    mask = torch.unsqueeze(mask, -1).type_as(features)
    features *= mask
    return features


def plot_gt_boxes(points, bev_range, name=None):
    """ Visualize the ground truth boxes.
    :param points: lidar points, [N, 3]
    :param gt_boxes: gt boxes, [N, [x, y, z, l, w, h, r]]
    :param bev_range: bev range, [x_min, y_min, z_min, x_max, y_max, z_max]
    lidar_camera_idx: shape: (lidar_pt_nums,) for camera fov
    :return: None
    """
    steps = 0.1
    # Initialize the plotting canvas
    pixels_x = int((bev_range[3] - bev_range[0]) / steps)
    pixels_y = int((bev_range[4] - bev_range[1]) / steps)
    canvas = np.zeros((pixels_x, pixels_y, 3), np.uint8)
    canvas.fill(0)

    # Plot the point cloud
    loc_x = ((points[:, 0] - bev_range[0]) / steps).astype(int)
    loc_x = np.clip(loc_x, 0, pixels_x - 1)
    loc_y = ((points[:, 1] - bev_range[1]) / steps).astype(int)
    loc_y = np.clip(loc_y, 0, pixels_y - 1)
    canvas[loc_x, loc_y] = [0, 255, 255]

    # Rotate the canvas to correct direction
    # canvas = cv2.rotate(canvas, cv2.cv2.ROTATE_90_CLOCKWISE)
    canvas = cv2.flip(canvas, 0)
    canvas = cv2.flip(canvas, 1)
    canvas = cv2.rotate(canvas, cv2.ROTATE_90_CLOCKWISE)
    canvas = cv2.flip(canvas, 0)
    # canvas = cv2.resize(canvas, dsize=(200, 40))
    cv2.imwrite("%s.jpg" % name, canvas)
    
    gt_mask = cv2.imread('/mnt/intel/jupyterhub/swc/datasets/L4E_wo_tele/L4E_origin_data/training/gt_masks/013693.1620546617.898512.20210509T151749_j7-l4e-00011_6_253to273.jpg', cv2.IMREAD_GRAYSCALE)
    gt_mask = cv2.rotate(gt_mask, cv2.ROTATE_90_CLOCKWISE)
    gt_mask = cv2.flip(gt_mask, 0)
    gt_mask = cv2.resize(gt_mask, dsize=(200, 40))
    cv2.imwrite("mask.jpg", gt_mask)
    
    return canvas

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    args = parser.parse_args()
    return args


def onnx_inference_single(cfg, pcd_path):
    vfe_onnx_model = onnx.load(pfe_model_file)
    onnx.checker.check_model(vfe_onnx_model)
    onnx_vfe_session = onnxruntime.InferenceSession(pfe_model_file, providers=['CUDAExecutionProvider'])
    onnx_vfe_input_name = onnx_vfe_session.get_inputs()[0].name
    onnx_vfe_output_name = [onnx_vfe_session.get_outputs()[0].name]

    voxel_layer = Voxelization(**cfg.model.voxel_layer)
    middle_layer = PointPillarsScatter(in_channels=64, output_shape=[80, 400])

    points = np.fromfile(pcd_path).reshape(-1, 4)
    
    # vis_points:
    pcd_bev = plot_gt_boxes(points, bev_range=[0, -10, -2, 100, 10, 6], name="pcd_bev")
    points = torch.from_numpy(points)
    voxels, coors, num_points = [], [], []
    res_voxels, res_coors, res_num_points = voxel_layer(points)
    voxels.append(res_voxels)
    coors.append(res_coors)
    num_points.append(res_num_points)
    voxels = torch.cat(voxels, dim=0)
    num_points = torch.cat(num_points, dim=0)
    coors_batch = []
    for i, coor in enumerate(coors):
        coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
        coors_batch.append(coor_pad)
    coors_batch = torch.cat(coors_batch, dim=0)
    batch_size = coors_batch[-1, 0].item() + 1
    voxel_features = point_augment(voxels, num_points, coors_batch)
    voxel_features = voxel_features.unsqueeze(dim=0).float()
    print(f"shape: {voxel_features.shape} batch size: {batch_size}")
    if voxel_features.shape[1] != 6000:
        print("voxel < 6000!")
        return

    vfe_out_onnx = onnx_vfe_session.run(onnx_vfe_output_name, {onnx_vfe_input_name: voxel_features.numpy()})
    middle_input = torch.from_numpy(vfe_out_onnx[0]).float().cuda()
    middle_input = torch.squeeze(middle_input)
    pseudo_image = middle_layer(middle_input, coors_batch, batch_size)
    print(f"===================== pseudo_image: {pseudo_image.shape}")  # [1, 64, 80, 400]
    
    for i in range(64):
        print(f"current: {i}")
        sample_image = pseudo_image[0, i, ...].cpu().numpy()
        sample_idxs = sample_image > 0
        sample_image[sample_idxs] = 255
        cv2.imwrite(f"./sample_pseudeo/sample_image_c{i}.jpg", sample_image)
    
    return 1



def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg = compat_cfg(cfg)
    pts_path = '/mnt/intel/jupyterhub/swc/datasets/L4E_wo_tele/L4E_origin_data/training/pointcloud/013693.1620546617.898512.20210509T151749_j7-l4e-00011_6_253to273.bin'
    onnx_outputs = onnx_inference_single(cfg,pcd_path=pts_path)



if __name__ == '__main__':
    pfe_model_file = 'tools/export_onnx/mm3d_pps_pfe.onnx'
    main()
