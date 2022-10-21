# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F

from ..builder import DETECTORS
from .mvx_two_stage import MVXTwoStageDetector


@DETECTORS.register_module()
class BEVFusion(MVXTwoStageDetector):
    """Multi-modality VoxelNet using Faster R-CNN and dynamic voxelization."""

    def __init__(self, **kwargs):
        super(BEVFusion, self).__init__(**kwargs)

    def extract_pts_feat(self, points, img_feats, img_metas): #为什么要重写呢？
        """Extract point features."""
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(points)
        voxel_features = self.pts_voxel_encoder(
            voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        pts_feats = self.pts_middle_encoder(voxel_features, coors, batch_size) # pillar VFE
        fused_feats = torch.cat((img_feats, pts_feats), 1) # bev fusion
        x = self.pts_backbone(fused_feats) # second FPN
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x
