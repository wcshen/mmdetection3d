# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F

from ..builder import DETECTORS
from .mvx_two_stage import MVXTwoStageDetector
from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)

@DETECTORS.register_module()
class BEVFusion(MVXTwoStageDetector):
    """Multi-modality VoxelNet using Faster R-CNN and dynamic voxelization."""

    def __init__(self, **kwargs):
        super(BEVFusion, self).__init__(**kwargs)
        self.use_offline_img_feat = True
        self.use_Cam = True
        self.use_LiDAR = True
        self.use_Radar = False

    def extract_pts_feat(self, points):
        """Extract point features."""
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(points)
        voxel_features = self.pts_voxel_encoder(
            voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        pts_feats = self.pts_middle_encoder(voxel_features, coors, batch_size) # pillar VFE

        return pts_feats
    
    def extract_feat(self, points, img, img_feature, lidar2img, lidar2camera, camera_intrinsics, radar, img_metas):
        if self.use_Cam:
            img_feats = self.extract_img_feat(img, img_feature, lidar2img, lidar2camera, camera_intrinsics, img_metas)
        else:
            img_feats = None
        
        if self.use_LiDAR:
            pts_feats = self.extract_pts_feat(points)
        else:
            pts_feats = None
        
        if self.use_Radar:
            rad_feats = self.radar_encoder(radar)
        else:
            rad_feats = None
        
        return (img_feats, pts_feats, rad_feats)
    
    def forward_outs(self, pts_feats, img_feats, rad_feats):
        # featrue bev fusion
        fused_feats = torch.cat((img_feats, pts_feats), 1)
        
        x = self.pts_backbone(fused_feats) # second FPN
        if self.with_pts_neck:
            x = self.pts_neck(x)
        
        outs = self.pts_bbox_head(x)
        return outs
    
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      img_feature=None,
                      lidar2img=None,
                      lidar2camera=None, 
                      camera_intrinsics=None,
                      radar=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None):
   
        # extract feat
        img_feats, pts_feats, rad_feats = self.extract_feat(points, img, img_feature, lidar2img, lidar2camera, camera_intrinsics, radar, img_metas)
        # calculate loss
        losses = dict()
        loss_fused = self.forward_mdfs_train(pts_feats, img_feats, rad_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(loss_fused)
        return losses    
    
    def forward_mdfs_train(self,
                          pts_feats,
                          img_feats,
                          rad_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        outs = self.forward_outs(pts_feats, img_feats, rad_feats)
        
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.pts_bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses    

    def extract_img_feat(self, img, offline_img_feat, lidar2img, lidar2camera, camera_intrinsics, img_metas):
        """Extract features of images."""
        if self.use_offline_img_feat:
            img_feats = offline_img_feat.squeeze(2)
        else:
            if self.with_img_backbone and img is not None:
                # input_shape = img.shape[-2:]
                # # update real input shape of each single img
                # for img_meta in img_metas:
                #     img_meta.update(input_shape=input_shape)
                if img.dim() == 5 and img.size(0) == 1:
                    img.squeeze_()
                elif img.dim() == 5 and img.size(0) > 1:
                    B, N, C, H, W = img.size()
                    img = img.view(B * N, C, H, W)
                img_feats = self.img_backbone(img)
            else:
                return None
        
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats, img_metas, lidar2img, lidar2camera, camera_intrinsics)
        return img_feats
    
    def simple_test(self, points, img_metas, img=None, radar=None, rescale=False, img_feature=None, lidar2img=None, lidar2camera=None, camera_intrinsics=None):
        """Test function without augmentaiton."""
        img_feats, pts_feats, rad_feats = self.extract_feat(points, img, img_feature, lidar2img, lidar2camera, camera_intrinsics, radar, img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        
        bbox_pts = self.simple_test_mdfs(
            pts_feats, img_feats, rad_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list
 
    def simple_test_mdfs(self, pts_feats, img_feats, rad_feats, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.forward_outs(pts_feats, img_feats, rad_feats)
        bbox_list = self.pts_bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results
    