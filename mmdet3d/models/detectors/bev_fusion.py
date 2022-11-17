# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F

from ..builder import DETECTORS
from .. import builder
from .mvx_two_stage import MVXTwoStageDetector
from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)

@DETECTORS.register_module()
class BEVFusion(MVXTwoStageDetector):
    """Multi-modality VoxelNet using Faster R-CNN and dynamic voxelization."""

    def __init__(self,used_sensors=None, use_offline_img_feat=True, img_view_transformer=None, **kwargs):
        super(BEVFusion, self).__init__(**kwargs)
        self.use_offline_img_feat = use_offline_img_feat
        self.use_LiDAR = used_sensors.get('use_lidar', False)
        self.use_Cam = used_sensors.get('use_camera', False)
        self.use_Radar = used_sensors.get('use_radar', False)
        if img_view_transformer is not None:
            self.img_view_transformer = builder.build_neck(img_view_transformer)
    
    @property
    def with_img_view_transformer(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'img_view_transformer') and self.img_view_transformer is not None
    
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
            img_feats = self.extract_img_feat(points, img, img_feature, lidar2img, lidar2camera, camera_intrinsics, img_metas)
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
        if self.use_LiDAR and self.use_Cam and not self.use_Radar:
            fused_feats = torch.cat((img_feats, pts_feats), 1)
        elif self.use_LiDAR and not self.use_Cam and not self.use_Radar:
            fused_feats = pts_feats
        elif not self.use_LiDAR and self.use_Cam and not self.use_Radar:
            fused_feats = img_feats
        else: # todo
            fused_feats = torch.cat((img_feats, pts_feats, rad_feats), 1)
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
                      side_img_feature=None,
                      lidar2img=None,
                      lidar2camera=None, 
                      camera_intrinsics=None,
                      radar=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None):
   
        # extract feat
        offline_img_features = [img_feature]
        if side_img_feature is not None:
            offline_img_features.append(side_img_feature)
        img_feats, pts_feats, rad_feats = self.extract_feat(points, img, offline_img_features, lidar2img, lidar2camera, camera_intrinsics, radar, img_metas)
        # calculate loss
        losses = dict()
        loss_fused = self.forward_mdfs_train(pts_feats, img_feats, rad_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(loss_fused)
        return losses    
    
    
    def forward_test(self, points, img_metas, img=None, radar=None, 
                     img_feature=None, side_img_feature=None, lidar2img=None, lidar2camera=None,
                     camera_intrinsics=None, **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(points, 'points'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(points)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(points), len(img_metas)))

        if num_augs == 1:
            img = [img] if img is None else img
            radar =[radar] if radar is None else radar
            img_feature =[img_feature] if img_feature is None else img_feature
            side_img_feature =[side_img_feature] if side_img_feature is None else side_img_feature
            
            offline_img_features = [img_feature[0]]
            if side_img_feature is not None:
                offline_img_features.append(side_img_feature[0])
            lidar2img = [lidar2img] if lidar2img is None else lidar2img
            lidar2camera = [lidar2camera] if lidar2camera is None else lidar2camera
            camera_intrinsics = [camera_intrinsics] if camera_intrinsics  is None else camera_intrinsics
            return self.simple_test(points=points[0], 
                                    img_metas=img_metas[0],
                                    img=img[0], radar=radar[0],
                                    img_feature=offline_img_features, 
                                    lidar2img=lidar2img[0],
                                    lidar2camera=lidar2camera[0], 
                                    camera_intrinsics=camera_intrinsics[0],
                                    **kwargs)
        else:
            return self.aug_test(points, img_metas, img, **kwargs)
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

    def extract_img_feat(self, points, img, offline_img_feat, lidar2img, lidar2camera, camera_intrinsics, img_metas):
        """Extract features of images."""
        if self.use_offline_img_feat:
            img_feats = []
            for feat in offline_img_feat:
                img_feats.append(feat.squeeze(2)) # todo
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
            if self.with_img_neck:
                img_feats = self.img_neck(img_feats)
            img_feats = img_feats.view(B, N, img_feats.shape[-3], img_feats.shape[-2], img_feats.shape[-1])
        
        if self.with_img_view_transformer:
            img_feats = self.img_view_transformer(points, img_feats, img_metas, lidar2img, lidar2camera, camera_intrinsics)
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
    