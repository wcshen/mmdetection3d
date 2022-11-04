# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn as nn

from mmdet3d.models.backbones import PcdetBackbone
from mmdet3d.models.dense_heads import Anchor3DHead

class RPN(nn.Module):
    def __init__(self, backbone_cfg, head_cfg, model_setting) -> None:
        super().__init__()
        self.backbone = PcdetBackbone(in_channels=backbone_cfg.in_channels,
                                      layer_nums=backbone_cfg.layer_nums,
                                      layer_strides=backbone_cfg.layer_strides,
                                      num_filters=backbone_cfg.num_filters,
                                      upsample_strides=backbone_cfg.upsample_strides,
                                      num_upsample_filters=backbone_cfg.num_upsample_filters)
        
        self.head = Anchor3DHead(num_classes=head_cfg.num_classes,
                                 in_channels=head_cfg.in_channels,
                                 feat_channels=head_cfg.feat_channels,
                                 use_direction_classifier=head_cfg.use_direction_classifier,
                                 anchor_generator=head_cfg.anchor_generator,
                                 diff_rad_by_sin=head_cfg.diff_rad_by_sin,
                                 bbox_coder=head_cfg.bbox_coder,
                                 loss_cls=head_cfg.loss_cls,
                                 loss_bbox=head_cfg.loss_bbox,
                                 loss_dir=head_cfg.loss_dir,
                                 train_cfg=model_setting.train_cfg,
                                 test_cfg=model_setting.test_cfg
                                 )
        
    def forward(self, input):
        x = self.backbone(input) # return [out]
        out = self.head(x) # return cls_score, bbox_pred, dir_cls_preds
        cls_score, bbox_pred, dir_cls_preds = out
        cls_score = cls_score[0]
        bbox_pred = bbox_pred[0]
        dir_cls_preds = dir_cls_preds[0]
        
        cls_score = cls_score.permute(0, 2, 3, 1).contiguous()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).contiguous()
        dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
        return cls_score, bbox_pred, dir_cls_preds