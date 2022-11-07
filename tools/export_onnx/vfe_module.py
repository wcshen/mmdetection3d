# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import build_norm_layer
from torch import nn
from torch.nn import functional as F


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

class VFELayer(nn.Module):
    """Voxel Feature Encoder layer.

    The voxel encoder is composed of a series of these layers.
    This module do not support average pooling and only support to use
    max pooling to gather features inside a VFE.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict): Config dict of normalization layers
        max_out (bool): Whether aggregate the features of points inside
            each voxel and only return voxel features.
        cat_max (bool): Whether concatenate the aggregated features
            and pointwise features.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 max_out=True,
                 cat_max=True):
        super(VFELayer, self).__init__()
        self.fp16_enabled = False
        self.cat_max = cat_max
        self.max_out = max_out
        # self.units = int(out_channels / 2)

        self.norm = build_norm_layer(norm_cfg, out_channels)[1]
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, inputs):
        """Forward function.

        Args:
            inputs (torch.Tensor): Voxels features of shape (N, M, C).
                N is the number of voxels, M is the number of points in
                voxels, C is the number of channels of point features.

        Returns:
            torch.Tensor: Voxel features. There are three mode under which the
                features have different meaning.
                - `max_out=False`: Return point-wise features in
                    shape (N, M, C).
                - `max_out=True` and `cat_max=False`: Return aggregated
                    voxel features in shape (N, C)
                - `max_out=True` and `cat_max=True`: Return concatenated
                    point-wise features in shape (N, M, C).
        """
        # [K, T, 7] tensordot [7, units] = [K, T, units]
        voxel_count = inputs.shape[1]

        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()
        pointwise = F.relu(x)
        # [K, T, units]
        if self.max_out:
            aggregated = torch.max(pointwise, dim=1, keepdim=True)[0]
        else:
            # this is for fusion layer
            return pointwise

        if not self.cat_max:
            return aggregated.squeeze(1)
        else:
            # [K, 1, units]
            repeated = aggregated.repeat(1, voxel_count, 1)
            concatenated = torch.cat([pointwise, repeated], dim=2)
            # [K, T, 2 * units]
            return concatenated


class PFNLayer(nn.Module):
    """Pillar Feature Net Layer.

    The Pillar Feature Net is composed of a series of these layers, but the
    PointPillars paper results only used a single PFNLayer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict, optional): Config dict of normalization layers.
            Defaults to dict(type='BN1d', eps=1e-3, momentum=0.01).
        last_layer (bool, optional): If last_layer, there is no
            concatenation of features. Defaults to False.
        mode (str, optional): Pooling model to gather features inside voxels.
            Defaults to 'max'.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 last_layer=False,
                 mode='max'):

        super().__init__()
        self.fp16_enabled = False
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        self.norm = build_norm_layer(norm_cfg, self.units)[1]
        self.linear = nn.Linear(in_channels, self.units, bias=False)

        assert mode in ['max', 'avg']
        self.mode = mode

    def forward(self, inputs, num_voxels=None, aligned_distance=None):
        """Forward function.

        Args:
            inputs (torch.Tensor): Pillar/Voxel inputs with shape (N, M, C).
                N is the number of voxels, M is the number of points in
                voxels, C is the number of channels of point features.
            num_voxels (torch.Tensor, optional): Number of points in each
                voxel. Defaults to None.
            aligned_distance (torch.Tensor, optional): The distance of
                each points to the voxel center. Defaults to None.

        Returns:
            torch.Tensor: Features of Pillars.
        """
        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()
        x = F.relu(x)

        if self.mode == 'max':
            if aligned_distance is not None:
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = torch.max(x, dim=1, keepdim=True)[0]
        elif self.mode == 'avg':
            if aligned_distance is not None:
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = x.sum(
                dim=1, keepdim=True) / num_voxels.type_as(inputs).view(
                    -1, 1, 1)

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PcdetPFNLayer(nn.Module):
    """Pillar Feature Net Layer.

    The Pillar Feature Net is composed of a series of these layers, but the
    PointPillars paper results only used a single PFNLayer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict, optional): Config dict of normalization layers.
            Defaults to dict(type='BN1d', eps=1e-3, momentum=0.01).
        last_layer (bool, optional): If last_layer, there is no
            concatenation of features. Defaults to False.
        mode (str, optional): Pooling model to gather features inside voxels.
            Defaults to 'max'.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False
                 ):

        super().__init__()
        self.fp16_enabled = False
        self.name = 'PcdetPFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        self.use_norm = use_norm
        if self.use_norm:
            self.conv_layer1 = nn.Conv2d(in_channels, out_channels,  kernel_size=1)
            self.norm = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)

    def forward(self, inputs, num_voxels=None, aligned_distance=None):
        """Forward function.

        Args:
            inputs (torch.Tensor): Pillar/Voxel inputs with shape (N, M, C).
                N is the number of voxels, M is the number of points in
                voxels, C is the number of channels of point features.
            num_voxels (torch.Tensor, optional): Number of points in each
                voxel. Defaults to None.
            aligned_distance (torch.Tensor, optional): The distance of
                each points to the voxel center. Defaults to None.

        Returns:
            torch.Tensor: Features of Pillars.
        """
        # BxFxPxN
        x = self.conv_layer1(inputs)
        x = self.norm(x)
        x = F.relu(x)
        x = torch.max(x, dim=-1, keepdim=True)[0]
        return x
    
class PillarFeatureNet(nn.Module):
    """Pillar Feature Net.

    The network prepares the pillar features and performs forward pass
    through PFNLayers.

    Args:
        in_channels (int, optional): Number of input features,
            either x, y, z or x, y, z, r. Defaults to 4.
        feat_channels (tuple, optional): Number of features in each of the
            N PFNLayers. Defaults to (64, ).
        with_distance (bool, optional): Whether to include Euclidean distance
            to points. Defaults to False.
        with_cluster_center (bool, optional): [description]. Defaults to True.
        with_voxel_center (bool, optional): [description]. Defaults to True.
        voxel_size (tuple[float], optional): Size of voxels, only utilize x
            and y size. Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): Point cloud range, only
            utilizes x and y min. Defaults to (0, -40, -3, 70.4, 40, 1).
        norm_cfg ([type], optional): [description].
            Defaults to dict(type='BN1d', eps=1e-3, momentum=0.01).
        mode (str, optional): The mode to gather point features. Options are
            'max' or 'avg'. Defaults to 'max'.
        legacy (bool, optional): Whether to use the new behavior or
            the original behavior. Defaults to True.
    """

    def __init__(self,
                 in_channels=4,
                 feat_channels=(64, ),
                 with_distance=False,
                 with_cluster_center=True,
                 with_voxel_center=True,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max',
                 legacy=True,
                 use_pcdet=False,
                 use_norm=True):
        super(PillarFeatureNet, self).__init__()
        assert len(feat_channels) > 0
        self.legacy = legacy
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 3
        if with_distance:
            in_channels += 1
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self.use_pcdet = use_pcdet
        self.use_norm = use_norm
        self.fp16_enabled = False
        # Create PillarFeatureNet layers
        self.in_channels = in_channels
        feat_channels = [in_channels] + list(feat_channels)
        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = True
            if self.use_pcdet:
                pfn_layers.append(
                PcdetPFNLayer(
                    in_filters,
                    out_filters,
                    use_norm=self.use_norm,
                    last_layer=last_layer))
            else:
                pfn_layers.append(
                    PFNLayer(
                        in_filters,
                        out_filters,
                        norm_cfg=norm_cfg,
                        last_layer=last_layer,
                        mode=mode))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range

    def forward(self, features):
        """Forward function.

        Args:
            features (torch.Tensor): Point features or raw points in shape
                (N, M, C).
            num_points (torch.Tensor): Number of points in each pillar.
            coors (torch.Tensor): Coordinates of each voxel.

        Returns:
            torch.Tensor: Features of pillars.
        """
        if self.use_pcdet:
            features = features.permute((0, 3, 1, 2))
        for pfn in self.pfn_layers:
            features = pfn(features)
        if self.use_pcdet:
            return features.permute((0, 2, 1, 3))
        else:
            return features.squeeze(1)