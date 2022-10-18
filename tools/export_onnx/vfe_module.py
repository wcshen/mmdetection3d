# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
from torch.nn import functional as F

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
        self.name = 'PcdetPFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        self.use_norm = use_norm
        if self.use_norm:
            self.conv_layer1 = nn.Conv2d(in_channels, out_channels,  kernel_size=1)
            self.norm = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)

    def forward(self, inputs):
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

    def __init__(self,
                 in_channels=4,
                 feat_channels=(64, ),
                 with_distance=False,
                 with_cluster_center=True,
                 with_voxel_center=True,
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
        self.use_pcdet = use_pcdet
        self.use_norm = use_norm

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
        self.pfn_layers = nn.ModuleList(pfn_layers)

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