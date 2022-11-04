from typing import Tuple

import torch
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.models.builder import NECKS

from .base import BaseDepthTransform

__all__ = ["DepthLSSTransform"]


@NECKS.register_module()
class DepthLSSTransform(BaseDepthTransform):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        self.dtransform = nn.Sequential( # conv2d,为啥是8? 8是输出通道 in_channel out_channel kernel
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels + 64, in_channels, 3, padding=1), # 64是depth通道个数
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.D + self.C, 1),
        )
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    @force_fp32()
    def get_cam_feats(self, x, d):
        B, N, C, fH, fW = x.shape # torch.Size([1, 6, 256, 32, 88])

        d = d.view(B * N, *d.shape[2:]) 
        x = x.view(B * N, C, fH, fW) # torch.Size([6, 256, 32, 88])

        d = self.dtransform(d) # torch.Size([6, 64, 32, 88])
        x = torch.cat([d, x], dim=1) # torch.Size([6, 320, 32, 88])
        x = self.depthnet(x) # [6, 198, 32, 88]

        depth = x[:, : self.D].softmax(dim=1) #通过soft max获得概率最大的深度 [6, 118, 32, 88]
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2) # camera feature * depth

        x = x.view(B, N, self.C, self.D, fH, fW)  # torch.Size([1, 6, 80, 118, 32, 88])
        x = x.permute(0, 1, 3, 4, 5, 2) # torch.Size([1, 6, 118, 32, 88, 80])
        return x
    
    def forward(self, *args, **kwargs):
        x = super().forward(*args, **kwargs)
        x = self.downsample(x)
        return x