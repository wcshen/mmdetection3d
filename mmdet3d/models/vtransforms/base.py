from threading import currentThread
from typing import Tuple

import torch
from mmcv.runner import force_fp32
from torch import nn
from torch.nn import functional as F

from mmdet3d.ops.bev_pool import bev_pool
from mmdet3d.models import apply_3d_transformation

__all__ = ["BaseTransform", "BaseDepthTransform"]


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]]) # grid间隔
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.Tensor( # LongTensor will report error
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]
    )
    return dx, bx, nx


class BaseTransform(nn.Module):
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
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound

        dx, bx, nx = gen_dx_bx(self.xbound, self.ybound, self.zbound)
        self.dx = nn.Parameter(dx, requires_grad=False) # [ 0.3000,  0.3000, 20.0000]
        self.bx = nn.Parameter(bx, requires_grad=False) # [-53.8500, -53.8500,   0.0000]
        self.nx = nn.Parameter(nx, requires_grad=False) # [360, 360,   1]

        self.C = out_channels
        
        self.D = torch.arange(*self.dbound, dtype=torch.float).shape[0] #深度范围，[0,100,1]的话是99
        self.fp16_enabled = False

    @force_fp32()
    def create_frustum(self, img_size, feature_size):
        fH, fW = feature_size #32x88
        iH, iW = img_size[0:2]
        ds = (
            torch.arange(*self.dbound, dtype=torch.float) # 1 60 0.5
            .view(-1, 1, 1)
            .expand(-1, fH, fW) # 118 x 32 x 88
        )
        D, _, _ = ds.shape

        xs = (
            torch.linspace(0, iW - 1, fW, dtype=torch.float) # 0-iw-1之间生成fw个数
            .view(1, 1, fW)
            .expand(D, fH, fW) 
        )  # 118 x 32 x 88
        ys = (
            torch.linspace(0, iH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)  # 118 x 32 x 88
        )

        frustum = torch.stack((xs, ys, ds), -1) # torch.Size([118, 32, 88, 3]) 最后一维3是(u,v,d)
        return nn.Parameter(frustum, requires_grad=False)

    @force_fp32()
    def get_geometry(
        self, rots, trans, cam2imgs, post_rots, post_trans, img_metas, start=0, end=-1, **kwargs):
        B, N, _ = trans.shape

        # post-transformation
        post_trans = torch.zeros(B,N,3).to(rots)
        post_rots = torch.eye(3, 3).repeat(B,N,1,1).to(rots)
        # B x N x D x H x W x 3
        frustums = []
        for i in range(B):
            single_frustums=[]
            for img_shape, img_feature_shape in zip(img_metas[i]['img_shape'][start:end],  img_metas[i]['img_feature_shape'][start:end]): # 多个camera
                single_frustum = self.create_frustum(img_shape[0:2], img_feature_shape[-2:])
                single_frustums.append(single_frustum)
            frustums.append(torch.stack(single_frustums))

        frustum = torch.stack(frustums)
        points = frustum.to(rots) - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3)\
            .matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
        combine = rots.matmul(torch.inverse(cam2imgs))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        raw_points_shape = points.shape[1:]
        new_points = []
        for i in range(B):
            x = apply_3d_transformation(points[i].view(-1, 3), 'LIDAR', img_metas[i], reverse=False)
            x = x.view(raw_points_shape)
            new_points.append(x)
        points = torch.stack(new_points)
        
        # if "extra_rots" in kwargs:  # todo
        #     extra_rots = kwargs["extra_rots"]
        #     points = (
        #         extra_rots.view(B, 1, 1, 1, 1, 3, 3)
        #         .repeat(1, N, 1, 1, 1, 1, 1)
        #         .matmul(points.unsqueeze(-1))
        #         .squeeze(-1)
        #     )
        # if "extra_trans" in kwargs:
        #     extra_trans = kwargs["extra_trans"]
        #     points += extra_trans.view(B, 1, 1, 1, 1, 3).repeat(1, N, 1, 1, 1, 1)

        return points #torch.Size([1, 6, 118, 32, 88, 3])

    def get_cam_feats(self, x):
        raise NotImplementedError

    @force_fp32()
    def bev_pool(self, geom_feats, x):
        B, N, D, H, W, C = x.shape # torch.Size([1, 6, 118, 32, 88, 80])
        Nprime = B * N * D * H * W 

        # flatten x
        x = x.reshape(Nprime, C) # 1993728 * 80

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3) #torch.Size([1993728, 3])
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
                for ix in range(B)
            ]
        )
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1]) # self.nx: 360*360*1
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]

        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1) # 1*80*360*360

        return final

    @force_fp32()
    def forward(
        self, points, img_feats, img_metas, lidar2img, lidar2camera, camera_intrinsics, **kwargs):
        
        camera2lidar = torch.inverse(lidar2camera)
        rots = camera2lidar[..., :3, :3]
        trans = camera2lidar[..., :3, 3]
        intrins = camera_intrinsics[..., :3, :3]
        
        if not isinstance(img_feats, list): # 从头开始训练
            geom = self.get_geometry(rots, trans, intrins, None, None, img_metas, **kwargs)
            x = self.get_cam_feats(img_feats)
            x = self.bev_pool(geom, x)
            return x
        else: # 加载离线feature,front和side的feature大小不一致
            start, end = 0, 2
            front_geom = self.get_geometry(rots[:, start:end, :, :], trans[:, start:end, :], intrins[:, start:end, :, :], None, None, img_metas, start, end, **kwargs)
            start, end = 2, 4        
            side_geom = self.get_geometry(rots[:, start:end, :, :], trans[:, start:end, :], intrins[:, start:end, :, :], None, None, img_metas, start, end, **kwargs)
            geoms =[front_geom, side_geom]
            
            bev_feats = []
            for img_feat, geom in zip(img_feats, geoms):
                x = self.get_cam_feats(img_feat)
                x = self.bev_pool(geom, x)
                bev_feats.append(x)
            
            result = torch.max(bev_feats[0], bev_feats[1])
            return result



class BaseDepthTransform(BaseTransform):
    @force_fp32()
    def forward(
        self, points, img_feats, img_metas, lidar2img, lidar2camera, camera_intrinsics, **kwargs
    ):
        camera2lidar = torch.inverse(lidar2camera)
        rots = camera2lidar[..., :3, :3]
        trans = camera2lidar[..., :3, 3]
        
        intrins = camera_intrinsics[..., :3, :3]
        # post_rots = img_aug_matrix[..., :3, :3]
        # post_trans = img_aug_matrix[..., :3, 3]

        # extra_rots = lidar_aug_matrix[..., :3, :3]
        # extra_trans = lidar_aug_matrix[..., :3, 3]

        batch_size = len(points)
        num_cam = len(img_metas[0]['img_shape'])
        depth = torch.zeros(batch_size, num_cam, 1, *self.image_size).to(points[0].device)

        for b in range(batch_size):
            # cur_coords = points[b][:, :3].transpose(1, 0)
            cur_coords = apply_3d_transformation(points[b][:, :3].view(-1, 3), 'LIDAR', img_metas[b], reverse=True)
            cur_coords = cur_coords.transpose(1, 0)
            # cur_img_aug_matrix = img_aug_matrix[b]
            # cur_lidar_aug_matrix = lidar_aug_matrix[b] # not used?
            cur_lidar2image = lidar2img[b].float()

            # lidar2image
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords) #投影到6个相机上
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1) # ?
            # get 2d coords
            dist = cur_coords[:, 2, :] # 相机坐标系下的深度
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :] # 这都是投影到图像上的点了

            # imgaug
            # cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords) # todo?
            # cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)

            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]]

            on_img = (
                (cur_coords[..., 0] < self.image_size[0])
                & (cur_coords[..., 0] >= 0)
                & (cur_coords[..., 1] < self.image_size[1])
                & (cur_coords[..., 1] >= 0)
            )
            for c in range(num_cam):
                masked_coords = cur_coords[c, on_img[c]].long()
                masked_dist = dist[c, on_img[c]]
                depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist

        geom = self.get_geometry(rots, trans, intrins, None, None, img_metas, **kwargs) #torch.Size([1, 6, 118, 32, 88, 3])

        depth_resize_list = []
        for i in range(batch_size):
            out = self.resize_feature(self.feature_size[0]*4, self.feature_size[1]*4, depth[i])
            depth_resize_list.append(out)
            
        depth = torch.stack(depth_resize_list)

        x = self.get_cam_feats(img_feats, depth) # img: [1, 6, 256, 32, 88]  depth: [1, 6, 1, 256, 704]  x: torch.Size([1, 6, 118, 32, 88, 80])
        x = self.bev_pool(geom, x) # geom: [1, 6, 118, 32, 88, 3]   x: [1, 6, 118, 32, 88, 80], 118个深度离散值, 通道是80
        return x
    
    def resize_feature(self, out_h, out_w, in_feat):
        new_h = torch.linspace(-1, 1, out_h).view(-1, 1).repeat(1, out_w)
        new_w = torch.linspace(-1, 1, out_w).repeat(out_h, 1)
        grid = torch.cat((new_h.unsqueeze(2), new_w.unsqueeze(2)), dim=2)
        grid = grid.unsqueeze(0)
        grid = grid.expand(in_feat.shape[0], *grid.shape[1:]).to(in_feat)
        
        out_feat = F.grid_sample(in_feat, grid=grid, mode='bilinear', align_corners=True)
        
        return out_feat
        