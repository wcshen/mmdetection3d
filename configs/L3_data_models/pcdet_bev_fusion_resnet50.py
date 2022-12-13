_base_ = [
    '../_base_/schedules/cyclic_40e.py', '../_base_/default_runtime.py'
]

# model settings
voxel_size = [0.25, 0.25, 8]
point_cloud_range = [0, -10, -2, 100, 10, 6]
used_cameras=2
use_offline_img_feat=False
used_sensors = {'use_lidar': False,
               'use_camera': True,
               'use_radar': False}
grid_config = {
    'x': [0, 100, voxel_size[0]],
    'y': [-10, 10, voxel_size[1]],
    'z': [-10.0, 10.0, 20.0],
    'depth': [1.0, 100.0, 1],
}
bev_grid_map_size = [
    int((grid_config['y'][1] - grid_config['y'][0]) / voxel_size[1]),
    int((grid_config['x'][1] - grid_config['x'][0]) / voxel_size[0]),
    ]


model = dict(
    type='BEVFusion',
    used_sensors=used_sensors,
    use_offline_img_feat=use_offline_img_feat,
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
        strides=(1, 1, 2, 2),
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='FPNForBEVDet',
        in_channels=[256, 512, 1024, 2048],
        out_channels=64,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(type='LSSTransform',
        in_channels=64,
        out_channels=64,
        image_size=(540, 960),
        feature_size=(135, 240),
        xbound=grid_config['x'],
        ybound=grid_config['y'],
        zbound=grid_config['z'],
        dbound=grid_config['depth'],
        ),
    pts_voxel_layer=dict(
        max_num_points=8,  # max_points_per_voxel
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(6000, 6000)  # (training, testing) max_voxels
    ),
    pts_voxel_encoder=dict( #lss加到这里？
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        use_pcdet=True,
        point_cloud_range=point_cloud_range),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=bev_grid_map_size),
    pts_backbone=dict(
        type='PcdetBackbone',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        num_filters=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        num_upsample_filters=[128, 128, 128],
        ),
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=2,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        assign_per_class=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                [0, -10.0, -0.4, 100.0, 10.0, -0.4],
                [0, -10.0, -0.6, 100.0, 10.0, -0.6]
            ],
            sizes=[[4.63, 1.97, 1.74], # car
                   [12.5, 2.94, 3.47],  # truck
                   ],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(assigner=[
            dict(  # for Car
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            dict(  # for Truck
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.55,
                neg_iou_thr=0.4,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False)),
    test_cfg=dict(
        pts=dict(use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.3,
        min_bbox_size=0,
        nms_pre=4096,
        max_num=500)))
# dataset settings
dataset_type = 'PlusKittiDataset'
l4_data_root = '/home/wancheng.shen/datasets/CN_L4_origin_data/'
l4_benchmark_root = '/home/wancheng.shen/datasets/CN_L4_origin_benchmark/'
l3_data_root = '/mnt/intel/jupyterhub/mrb/datasets/L4E_wo_tele/L4E_origin_data/'
l3_benchmark_root = '/mnt/intel/jupyterhub/swc/datasets/L4E_wo_tele/L4E_origin_benchmark/'
# l3_mini_data='/mnt/intel/jupyterhub/mrb/l4e_mini_data/'
# l3_benchmark_root=l3_mini_data

class_names = ['Car', 'Truck']
input_modality = dict(use_lidar=True, use_camera=False)

file_client_args = dict(backend='disk')

db_sampler = dict(
    data_root=l3_data_root,
    info_path=l3_data_root + 'Kitti_L4_data_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=5, Cyclist=5, Truck=5)),
    classes=class_names,
    sample_groups=dict(Car=15, Pedestrian=15, Cyclist=15, Truck=5),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        point_type='float64',
        file_client_args=file_client_args),
    file_client_args=file_client_args)

# PointPillars uses different augmentation hyper parameters
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        point_type='float64',
        file_client_args=file_client_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        file_client_args=file_client_args),
    # dict(type='ObjectSample', db_sampler=db_sampler, use_ground_plane=True),
    dict(type='LoadMultiCamImagesFromFile', to_float32=True),
    # dict(type='PaintPointsWithImageFeature', used_cameras=used_cameras, drop_camera_prob=100),
    dict(type='RandomFlipLidarOnly', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.4, 0.4],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 
                                 'img_feature', 'lidar2img', 'lidar2camera', 'camera_intrinsics'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args,
        point_type='float64'),
    dict(type='LoadMultiCamImagesFromFile', to_float32=True),
    # dict(type='PaintPointsWithImageFeature', used_cameras=used_cameras, drop_camera_prob=0),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            # dict(
            #     type='GlobalRotScaleTrans',
            #     rot_range=[0, 0],
            #     scale_ratio_range=[1., 1.],
            #     translation_std=[0, 0, 0]),
            # dict(type='RandomFlipLidarOnly'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img', 'img_feature', 'lidar2img', 'lidar2camera', 'camera_intrinsics'])
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=l3_data_root,
            ann_file=l3_data_root + 'Kitti_L4_data_mm3d_infos_train.pkl',
            # ann_file=l3_data_root + 'l4e_mini_data_train.pkl',
            split='training',
            pts_prefix='pointcloud',
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            pcd_limit_range=point_cloud_range,
            used_cameras=used_cameras,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR',
            file_client_args=file_client_args)),
    val=dict(
        type=dataset_type,
        data_root=l3_data_root,
        ann_file=l3_data_root + 'Kitti_L4_data_mm3d_infos_val.pkl',
        # ann_file=l3_data_root + 'l4e_mini_data_val.pkl',
        split='training',
        pts_prefix='pointcloud',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        pcd_limit_range=point_cloud_range,
        used_cameras=used_cameras,
        box_type_3d='LiDAR',
        file_client_args=file_client_args),
    test=dict(
        type=dataset_type,
        data_root=l3_benchmark_root,
        ann_file=l3_benchmark_root + 'Kitti_L4_data_mm3d_infos_val.pkl',
        # data_root=l3_data_root,
        # ann_file=l3_data_root + 'l4e_mini_data_test.pkl',
        split='training',
        pts_prefix='pointcloud',
        samples_per_gpu=8,
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        pcd_limit_range=point_cloud_range,
        test_mode=True,
        used_cameras=used_cameras,
        box_type_3d='LiDAR',
        file_client_args=file_client_args))
# In practice PointPillars also uses a different schedule
# optimizer
lr = 0.001
optimizer = dict(lr=lr)
# max_norm=35 is slightly better than 10 for PointPillars in the earlier
# development of the codebase thus we keep the setting. But we does not
# specifically tune this parameter.
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# PointPillars usually need longer schedule than second, we simply double
# the training schedule. Do remind that since we use RepeatDataset and
# repeat factor is 2, so we actually train 160 epochs.
runner = dict(max_epochs=80)

# Use evaluation interval=2 reduce the number of evaluation timese
evaluation = dict(interval=10)
checkpoint_config = dict(interval=10)
workflow = [('train', 2), ('val', 1)]
# resume_from ='/mnt/intel/jupyterhub/mrb/code/mm3d_bevfusion/train_log/mm3d/pcdet_bev_fusion/20221020-095511/epoch_4.pth'
find_unused_parameters=True