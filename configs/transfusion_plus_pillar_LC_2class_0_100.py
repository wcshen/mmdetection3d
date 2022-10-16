point_cloud_range = [0, -10.0, -2.0, 100.0, 10.0, 6.0]
file_client_args = dict(backend='disk')

class_names = ['Car', 'Truck']
voxel_size = [0.25, 0.25, 8]
out_size_factor = 4
evaluation = dict(interval=1)
dataset_type = 'PlusKittiDataset'
data_root = 'data/L4E_origin_benchmark/'
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
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
    dict(type='LoadMultiCamImagesFromFile'),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925 * 2, 0.3925 * 2],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.5, 0.5, 0.5]),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundleMultiCam3D', class_names=class_names),
    # dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args,
        point_type='float64'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='DefaultFormatBundleMultiCam3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img', 'points'])
        ])
]
data = dict(
    samples_per_gpu=1, # todo?
    workers_per_gpu=1,
    train=dict(
        # type='CBGSDataset', # todo?
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + '/Kitti_L4_data_mm3d_infos_trainval.pkl',
            split='training',
            pts_prefix='pointcloud',
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/Kitti_L4_data_mm3d_infos_val.pkl',
        split='training',
        pts_prefix='pointcloud',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root='data/L4E_origin_benchmark/',
        ann_file='data/L4E_origin_benchmark/Kitti_L4_data_mm3d_infos_val.pkl',
        split='training',
        pts_prefix='pointcloud',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'))
model = dict(
    type='TransFusionDetector',
    freeze_img=True,
    # img_backbone=dict(
    #     type='DLASeg',
    #     num_layers=34,
    #     heads={},
    #     head_convs=-1,
    #     ),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    pts_voxel_layer=dict(
        max_num_points=20,
        voxel_size=voxel_size,
        max_voxels=(30000, 60000), # todo?
        point_cloud_range=point_cloud_range,
        deterministic=True,
        ),
    pts_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        point_cloud_range=point_cloud_range,
    ),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=(80, 400) # todo
    ),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        out_channels=[64, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        out_channels=[128, 128, 128],
        upsample_strides=[0.5, 1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='TransFusionHead', # 
        fuse_img=True,
        num_views=2,
        in_channels_img=64,
        out_size_factor_img=4,
        num_proposals=200,
        auxiliary=True,
        in_channels=128 * 3,
        hidden_channel=128,
        num_classes=len(class_names),
        num_decoder_layers=1,
        num_heads=8, #?
        learnable_query_pos=False,
        initialize_by_heatmap=True,
        nms_kernel_size=3,
        ffn_channel=256,
        dropout=0.1,
        bn_momentum=0.1,
        activation='relu',
        common_heads=dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=point_cloud_range[:2],
            voxel_size=voxel_size[:2],
            out_size_factor=out_size_factor,
            post_center_range=[-10.0, -15.0, -4.0, 110.0, 15.0, 10.0], # todo [0, -10.0, -2.0, 100.0, 10.0, 6.0]
            score_threshold=0.0,
            code_size=8,
        ),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2, alpha=0.25, reduction='mean', loss_weight=1.0),
        # loss_iou=dict(type='CrossEntropyLoss', use_sigmoid=True, reduction='mean', loss_weight=0.0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
    ),
    train_cfg=dict( # 训练时和测试时传递的一些参数, 所有层都能获取到?
        pts=dict(
            dataset='nuScenes', # todo, 已注释, 对ped和cys专门处理的    
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(type='FocalLossCost', gamma=2, alpha=0.25, weight=0.15),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25)
            ),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size=[400, 80, 1],  # [x_len, y_len, 1] # 生成heatmap
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], # 各个loss的权重
            point_cloud_range=point_cloud_range)),
    test_cfg=dict(
        pts=dict(
            dataset='nuScenes', #有影响吗？
            grid_size=[400, 80, 1],
            out_size_factor=out_size_factor,
            pc_range=point_cloud_range[0:2],
            voxel_size=voxel_size[:2],
            nms_type=None,
        )))
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.01)  # for 8gpu * 2sample_per_gpu
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 0.0001),
    cyclic_times=1,
    step_ratio_up=0.4)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4)
total_epochs = 80
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 8)
