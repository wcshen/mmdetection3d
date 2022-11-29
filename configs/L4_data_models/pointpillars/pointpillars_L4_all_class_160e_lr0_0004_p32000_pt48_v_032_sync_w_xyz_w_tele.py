_base_ = [
    '../../_base_/schedules/cyclic_40e.py', '../../_base_/default_runtime.py'
]
using_tele=True
# model settings
voxel_size = [0.32, 0.32, 8]
point_cloud_range = [-50, -51.2, -2, 154.8, 51.2, 6]
use_sync_bn=True
model = dict(
    type='VoxelNet',
    voxel_layer=dict(
        max_num_points=48,  # max_points_per_voxel
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(32000, 32000)  # (training, testing) max_voxels
    ),
    voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        use_pcdet=True,
        legacy=False,
        with_camera_feature=False,
        point_cloud_range=point_cloud_range),
    middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[320, 640]),
    backbone=dict(
        type='PcdetBackbone',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        num_filters=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        num_upsample_filters=[128, 128, 128],
    ),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=4,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        assign_per_class=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                [-50, -51.2, -0.4, 154.8, 51.2, -0.4],
                [-50, -51.2, -0.6, 154.8, 51.2, -0.6],
                [-50, -51.2, -0.4, 154.8, 51.2, -0.4],
                [-50, -51.2, -0.4, 154.8, 51.2, -0.4],
            ],
            sizes=[
                   [4.63, 1.97, 1.74], # car
                   [12.5, 2.94, 3.47],  # truck
                   [0.8, 0.6, 1.73], # ped
                   [1.76, 0.6, 1.73], # cyclist
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
        assigner=[
            dict(  # for Car
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.2,
                ignore_iof_thr=-1),
            dict(  # for Truck
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.55,
                neg_iou_thr=0.4,
                min_pos_iou=0.2,
                ignore_iof_thr=-1),
            dict(  # for Pedestrian
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.2,
                ignore_iof_thr=-1),
            dict(  # for Cyclist
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.2,
                ignore_iof_thr=-1),
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=4096,
        max_num=500))

# dataset settings
dataset_type = 'PlusKittiDataset'
data_root = '/mnt/intel/jupyterhub/swc/datasets/L4_extracted_data/CN_L4_origin_data/'
hard_case_data = '/mnt/intel/jupyterhub/swc/datasets/L4_extracted_data/hard_case_origin_data/'
side_vehicle_data = '/mnt/intel/jupyterhub/swc/datasets/L4_extracted_data/side_vehicle_origin_data/'
under_tree_data = '/mnt/intel/jupyterhub/swc/datasets/L4_extracted_data/under_tree_origin_data/'

benchmark_root = '/mnt/intel/jupyterhub/swc/datasets/L4_extracted_data/CN_L4_origin_benchmark/'

class_names = ['Car', 'Truck', 'Pedestrian', 'Cyclist']
input_modality = dict(use_lidar=True, use_camera=False)

file_client_args = dict(backend='disk')

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'Kitti_L4_data_dbinfos_train.pkl',
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
        using_tele=using_tele,
        file_client_args=file_client_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        file_client_args=file_client_args),
    # dict(type='ObjectSample', db_sampler=db_sampler, use_ground_plane=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args,
        point_type='float64',
        using_tele=using_tele),    
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        point_type='float64',
        using_tele=using_tele,
        file_client_args=file_client_args),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]

concat_train_data = dict(
    type='ConcatDataset',
    datasets=[
        dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'Kitti_L4_data_mm3d_infos_train.pkl',
            split='training',
            pts_prefix='pointcloud',
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            pcd_limit_range=point_cloud_range,
            box_type_3d='LiDAR',
            file_client_args=file_client_args
        ),
        dict(
            type=dataset_type,
            data_root=hard_case_data,
            ann_file=hard_case_data + 'Kitti_L4_data_mm3d_infos_train.pkl',
            split='training',
            pts_prefix='pointcloud',
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            pcd_limit_range=point_cloud_range,
            box_type_3d='LiDAR',
            file_client_args=file_client_args
        ),
        dict(
            type=dataset_type,
            data_root=side_vehicle_data,
            ann_file=side_vehicle_data + 'Kitti_L4_data_mm3d_infos_train.pkl',
            split='training',
            pts_prefix='pointcloud',
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            pcd_limit_range=point_cloud_range,
            box_type_3d='LiDAR',
            file_client_args=file_client_args
        ),
        dict(
            type=dataset_type,
            data_root=under_tree_data,
            ann_file=under_tree_data + 'Kitti_L4_data_mm3d_infos_train.pkl',
            split='training',
            pts_prefix='pointcloud',
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            pcd_limit_range=point_cloud_range,
            box_type_3d='LiDAR',
            file_client_args=file_client_args
        ),
    ]
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=concat_train_data),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'Kitti_L4_data_mm3d_infos_val.pkl',
        split='training',
        pts_prefix='pointcloud',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        pcd_limit_range=point_cloud_range,
        box_type_3d='LiDAR',
        file_client_args=file_client_args),
    test=dict(
        type=dataset_type,
        data_root=benchmark_root,
        ann_file=benchmark_root + 'Kitti_L4_data_mm3d_infos_val.pkl',
        split='training',
        pts_prefix='pointcloud',
        samples_per_gpu=8,
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        pcd_limit_range=point_cloud_range,
        test_mode=True,
        box_type_3d='LiDAR',
        file_client_args=file_client_args))

# In practice PointPillars also uses a different schedule
# optimizer
lr = 0.0004
optimizer = dict(lr=lr)
# max_norm=35 is slightly better than 10 for PointPillars in the earlier
# development of the codebase thus we keep the setting. But we does not
# specifically tune this parameter.
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
runner = dict(max_epochs=80)

# Use evaluation interval=2 reduce the number of evaluation timese
evaluation = dict(interval=10, pipeline=eval_pipeline)
checkpoint_config = dict(interval=2)
workflow = [('train', 2), ('val', 1)]
# resume_from = '/mnt/intel/jupyterhub/swc/train_log/mm3d/pointpillars_L4_all_class_160e_lr0_001_p32000_pt48_v_025/20220926-173323/epoch_10.pth'
