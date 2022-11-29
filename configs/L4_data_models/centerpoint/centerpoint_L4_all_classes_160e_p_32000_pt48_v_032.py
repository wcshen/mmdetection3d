_base_ = [
    '../../_base_/schedules/cyclic_40e.py', '../../_base_/default_runtime.py'
]
use_sync_bn=True
using_tele=False
# model settings
voxel_size = [0.32, 0.32, 8]
point_cloud_range = [-50, -51.2, -2, 154.8, 51.2, 6]
model = dict(
    type='CenterPoint',
    pts_voxel_layer=dict(
        max_num_points=48,  # max_points_per_voxel
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(32000, 32000)  # (training, testing) max_voxels
    ),
    pts_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        use_pcdet=True,
        point_cloud_range=point_cloud_range,
        legacy=False),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[320, 640]),
    pts_backbone=dict(
        type='PcdetBackbone',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        num_filters=[64, 128, 256],
        upsample_strides=[0.5, 1, 2],
        num_upsample_filters=[128, 128, 128],
    ),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=sum([128, 128, 128]),
        tasks=[
            dict(num_class=1, class_names=['Pedestrian']),
            dict(num_class=1, class_names=['Cyclist']),
            dict(num_class=1, class_names=['Car']),
            dict(num_class=1, class_names=['Truck']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=point_cloud_range,
            max_num=500,
            score_threshold=0.1,
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            code_size=7,
            pc_range=point_cloud_range[:2],),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[640, 320, 1],
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            out_size_factor=4,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=point_cloud_range,
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[0.175, 0.85, 4, 12],
            score_threshold=0.1,
            pc_range=point_cloud_range[:2],
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2)))
# dataset settings
dataset_type = 'PlusKittiDataset'
data_root = '/mnt/intel/jupyterhub/swc/datasets/L4_extracted_data/CN_L4_origin_data/'
hard_case_data = '/mnt/intel/jupyterhub/swc/datasets/L4_extracted_data/hard_case_origin_data/'
side_vehicle_data = '/mnt/intel/jupyterhub/swc/datasets/L4_extracted_data/side_vehicle_origin_data/'
under_tree_data = '/mnt/intel/jupyterhub/swc/datasets/L4_extracted_data/under_tree_origin_data/'

benchmark_root = '/mnt/intel/jupyterhub/swc/datasets/L4_extracted_data/CN_L4_origin_benchmark/'

class_names = ['Pedestrian', 'Cyclist', 'Car', 'Truck']
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
    samples_per_gpu=8,
    workers_per_gpu=8,
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
runner = dict(max_epochs=40)

# Use evaluation interval=2 reduce the number of evaluation timese
evaluation = dict(interval=5, pipeline=eval_pipeline)
checkpoint_config = dict(interval=2)
workflow = [('train', 2), ('val', 1)]
# resume_from = '/mnt/intel/jupyterhub/swc/train_log/mm3d/pointpillars_L4_all_class_160e_lr0_001_p32000_pt48_v_025/20220926-173323/epoch_10.pth'
