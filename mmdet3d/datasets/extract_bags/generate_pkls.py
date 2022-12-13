import os
import cv2
import sys
import yaml
import torch
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from easydict import EasyDict

from mmdet3d.ops import points_in_boxes_cpu


class L4Dataset():
    def __init__(self, dataset_cfg, class_names, mode='train', root_path=None, logger=None, debug_frame_id=None, used_camera_names=None):
        """

        Args:
            dataset_cfg:
            class_names:
            mode:
            root_path:
            logger:
            debug_frame_id:
        """
        self.dataset_cfg = dataset_cfg
        self.mode = mode
        self.class_names = class_names
        self.logger = logger
        self.root_path = root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)
        self.logger = logger 
        if self.dataset_cfg is None or class_names is None:
            return

        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float)
        self.num_points_in_object_threshold = 4
        # min_size:
        self.length_threshold = 4.0
        self.width_threshold = 1.4
        self.height_threshold = 1.2
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        self.L4_data_infos = []
        self.debug_frame_id = debug_frame_id
        self.all_label_count = 0
        self.filter_label_count = 0
        self.camera_names = used_camera_names
        

    def set_split(self, split):
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        # lidar_bin names without '.bin'
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def get_lidar(self, idx):
        lidar_file = self.root_split_path / 'pointcloud' / ('%s.bin' % idx)
        assert lidar_file.exists()
        lidar_data = np.fromfile(str(lidar_file)).reshape(-1, 4)
        # ignore intensity value
        lidar_data = np.concatenate([lidar_data[:, 0:3], np.zeros((lidar_data.shape[0], 1))], axis=1)

        return lidar_data

    def get_camera_features(self, featrues_path):
        feature_list = []
        for name in featrues_path:
            feature = np.load(name)
            feature_list.append(feature)

            img_name = name.replace('front_left_camera_feature', 'front_left_camera')
            img_name = img_name.replace('_0.npy', '.jpg')
            img = cv2.imread(img_name)
            img = np.expand_dims(img, 0).transpose(0, 3, 1, 2)
            feature_list.append(img)
            break

        return feature_list

    def get_label(self, idx):
        label_file = self.root_split_path / 'label' / ('%s.pkl' % idx)
        try:
            assert label_file.exists()
        except AssertionError:
            print('[ERROR] get label failed:', label_file)
        with open(label_file, 'rb') as f:
            labels = pickle.load(f, encoding='latin1')
            for label in labels:
                # Combine Livox label names
                if label['name'] in ['Car', 'car', 'police_car']:
                    label['name'] = 'Car'
                elif label['name'] in ['Truck', 'bus', 'truck', 'Engineering_vehicles', 'trailer']:
                    label['name'] = 'Truck'
                elif label['name'] in ['Pedestrian', 'pedestrians', 'wheelchair', 'stroller']:
                    label['name'] = 'Pedestrian'
                elif label['name'] in ['Cyclist', 'bicycle', 'motorcycle', 'Portable_personal_motor_vehicle']:
                    label['name'] = 'Cyclist'

        return labels

    def is_size_valid(self, size):
        length = size[3]
        width = size[4]
        height = size[5]
        if length <= 3 or width <= 1.2 or height <= 1:
            return False
        else:
            return True

    def check_size(self, size):
        size[3] = max(size[3], self.length_threshold)
        size[4] = max(size[4], self.width_threshold)
        size[5] = max(size[5], self.height_threshold)
        return size

    def filter_object_by_rule(self, points, labels):  # todo
        num_objects = len(labels)
        if num_objects == 0:
            print("[Warning] No objects in this frame.")
            return labels

        # calculate number of points in each object box
        box_array = []
        for label in labels:
            box_array.append(label['box3d_lidar'])
        box_array = np.array(box_array, dtype=np.float)
        point_indices = points_in_boxes_cpu(torch.from_numpy(points[:, 0:3]),
                                                                  torch.from_numpy(box_array)
                                                                  ).numpy()  # (nboxes, npoints)
        num_points = np.sum(point_indices, axis=1)

        filtered_label_list = []
        for idx in range(num_objects):
            labels[idx]['box3d_lidar'] = np.array(labels[idx]['box3d_lidar'], dtype=np.float)
            loc_x = labels[idx]['box3d_lidar'][0]
            loc_y = labels[idx]['box3d_lidar'][1]
            num_pts_in_obj = num_points[idx]
            self.all_label_count += 1
            if self.range[0] <= loc_x <= self.range[3] \
                    and self.range[1] <= loc_y <= self.range[4] \
                    and num_pts_in_obj >= self.num_points_in_object_threshold:
                filtered_label_data = {'name': labels[idx]['name'],
                                       'box3d_lidar': self.check_size(labels[idx]['box3d_lidar']),
                                       'num_points_in_gt': num_pts_in_obj}
                filtered_label_list.append(filtered_label_data)
            else:
                self.filter_label_count += 1
                # if loc_x < 0:
                # print('filter label', self.filter_label_count,'/',self.all_label_count, num_pts_in_obj,
                #     loc_x, loc_y)

        return filtered_label_list

    def prepare_labels(self):
        split_dir = self.root_path / 'ImageSets' / 'trainval.txt'
        (self.root_split_path / 'filtered_label').mkdir(parents=True, exist_ok=True)

        sample_id_list = [x.strip() for x in open(
            split_dir).readlines()] if split_dir.exists() else None
        for sample_idx in tqdm(sample_id_list):
            obj_labels = self.get_label(sample_idx)
            lidar_pts = self.get_lidar(sample_idx)
            filtered_obj_labels = self.filter_object_by_rule(
                lidar_pts, obj_labels)

            # Save the filtered labels
            label_file = self.root_split_path / \
                         'filtered_label' / ('%s.pkl' % sample_idx)
            with open(label_file, 'wb') as f:
                pickle.dump(filtered_obj_labels, f)

    def load_images_and_calibs(self):

        pkl_name = str(self.root_split_path / 'cam_img_timestamp.pkl')
        pkl_file = open(pkl_name, 'rb')
        data = pickle.load(pkl_file, encoding='latin1')
        # NOTE(swc): only used here right now
        self.camera_lidar_timestamp = data

        self.cameras_img_name = {}
        for camera_name in self.camera_names:
            img_names = sorted(os.listdir(str(self.root_split_path / camera_name)))

            self.cameras_img_name[camera_name] = img_names

        pkl_name = str(self.root_split_path / 'sensor_calibs.pkl')
        pkl_file = open(pkl_name, 'rb')
        self.sensor_calibs = pickle.load(pkl_file, encoding='latin1')

    def get_image_features(self, sample_idx, idx):
        feature_list = []
        # root_path = '/mnt/jupyterhub/mrb/plus/pc_label_trainval/L4E_data_only_livox_front/training'
        root_path = self.root_path
        # front_camera_img_name = self.camera_lidar_timestamp[sample_idx+'.bin']['/front_left_camera/image_color/compressed']
        front_camera_img_name = self.camera_img_name[idx]
        front_camera_img_name = str(
            self.root_split_path / 'front_left_camera_feature' / front_camera_img_name.replace('.jpg', '_'))

        for i in range(1):
            featrue_i = front_camera_img_name + str(i) + '.npy'

            # if (os.path.exists(featrue_i)):
            feature_list.append(featrue_i)
            # else:
            #     print(featrue_i, ' not exist!!!!!!!!!!!!11')

        featrues = {'front_left_camera': feature_list}
        return featrues

    def get_image_names(self, idx):
        image_names = {}
        for camera_name in self.camera_names:
            image_names[camera_name] = self.cameras_img_name[camera_name][idx]
        return image_names

    def get_sensor_calib(self, sample_idx):

        bag_name = sample_idx.split('.')[-1]
        return self.sensor_calibs[bag_name]

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        if not self.sample_id_list:
            return []
        self.load_images_and_calibs()

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            image_info = {'image_idx': sample_idx, 'image_shape': np.array([1920, 1080])}
            info['image'] = image_info

            calib_info = {'P2': np.eye(4), 'R0_rect': np.eye(4), 'Tr_velo_to_cam': np.eye(4)}
            info['calib'] = calib_info

            idx = int(sample_idx.split('.')[0])
            info['camera_feature'] = self.get_image_features(sample_idx, idx)
            info['image_names'] = self.get_image_names(sample_idx, idx)
            info['sensor_calib'] = self.get_sensor_calib(sample_idx)
            if has_label:
                obj_labels = self.get_label(sample_idx)
                lidar_pts = self.get_lidar(sample_idx)
                obj_labels = self.filter_object_by_rule(lidar_pts, obj_labels)  # list
                annotations = {}
                # Fuse some categories
                anno_names = []
                for label in obj_labels:
                    if label['name'] in ['Car', 'car', 'police_car']:
                        anno_names.append('Car')
                    elif label['name'] in ['Truck', 'bus', 'truck', 'Engineering_vehicles', 'trailer']:
                        anno_names.append('Truck')
                    elif label['name'] in ['Pedestrian', 'pedestrians', 'wheelchair', 'stroller']:
                        anno_names.append('Pedestrian')
                    elif label['name'] in ['Cyclist', 'bicycle', 'motorcycle', 'Portable_personal_motor_vehicle']:
                        anno_names.append('Cyclist')
                    else:
                        anno_names.append(label['name'])
                annotations['name'] = np.array(anno_names)
                # annotations['name'] = np.array([label['name'] for label in obj_labels])
                annotations['truncated'] = np.array([0 for label in obj_labels])
                annotations['occluded'] = np.array([0 for label in obj_labels])
                annotations['alpha'] = np.array([0 for label in obj_labels])
                annotations['bbox'] = np.array([[1, 1, 1, 1] for label in obj_labels])
                annotations['dimensions'] = np.array([label['box3d_lidar'][3:6] for label in obj_labels],
                                                     dtype=np.float)  # lwh(lidar) format
                annotations['location'] = np.array([label['box3d_lidar'][0:3] for label in obj_labels], dtype=np.float)
                annotations['rotation_y'] = np.array([label['box3d_lidar'][6] for label in obj_labels], dtype=np.float)
                annotations['score'] = np.array([1 for label in obj_labels])
                annotations['difficulty'] = np.array([0 for label in obj_labels], np.int32)

                num_objects = len([label['name'] for label in obj_labels if label['name'] != 'DontCare'])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                if num_objects == 0:
                    return None
                else:
                    loc = annotations['location'][:num_objects]
                    dims = annotations['dimensions'][:num_objects]
                    rots = annotations['rotation_y'][:num_objects]
                loc_lidar = loc
                l, w, h = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                # loc_lidar[:, 2] += h[:, 0] / 2
                # gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, rots[..., np.newaxis]], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar
                info['annos'] = annotations

                if count_inside_pts:
                    annotations['num_points_in_gt'] = np.array([label['num_points_in_gt'] for label in obj_labels])

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        valid_infos = []
        if sample_id_list:
            with futures.ThreadPoolExecutor(1) as executor:
                infos = executor.map(process_single_scene, sample_id_list)
            for info in infos:
                if info is not None:
                    valid_infos.append(info)
        return valid_infos

    def get_mm3d_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures
        # NOTE(swc): read from (self.split + '.txt')
        if not self.sample_id_list:
            return []
        self.load_images_and_calibs()

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            idx = int(sample_idx.split('.')[0])

            info = {}
            sensor_calib = self.get_sensor_calib(sample_idx)
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            # NOTE(swc): dict(camera0:path0, camera1:path1, ...)
            camreas_image_path = self.get_image_names(idx)
            image_info = {}
            calib_info = {}
            image_info['image_idx'] = sample_idx
            # NOTE(swc): all camera infos
            for camera_name in self.camera_names:
                camera_img = cv2.imread(str(self.root_split_path / camera_name / camreas_image_path[camera_name]))
                img_shape = camera_img.shape
                image_info[camera_name] = {}
                image_info[camera_name]['image_path'] = str(Path(camera_name) / camreas_image_path[camera_name])
                image_info[camera_name]['image_shape'] = img_shape

                if img_shape[0] == 1080:
                    cam_calib = sensor_calib[camera_name + '_fullres']
                else:
                    cam_calib = sensor_calib[camera_name]

                calib_info[camera_name] = {}
                calib_info[camera_name]['P2'] = cam_calib['P_4x4']
                calib_info[camera_name]['R0_rect'] = cam_calib['Tr_imu_to_cam']
                calib_info[camera_name]['Tr_velo_to_cam'] = np.eye(4)

            info['image'] = image_info
            info['calib'] = calib_info

            if has_label:
                obj_labels = self.get_label(sample_idx)
                lidar_pts = self.get_lidar(sample_idx)
                obj_labels = self.filter_object_by_rule(lidar_pts, obj_labels)
                annotations = {}
                # Fuse some categories
                anno_names = []
                for label in obj_labels:
                    if label['name'] in ['car', 'police_car']:
                        anno_names.append('Car')
                    elif label['name'] in ['bus', 'truck', 'Engineering_vehicles', 'trailer']:
                        anno_names.append('Truck')
                    elif label['name'] in ['pedestrians', 'wheelchair', 'stroller']:
                        anno_names.append('Pedestrian')
                    elif label['name'] in ['bicycle', 'motorcycle', 'Portable_personal_motor_vehicle']:
                        anno_names.append('Cyclist')
                    else:
                        anno_names.append(label['name'])
                annotations['name'] = np.array(anno_names)
                annotations['truncated'] = np.array([0 for label in obj_labels])
                annotations['occluded'] = np.array([0 for label in obj_labels])
                annotations['alpha'] = np.array([0 for label in obj_labels])
                annotations['bbox'] = np.array([[1, 1, 1, 1] for label in obj_labels])
                annotations['dimensions'] = np.array([label['box3d_lidar'][3:6] for label in obj_labels],
                                                     dtype=np.float)  # lwh(lidar) format
                annotations['location'] = np.array([label['box3d_lidar'][0:3] for label in obj_labels], dtype=np.float)
                annotations['rotation_y'] = np.array([label['box3d_lidar'][6] for label in obj_labels], dtype=np.float)
                annotations['score'] = np.array([1 for label in obj_labels])
                annotations['difficulty'] = np.array([0 for label in obj_labels], np.int32)

                num_objects = len([label['name'] for label in obj_labels if label['name'] != 'DontCare'])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                if num_objects == 0:
                    return None
                else:
                    loc = annotations['location'][:num_objects]
                    dims = annotations['dimensions'][:num_objects]
                    rots = annotations['rotation_y'][:num_objects]
                loc_lidar = loc
                l, w, h = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                # loc_lidar[:, 2] += h[:, 0] / 2
                # gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, rots[..., np.newaxis]], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar
                info['annos'] = annotations

                if count_inside_pts:
                    annotations['num_points_in_gt'] = np.array([label['num_points_in_gt'] for label in obj_labels])

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        valid_infos = []
        if sample_id_list:
            with futures.ThreadPoolExecutor(1) as executor:
                infos = executor.map(process_single_scene, sample_id_list)
            for info in infos:
                if info is not None:
                    valid_infos.append(info)
        return valid_infos

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('Kitti_L4_data_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        print("Generating ground truth database ...")
        for k in tqdm(range(len(infos))):
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)


def clean_labels(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = L4Dataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path)
    dataset.prepare_labels()


def create_L4_data_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = L4Dataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, mode='val')
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('L4_data_infos_%s.pkl' % train_split)
    val_filename = save_path / ('L4_data_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'L4_data_infos_trainval.pkl'
    test_filename = save_path / 'L4_data_infos_test.pkl'

    print('---------------Start to generate data infos---------------')
    # from ..utils.extract_bags import BagExtractor
    dataset.set_split(train_split)
    L4_data_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=False)
    with open(train_filename, 'wb') as f:
        pickle.dump(L4_data_infos_train, f)
    print('L4 info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    L4_data_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=False)
    with open(val_filename, 'wb') as f:
        pickle.dump(L4_data_infos_val, f)
    print('L4 info val file is saved to %s' % val_filename)

    with open(trainval_filename, 'wb') as f:
        pickle.dump(L4_data_infos_train + L4_data_infos_val, f)
    print('L4 info trainval file is saved to %s' % trainval_filename)

    dataset.set_split('test')
    L4_data_infos_test = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=False)
    with open(test_filename, 'wb') as f:
        pickle.dump(L4_data_infos_test, f)
    print('L4 info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


def create_mm3d_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = L4Dataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, mode='val', used_camera_names=dataset_cfg.CAMERA_NMAES)
    train_split, val_split = 'train', 'val'
    # L4/L4E
    train_filename = save_path / ('Kitti_L4_data_mm3d_infos_%s.pkl' % train_split)
    val_filename = save_path / ('Kitti_L4_data_mm3d_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'Kitti_L4_data_mm3d_infos_trainval.pkl'
    test_filename = save_path / 'Kitti_L4_data_mm3d_infos_test.pkl'

    print('---------------Start to generate data infos---------------')
    dataset.set_split(train_split)
    mm3d_data_infos_train = dataset.get_mm3d_infos(num_workers=workers, has_label=True, count_inside_pts=False)
    with open(train_filename, 'wb') as f:
        pickle.dump(mm3d_data_infos_train, f)
    print('info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    mm3d_data_infos_val = dataset.get_mm3d_infos(num_workers=workers, has_label=True, count_inside_pts=False)
    with open(val_filename, 'wb') as f:
        pickle.dump(mm3d_data_infos_val, f)
    print('info val file is saved to %s' % val_filename)

    with open(trainval_filename, 'wb') as f:
        pickle.dump(mm3d_data_infos_train + mm3d_data_infos_val, f)
    print('info trainval file is saved to %s' % trainval_filename)

    dataset.set_split('test')
    mm3d_data_infos_test = dataset.get_mm3d_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    with open(test_filename, 'wb') as f:
        pickle.dump(mm3d_data_infos_test, f)
    print('info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


def create_bag_data_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = L4Dataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, mode='test', used_camera_names=dataset_cfg.CAMERA_NMAES)

    test_filename = save_path / 'L4_data_infos_test.pkl'

    dataset.set_split('test')
    L4_data_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    with open(test_filename, 'wb') as f:
        pickle.dump(L4_data_infos_test, f)
    print('L4 info test file is saved to %s' % test_filename)


if __name__ == '__main__':
    print(sys.argv)
    # pcdet format
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_L4_data_infos':
        dataset_cfg = EasyDict(yaml.load(open(sys.argv[2])))
        create_L4_data_infos(
            class_names=dataset_cfg.VEHICLE_CLASS_NAMES,
            data_path=Path(dataset_cfg.DATA_PATH),
            save_path=Path(dataset_cfg.DATA_PATH)
        )
    # mm3d format
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_L4_data_mm3d_infos':
        dataset_cfg = EasyDict(yaml.load(open(sys.argv[2])))
        subfolder = sys.argv[3]
        data_path = dataset_cfg.DATA_PATH
        old_tail = data_path.split('/')[-1]
        data_path = data_path.replace(old_tail, subfolder)
        create_mm3d_infos(
            class_names=dataset_cfg.ALL_CLASS_NAMES,
            data_path=Path(data_path),
            save_path=Path(data_path)
        )
        
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_bag_mm3d_infos':
        dataset_cfg = EasyDict(yaml.load(open(sys.argv[2])))
        data_path = dataset_cfg.BAG_DATA_PATH
        create_mm3d_infos(
            class_names=dataset_cfg.ALL_CLASS_NAMES, 
            data_path=Path(data_path),
            save_path=Path(data_path)
        )
    
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_bag_data_infos':
        dataset_cfg = EasyDict(yaml.load(open(sys.argv[2])))
        data_path = sys.argv[3]
        dataset_cfg.DATA_PATH = data_path
        create_bag_data_infos(
            class_names=dataset_cfg.ALL_CLASS_NAMES,
            data_path=Path(dataset_cfg.DATA_PATH),
            save_path=Path(dataset_cfg.DATA_PATH)
        )
    elif sys.argv.__len__() > 1 and sys.argv[1] == 'clean_labels':
        dataset_cfg = EasyDict(yaml.load(open(sys.argv[2])))
        clean_labels(
            class_names=dataset_cfg.ALL_CLASS_NAMES,
            data_path=Path(dataset_cfg.DATA_PATH),
            save_path=Path(dataset_cfg.DATA_PATH)
        )
