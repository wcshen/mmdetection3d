import os
from xmlrpc.client import FastMarshaller
import cv2
import copy
import math
import json
import fire
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path

import rosbag
import fastbag
from pluspy.calib import Calib
from pluspy import calib_utils
from sensor_msgs import point_cloud2
from pluspy.bag_utils import extract_bag_components

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from visulize import plot_gt_boxes
# import torch

import time

dataset_name = ''

type_hub = {
    0: "DONTCARE",
    1: "UNKNOWN",
    11: "CAR",
    12: "PEDESTRIAN",
    14: "BICYCLE",
    15: "VAN",
    16: "BUS",
    17: "TRUCK",
    18: "TRAM",
    19: "MOTO",
    20: "BARRIER",
    21: "CONE",
    23: "MOVABLE_SIGN",
    26: "LICENSE_PLATE",
    27: "SUV",
    28: "LIGHTTRUCK",
    29: "TRAILER",
    # deprecated types, don't use these. Use Attribute instead
    22: "HEAVY_EQUIPMENT",
    3: "UNKNOWN_STATIONARY",
    13: "PERSON_SITTING",
}

class LidarCollector(object):
    def __init__(self, topic_names, calib_names, calib, lidar_related_topic_num, collection_frequency=5):
        self.topic_names = topic_names
        self.calib_names = calib_names
        self.calib = calib
        self.lidar_related_topic_num = lidar_related_topic_num
        self.start_timestamp = 0.
        self.collected_timestamp = 0.
        # Timestamp interval larger than this value would trigger new collection  # NOTE(swc): in second
        self.max_interval = 0.05
        self.collection_frequency = collection_frequency  # Collect one time in ? frames
        self.frame_count = 1  # Skip first frame due to lost topic
        self.collected_topic = set()
        self.pts_list = []
        self.tele_pts = None
        self.is_collected = False
        self.collected_pts = None
        self.pts_with_ts = {}

    def push(self, topic, msg):
        if topic not in self.topic_names:
            return
        topic_timestamp = msg.header.stamp.to_sec()
        self.is_collected = False

        # When the timestamp of new message is larger than certain threshold,
        # deal with collected frame and start new collection
        if abs(topic_timestamp - self.start_timestamp) >= self.max_interval:
            
            if self.lidar_related_topic_num - len(self.collected_topic) <= 1 \
                    and len(self.collected_topic) > 0:
                self.is_collected = True
                self.collected_pts = np.concatenate(self.pts_list, axis=0)
                self.collected_timestamp = self.start_timestamp
                self.pts_with_ts[self.collected_timestamp] = [self.collected_pts, self.tele_pts]
                self.tele_pts = None
            self.start_timestamp = topic_timestamp
            self.collected_topic = {topic}
            if 'tele' in topic:
                self.tele_pts = self._extract_points(topic, msg)
                self.pts_list = []
            else:
                # NOTE(swc): _extract_points lidar -> IMU
                self.pts_list = [self._extract_points(topic, msg)]
            self.frame_count += 1
            return

        # Otherwise, keep collecting new message
        self.collected_topic.add(topic)
        if 'tele' in topic:
            self.tele_pts = self._extract_points(topic, msg)
        else:
            self.pts_list.append(self._extract_points(topic, msg))

    def get_tr_lidar_to_imu(self, topic):
        return self.calib.sensors[self.calib_names[topic]].Tr_lidar_to_imu

    def get_timestamp(self):
        return self.collected_timestamp

    def get_points(self):
        return self.collected_pts

    def reset(self):
        self.start_timestamp = 0.
        self.collected_topic = set()
        self.pts_list = []
    
    def get_points_with_ts(self):
        return self.pts_with_ts

    def _extract_points(self, topic, msg):
        # TODO(swc): transform lidar_point from ros_msg to list will be time consuming 
        lidar_pts = np.array(
            list(point_cloud2.read_points(msg, field_names=['x', 'y', 'z', 'intensity'], skip_nans=True)))
        if lidar_pts.dtype == 'int64':
            lidar_pts = lidar_pts.astype(np.float)
            lidar_pts[:, 0:3] = lidar_pts[:, 0:3] * 0.01
            print("extract new pointcloud!!!!! [x, y, z] /= 100")

        intensity = copy.deepcopy(lidar_pts[:, 3])
        # NOTE(swc): normalize intensity of ouster lidar from [0,2^16-1] to [0, 2^8-1]
        if 'os1' in topic:
            intensity = intensity / 9600.0 * 255.0
            intensity = np.clip(intensity, 0, 255)

        # perform transformation from sensor to IMU coordinates
        lidar_pts[:, 3] = 1
        lidar_pts = np.matmul(self.calib.sensors[self.calib_names[topic]].Tr_lidar_to_imu, lidar_pts.T).T
        lidar_pts[:, 3] = intensity

        return lidar_pts


class BagExtractor(object):
    def __init__(self, root_path, output_path, bev_range, calib_db, lidar_topics,used_cameras=None,
                 mod=1, mod_res=0, collection_frequency=5, training=True, has_label=True, extract_img=True, is_benchmark=False):
        self.root_path = root_path
        self.output_path = output_path
        self.bev_range = bev_range
        self.training = training
        self.frame_cnt = 0  # start_idx
        self.delta_timestamp_threshold = 0.005  # in second
        self.object_type_by_length_threshold = 6
        self.num_object_in_frame_threshold = 2
        self.length_stat_hist = np.zeros(25)

        self.box_size_stat_hist = {'length_stat': np.zeros(25),
                                   'width_stat': np.zeros(25),
                                   'height_stat': np.zeros(25)}

        self.calib_db = calib_db  # path
        self.lidar_topics = list(lidar_topics)
        self.mod = mod
        self.mod_res = mod_res  # for parallel process usage
        self.collection_frequency = collection_frequency
        if self.training:
            self.split = 'training'
        else:
            self.split = 'testing'

        self.save_cam_img_time_pkl = True
        self.bag_calib_matrixs = {}

        self.has_label = has_label
        if has_label:
            sub_dirs = ['pointcloud', 'label', 'gt_image']
        else:
            sub_dirs = ['pointcloud']

        self.extract_img = extract_img
        self.extract_camera_topics = list(camera_topics)
        for topic in camera_topics:
            sub_dirs.append(topic.split('/')[1])
            sub_dirs.append(topic.split('/')[1]+'_painted')

        sub_dirs.append('full_img')
        sub_dirs.append('tele_points')

        # NOTE(swc): make dirs here  // one lidar and several cameras
        for d in sub_dirs:
            dd = os.path.join(str(self.output_path), self.split, d)
            # print('create bags_dir',dd)
            if not os.path.exists(dd):
                os.makedirs(dd)

        self.visualize_data = True
        self.visualize_rate = 0.2  # Should between 0-1
        self.lidar_camera_timestamp = {}

        self.camera_names = used_cameras
        self.file_base_names = []
        self.one_all_painted_imgs = {}

        self.num_car = 0
        self.num_truck = 0
        self.num_ped = 0
        self.num_cyclist = 0

        self.all_topic_nums = dict()
        for cam_topic in camera_topics:
            self.all_topic_nums[cam_topic] = 0
        for lidar_topic in lidar_topics:
            self.all_topic_nums[lidar_topic] = 0

        self.min_delta_t = 0.05  # s
        self.lidar_cam_fp = 0
        self.lidar_cam_fp_dict = {}
        
        self.is_benchmark = is_benchmark

    def is_size_valid(self, label):
        box = label['box3d_lidar']
        length = box[3]
        width = box[4]
        height = box[5]
        if label['name'] == 'Car' and (length <= 2.5 or width <= 1.2 or width >= 3 or height <= 0.7):
            # print("[Warning] Invalid object size ignored:", label['name'], length, width, height)
            return False
        elif label['name'] == 'Truck' and (width <= 1.5 or width >= 5 or height <= 1):
            # print("[Warning] Invalid object size ignored:", label['name'], length, width, height)
            return False
        else:
            return True

    def filter_object_by_rule(self, labels):
        num_objects = len(labels)
        if num_objects == 0:
            return labels

        filtered_label_list = []
        for idx in range(num_objects):
            loc_x = labels[idx]['box3d_lidar'][0]
            loc_y = labels[idx]['box3d_lidar'][1]
            if self.bev_range[0] <= loc_x <= self.bev_range[2] \
                    and self.bev_range[1] <= loc_y <= self.bev_range[3] \
                    and labels[idx]['has_3d_label']:
                filtered_label_data = {'name': labels[idx]['name'], 'box3d_lidar': labels[idx]['box3d_lidar']}
                if 'type' in labels[idx]:
                    if labels[idx]['type']:
                        filtered_label_data['type'] = labels[idx]['type']
                filtered_label_list.append(filtered_label_data)

                if labels[idx]['name'] == 'Car':
                    self.num_car += 1
                elif labels[idx]['name'] == 'Truck':
                    self.num_truck += 1
                elif labels[idx]['name'] == 'Pedestrian':
                    self.num_ped += 1
                elif labels[idx]['name'] == 'Cyclist':
                    self.num_cyclist += 1
                else:
                    pass
                    print("[Warning] Unknown label:", labels[idx]['name'])

                length = filtered_label_data['box3d_lidar'][3]
                width = filtered_label_data['box3d_lidar'][4]
                height = filtered_label_data['box3d_lidar'][5]
                if math.floor(length) < self.length_stat_hist.shape[0]:
                    self.box_size_stat_hist['length_stat'][int(math.floor(length))] += 1
                else:
                    pass
                    print("[Warning] Oversized object:", filtered_label_data['box3d_lidar'])

                if math.floor(width) < self.length_stat_hist.shape[0]:
                    self.box_size_stat_hist['width_stat'][int(math.floor(width))] += 1
                else:
                    pass
                    print("[Warning] Oversized object:", filtered_label_data['box3d_lidar'])

                if math.floor(height) < self.length_stat_hist.shape[0]:
                    self.box_size_stat_hist['height_stat'][int(math.floor(height))] += 1
                else:
                    pass
                    print("[Warning] Oversized object:", filtered_label_data['box3d_lidar'])

        return filtered_label_list


    def type_map(self, type_name):
        if type_name in ['VAN','SUV','CAR']:
            label_name = 'Car'
        elif type_name in ['TRUCK','BUS','HEAVY_EQUIPMENT','LIGHTTRUCK','TRAILER']:
            label_name = 'Truck'
        elif type_name in ['PEDESTRIAN','PERSON_SITTING']:
            label_name = 'Pedestrian'
        elif type_name in ['MOTO','BICYCLE']:
            label_name = 'Cyclist'
        else:
            label_name = 'Dontcare'
        return label_name
    
    def parse_label(self, bag_name):
        label_file = str(self.root_path / dataset_name / 'labels' / (bag_name + '.json'))
        with open(label_file, 'r') as f:
            label_data = json.load(f, encoding='utf-8')

        parsed_labels = {}
        for obj in label_data['objects']:
            for obj_box in obj['bounds']:
                # find the box in closest timestamp
                box_timestamp = float(str(obj_box['timestamp']) + '.' + ('%09d' % obj_box['timestamp_nano']))
                loc_x = obj_box['position']['x']
                loc_y = obj_box['position']['y']
                loc_z = obj_box['position']['z']
                dim_l = obj['size']['x']
                dim_w = obj['size']['y']
                dim_h = obj['size']['z']

                if 'status_flags' in obj_box:
                    has_3d_label = True if not ('has_3d_label' in obj_box['status_flags']) else (
                        obj_box['status_flags']['has_3d_label'])
                else:
                    has_3d_label = True

                if obj_box['direction']['x'] == 0.:
                    obj_box['direction']['x'] += 0.000001
                rot_z = math.atan2(obj_box['direction']['y'], obj_box['direction']['x'])
                parsed_label_data = {'box3d_lidar': np.array([loc_x, loc_y, loc_z, dim_l, dim_w, dim_h, rot_z])}
                if dim_l <= 1 and dim_w <= 1:
                    # print("Pedestrian:", dim_l, dim_w, dim_h)
                    type_name = 'Pedestrian'
                elif 1 < dim_l <= 2 and dim_w <= 1 and dim_h > 1:
                    # print("Cyclist:", dim_l, dim_w, dim_h)
                    type_name = 'Cyclist'
                elif 3.5 <= dim_l < self.object_type_by_length_threshold:
                    type_name = 'Car'
                elif dim_l >= self.object_type_by_length_threshold:
                    type_name = 'Truck'
                else:
                    # print("[Warning] abnormal object size:", dim_l, dim_w, dim_h)
                    type_name = 'Car'
                
                if 'type' in obj:
                    parsed_label_data['type'] = self.type_map(type_hub[obj['type']])
                else:
                    parsed_label_data['type'] = None
                parsed_label_data['name'] = type_name
                parsed_label_data['has_3d_label'] = has_3d_label
                
                if box_timestamp not in parsed_labels:
                    parsed_labels[box_timestamp] = []
                parsed_labels[box_timestamp].append(parsed_label_data)
        # downsample
        time_order_keys = sorted(parsed_labels)
        output_keys = []
        time_skip = 0.45 # s
        last_time = 0
        for time_order_key in time_order_keys:
            if time_order_key - last_time >= time_skip:
                output_keys.append(time_order_key)
                last_time = time_order_key

        return parsed_labels, output_keys

    def find_nearest(self, a, a0):
        "Element in nd array `a` closest to the scalar value `a0`"
        delta_t = np.abs(a - a0)
        idx = delta_t.argmin()
        min_delta_t = delta_t[idx]
        if min_delta_t > self.min_delta_t:
            idx = -1
        # return a.flat[idx]
        return idx

    def get_camera_timestamps(self, bag_path, save_imgs=False):

        if self.extract_img:
            from cv_bridge import CvBridge
            bridge = CvBridge()

        cameras_ts = {}
        self.cameras_ts = {}
        self.image_path_names = {}
        self.images_cv2 = {}  # in docker; load every frame imgs to the memory
        try:
            with fastbag.Reader(bag_path) if fastbag.Reader.is_readable(bag_path) else rosbag.Bag(bag_path) as bag:
                for topic, msg, _ in bag.read_messages(topics=self.extract_camera_topics):
                    if topic not in cameras_ts:  # if new topic, init 
                        cameras_ts[topic] = []
                        self.image_path_names[topic] = []
                        self.images_cv2[topic] = {}
                    camera_name = topic.split('/')[1]
                    raw_img_path = str(self.output_path / self.split / (camera_name + '_raw_img'))
                    image_name = str(raw_img_path + '/' + ('%06d.%06f.%s.jpg' % (len(cameras_ts[topic]),
                                                                                 msg.header.stamp.to_sec(),
                                                                                 bag_path.split('/')[-1])))
                    if self.extract_img and not os.path.exists(image_name):
                        cv_image = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
                        self.images_cv2[topic].update({image_name.split('/')[-1]: cv_image})  # img_name -> cv_img
                        if save_imgs:
                            cv2.imwrite(image_name, cv_image)  # for local debug
                    cameras_ts[topic].append(msg.header.stamp.to_sec())
                    self.image_path_names[topic].append(image_name.split('/')[-1])
        except IOError:
            print("[Warning] couldn't open ", bag_path, ", skipping...")
            return
        for k in cameras_ts.keys():
            self.cameras_ts[k] = np.array(cameras_ts[k])
            # print(k, len(cameras_ts[k]))

    def cal_lidar_cam_timestamp(self, lidar_collector, label_ts, lidar_raw, file_base_name, raw_bag_name, boxes):
        cam_tms = {}
        
        lidar_pts_num = lidar_raw.shape[0]
        lidar_camera_idx = np.zeros(shape=(lidar_pts_num,), dtype=np.int)
        lidar_camera_idx.fill(0)
        # for each camera, find out the nearest to the lidar_time
        lidar_cam_time_aligned = True
        imgs4saving = {}
        for topic in self.cameras_ts.keys():
            camera_name = topic.split('/')[1]
            camera_idx = self.camera_names.index(camera_name)
            cam_tm_idx = self.find_nearest(self.cameras_ts[topic], label_ts)  # lidar -> cameras

            if cam_tm_idx == -1:
                lidar_cam_time_aligned = False
                break

            cam_tms[topic] = self.image_path_names[topic][cam_tm_idx]

            raw_img_path = str(self.output_path / self.split / camera_name)
            painted_camera_name = camera_name+'_painted'
            painted_img_path = str(self.output_path / self.split / painted_camera_name)

            if not os.path.exists(raw_img_path):
                os.makedirs(raw_img_path)

            t = cam_tms[topic].split('.')
            cam_time = t[1] + '.' + t[2]
            img_base_name = ('%06d.%s.%s' % (
                self.frame_cnt, cam_time, raw_bag_name))  # NOTE(swc): only bag_name without .bag
            raw_img_name = str(raw_img_path + '/' + ('%s.jpg' % img_base_name))
            painted_img_name = str(painted_img_path + '/' + ('%s.jpg' % img_base_name))

            if self.extract_img:
                raw_img = self.images_cv2[topic][cam_tms[topic]]
                width = raw_img.shape[0]
                height = raw_img.shape[1]
                # NOTE(swc): rectify image here
                if width == 1080:
                    image_rectified = self.leo_full_calib[camera_name].rectify(raw_img, flag_rotate=True)
                else:
                    image_rectified = self.leo_calib[camera_name].rectify(raw_img, flag_rotate=True)
            else:
                image_rectified = cv2.imread(raw_img_name)
            # NOTE(swc): paint_img
            paint_img, lidar_camera_idx = self.project_new(topic, image_rectified, lidar_collector, lidar_raw, camera_idx,
                                                           lidar_camera_idx, boxes)

            self.one_all_painted_imgs.update({camera_name: paint_img})

            imgs4saving[raw_img_name] = image_rectified
            imgs4saving[painted_img_name] = paint_img

        if lidar_cam_time_aligned and self.extract_img:
            for img_name, img in imgs4saving.items():
                cv2.imwrite(img_name, img)

        lidar_bin_name = str('%s.bin' % file_base_name)
        self.lidar_camera_timestamp[lidar_bin_name] = cam_tms
        return cam_tms, lidar_camera_idx, lidar_cam_time_aligned

    def get_sensor_calibs(self, bag_info, raw_bag_name):
        from leo_calib import Mono
        from leo_extract_img_from_bag_sh import searchCalibFile

        if bag_info is not None:
            try:
                # used to rectify_img
                cam_yamls = searchCalibFile(bag_name=raw_bag_name,
                                            camera_names=self.camera_names,
                                            yaml_dir=self.calib_db,
                                            fullres=False)

                full_cam_yamls = searchCalibFile(bag_name=raw_bag_name,
                                                 camera_names=self.camera_names,
                                                 yaml_dir=self.calib_db,
                                                 fullres=True)
                                
                self.leo_calib = {}
                self.leo_full_calib = {}
                for idx, camera_name in enumerate(self.camera_names):
                    self.leo_calib[camera_name] = Mono(cam_yamls[idx])
                    if full_cam_yamls[idx] != '':
                        self.leo_full_calib[camera_name] = Mono(full_cam_yamls[idx])

                # NOTE(swc): calib files of all vehicles in different time 
                # {vehicle1:{calib_type1:{time1_file_path,...},calib_type2:{time1_file_path,...}},..}
                all_calib_files = calib_utils.find_calibration_files(self.calib_db)

                if not all_calib_files:
                    raise RuntimeError("Found no calibration files in directory: %s" % self.calib_db)
                # Load all the vehicle sensors by the vehicle name
                sensors = all_calib_files[bag_info.vehicle].keys()

                load_calib_name = []
                for s in sensors:
                    if 'camera' in s or 'lidar' in s or 'radar' in s:
                        load_calib_name.append(s)

                calib = Calib.load(bag_info.vehicle, bag_info.date, calib_dir=self.calib_db, sensors=load_calib_name)
                calib_matrix = {}
                for topic, cal in calib.sensors.items():
                    if 'camera' in topic \
                            and hasattr(cal, 'P_4x4') and hasattr(cal, 'Tr_imu_to_cam'):
                        calib_matrix[topic] = {'P_4x4': cal.P_4x4, 'Tr_imu_to_cam': cal.Tr_imu_to_cam}
                # NOTE(swc): only camera
                self.bag_calib_matrixs[raw_bag_name] = calib_matrix  # used to save 'sensor_calibs.pkl' files
                self.sensor_calibs = calib_matrix  # didn't use
                return calib
            except Exception as e:
                print(raw_bag_name, e)
                exit()
                return None
        else:
            print('[Warning] got None bag info for bag ', raw_bag_name)
            exit()
            return None
    
    @staticmethod
    def align_pcd_labels(pcd_ts, label_ts):
        th = 0.005 # 5ms
        delta_t = np.abs(pcd_ts - label_ts)
        idx = delta_t.argmin()
        min_delta_t = delta_t[idx]
        if min_delta_t > th:
            res = -1
        else:
            res = pcd_ts[idx]
        return res

    def process_one_bag(self, bag_name, save_imgs):
        """

        Args:
            bag_name:

        Returns:

        """
        # load calibration info
        bag_path = str(self.root_path / dataset_name / 'bags' / bag_name)
        print('current_bag_path: ', bag_path)
        bag_info = extract_bag_components(bag_path)
        raw_bag_name = bag_name.replace(".bag", '').replace(".db", '')

        calib = self.get_sensor_calibs(bag_info, raw_bag_name)  # calib param of each camera, lidar and radar

        self.get_camera_timestamps(bag_path, save_imgs)  # get the cv_image and image_ts of all imgs in one bag
        try:
            with fastbag.Reader(bag_path) if fastbag.Reader.is_readable(bag_path) else rosbag.Bag(bag_path) as bag:
                # get pointcloud topic count
                # get intersection in given topic and bag topic, here not has unified point
                all_topics = bag.get_type_and_topic_info().topics.keys()
                pc_topic = filter(lambda x: x in self.lidar_topics, all_topics)
                pc_topic = list(pc_topic)
                
                bag_lidar_topic = filter(lambda x: 'lidar' in x, all_topics)
                bag_lidar_topic = list(bag_lidar_topic)

                filtered_topics = filter(lambda x: x in lidar_topics or x in camera_topics, all_topics)
                filtered_topics = list(filtered_topics)
                for topic in filtered_topics:
                    if not (topic in self.all_topic_nums):
                        self.all_topic_nums[topic] = 1
                    else:
                        self.all_topic_nums[topic] += 1

                if len(pc_topic) == 0:
                    print('topics ', self.lidar_topics, 'not in bag, check topic name!!!!!!!!!!')
                    print('bag_lidar_topic ', bag_lidar_topic)
                    return
                elif len(pc_topic) == 1 and 'tele' in pc_topic[0]:
                    return

                lidar_collector = LidarCollector(self.lidar_topics, calib_name, calib, len(pc_topic),
                                                 self.collection_frequency)

                obj_names = ['Car', 'Truck', 'Pedestrian', 'Cyclist']
                
                # collect pcds
                for topic, msg, temp in bag.read_messages(topics=pc_topic):
                    # Collect lidar data continuously
                    lidar_collector.push(topic, msg)
                # dict: [self.collected_timestamp] = [self.collected_pts, self.tele_pts]
                points_with_ts = lidar_collector.get_points_with_ts()
                pcd_ts_list = sorted(points_with_ts)
                pcd_ts_array = np.asarray(pcd_ts_list)
                
                # collect labels
                if self.has_label:
                    # parse corresponding label file
                    parsed_labels, downsampled_keys = self.parse_label(bag_name)
                    # align pcd and label
                    for label_ts in downsampled_keys:
                        aligned_pcd_ts = self.align_pcd_labels(pcd_ts_array, label_ts)
                        if aligned_pcd_ts == -1:
                            continue
                        lidar_pts = points_with_ts[aligned_pcd_ts][0]
                        tele_pts = points_with_ts[aligned_pcd_ts][1]
                        file_base_name = ('%06d.%06f.%s' % (self.frame_cnt, label_ts, raw_bag_name))
                        
                        cur_labels = parsed_labels[label_ts]
                        # pcd range filter
                        cur_labels = self.filter_object_by_rule(cur_labels)

                        # ignore frame with too few valid objects
                        if len(cur_labels) < self.num_object_in_frame_threshold and not self.is_benchmark:
                            continue

                        if not os.path.exists(
                                str(self.output_path / self.split / 'label' / ('%s.pkl' % file_base_name))):
                            with open(str(self.output_path / self.split / 'label' / ('%s.pkl' % file_base_name)),
                                        'wb') as f:
                                pickle.dump(cur_labels, f)
                        
                        boxes4vis = []
                        for cur_label in cur_labels:
                            obj_name_index = obj_names.index(cur_label['type'])
                            # obj_name_index: for vis color in the painted image
                            this_box = np.append(cur_label['box3d_lidar'], obj_name_index)
                            boxes4vis.append(this_box)
                        # np.array([loc_x, loc_y, loc_z, dim_l, dim_w, dim_h, rot_z])
                        boxes4vis = np.array(boxes4vis)
                        
                        if self.save_cam_img_time_pkl:
                            _, lidar_camera_idx, lidar_cam_time_aligned = self.cal_lidar_cam_timestamp(lidar_collector,label_ts, lidar_pts, file_base_name, raw_bag_name, boxes4vis)

                        if not lidar_cam_time_aligned:
                            self.lidar_cam_fp += 1
                            if bag_name in self.lidar_cam_fp_dict.keys():
                                self.lidar_cam_fp_dict[bag_name] += 1
                            else:
                                self.lidar_cam_fp_dict[bag_name] = 1
                            continue

                        # save pointcloud
                        if not os.path.exists(
                                str(self.output_path / self.split / 'pointcloud' / ('%s.bin' % file_base_name))):
                            with open(
                                    str(self.output_path / self.split / 'pointcloud' / ('%s.bin' % file_base_name)),
                                    'wb') as f:
                                lidar_pts.tofile(f)
                        if (tele_pts is not None) and not os.path.exists(
                                str(self.output_path / self.split / 'tele_points' / ('%s.bin' % file_base_name))):
                            with open(
                                    str(self.output_path / self.split / 'tele_points' / ('%s.bin' % file_base_name)),
                                    'wb') as f:
                                tele_pts.tofile(f)

                        # ============= DEBUG: plot box in pointcloud =============
                        if self.visualize_data and self.has_label:
                            # if (random.randint(0, 100) / 100) <= self.visualize_rate:
                            if True:
                                from data_viz_in_docker import plot_gt_boxes
                                image_name = str(
                                    self.output_path / self.split / 'gt_image' / str('%s' % file_base_name))
                                gt_img = plot_gt_boxes(lidar_pts,
                                                       boxes4vis,
                                                       [-100, -50, -2, 200, 50, 6],
                                                       lidar_camera_idx,
                                                       image_name)
                                self.one_all_painted_imgs.update({'gt_img': gt_img})
                                # NOTE(swc): stitching images here
                                self.stitching_all_imgs(file_base_name=file_base_name)
                        # ==========================================================
                        self.frame_cnt += 1
                else:
                    # save pcd and images etc.. every frames
                    for pcd_ts in pcd_ts_list:
                        lidar_pts = points_with_ts[pcd_ts][0]
                        tele_pts = points_with_ts[pcd_ts][1]
                        file_base_name = ('%06d.%06f.%s' % (self.frame_cnt, pcd_ts, raw_bag_name))
                        if self.save_cam_img_time_pkl:
                            _, lidar_camera_idx, lidar_cam_time_aligned = self.cal_lidar_cam_timestamp(lidar_collector, pcd_ts, lidar_pts, file_base_name, raw_bag_name, None)

                        if not lidar_cam_time_aligned:
                            self.lidar_cam_fp += 1
                            if bag_name in self.lidar_cam_fp_dict.keys():
                                self.lidar_cam_fp_dict[bag_name] += 1
                            else:
                                self.lidar_cam_fp_dict[bag_name] = 1
                            continue

                        # save pointcloud
                        if not os.path.exists(
                                str(self.output_path / self.split / 'pointcloud' / ('%s.bin' % file_base_name))):
                            with open(
                                    str(self.output_path / self.split / 'pointcloud' / ('%s.bin' % file_base_name)),
                                    'wb') as f:
                                lidar_pts.tofile(f)
                        if (tele_pts is not None) and not os.path.exists(
                                str(self.output_path / self.split / 'tele_points' / ('%s.bin' % file_base_name))):
                            with open(
                                    str(self.output_path / self.split / 'tele_points' / ('%s.bin' % file_base_name)),
                                    'wb') as f:
                                tele_pts.tofile(f)

        except IOError:
            print("[Warning] couldn't open ", bag_path, ", skipping...")
            return

    def save_cam_img_timestamp(self):
        if self.save_cam_img_time_pkl:
            name = str(self.output_path / self.split / 'cam_img_timestamp.pkl')
            output = open(name, 'wb')
            pickle.dump(self.lidar_camera_timestamp, output)

    def save_sensor_calibs(self):
        name = str(self.output_path / self.split / 'sensor_calibs.pkl')
        output = open(name, 'wb')
        pickle.dump(self.bag_calib_matrixs, output)

    def project_new(self, camera_topic, img, lidar_collector, lidar_raw, camera_idx, lidar_camera_idx, boxes):
        from paint_img import get_paint_image
        cols = img.shape[0]
        rows = img.shape[1]

        if cols == 1080:
            camera_calib = lidar_collector.calib.sensors[lidar_collector.calib_names[camera_topic] + '_fullres']
        else:
            camera_calib = lidar_collector.calib.sensors[lidar_collector.calib_names[camera_topic]]

        paint_img, lidar_camera_idx = get_paint_image(img, lidar_raw, camera_calib.P_4x4,
                                                      camera_calib.Tr_imu_to_cam, camera_idx, lidar_camera_idx, boxes)
        return paint_img, lidar_camera_idx


    def process_bags(self, save_img=False):
        print(self.root_path)
        bag_list = os.listdir(str(self.root_path / dataset_name / 'bags'))
        bag_list = sorted(bag_list)
        all_start_time = time.time()
        for bag_name in tqdm(bag_list):
            start_time = time.time()
            if bag_name in black_list:
                print("[Warning] Bag file %s is in blacklist, skipping ..." % bag_name)
                continue
            n = hash(bag_name) % self.mod
            if n not in self.mod_res:
                print("continue")
                continue
            # check whether bag and label both exist
            if Path(self.root_path / dataset_name / 'bags' / bag_name).exists() \
                    and Path(self.root_path / dataset_name / 'labels' / (bag_name + '.json')).exists():
                print('-------------------process bag -ing', bag_name)

                self.process_one_bag(bag_name, save_img)
            else:
                print("[Warning] Label file not found, skipping ...")
                continue
            end_time = time.time()
            cost_time = (end_time - start_time) / 60.0
            print("this bag cost time: {:.2f}min".format(cost_time))
            # exit()
            # break

        all_end_time = time.time()
        cost_time = (all_end_time - all_start_time) / 60.0
        print("this dataset cost time: {:.2f}min".format(cost_time))

        obj_statistics = {}
        obj_statistics['num_frames'] = self.frame_cnt
        obj_statistics['num_full'] = self.num_car + self.num_truck + self.num_ped + self.num_cyclist
        obj_statistics['num_car'] = self.num_car
        obj_statistics['num_truck'] = self.num_truck
        obj_statistics['num_ped'] = self.num_ped
        obj_statistics['num_cyclist'] = self.num_cyclist
        # obj_statistics['label_size']
        json_str = json.dumps(obj_statistics, indent=4)
        obj_statistics_name = str(self.output_path / self.split / 'obj_statistics.json')
        with open(obj_statistics_name, 'w') as json_file:
            json_file.write(json_str)

        json_str = json.dumps(self.all_topic_nums, indent=4)
        all_topic_nums_name = str(self.output_path / self.split / 'all_topic_nums.json')
        with open(all_topic_nums_name, 'w') as json_file:
            json_file.write(json_str)

        for key, value in self.box_size_stat_hist.items():
            self.box_size_stat_hist[key] = value.tolist()

        json_str = json.dumps(self.box_size_stat_hist, indent=4)
        box_size_stat_name = str(self.output_path / self.split / 'box_size_stat.json')
        with open(box_size_stat_name, 'w') as json_file:
            json_file.write(json_str)

        self.lidar_cam_fp_dict['all'] = self.lidar_cam_fp
        json_str = json.dumps(self.lidar_cam_fp_dict, indent=4)
        lidar_cam_fp_dict_name = str(self.output_path / self.split / 'lidar_cam_fp_dict.json')
        with open(lidar_cam_fp_dict_name, 'w') as json_file:
            json_file.write(json_str)

    def extract_and_save_bags_calibs(self):
        bag_list = os.listdir(str(self.root_path / dataset_name / 'bags'))
        # bag_list = debug_bags
        for bag_name in tqdm(bag_list):
            if bag_name in black_list:
                print("[Warning] Bag file %s is in blacklist, skipping ..." % bag_name)
                continue
            n = hash(bag_name) % self.mod
            if n not in self.mod_res:
                continue
            # check whether bag and label both exist
            if Path(self.root_path / dataset_name / 'bags' / bag_name).exists() \
                    and Path(self.root_path / dataset_name / 'labels' / (bag_name + '.json')).exists():
                print('-------------------extract bag calib', bag_name)
                bag_path = str(self.root_path / dataset_name / 'bags' / bag_name)
                bag_info = extract_bag_components(bag_path)
                raw_bag_name = bag_name.replace(".bag", '').replace(".db", '')

                self.get_sensor_calibs(bag_info, raw_bag_name)
            else:
                print("[Warning] Label file not found, skipping ...")
                continue

        self.save_sensor_calibs()

    def stitching_all_imgs(self, file_base_name):
        root_dir = str(os.path.join(str(self.output_path), self.split))

        gt_img = self.one_all_painted_imgs['gt_img'][:, :, ::-1]
        plt.figure(figsize=(15, 15))
        ax1 = plt.subplot2grid(shape=(3, 3), loc=(0, 1), rowspan=3)
        ax1.imshow(gt_img)
        ax1.set_title('gt_bev')
        plt.xticks([])
        plt.yticks([])
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        if 'front_left_camera' in self.one_all_painted_imgs.keys():
            front_left_img = self.one_all_painted_imgs['front_left_camera'][:, :, ::-1]
            ax2 = plt.subplot2grid(shape=(3, 3), loc=(0, 0), colspan=1)
            ax2.imshow(front_left_img)
            ax2.set_title('front_left_cam')
            plt.xticks([])
            plt.yticks([])
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        if 'front_right_camera' in self.one_all_painted_imgs.keys():
            front_right_img = self.one_all_painted_imgs['front_right_camera'][:, :, ::-1]
            ax3 = plt.subplot2grid(shape=(3, 3), loc=(0, 2), colspan=1)
            ax3.imshow(front_right_img)
            ax3.set_title('front_right_cam')
            plt.xticks([])
            plt.yticks([])
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        if 'side_left_camera' in self.one_all_painted_imgs.keys():    
            side_left_img = self.one_all_painted_imgs['side_left_camera'][:, :, ::-1]
            ax4 = plt.subplot2grid(shape=(3, 3), loc=(1, 0), colspan=1)
            ax4.imshow(side_left_img)
            ax4.set_title('side_left_cam')
            plt.xticks([])
            plt.yticks([])
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        if 'side_right_camera' in self.one_all_painted_imgs.keys():
            side_right_img = self.one_all_painted_imgs['side_right_camera'][:, :, ::-1]
            ax5 = plt.subplot2grid(shape=(3, 3), loc=(1, 2), colspan=1)
            ax5.imshow(side_right_img)
            ax5.set_title('side_right_cam')
            plt.xticks([])
            plt.yticks([])
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        if 'rear_left_camera' in self.one_all_painted_imgs.keys():
            rear_left_img = self.one_all_painted_imgs['rear_left_camera'][:, :, ::-1]
            ax6 = plt.subplot2grid(shape=(3, 3), loc=(2, 0), colspan=1)
            ax6.imshow(rear_left_img)
            ax6.set_title('rear_left_cam')
            plt.xticks([])
            plt.yticks([])
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        if 'rear_right_camera' in self.one_all_painted_imgs.keys():
            rear_right_img = self.one_all_painted_imgs['rear_right_camera'][:, :, ::-1]
            ax7 = plt.subplot2grid(shape=(3, 3), loc=(2, 2), colspan=2)
            ax7.imshow(rear_right_img)
            ax7.set_title('rear_right_cam')
            plt.xticks([])
            plt.yticks([])
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        plt.tight_layout()
        full_img_path = root_dir + '/full_img/' + file_base_name + '.jpg'
        # print(full_img_path)
        plt.savefig(full_img_path, bbox_inches='tight', pad_inches=0.02)
        plt.close('all')

class mains():
    def main(self,
             raw_bag_path='/home/wancheng.shen/datasets',
             out_path='/mnt/jupyterhub/swc/plus/pc_label_trainval',  # change this
             subfolders=['L4E_origin_data', 'L4E_origin_data'],  # change this
             calib_db='/mnt/jupyterhub/swc/plus/calib_db',
             unified=False,
             mod=1, mod_res=0, extract_img=True, extract_calibs=0):
        if type(mod_res) == int:
            mod_res = [int(a) for a in str(mod_res).split(',')]

        global lidar_topics, calib_name
        if unified:  # for US
            lidar_topics = list(filter(lambda x: x.find('unified') != -1, lidar_topics))
            calib_name = dict([(t, calib_name[t]) for t in lidar_topics])  # lidar_topic -> calib

        for subfolder in subfolders:
            is_benchmark = False
            print('\033[31mNow subfolder: {}\033[0m'.format(subfolder))
            root_path = Path(raw_bag_path + "/%s" % subfolder)
            print("root_path: {}".format(str(root_path)))
            if 'benchmark' in subfolder:
                is_benchmark = True
            output_path = Path(out_path + "/%s" % subfolder)
            bag_extractor = BagExtractor(root_path, output_path, bev_range, calib_db, lidar_topics, mod=mod, mod_res=mod_res, collection_frequency=5, training=True, has_label=True, extract_img=extract_img, used_cameras=used_camera_names, is_benchmark=is_benchmark)

            if int(extract_calibs):
                bag_extractor.extract_and_save_bags_calibs()
            else:
                bag_extractor.process_bags()
                bag_extractor.save_cam_img_timestamp()
                bag_extractor.save_sensor_calibs()
            print("\033[31m{} done!\033[0m".format(subfolder))
        

    def main_one_bag_end2end(self,
                             bag_path='/mnt/jupyterhub/mrb/plus/bags/20220426T153540_j7-l4e-LFWSRXSJ1M1F50506_3_443to453.db',
                             calib_db='/mnt/jupyterhub/mrb/plus/road_test_calib_db',
                             out_path='/home/rongbo.ma/dataset/lidar',
                             subfolder='',
                             unified=False,
                             mod=1, mod_res=0, extract_img=True):

        pos = bag_path.rfind('/')
        root_path = Path(bag_path[:pos - 4])
        print(bag_path.split('/')[-1])
        output_path = Path(os.path.join(out_path, bag_path.split('/')[-1]))
        bag_name = bag_path[pos + 1:]
        global lidar_topics
        if unified:  # for US
            lidar_topics = list(filter(lambda x: x.find('unified') != -1, lidar_topics))
            print(lidar_topics)

        bag_extractor = BagExtractor(root_path, output_path, bev_range, calib_db, lidar_topics, mod, mod_res, \
                                     collection_frequency=1, training=False, has_label=False, extract_img=extract_img)
        bag_extractor.process_one_bag(bag_name, True)
        bag_extractor.save_cam_img_timestamp()
        bag_extractor.save_sensor_calibs()


if __name__ == "__main__":
    black_list = ['20210419T172200_paccar-amazonph2-1XKYDP9X9LJ407494_k0002dm_15_0to20.db',
                  '20210419T172200_paccar-amazonph2-1XKYDP9X9LJ407494_k0002dm_17_0to20.db',
                  '20210419T184837_paccar-p005sc_22_60to80.bag',
                  '20210419T184837_paccar-p005sc_23_20to40.bag',
                  '20210419T184837_paccar-p005sc_23_80to100.bag',
                  '20210419T184837_paccar-p005sc_24_0to20.bag',
                  '20210419T184837_paccar-p005sc_26_40to60.bag']
    debug_bags = ['20210901T170211_j7-00010_12_43to63.db']
    bev_range = [-100, -50, 200, 50]
    start_idx = 0
    
    # CN_L4E
    # lidar_topics = ['/livox/lidar/horizon_front',
    #                 '/livox/lidar/tele_front',
    #                 '/livox/lidar/horizon_left',
    #                 '/livox/lidar/horizon_right',
    #                 '/livox/lidar']
    
    # US_L4
    # lidar_topics = ['/unified/lidar_points']

    # CN_L4
    lidar_topics = ['/livox/lidar/horizon_front',
                    '/livox/lidar/tele_front',
                    '/os1_left/points',
                    '/os1_right/points',
                    '/livox/lidar/horizon_left',
                    '/livox/lidar/horizon_right',
                    '/livox/lidar',
                    '/rslidar_points',
                    ]

    camera_topics = ['/side_left_camera/image_color/compressed',
                     '/side_right_camera/image_color/compressed',
                     '/front_left_camera/image_color/compressed',
                     '/front_right_camera/image_color/compressed',
                     '/rear_left_camera/image_color/compressed',
                     '/rear_right_camera/image_color/compressed',
                     ]

    calib_name = {'/livox/lidar/horizon_front': 'lidar',
                  '/livox/lidar': 'lidar',
                  '/rslidar_points': 'lidar',
                  '/livox/lidar/tele_front': 'tele_lidar',
                  '/os1_left/points': 'side_left_lidar',
                  '/os1_right/points': 'side_right_lidar',
                  '/unified/lidar_points': 'lidar',
                  '/livox/lidar/horizon_left': 'side_left_lidar',
                  '/livox/lidar/horizon_right': 'side_right_lidar',
                  '/front_left_camera/image_color/compressed': 'front_left_camera',
                  '/front_right_camera/image_color/compressed': 'front_right_camera',
                  '/side_left_camera/image_color/compressed': 'side_left_camera',
                  '/side_right_camera/image_color/compressed': 'side_right_camera',
                  '/rear_left_camera/image_color/compressed': 'rear_left_camera',
                  '/rear_right_camera/image_color/compressed': 'rear_right_camera',
                  }
    
    camera_calib_name = ['front_left_camera', 'front_right_camera', 'front_left_right_camera',
                         'side_left_camera', 'side_right_camera',
                         'rear_left_camera', 'rear_right_camera']
    
    # FIXME: should be changed
    used_camera_names = [
            'front_left_camera',  # 0
            'front_right_camera',  # 1
            'side_left_camera',  # 2
            'side_right_camera',  # 3
            'rear_left_camera',  # 4
            'rear_right_camera'  # 5
        ]
    fire.Fire(mains)
