#!/usr/bin/env python
import glob
import fastbag
from fastbag.readers import Reader
import cv2
# from cv_bridge import CvBridge
import os
import argparse
from multiprocessing.pool import ThreadPool as Pool
from functools import partial
import yaml
import numpy as np
from leo_calib import Mono

# yaml_root = '/home/plusai/workspace/gpu6_local/data_process/common/sensor/calib_db'
# yaml_root = '/mnt/jupyterhub/mrb/plus/calib_db'

# TODO(swc): need to be changed
# yaml_root = '/mnt/jupyterhub/mrb/plus/road_test_calib_db'  # origin
yaml_root = '/home/plusai/Plus_2022/dataset/lidar/raw/road_test_calib_db'  # local_debug


# opencv yml cannot be loaded directly, has to:
# 1. ignore the first two line
# 2. registry constructor function for !!opencv_matrix
def opencv_matrix(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    mat = np.array(mapping["data"])
    mat.resize(mapping["rows"], mapping["cols"])
    return np.mat(mat)


yaml.add_constructor("tag:yaml.org,2002:opencv-matrix", opencv_matrix)


def load_opencv_yaml(yaml_path):
    with open(yaml_path) as f:
        # ignore first two line
        f.readline()  # %YAML:1.0
        f.readline()  # ---
        data = yaml.load(f)
    return data


class monoCalib(object):
    def __init__(self, bag_name):
        super(monoCalib, self).__init__()

        def searchCalibFile(car_name, date):
            yaml_side_left_cam = ""
            yaml_side_right_cam = ""
            yaml_front_left_cam = ""
            dirs = os.listdir(yaml_root)
            latest = [0, 0, 0]
            for file in dirs:
                if file.find('.yml') != -1 and file.find(car_name) != -1 and file.find('fullres') == -1:
                    calib_date = int(file.split('_')[1])
                    if file.find('side_left_camera') != -1:
                        if calib_date < date:
                            if latest[0] < calib_date:
                                latest[0] = calib_date
                                yaml_side_left_cam = os.path.join(yaml_root, file)
                    if file.find('side_right_camera') != -1:
                        if calib_date < date:
                            if latest[1] < calib_date:
                                latest[1] = calib_date
                                yaml_side_right_cam = os.path.join(yaml_root, file)
                    if file.find('front_left_camera') != -1:
                        if calib_date < date:
                            if latest[2] < calib_date:
                                latest[2] = calib_date
                                yaml_front_left_cam = os.path.join(yaml_root, file)
            print('------------------------------------')
            print('yaml_side_left_cam:', yaml_side_left_cam)
            print('yaml_side_right_cam:', yaml_side_right_cam)
            print('yaml_front_left_cam:', yaml_front_left_cam)
            print('------------------------------------')
            return yaml_side_left_cam, yaml_side_right_cam, yaml_front_left_cam

        def loadSideLeftCamConf(yaml_file):
            # f = open(yaml_file)
            # print(yaml_file)
            d = load_opencv_yaml(yaml_file)
            size = (d['width'], d['height'])
            # print('d:', d)
            M, D, R, P, Tr_cam_to_imu = [d[k] for k in ['M', 'D', 'R', 'P', 'Tr_cam_to_imu']]
            # print('m:', M)
            # print('Tr_cam_to_imu1111:', Tr_cam_to_imu)
            return size, P, R, M, D, Tr_cam_to_imu

        def loadSideRightCamConf(yaml_file):
            # f = open(yaml_file)
            # print(yaml_file)
            d = load_opencv_yaml(yaml_file)
            size = (d['width'], d['height'])
            M, D, R, P, Tr_cam_to_imu = [d[k] for k in ['M', 'D', 'R', 'P', 'Tr_cam_to_imu']]
            return size, P, R, M, D, Tr_cam_to_imu

        car_name = bag_name.split('_')[1]
        date = int(bag_name.split('_')[0].split('T')[0])
        print('car_name:', car_name)
        print('date:', date)
        self.yaml_side_left_cam, self.yaml_side_right_cam, self.yaml_front_left_cam = searchCalibFile(car_name, date)
        if self.yaml_side_left_cam == "" or self.yaml_side_right_cam == "":
            return
        self.side_left_size, self.side_left_P, self.side_left_R, self.side_left_M, self.side_left_D, self.side_left_Tr_cam_to_imu = \
            loadSideLeftCamConf(self.yaml_side_left_cam)
        self.side_right_size, self.side_right_P, self.side_right_R, self.side_right_M, self.side_right_D, self.side_right_Tr_cam_to_imu = \
            loadSideRightCamConf(self.yaml_side_right_cam)

    def getCalibParams(self, index):
        if index == 0:
            return self.side_left_size, self.side_left_P, self.side_left_R, self.side_left_M, self.side_left_D, self.side_left_Tr_cam_to_imu
        elif index == 1:
            return self.side_right_size, self.side_right_P, self.side_right_R, self.side_right_M, self.side_right_D, self.side_right_Tr_cam_to_imu

    def unwarp(self, image, index):
        h, w, c = image.shape
        origin_size = (w, h)
        size, P, R, M, D, _ = self.getCalibParams(index)
        if size != origin_size:
            image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(M, D, R, P, size, cv2.CV_16SC2)
        image_rectified = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
        return image_rectified

    def unwarp_front(self, image):
        # from calib import Mono
        calib = Mono(self.yaml_front_left_cam)
        image_rectified = calib.rectify(image)
        return image_rectified


def searchCalibFile(bag_name, camera_names, yaml_dir, fullres=False):
    car_name = bag_name.split('_')[1]
    date = int(bag_name.split('_')[0].split('T')[0])

    yaml_paths = [''] * len(camera_names)
    latest = [0] * len(camera_names)
    dirs = os.listdir(yaml_dir)
    # NOTE(swc): file -> camera
    for file in dirs:
        for i, camera_name in enumerate(camera_names):
            if fullres and file.find('fullres') == -1:
                continue
            if not fullres and file.find('fullres') != -1:
                continue

            if file.find('.yml') != -1 and file.find(car_name) != -1:
                if 'T' in file.split('_')[1]:
                    calib_date = int(file.split('_')[1].split('T')[0])
                else:
                    calib_date = int(file.split('_')[1])
                if file.find(camera_name) != -1:
                    if calib_date < date:
                        if latest[i] < calib_date:  # NOTE(swc): find out the latest calib file
                            latest[i] = calib_date
                            yaml_paths[i] = os.path.join(yaml_dir, file)
    return yaml_paths


def mkdir(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


def msgToPng(msg):
    bridge = CvBridge()

    timestamp = msg.header.stamp.to_sec()
    image = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")

    return image, timestamp


def extract_sample(bag_path, out_dir, topics, rectify, rotate, sep):
    sep = int(sep)
    if sep <= 0:
        sep = 1
    # endwish = bag_path.split('/')[-1].split('.')[-1]
    # data_dir = bag_path.split('/')[-3]
    # bag_path = bag_path.replace(endwish, 'db')
    # out_dirs = bag_path.replace('bag', '/traffic_light/plusai_image_2021-06-03')
    bag_name = bag_path.split('/')[-1]
    # out_dirs = out_dirs.replace(bag_name, '')
    # out_dir = '/home/plusai/data/test_bags_side_b8'
    # print(out_dir)
    mkdir(out_dir)
    # calib = monoCalib(bag_name)

    yaml_front_left_cam, yaml_rear_left_path, yaml_rear_right_path = searchCalibFile(bag_name=bag_name,
                                                                                     camera_names=['front_left_camera',
                                                                                                   'rear_left_camera',
                                                                                                   'rear_right_camera'])
    # yaml_front_left_cam = '/home/plusai/workspace/repos/common/sensor/calib_db/j7-l4e-c0003_LFWSRXSJ6M1F44913_20210924_front_left_camera.yml'
    # yaml_rear_left_path = '/home/plusai/workspace/repos/common/sensor/calib_db/j7-l4e-c0003_LFWSRXSJ6M1F44913_20210907_rear_left_camera.yml'
    # yaml_rear_right_path = '/home/plusai/workspace/repos/common/sensor/calib_db/j7-l4e-c0003_LFWSRXSJ6M1F44913_20210907_rear_right_camera.yml'

    count = -1
    with Reader(bag_path) as reader:
        for topic, msg, t in reader.iter_messages(topics=topics,
                                                  use_envelope_timestamp=True):
            count += 1
            if count % sep == 0:
                image, timestamp = msgToPng(msg)
                if rectify:
                    if topic.find('side_left_camera') != -1:
                        # image_rectified = calib.unwarp(image, 0)
                        image_rectified = image
                    elif topic.find('side_right_camera') != -1:
                        # image_rectified = calib.unwarp(image, 1)
                        image_rectified = image
                    elif topic.find('front_left_camera') != -1:
                        calib = Mono(yaml_front_left_cam)
                        image_rectified = calib.rectify(image)
                        # image_rectified = calib.unwarp_front(image)
                    elif topic.find('front_right_camera') != -1 or topic.find('front_center_camera') != -1:
                        image_rectified = image  # add by me
                    elif topic.find('rear_left_camera') != -1:
                        calib = Mono(yaml_rear_left_path)
                        image_rectified = calib.rectify(image, rotate)
                    elif topic.find('rear_right_camera') != -1:
                        calib = Mono(yaml_rear_right_path)
                        image_rectified = calib.rectify(image, rotate)
                    else:
                        print('WARNING:unsupported type of camera in rectify')
                    image = image_rectified

                topic_name = topic.split('/')[1]
                # img_path = out_dir + '/' + bag_name + '-' + topic_name + '-' + str(timestamp)+'.jpg'
                img_save_dir = os.path.join(out_dir, topic_name)
                if not os.path.exists(img_save_dir):
                    os.makedirs(img_save_dir)
                img_path = os.path.join(img_save_dir, str(timestamp) + '.jpg')
                cv2.imwrite(img_path, image)
            # if count > 100:
            #     break


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag_path', type=str, default="/home/plusai/Downloads/Bag/bags/20210831_j7-00010",
                        help='the labeled bag list')
    parser.add_argument('--out_dir', type=str, default="/home/plusai/Downloads/Bag/imgs/20210831_j7-00010_sep3",
                        help='output bags_dir')
    parser.add_argument('--rectify', type=bool, default=True)
    parser.add_argument('--rotate', type=bool, default=True)
    parser.add_argument('--sep', type=int, default=3)
    args = parser.parse_args()
    return args


def extract_img_from_bag(bag_path, save_root, sep):
    rectify = True
    rotate = True

    # topics = ['/rear_right_camera/image_color/compressed',
    #           '/rear_left_camera/image_color/compressed',
    #           '/front_left_camera/image_color/compressed']

    # topics = ['/front_left_camera/image_color/compressed',
    #           '/front_right_camera/image_color/compressed',
    #           '/front_center_camera/image_color/compressed']

    topics = ['/front_left_camera/image_color/compressed']

    if not os.path.exists(bag_path):
        print('The bag is not exist:', bag_path)
        return
    if not (bag_path.endswith('.db') or bag_path.endswith('.bag')):
        print('Not a bag file: ', bag_path)
        return

    bag_name = os.path.basename(bag_path)
    save_dir = os.path.join(save_root, bag_name)
    extract_sample(bag_path, save_dir, topics, rectify, rotate, sep)


if __name__ == '__main__':
    extract_img_from_bag(
        '/mnt/jupyterhub/chunming.wang/plus/lidar/raw/L4E_origin_benchmark/bags/20201113T154121_j7-l4e-00011_0_137to157.db', \
        '/home/rongbo.ma/bags_extract', 3)
