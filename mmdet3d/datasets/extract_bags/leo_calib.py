import os
from re import S
import yaml
import cv2
import numpy as np
# import euler
import math
# import enum
from collections import OrderedDict


class UnsortableList(list):
    def sort(self, *args, **kwargs):
        pass


class UnsortableOrderedDict(OrderedDict):
    def items(self, *args, **kwargs):
        return UnsortableList(OrderedDict.items(self, *args, **kwargs))


yaml.add_representer(
    UnsortableOrderedDict, yaml.representer.SafeRepresenter.represent_dict
)


def opencv_matrix_representer(dumper, mat):
    mat = np.atleast_2d(mat)
    mapping = UnsortableOrderedDict(
        [
            ("rows", mat.shape[0]),
            ("cols", mat.shape[1]),
            ("dt", "d"),
            ("data", mat.reshape(-1).tolist()),
        ]
    )
    return dumper.represent_mapping("tag:yaml.org,2002:opencv-matrix", mapping)


yaml.add_representer(np.ndarray, opencv_matrix_representer)


def read_yaml(yaml_path):
    # OpenCV writes in YAML 1.0 format, but PyYAML uses YAML 1.1 format
    yaml_raw = open(yaml_path).read()
    yaml_fix = yaml_raw.replace("%YAML:1.0", "%YAML 1.0").replace(
        " !!opencv-matrix", ""
    )
    return yaml.safe_load(yaml_fix)


def opencv_array(yaml_data, name):
    raw = yaml_data[name]
    return np.array(raw["data"]).reshape(raw["rows"], raw["cols"])


def load_calib(yaml_path):
    calib_obj = None
    calib_file = read_yaml(yaml_path)
    calib_type = calib_file["type"]
    if "camera" in calib_type and "stereo" not in calib_type:
        calib_obj = Mono(yaml_path)
    elif "stereo_camera" in calib_type:
        calib_obj = Stereo(yaml_path)
    elif "lidar" in calib_type and "sidelidar" not in calib_type:
        calib_obj = Lidar(yaml_path)
    elif "sidelidar" in calib_type:
        calib_obj = SideLidar(yaml_path)
    elif "radar" in calib_type:
        calib_obj = Radar(yaml_path)
    else:
        print("{} is not a valid calib_objration file".format(yaml_path))
    return calib_obj


class Calibration:
    def __init__(self, yaml_path):
        calib = read_yaml(yaml_path)
        self.yaml_path = yaml_path
        for key in ("car", "sensor_name", "date", "type"):
            setattr(self, key, calib[key])


class Stereo(Calibration):
    def __init__(self, yaml_path):
        Calibration.__init__(self, yaml_path)
        stereo_calib = read_yaml(yaml_path)
        for key in ("height", "width", "upside_down"):
            setattr(self, key, int(stereo_calib[key]))
        self.imgsize = (self.width, self.height)
        for key in (
                "R1",
                "M1",
                "P1",
                "D1",
                "R2",
                "M2",
                "P2",
                "D2",
                "Q",
                "Tr_cam_to_imu",
                "R",
                "T",
        ):
            setattr(self, key, opencv_array(stereo_calib, key))
        # use the self params
        self.reset_params()

    def undistort(self, left_img, right_img=None):
        undist_left_img = cv2.undistort(
            left_img, self.M1, self.D1, newCameraMatrix=self.P1
        )
        P2 = self.P2
        P2[0, 3] = 0.0
        undist_right_img = cv2.undistort(
            right_img, self.M2, self.D2, newCameraMatrix=self.P2
        )
        return undist_left_img, undist_right_img

    def rectify(self, left_img, right_img=None):
        if right_img is None:
            return cv2.remap(left_img, self.left_map1, self.left_map2, cv2.INTER_CUBIC)
        else:
            return (
                cv2.remap(left_img, self.left_map1, self.left_map2, cv2.INTER_CUBIC),
                cv2.remap(right_img, self.right_map1, self.right_map2, cv2.INTER_CUBIC),
            )

    def rectify_left(self, left_img):
        return cv2.remap(left_img, self.left_map1, self.left_map2, cv2.INTER_CUBIC)

    def rectify_right(self, right_img):
        return cv2.remap(right_img, self.right_map1, self.right_map2, cv2.INTER_CUBIC)

    def write_yaml(self, output):
        stereo_calib = UnsortableOrderedDict(
            [
                ("type", self.type),
                ("car", self.car),
                ("date", self.date),
                ("sensor_name", self.sensor_name),
                ("upside_down", self.upside_down),
                ("height", self.height),
                ("width", self.width),
                ("R", self.R),
                ("T", self.T),
                ("Q", self.Q),
                ("R1", self.R1),
                ("P1", self.P1),
                ("M1", self.M1),
                ("D1", self.D1),
                ("R2", self.R2),
                ("M2", self.M2),
                ("P2", self.P2),
                ("D2", self.D2),
                ("Tr_cam_to_imu", self.Tr_cam_to_imu),
            ]
        )

        with open(output, "w") as f:
            f.write("%YAML:1.0\n---\n")
            yaml.dump(stereo_calib, f)

    def reset_params(self):
        self.imgsize = (self.width, self.height)
        # put here as a reminder
        # [R T] = Tr_ref_cam_to_sec_cam
        # R1 = Tr_ref_cam_to_rectify_cam
        # R2 = Tr_sec_cam_to_rectify_cam
        # R = inv(R2)*R1

        self.Qinv = np.linalg.inv(self.Q)
        self.Tr_imu_to_cam = np.linalg.inv(self.Tr_cam_to_imu)
        self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(
            self.M1, self.D1, self.R1, self.P1, self.imgsize, cv2.CV_32FC1
        )
        self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(
            self.M2, self.D2, self.R2, self.P2, self.imgsize, cv2.CV_32FC1
        )
        self.fov_d = np.array(
            [
                2.0 * np.arctan2(self.width / 2.0 / self.M1[0, 0], 1) * 180.0 / math.pi,
                2.0
                * np.arctan2(self.height / 2.0 / self.M1[1, 1], 1)
                * 180.0
                / math.pi,
            ]
        )

        self.colmap_camera_params = {
            "model_id": 3,
            "width": str(self.width),
            "height": str(self.height),
            "params": np.array(
                [
                    self.M1[0, 0],
                    self.M1[0, 2],
                    self.M1[1, 2],
                    self.D1.ravel()[0],
                    self.D2.ravel()[1],
                ]
            ),
        }
        self.colmap_left_camera_params = {
            "model_id": 3,
            "width": str(self.width),
            "height": str(self.height),
            "params": np.array(
                [
                    self.M1[0, 0],
                    self.M1[0, 2],
                    self.M1[1, 2],
                    self.D1.ravel()[0],
                    self.D1.ravel()[1],
                ]
            ),
        }
        self.colmap_right_camera_params = {
            "model_id": 3,
            "width": str(self.width),
            "height": str(self.height),
            "params": np.array(
                [
                    self.M2[0, 0],
                    self.M2[0, 2],
                    self.M2[1, 2],
                    self.D2.ravel()[0],
                    self.D2.ravel()[1],
                ]
            ),
        }


class Mono(Calibration):
    def __init__(self, yaml_path):
        Calibration.__init__(self, yaml_path)
        # car_name = bag_name.split('_')[1]
        # date = int(bag_name.split('_')[0].split('T')[0])
        # self.yaml_left_path, self.yaml_right_path = self.searchCalibFile(car_name, date)
        # assert 'left' in bag_name or 'right' in bag_name
        # if 'left' in bag_name:
        #     yaml_path = self.yaml_left_path
        # elif 'right' in bag_name:
        #     yaml_path = self.yaml_right_path
        mono_calib = read_yaml(yaml_path)
        # print("yaml_path:", yaml_path)
        # print("mono_calib:\n")
        # print(mono_calib)

        # for key in ("height", "width", "upside_down"):
        for key in ("height", "width", "rotate"):
            setattr(self, key, int(mono_calib[key]))
        for key in ("R", "P", "M", "D", "Tr_cam_to_imu"):
            setattr(self, key, opencv_array(mono_calib, key))

        self.reset_params()

    def searchCalibFile(car_name, date, yaml_root):
        yaml_side_left_cam = ""
        yaml_side_right_cam = ""
        dirs = os.listdir(yaml_root)
        latest = [0, 0]
        for file in dirs:
            if file.find('.yml') != -1 and file.find(car_name) != -1 and file.find('fullres') == -1:
                calib_date = int(file.split('_')[1])
                if file.find('rear_left_camera') != -1:
                    if calib_date < date:
                        if latest[0] < calib_date:
                            latest[0] = calib_date
                            yaml_side_left_cam = os.path.join(yaml_root, file)
                if file.find('rear_right_camera') != -1:
                    if calib_date < date:
                        if latest[1] < calib_date:
                            latest[1] = calib_date
                            yaml_side_right_cam = os.path.join(yaml_root, file)
        # print('yaml_side_left_cam:', yaml_side_left_cam)
        # print('yaml_side_right_cam:', yaml_side_right_cam)
        return yaml_side_left_cam, yaml_side_right_cam

    def rectify(self, img, flag_rotate=False):
        # if self.upside_down:
        # if hasattr(self, 'upside_down') and self.upside_down:
        #     img = cv2.flip(img, -1)
        # img = np.rot90(img)
        img = cv2.remap(img, self.map1, self.map2, cv2.INTER_CUBIC)
        # NOTE(swc): to rotate rear camera)
        if flag_rotate and hasattr(self, 'rotate'):
            if self.rotate == 0:  # left
                img = np.rot90(img, -1)
            elif self.rotate == 2:  # right
                img = np.rot90(img)
            else:
                pass
                # print("rotation params wrong!should be 0 or 2, but get:", self.rotate)

        return img

    def reset_params(self):
        # self.upside_down = None
        self.imgsize = (self.width, self.height)
        self.Tr_imu_to_cam = np.linalg.inv(self.Tr_cam_to_imu)
        # P_4x4:
        # [  P_3x4 ]
        # [ 0 | 1  ]
        self.P_4x4 = np.zeros((4, 4), dtype=float)
        self.P_4x4[:3, :4] = self.P
        self.P_4x4[3, 3] = 1.0

        self.fov_d = np.array(
            [
                2.0 * np.arctan2(self.width / 2.0 / self.M[0, 0], 1) * 180.0 / math.pi,
                2.0 * np.arctan2(self.height / 2.0 / self.M[1, 1], 1) * 180.0 / math.pi,
            ]
        )
        # FIXME(swc): just for debug
        # D = np.array([-5.7913453585740038e-01, 2.7550574459516303e-01, 0., 0., 0.]).reshape(5, 1)

        # M = np.array([9.8616529615759885e+02, 0., 302., 0., 9.8242576769520190e+02,
        #               480., 0., 0., 1.]).reshape((3, 3))

        # P = np.array([9.2882531738281250e+02, 0., 3.0200000703948535e+02, 0., 0.,
        #               8.1932165527343750e+02, 4.7999998273749952e+02, 0., 0., 0., 1.,
        #               0.]).reshape((3, 4))

        if self.type == "fisheye_camera":
            self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
                self.M, self.D, self.R, self.P, self.imgsize, cv2.CV_32FC1
            )
            self.colmap_camera_params = {
                "model_id": 9,
                "width": str(self.width),
                "height": str(self.height),
                "params": np.array(
                    [
                        self.M[0, 0],
                        self.M[0, 2],
                        self.M[1, 2],
                        self.D.ravel()[0],
                        self.D.ravel()[1],
                    ]
                ),
            }
        else:
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                self.M, self.D, self.R, self.P, self.imgsize, cv2.CV_32FC1
            )
            self.colmap_camera_params = {
                "model_id": 3,
                "width": str(self.width),
                "height": str(self.height),
                "params": np.array(
                    [
                        self.M[0, 0],
                        self.M[0, 2],
                        self.M[1, 2],
                        self.D.ravel()[0],
                        self.D.ravel()[1],
                    ]
                ),
            }

    def write_yaml(self, output):
        mono_calib = UnsortableOrderedDict(
            [
                ("type", self.type),
                ("car", self.car),
                ("date", self.date),
                ("sensor_name", self.sensor_name),
                ("upside_down", self.upside_down),
                ("height", self.height),
                ("width", self.width),
                ("R", self.R),
                ("P", self.P),
                ("M", self.M),
                ("D", self.D),
                ("Tr_cam_to_imu", self.Tr_cam_to_imu),
            ]
        )

        with open(output, "w") as f:
            f.write("%YAML:1.0\n---\n")
            yaml.dump(mono_calib, f)


class Lidar(Calibration):
    def __init__(self, yaml_path):
        Calibration.__init__(self, yaml_path)
        lidar_calib = read_yaml(yaml_path)
        for key in ["Tr_lidar_to_imu"]:
            setattr(self, key, opencv_array(lidar_calib, key))
        for key in ["sensor_id"]:
            setattr(self, key, int(lidar_calib[key]))

    def write_yaml(self, output):
        lidar_calib = UnsortableOrderedDict(
            [
                ("type", self.type),
                ("car", self.car),
                ("date", self.date),
                ("sensor_name", self.sensor_name),
                ("sensor_id", self.sensor_id),
                ("Tr_lidar_to_imu", self.Tr_lidar_to_imu),
            ]
        )

        with open(output, "w") as f:
            f.write("%YAML:1.0\n---\n")
            yaml.dump(lidar_calib, f)


class SideLidar(Calibration):
    def __init__(self, yaml_path):
        Calibration.__init__(self, yaml_path)
        sidelidar_calib = read_yaml(yaml_path)
        for key in ["Tr_side_to_center", "Tr_lidar_to_imu"]:
            setattr(self, key, opencv_array(sidelidar_calib, key))
        for key in ["sensor_id"]:
            setattr(self, key, int(sidelidar_calib[key]))

    def write_yaml(self, output):
        sidelidar_calib = UnsortableOrderedDict(
            [
                ("type", self.type),
                ("car", self.car),
                ("date", self.date),
                ("sensor_name", self.sensor_name),
                ("sensor_id", self.sensor_id),
                ("Tr_side_to_center", self.Tr_side_to_center),
            ]
        )

        with open(output, "w") as f:
            f.write("%YAML:1.0\n---\n")
            yaml.dump(sidelidar_calib, f)


class Radar(Calibration):
    def __init__(self, yaml_path):
        Calibration.__init__(self, yaml_path)
        radar_calib = read_yaml(yaml_path)
        key = "Tr_radar_to_imu"
        setattr(self, key, opencv_array(radar_calib, key))

    def write_yaml(self, output):
        radar_calib = UnsortableOrderedDict(
            [
                ("type", self.type),
                ("car", self.car),
                ("date", self.date),
                ("sensor_name", self.sensor_name),
                ("Tr_radar_to_imu", self.Tr_radar_to_imu),
            ]
        )

        with open(output, "w") as f:
            f.write("%YAML:1.0\n---\n")
            yaml.dump(radar_calib, f)
