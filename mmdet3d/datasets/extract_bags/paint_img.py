import copy
import math
import numpy as np
import cv2

box_colors = [[0, 255, 0],  # car
              [255, 0, 0],  # truck
              [255, 255, 0],  # pedestrian
              [0, 0, 255]]  # cyclist


def color_map(intensity):
    intensity /= 255.0
    color_nums = 5
    color_step = 1.0 / color_nums
    idx = math.floor(intensity * color_nums)
    bias = (intensity - color_step * idx) / color_step

    if intensity == 1.0:
        idx = 4
    if idx == 0:
        r = 0
        g = int(bias * 255)
        b = 255
    if idx == 1:
        r = 0
        g = 255
        b = int((1 - bias) * 255)
    if idx == 2:
        r = int(bias * 255)
        g = 255
        b = 0
    if idx == 3:
        r = 255
        g = int((1 - bias) * 255)
        b = 0
    if idx == 4:
        r = 255
        g = 0
        b = int(bias * 255)
    return np.asarray([r, g, b])


def get_3d_box(box_size, heading_angle, center):
    ''' Calculate 3D bounding box corners from its parameterization.

    Input:
        box_size: tuple of (l,w,h)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''

    def rotz(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, s, 0],
                         [-s, c, 0],
                         [0, 0, 1]])

    R = rotz(heading_angle)
    l, w, h = box_size
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    z_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = np.transpose(corners_3d)
    return corners_3d


def draw_boxes(box, gt_box_color, img):
    us, vs = box
    corners_2d = []
    height = img.shape[0]
    width = img.shape[1]
    valid_corners_num = 0
    for u, v in zip(us, vs):
        corners_2d.append((u, v))
        if (u >= 0) & (u < width) & (v >= 0) & (v < height):
            valid_corners_num += 1
    if valid_corners_num < 4:
        return img
    up_plane = [0, 1, 2, 3, 0]
    blew_plane = [4, 5, 6, 7, 4]

    thickness = 3

    for i in range(1, 5):
        up_p1 = corners_2d[up_plane[i - 1]]
        up_p2 = corners_2d[up_plane[i]]

        blew_p1 = corners_2d[blew_plane[i - 1]]
        blew_p2 = corners_2d[blew_plane[i]]

        cv2.line(img, pt1=up_p1, pt2=up_p2, color=gt_box_color, thickness=thickness)
        cv2.line(img, pt1=blew_p1, pt2=blew_p2, color=gt_box_color, thickness=thickness)

    pt1 = corners_2d[0]
    pt5 = corners_2d[4]
    up2down = [pt1, pt5]
    cv2.line(img, pt1=up2down[0], pt2=up2down[1], color=gt_box_color, thickness=thickness)
    pt2 = corners_2d[1]
    pt6 = corners_2d[5]
    up2down = [pt2, pt6]
    cv2.line(img, pt1=up2down[0], pt2=up2down[1], color=gt_box_color, thickness=thickness)
    pt4 = corners_2d[3]
    pt8 = corners_2d[7]
    up2down = [pt4, pt8]
    cv2.line(img, pt1=up2down[0], pt2=up2down[1], color=gt_box_color, thickness=thickness)
    pt3 = corners_2d[2]
    pt7 = corners_2d[6]
    up2down = [pt3, pt7]
    cv2.line(img, pt1=up2down[0], pt2=up2down[1], color=gt_box_color, thickness=thickness)
    return img


def draw_corners(box, img):
    us, vs = box
    height = img.shape[0]
    width = img.shape[1]

    for u, v in zip(us, vs):
        if (u >= 0) & (u < width) & (v >= 0) & (v < height):
            cv2.circle(img, center=(u, v), radius=2, color=(0, 255, 0), thickness=2)

    return img


def get_lidar_colors(lidar_intensity):
    # lidar_intensity = lidar_intensity % 255
    pt_colors = map(lambda x: color_map(x), lidar_intensity)
    return pt_colors


def get_paint_image(img_raw, lidar_raw, intrc, imu_to_cam, camera_idx, lidar_camera_idx, boxes):
    paint_img = copy.deepcopy(img_raw)
    lidar_pts = copy.deepcopy(lidar_raw)

    Tr_imu_to_pixel = np.dot(intrc, imu_to_cam)
    # NOTE(swc): colored by the intensity
    color_map_ = np.frompyfunc(color_map, 1, 1)
    lidar_colors = color_map_(lidar_pts[:, 3])
    lidar_pts[:, 3] = 1
    lidar_pts = np.matmul(Tr_imu_to_pixel, lidar_pts.T).T

    us = lidar_pts[:, 0] / lidar_pts[:, 2]
    vs = lidar_pts[:, 1] / lidar_pts[:, 2]
    us = us.astype(int)
    vs = vs.astype(int)

    height = paint_img.shape[0]
    width = paint_img.shape[1]
    true_where_point_on_img = (us >= 0) & (us < width) \
                              & (vs >= 0) & (vs < height) & (lidar_pts[:, 2] > 0)
    # NOTE(swc): get the camera idx of every pcd point +=1 to
    #  avoid the conflict between default color and the first cam color
    lidar_camera_idx[true_where_point_on_img] = camera_idx + 1

    img4painting = paint_img.copy()
    # NOTE(swc): set points transparency
    alpha = 0.7
    for u, v, on_img, c in zip(us, vs, true_where_point_on_img, lidar_colors):
        if not on_img:
            continue
        p = (u, v)
        cv2.circle(img4painting, p, 1, c, 2)
    paint_img = cv2.addWeighted(img4painting, alpha, paint_img, 1 - alpha, 0)
    # NOTE(swc): draw 3d boxes
    # np.array([loc_x, loc_y, loc_z, dim_l, dim_w, dim_h, rot_z])
    if boxes is not None:
        ones = np.ones(shape=(8, 1))
        gt_boxes_3d_pixel = []
        gt_boxes_colors = []
        gt_boxes_3d = []
        for box in boxes:
            box_size = (box[3], box[4], box[5])
            heading_angle = box[6]
            center_3d = (box[0], box[1], box[2])
            box_3d = get_3d_box(box_size, -heading_angle, center_3d)  # shape (8,3)
            box_3d = np.concatenate([box_3d, ones], axis=1)

            box_3d_pixel = np.matmul(Tr_imu_to_pixel, box_3d.T).T

            have_negative_z = False
            for i in range(8):
                camera_z = box_3d_pixel[i][2]
                if camera_z < 0:
                    have_negative_z = True
                    break
            if have_negative_z:
                continue

            box_color = box_colors[int(box[-1])]
            gt_boxes_colors.append(box_color)

            center_3d = np.asarray([box[0], box[1], box[2], 1])
            center_3d = np.reshape(center_3d, newshape=(1, 4))
            center_3d = np.matmul(Tr_imu_to_pixel, center_3d.T).T
            # NOTE(swc): filter out the boxes without the camera fov
            if center_3d[0, 2] < 0:
                continue

            us = box_3d_pixel[:, 0] / box_3d_pixel[:, 2]
            vs = box_3d_pixel[:, 1] / box_3d_pixel[:, 2]
            us = us.astype(int)
            vs = vs.astype(int)
            gt_boxes_3d_pixel.append([us, vs])
            gt_boxes_3d.append(box_3d)

        for gt_box_3d, gt_box_color in zip(gt_boxes_3d_pixel, gt_boxes_colors):
            paint_img = draw_boxes(gt_box_3d, gt_box_color, paint_img)

    return paint_img, lidar_camera_idx


if __name__ == '__main__':
    lidar_pts = np.random.random((200, 4))
    lidar_colors = get_lidar_colors(lidar_pts[:, 3])
    print(lidar_colors.shape)
