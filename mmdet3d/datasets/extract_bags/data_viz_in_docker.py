from time import time
import cv2
import numpy as np

camera_names = [
    'camera_blind',
    'front_left_camera',  # 1
    'front_right_camera',  # 2
    'side_left_camera',  # 3
    'side_right_camera',  # 4
    'rear_left_camera',  # 5
    'rear_right_camera'  # 6
]

camera_color = np.asarray([[0, 255, 255],  # defalut
                           [255, 0, 0],
                           [255, 255, 0],
                           [0, 255, 0],
                           [255, 0, 255],
                           [0, 0, 255],
                           [255, 255, 255]
                           ], dtype=np.uint8)


# def get_camera_fov_color(lidar_camera_idx):
#     lidar_color = map(lambda x: camera_color[x], lidar_camera_idx)
#     lidar_color = list(lidar_color)
#     lidar_color = np.asarray(lidar_color, dtype=np.uint8)
#     return lidar_color


def draw_boxes_on_canvas(canvas, boxes, bev_range, scores=None, label_strings=None, resolution=0.1, color=(0, 255, 0)):
    box_centers = np.copy(boxes[:, 0:2])
    box_centers[:, 0] -= bev_range[0]
    box_centers[:, 1] -= bev_range[1]
    box_centers /= resolution
    height = canvas.shape[0]
    width = canvas.shape[1]
    box_centers[:, 0] = height - box_centers[:, 0]
    box_centers[:, 1] = width - box_centers[:, 1]
    for idx, box in enumerate(boxes):
        box2d = get_corners_2d(box)
        box2d[:, 0] -= bev_range[0]
        box2d[:, 1] -= bev_range[1]
        # in canvas coordinates
        p1 = (width - int(box2d[0, 1] / resolution), height - int(box2d[0, 0] / resolution))
        p2 = (width - int(box2d[1, 1] / resolution), height - int(box2d[1, 0] / resolution))
        p3 = (width - int(box2d[2, 1] / resolution), height - int(box2d[2, 0] / resolution))
        p4 = (width - int(box2d[3, 1] / resolution), height - int(box2d[3, 0] / resolution))
        # Plot box
        cv2.line(canvas, p1, p2, color, 2)
        cv2.line(canvas, p2, p3, color, 2)
        cv2.line(canvas, p3, p4, color, 2)
        cv2.line(canvas, p4, p1, color, 2)
        # Plot heading
        heading_points = rot_line_90(p1, p2)  # bit of a hack: draw heading as just the front edge rotated by 90 degrees
        # opency internal type stuff
        heading_points = (
            (int(heading_points[0][0]), int(heading_points[0][1])),
            (int(heading_points[1][0]), int(heading_points[1][1])))
        cv2.line(canvas, heading_points[0],
                 heading_points[1], color, thickness=2)

        text = ""
        if label_strings is not None:
            text += label_strings[idx]
        if scores is not None:
            if label_strings is not None:
                text += ", "
            text += "%.2f" % scores[idx]
        if text:
            # note y and x are switched, due to difference in convention between imu and images
            origin = (int(box_centers[idx][1]), int(box_centers[idx][0]))
            cv2.putText(canvas, text, origin, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.6, color=color, thickness=1)


def plot_boxes(points, boxes, bev_range, scores=None, label_strings=None, path=None):
    """ Setup a canvas covering bev_range draw boxes on it, and save it to path.

    :param points: lidar points, [N, 3]
    :param boxes: boxes, [N, [x, y, z, l, w, h, r]]
    :param bev_range: bev range, [x_min, y_min, z_min, x_max, y_max, z_max]
    :param scores: [N]
    :return: None
    """

    # Configure the resolution
    resolution = 0.1
    # Initialize the plotting canvas
    pixels_x = int((bev_range[3] - bev_range[0]) / resolution) + 1
    pixels_y = int((bev_range[4] - bev_range[1]) / resolution) + 1
    canvas = np.zeros((pixels_x, pixels_y, 3), np.uint8)
    canvas.fill(0)

    # Plot the point cloud
    loc_x = ((points[:, 0] - bev_range[0]) / resolution).astype(int)
    loc_y = ((points[:, 1] - bev_range[1]) / resolution).astype(int)
    canvas[loc_x, loc_y] = [0, 255, 255]

    # Rotate the canvas to correct direction
    # canvas = cv2.rotate(canvas, cv2.cv2.ROTATE_90_CLOCKWISE)
    canvas = cv2.flip(canvas, 0)
    canvas = cv2.flip(canvas, 1)

    # Plot the gt boxes
    gt_color = (0, 255, 0)
    draw_boxes_on_canvas(canvas, boxes, bev_range, scores=scores, label_strings=label_strings, resolution=resolution,
                         color=gt_color)

    cv2.putText(canvas, "Green: Ground Truth", (10, 35), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6, color=gt_color, thickness=1)

    cv2.imwrite(path, canvas)


def plot_gt_boxes(points, gt_boxes, bev_range, lidar_camera_idx, name=None):
    """ Visualize the ground truth boxes.
    :param points: lidar points, [N, 3]
    :param gt_boxes: gt boxes, [N, [x, y, z, l, w, h, r]]
    :param bev_range: bev range, [x_min, y_min, z_min, x_max, y_max, z_max]
    lidar_camera_idx: shape: (lidar_pt_nums,) for camera fov
    :return: None
    """
    # NOTE(swc): bev here
    # Configure the resolution
    steps = 0.1
    # Initialize the plotting canvas
    pixels_x = int((bev_range[3] - bev_range[0]) / steps) + 1
    pixels_y = int((bev_range[4] - bev_range[1]) / steps) + 1
    canvas = np.zeros((pixels_x, pixels_y, 3), np.uint8)
    canvas.fill(0)

    x_range = np.array([-25, -10, 10, 25, 50, 75, 100, 125])  # m
    y1_range = np.ones(len(x_range)) * bev_range[1]
    y2_range = np.ones(len(x_range)) * bev_range[4]
    x_range = ((x_range - bev_range[0]) / steps).astype(int)
    y1_range = ((y1_range - bev_range[1]) / steps).astype(int)
    y2_range = ((y2_range - bev_range[1]) / steps).astype(int)
    for x, y1, y2 in zip(x_range, y1_range, y2_range):
        cv2.line(canvas, (y1, x), (y2, x), (255, 255, 255), 2)

    point_colors = camera_color[lidar_camera_idx]

    # Plot the point cloud
    loc_x = ((points[:, 0] - bev_range[0]) / steps).astype(int)
    loc_x = np.clip(loc_x, 0, pixels_x - 1)
    loc_y = ((points[:, 1] - bev_range[1]) / steps).astype(int)
    loc_y = np.clip(loc_y, 0, pixels_y - 1)
    canvas[loc_x, loc_y] = [0, 255, 255]
    canvas[loc_x, loc_y] = point_colors

    # for idx in range(points.shape[0]):
    #     time0 = time()
    #     point = points[idx, :]
    #     time1 = time()
    #     print("checkpoint1:", time1 - time0)
    #     if bev_range[0] <= point[0] <= bev_range[3] and \
    #        bev_range[1] <= point[1] <= bev_range[4] and \
    #        bev_range[2] <= point[2] <= bev_range[5]:
    #         time2 = time()
    #         print("checkpoint2:", time2 - time1)
    #         loc_x = int((point[0] - bev_range[0]) / steps)
    #         loc_y = int((point[1] - bev_range[1]) / steps)
    #         time3 = time()
    #         print("checkpoint3:", time3 - time2)
    #         canvas[loc_x, loc_y] = [0, 255, 255]
    #         time4 = time()
    #         print("checkpoint4:", time4 - time3)

    # Plot the gt boxes
    gt_color = (0, 255, 0)
    for box in gt_boxes:
        box2d = get_corners_2d(box)
        box2d[:, 0] -= bev_range[0]
        box2d[:, 1] -= bev_range[1]
        # Plot box
        cv2.line(canvas, (int(box2d[0, 1] / steps), int(box2d[0, 0] / steps)),
                 (int(box2d[1, 1] / steps), int(box2d[1, 0] / steps)), gt_color, 3)
        cv2.line(canvas, (int(box2d[1, 1] / steps), int(box2d[1, 0] / steps)),
                 (int(box2d[2, 1] / steps), int(box2d[2, 0] / steps)), gt_color, 3)
        cv2.line(canvas, (int(box2d[2, 1] / steps), int(box2d[2, 0] / steps)),
                 (int(box2d[3, 1] / steps), int(box2d[3, 0] / steps)), gt_color, 3)
        cv2.line(canvas, (int(box2d[3, 1] / steps), int(box2d[3, 0] / steps)),
                 (int(box2d[0, 1] / steps), int(box2d[0, 0] / steps)), gt_color, 3)
        # Plot heading
        heading_points = rot_line_90(box2d[0], box2d[1])
        cv2.line(canvas, (int(heading_points[0, 1] / steps), int(heading_points[0, 0] / steps)),
                 (int(heading_points[1, 1] / steps), int(heading_points[1, 0] / steps)), gt_color, 3)

    # Rotate the canvas to correct direction
    # canvas = cv2.rotate(canvas, cv2.cv2.ROTATE_90_CLOCKWISE)
    canvas = cv2.flip(canvas, 0)
    canvas = cv2.flip(canvas, 1)

    cv2.putText(canvas, "Green: Ground Truth", (15, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0, color=gt_color, thickness=2)

    for idx in range(len(camera_names)):
        v = 40 * (idx + 2)
        # print(camera_color[idx])
        idx_color = tuple(int(x) for x in camera_color[idx])
        cv2.putText(canvas, camera_names[idx], (15, v), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0, color=idx_color, thickness=2)

    # cv2.namedWindow("gt_box", 2)
    # cv2.imshow("gt_box", canvas)
    # cv2.waitKey(10)

    # print(name)
    cv2.imwrite("%s.jpg" % name, canvas)
    return canvas


def plot_gt_det_cmp(points, gt_boxes, det_boxes, bev_range, path=None):
    """Visualize all gt boxes and det boxes for comparison.

    :param points: lidar points, [N, 3]
    :param gt_boxes: gt boxes, [N, [x, y, z, l, w, h, r]]
    :param det_boxes: det boxes, [N, [x, y, z, l, w, h, r]]
    :param bev_range: bev range, [x_min, y_min, z_min, x_max, y_max, z_max]
    :return:
    """

    # Configure the resolution
    resolution = 0.1
    # Initialize the plotting canvas
    pixels_x = int((bev_range[3] - bev_range[0]) / resolution) + 2
    pixels_y = int((bev_range[4] - bev_range[1]) / resolution) + 2
    canvas = np.zeros((pixels_x, pixels_y, 3), np.uint8)
    canvas.fill(0)

    # Plot the point cloud
    loc_x = ((points[:, 0] - bev_range[0]) / resolution).astype(int)
    loc_y = ((points[:, 1] - bev_range[1]) / resolution).astype(int)
    canvas[loc_x, loc_y] = [0, 255, 255]
    # for idx in range(points.shape[0]):
    #     point = points[idx, :]
    #     if bev_range[0] <= point[0] <= bev_range[3] and \
    #        bev_range[1] <= point[1] <= bev_range[4] and \
    #        bev_range[2] <= point[2] <= bev_range[5]:
    #         loc_x = int((point[0] - bev_range[0]) / resolution)
    #         loc_y = int((point[1] - bev_range[1]) / resolution)
    #         canvas[loc_x, loc_y] = [0, 255, 255]

    # Rotate the canvas to correct direction
    canvas = cv2.flip(canvas, 0)
    canvas = cv2.flip(canvas, 1)

    # Plot the gt boxes
    gt_color = (0, 255, 0)
    det_color = (0, 0, 255)  # BGR
    draw_boxes_on_canvas(canvas, gt_boxes, bev_range, resolution=resolution, color=gt_color)
    draw_boxes_on_canvas(canvas, det_boxes, bev_range, resolution=resolution, color=det_color)

    cv2.putText(canvas, "Green: Ground Truth", (10, 35), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=gt_color, thickness=1)
    cv2.putText(canvas, "Red: Detection", (10, 75), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=det_color, thickness=1)

    cv2.imwrite(path, canvas)


def rotz(t):
    """Rotation about the z-axis.

    :param t: rotation angle
    :return: rotation matrix
    """

    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s],
                     [s, c]])


def rot_line_90(point1, point2):
    """Rotate a line around its center for 90 degree in BEV plane.

    :param point1: End point1, [x, y]
    :param point2: End point2, [x, y]
    :return: rot_line
    """

    center_x = (point1[0] + point2[0]) / 2
    center_y = (point1[1] + point2[1]) / 2
    rot_point1 = np.dot(rotz(np.pi / 2), [point1[0] - center_x, point1[1] - center_y])
    rot_point2 = np.dot(rotz(np.pi / 2), [point2[0] - center_x, point2[1] - center_y])
    rot_point1 += [center_x, center_y]
    rot_point2 += [center_x, center_y]

    return np.array([rot_point1, rot_point2])


def get_corners_2d(box):
    """Takes an bounding box and calculate the 2D corners in BEV plane.

    0 --- 1
    |     |        x
    |     |        ^
    |     |        |
    3 --- 2  y <---o

    :param box: 3D bounding box, [x, y, z, l, w, h, r]
    :return: corners_2d: (4,2) array in left image coord.
    """

    # compute rotational matrix around yaw axis
    rz = box[6]
    R = rotz(rz)

    # 2d bounding box dimensions
    l = box[3]
    w = box[4]

    # 2d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_2d = np.dot(R, np.vstack([x_corners, y_corners]))
    corners_2d[0, :] = corners_2d[0, :] + box[0]
    corners_2d[1, :] = corners_2d[1, :] + box[1]

    return corners_2d.T


if __name__ == "__main__":
    bv_range = [0, -25.0, 50.0, 25.0]
    random_points = np.random.random((1000, 3)) * 50
    random_points[:, 1] -= 25.0
    gt_boxes = np.array([[10.0, 1.0, 0.0, 4.1, 1.7, 1.5, 0],
                         [3.0, -5.0, 0.0, 3.8, 1.67, 1.4, np.pi / 4],
                         [20.0, 13.0, 0.0, 8.1, 2.7, 4.5, np.pi / 2]])
    plot_boxes(random_points, gt_boxes, bv_range)

    # ================================================================
    det_boxes = np.array([[10.2, 1.05, 0.0, 4.15, 1.68, 1.5, 0.01],
                          [3.05, -5.04, 0.0, 3.84, 1.69, 1.4, np.pi / 4 - 0.04],
                          [20.3, 13.2, 0.0, 8.3, 2.69, 4.5, np.pi / 2 + 0.08]])
    plot_gt_det_cmp(random_points, gt_boxes, det_boxes, bv_range)
