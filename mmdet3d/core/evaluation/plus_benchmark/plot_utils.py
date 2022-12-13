import cv2
import numpy as np


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


def rotz(t):
    """Rotation about the z-axis.

    :param t: rotation angle
    :return: rotation matrix
    """

    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s],
                     [s, c]])
    

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


def draw_boxes_on_canvas(canvas, boxes, bev_range, scores=None, label_strings=None, resolution=0.1, color=(0, 255, 0), bev_title=""):
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
        
        # dt
        if text and scores is not None:
            # note y and x are switched, due to difference in convention between imu and images
            origin = (int(box_centers[idx][1]), int(box_centers[idx][0]))
            cv2.putText(canvas, text, origin, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.6, color=color, thickness=1)
        # gt
        elif text:
            # TODO(swc): 10
            origin = (int(box_centers[idx][1]), int(box_centers[idx][0]) + 10)
            cv2.putText(canvas, text, origin, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.6, color=color, thickness=1)
        if bev_title:
            cv2.putText(canvas, bev_title, (10, 115), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=color, thickness=1)
            
def plot_gt_dt_pcd(points, bev_range, gt, dt, fp_idx, fn_idx, title):
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

        # Rotate the canvas to correct direction
        canvas = cv2.flip(canvas, 0)
        canvas = cv2.flip(canvas, 1)

        # Plot the gt boxes
        gt_color = (0, 255, 0)
        det_color = (0, 0, 255)  # BGR
        
        all_gt_boxes = gt['gt_boxes']
        all_gt_names = gt['name']
        
        all_dt_boxes = dt['dt_boxes']
        all_dt_scores = dt['score']
        all_dt_names = dt['name']
        
        # plot all gt_boxes and dt_boxes
        alpha = 0.5
        canvas_gt_all = canvas.copy()
        draw_boxes_on_canvas(canvas_gt_all, all_gt_boxes, bev_range,label_strings=all_gt_names, resolution=resolution, color=gt_color, bev_title=title)
        canvas = cv2.addWeighted(canvas_gt_all, alpha, canvas, 1-alpha, 0)
        
        canvas_dt_all = canvas.copy()
        draw_boxes_on_canvas(canvas_dt_all, all_dt_boxes, bev_range, scores=all_dt_scores, label_strings=all_dt_names, resolution=resolution, color=det_color, bev_title=title)
        canvas = cv2.addWeighted(canvas_dt_all, alpha, canvas, 1-alpha, 0)
        
        # plot fp and fn
        gt_fn_boxes = all_gt_boxes[fn_idx]
        gt_fn_names = all_gt_names[fn_idx]
        draw_boxes_on_canvas(canvas, gt_fn_boxes, bev_range,label_strings=gt_fn_names, resolution=resolution, color=gt_color, bev_title=title)
        
        gt_fp_boxes = all_dt_boxes[fp_idx]
        gt_fp_names = all_dt_names[fp_idx]
        gt_fp_scores = all_dt_scores[fp_idx]
        draw_boxes_on_canvas(canvas, gt_fp_boxes, bev_range,label_strings=gt_fp_names,scores=gt_fp_scores, resolution=resolution, color=det_color, bev_title=title)

        cv2.putText(canvas, "Green: Ground Truth", (10, 35), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=gt_color, thickness=1)
        cv2.putText(canvas, "Red: Detection", (10, 75), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=det_color, thickness=1)
        
        return canvas