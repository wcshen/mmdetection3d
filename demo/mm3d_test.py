import numpy as np
import open3d as o3d

import cv2

def o3d_test():
    bin_file = '/home/wancheng.shen/mmdetection3d/demo/data/kitti/kitti_000008.bin'
    pcd_data = np.fromfile(bin_file).reshape(-1, 4)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pcd_data[:,:3])
    o3d.visualization.draw_geometries([point_cloud])

def cv_test():
    img = np.zeros(shape=(500, 500, 3), dtype=np.uint8)
    cv2.imshow("1", img)
    cv2.waitKey(0)

if __name__ == '__main__':
    # o3d_test()
    cv_test()