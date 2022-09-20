import copy
import os
import tempfile
from os import path as osp
from tkinter.messagebox import NO
from unittest.mock import NonCallableMagicMock

import mmcv
import numpy as np
import torch
from mmcv.utils import print_log

from ..core import show_multi_modality_result, show_result
from ..core.bbox import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
                         LiDARInstance3DBoxes, points_cam2img)
from .builder import DATASETS
from .custom_3d import Custom3DDataset
from .pipelines import Compose

from .kitti_dataset import KittiDataset
from tensorboardX import SummaryWriter

@DATASETS.register_module()
class PlusKittiDataset(KittiDataset):
    """
    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        split (str): Split of input data.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        pcd_limit_range (list, optional): The range of point cloud used to
            filter invalid predicted boxes.
            Default: [0, -40, -3, 70.4, 40, 0.0].
    """
    CLASSES = ['Pedestrian', 'Cyclist', 'Car', 'Truck']

    def __init__(self,
                 data_root,
                 ann_file,
                 split,
                 pts_prefix='pointcloud',
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 load_interval=1,
                 pcd_limit_range=[0, -10, -3, 100.0, 10, 6.0],
                 **kwargs):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            split=split,
            pts_prefix=pts_prefix,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            pcd_limit_range=pcd_limit_range,
            **kwargs)
        # data_infos is read in the custom_3d.py->Custom3DDataset __init__ function
        # through load_annotations()

        # to load a subset, just set the load_interval in the dataset config
        self.data_infos = self.data_infos[::load_interval]
        if hasattr(self, 'flag'):
            self.flag = self.flag[::load_interval]

        self.camera_names = ['front_left_camera', 'front_right_camera',
                             'side_left_camera', 'side_right_camera',
                             'rear_left_camera', 'rear_right_camera']
        self.eval_cnt = 0

    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file, file_format='pkl')
        data_infos = list(sorted(data, key=lambda e: int(e['point_cloud']['lidar_idx'].split('.')[0])))
        return data_infos

    def _get_pts_filename(self, idx):
        pts_filename = osp.join(self.root_split, self.pts_prefix, f'{idx}.bin')
        return pts_filename

    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        gt_names = set(info['annos']['name'])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_bboxes (np.ndarray): 2D ground truth bboxes.
                - gt_labels (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
                - difficulty (int): Difficulty defined by KITTI.
                    0, 1, 2 represent xxxxx respectively.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]

        difficulty = info['annos']['difficulty']
        annos = info['annos']
        # we need other objects to avoid collision when sample
        annos = self.remove_dontcare(annos)
        loc = annos['location']
        dims = annos['dimensions']
        rots = annos['rotation_y']
        gt_names = annos['name']

        # in imu coordinates
        gt_bboxes_3d = annos['gt_boxes_lidar']
        gt_bboxes_3d = LiDARInstance3DBoxes(gt_bboxes_3d, origin=(0.5, 0.5, 0.5))
        gt_bboxes = annos['bbox']  # [[1,1,1,1],...,]

        selected = self.drop_arrays_by_name(gt_names, ['DontCare'])
        gt_bboxes = gt_bboxes[selected].astype('float32')
        gt_names = gt_names[selected]

        gt_labels = []
        for cat in gt_names:
            if cat in self.CLASSES:
                gt_labels.append(self.CLASSES.index(cat))
            else:
                gt_labels.append(-1)
        # the index in the CLASSES of every box_label, -1: out of the CLASSES
        gt_labels = np.array(gt_labels).astype(np.int64)
        gt_labels_3d = copy.deepcopy(gt_labels)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            bboxes=gt_bboxes,
            labels=gt_labels,
            gt_names=gt_names,
            plane=None,  # TODO(swc): is it need?
            difficulty=difficulty)
        return anns_results

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
        """
        info = self.data_infos[index]
        sample_idx = info['image']['image_idx']
        pts_filename = self._get_pts_filename(sample_idx)

        img_filenames = []
        lidar2img_list = []
        for camera_name in self.camera_names:
            img_filename = os.path.join(self.root_split, info['image'][camera_name]['image_path'])
            img_filenames.append(img_filename)

            rect = info['calib'][camera_name]['R0_rect'].astype(np.float64)
            Trv2c = info['calib'][camera_name]['Tr_velo_to_cam'].astype(np.float64)
            P2 = info['calib'][camera_name]['P2'].astype(np.float64)
            lidar2img = np.dot(P2, rect)

            lidar2img_list.append(lidar2img)

        input_dict = dict(
            sample_idx=sample_idx,
            pts_filename=pts_filename,
            camera_names=self.camera_names,
            img_info=img_filenames,
            lidar2img=lidar2img_list)

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict
    
    
    # TODO(swc): only front_left_camera now
    def bbox2result_kitti(self,
                          net_outputs,
                          class_names,
                          pklfile_prefix=None,
                          submission_prefix=None):
        """Convert 3D detection results to kitti format for evaluation and test
        submission.

        Args:
            net_outputs (list[np.ndarray]): List of array storing the
                inferenced bounding boxes and scores.
            class_names (list[String]): A list of class names.
            pklfile_prefix (str): The prefix of pkl file.
            submission_prefix (str): The prefix of submission file.

        Returns:
            list[dict]: A list of dictionaries with the kitti format.
        """
        assert len(net_outputs) == len(self.data_infos), \
            'invalid list length of network outputs'
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        det_annos = []
        print('\nConverting prediction to KITTI format')
        for idx, pred_dicts in enumerate(
                mmcv.track_iter_progress(net_outputs)):
            annos = []
            info = self.data_infos[idx]
            sample_idx = info['image']['image_idx']
            image_shape = info['image']['front_left_camera']['image_shape'][:2]
            box_dict = self.convert_valid_bboxes(pred_dicts, info)
            anno = {
                'name': [],
                'truncated': [],
                'occluded': [],
                'alpha': [],
                'bbox': [],
                'dimensions': [],
                'location': [],
                'rotation_y': [],
                'score': [],
                'dt_boxes': [],
                'scores': []
            }
            if len(box_dict['bbox']) > 0:
                box_2d_preds = box_dict['bbox']
                box_preds = box_dict['box3d_camera']
                scores = box_dict['scores']
                box_preds_lidar = box_dict['box3d_lidar']
                label_preds = box_dict['label_preds']

                for box, box_lidar, bbox, score, label in zip(
                        box_preds, box_preds_lidar, box_2d_preds, scores,
                        label_preds):
                    bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                    bbox[:2] = np.maximum(bbox[:2], [0, 0])
                    anno['name'].append(class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    anno['alpha'].append(
                        -np.arctan2(-box_lidar[1], box_lidar[0]) + box[6])
                    anno['bbox'].append(bbox)
                    anno['dimensions'].append(box[3:6])
                    anno['location'].append(box[:3])
                    anno['rotation_y'].append(box[6])
                    anno['score'].append(score)

                    anno['dt_boxes'].append(box_lidar)
                    anno['scores'].append(score)
                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)
            else:
                anno = {
                    'name': np.array([]),
                    'truncated': np.array([]),
                    'occluded': np.array([]),
                    'alpha': np.array([]),
                    'bbox': np.zeros([0, 4]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.array([]),
                    'score': np.array([]),
                    'dt_boxes': np.zeros([0, 7]),
                    'scores': np.array([]),
                }
                annos.append(anno)

            if submission_prefix is not None:
                curr_file = f'{submission_prefix}/{sample_idx:06d}.txt'
                with open(curr_file, 'w') as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print(
                            '{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(
                                anno['name'][idx], anno['alpha'][idx],
                                bbox[idx][0], bbox[idx][1], bbox[idx][2],
                                bbox[idx][3], dims[idx][1], dims[idx][2],
                                dims[idx][0], loc[idx][0], loc[idx][1],
                                loc[idx][2], anno['rotation_y'][idx],
                                anno['score'][idx]),
                            file=f)
            # FIXME(swc): sample_idx
            sample_idx = int(sample_idx.split('.')[0])
            annos[-1]['sample_idx'] = np.array(
                [sample_idx] * len(annos[-1]['score']), dtype=np.int64)  # NOTE(swc): [sample_idx] * num_box

            det_annos += annos

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            mmcv.dump(det_annos, out)
            print(f'Result is saved to {out}.')

        return det_annos
    
    
    # TODO(swc): only front_left_camera now
    def convert_valid_bboxes(self, box_dict, info):
        """Convert the predicted boxes into valid ones.

        Args:
            box_dict (dict): Box dictionaries to be converted.

                - boxes_3d (:obj:`LiDARInstance3DBoxes`): 3D bounding boxes.
                - scores_3d (torch.Tensor): Scores of boxes.
                - labels_3d (torch.Tensor): Class labels of boxes.
            info (dict): Data info.

        Returns:
            dict: Valid predicted boxes.

                - bbox (np.ndarray): 2D bounding boxes.
                - box3d_camera (np.ndarray): 3D bounding boxes in
                    camera coordinate.
                - box3d_lidar (np.ndarray): 3D bounding boxes in
                    LiDAR coordinate.
                - scores (np.ndarray): Scores of boxes.
                - label_preds (np.ndarray): Class label predictions.
                - sample_idx (int): Sample index.
        """
        # TODO: refactor this function
        box_preds = box_dict['boxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        sample_idx = info['image']['image_idx']
        box_preds.limit_yaw(offset=0.5, period=np.pi * 2)

        if len(box_preds) == 0:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)

        rect = info['calib']['front_left_camera']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['front_left_camera']['Tr_velo_to_cam'].astype(np.float32)
        P2 = info['calib']['front_left_camera']['P2'].astype(np.float32)
        img_shape = info['image']['front_left_camera']['image_shape']
        P2 = box_preds.tensor.new_tensor(P2)

        box_preds_camera = box_preds.convert_to(Box3DMode.CAM, rect @ Trv2c)

        box_corners = box_preds_camera.corners
        box_corners_in_image = points_cam2img(box_corners, P2)
        # box_corners_in_image: [N, 8, 2]
        minxy = torch.min(box_corners_in_image, dim=1)[0]
        maxxy = torch.max(box_corners_in_image, dim=1)[0]
        box_2d_preds = torch.cat([minxy, maxxy], dim=1)
        # Post-processing
        # check box_preds_camera
        image_shape = box_preds.tensor.new_tensor(img_shape)
        valid_cam_inds = ((box_2d_preds[:, 0] < image_shape[1]) &
                          (box_2d_preds[:, 1] < image_shape[0]) &
                          (box_2d_preds[:, 2] > 0) & (box_2d_preds[:, 3] > 0))
        # check box_preds
        limit_range = box_preds.tensor.new_tensor(self.pcd_limit_range)
        valid_pcd_inds = ((box_preds.center > limit_range[:3]) &
                          (box_preds.center < limit_range[3:]))
        # valid_inds = valid_cam_inds & valid_pcd_inds.all(-1)
        valid_inds = valid_pcd_inds.all(-1)
        if valid_inds.sum() > 0:
            return dict(
                bbox=box_2d_preds[valid_inds, :].numpy(),
                box3d_camera=box_preds_camera[valid_inds].tensor.numpy(),
                box3d_lidar=box_preds[valid_inds].tensor.numpy(),
                scores=scores[valid_inds].numpy(),
                label_preds=labels[valid_inds].numpy(),
                sample_idx=sample_idx)
        else:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)
            

    # NOTE(swc): convert anno_box from lidar_coor(in the pkl) to camera_coor, the same with KITTI
    def anno_lidar2cam(self, anno, calib_info):
        cam_anno = {}
        rect = calib_info['front_left_camera']['R0_rect'].astype(np.float32)
        Trv2c = calib_info['front_left_camera']['Tr_velo_to_cam'].astype(np.float32)
        
        for k,v in anno.items():
            cam_anno[k] = v
        
        gt_bboxes_lidar = cam_anno['gt_boxes_lidar']
        gt_bboxes_cam= LiDARInstance3DBoxes(gt_bboxes_lidar, origin=(0.5, 0.5, 0.5)).convert_to(Box3DMode.CAM,rect @ Trv2c)
        
        cam_anno['dimensions'] = gt_bboxes_cam.dims.numpy()
        cam_anno['location'] = gt_bboxes_cam.center.numpy()
        cam_anno['rotation_y'] = gt_bboxes_cam.yaw.numpy()
        
        return cam_anno
    
    def bbox2result_pcdet(self,
                          net_outputs,
                          class_names,
                          pklfile_prefix=None,
                          submission_prefix=None):
        assert len(net_outputs) == len(self.data_infos), \
            'invalid list length of network outputs'
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        det_annos = []
        print('\nConverting prediction to pcdet format')
        for idx, pred_dicts in enumerate(
                mmcv.track_iter_progress(net_outputs)):
            annos = []
            info = self.data_infos[idx]
            sample_idx = info['image']['image_idx']
            anno = {
                    'dt_boxes': [],
                    'name': [],
                    'scores': []
            }
            if len(pred_dicts['boxes_3d']) > 0:
                scores = pred_dicts['scores_3d']
                box_preds_lidar = pred_dicts['boxes_3d']
                label_preds = pred_dicts['labels_3d']

                for box_lidar, score, label in zip(
                        box_preds_lidar, scores, label_preds):
                    anno['dt_boxes'].append(box_lidar)
                    anno['name'].append(class_names[int(label)])
                    anno['scores'].append(score)

                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)
            else:
                anno = {
                    'dt_boxes': np.zeros([0, 7]),
                    'name': np.array([]),
                    'scores': np.array([]),
                }
                annos.append(anno)

            det_annos += annos

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            mmcv.dump(det_annos, out)
            print(f'Result is saved to {out}.')

        return det_annos

    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None,
                 pipeline=None,
                 eval_result_dir=None):
        """Evaluation in KITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str, optional): The prefix of pkl files, including
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str, optional): The prefix of submission data.
                If not specified, the submission data will not be generated.
                Default: None.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        
        # to pcdet format
        self.eval_cnt+=1
        from mmdet3d.core.evaluation import get_formatted_results
        det_pcdet = self.bbox2result_pcdet(results, self.CLASSES, pklfile_prefix)
        gt_annos_pcdet = []
        for info in self.data_infos:
            gt_boxes = info['annos']['gt_boxes_lidar'] 
            gt_names = info['annos']['name'] 
            gt_anno = {'gt_boxes': gt_boxes, 'name': gt_names}
            gt_annos_pcdet.append(gt_anno)

        result_str, result_dict = get_formatted_results(self.pcd_limit_range, self.CLASSES, gt_annos_pcdet, det_pcdet, eval_result_dir, self.eval_cnt)
        
        print_log('\n' + '****************pcdet eval start.*****************', logger=logger)
        print_log('\n' + result_str, logger=logger)
        print_log('\n' + '****************pcdet eval done.*****************', logger=logger)

        if eval_result_dir is not None:
            with open(os.path.join(eval_result_dir, f'human_readable_results_{self.eval_cnt}.txt'), 'w') as f:
                f.write(result_str)
        if show or out_dir:
            self.show(results, out_dir, show=show, pipeline=pipeline)
        return result_dict
    
    
    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Whether to visualize the results online.
                Default: False.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            data_info = self.data_infos[i]
            pts_path = data_info['point_cloud']['lidar_idx']
            file_name = f"{pts_path}.bin"
            points, img_metas, img = self._extract_data(
                i, pipeline, ['points', 'img_metas', 'img'])
            points = points.numpy()
            # for now we convert points into depth mode
            points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                               Coord3DMode.DEPTH)
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
            show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
                                               Box3DMode.DEPTH)
            pred_bboxes = result['boxes_3d'].tensor.numpy()
            show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,
                                                 Box3DMode.DEPTH)
            show_result(points, show_gt_bboxes, show_pred_bboxes, out_dir,
                        file_name, show)

            # multi-modality visualization
            if self.modality['use_camera'] and 'lidar2img' in img_metas.keys():
                img = img.numpy()
                # need to transpose channel to first dim
                img = img.transpose(1, 2, 0)
                show_pred_bboxes = LiDARInstance3DBoxes(
                    pred_bboxes, origin=(0.5, 0.5, 0))
                show_gt_bboxes = LiDARInstance3DBoxes(
                    gt_bboxes, origin=(0.5, 0.5, 0))
                show_multi_modality_result(
                    img,
                    show_gt_bboxes,
                    show_pred_bboxes,
                    img_metas['lidar2img'],
                    out_dir,
                    file_name,
                    box_mode='lidar',
                    show=show)