import pickle
import shutil
import time
import io as sysio
import os

import numba
import numpy as np
import math

from tensorboardX import SummaryWriter
from mmdet3d.core.evaluation.kitti_utils.eval import bev_box_overlap

def print_str(value, *arg, sstream=None):
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    print(value, *arg, file=sstream)
    return sstream.getvalue()


@numba.jit
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if ((r_recall - current_recall) < (current_recall - l_recall)) and (
            i < (len(scores) - 1)
        ):
            continue
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1)

    return thresholds


def clean_data(gt_anno, dt_anno, current_cls_name, dist_threshold):    
    ignored_gt, ignored_dt = [], []
    num_gt = gt_anno["gt_boxes"].shape[0]
    num_dt = dt_anno["dt_boxes"].shape[0]
    num_valid_gt = 0

    for i in range(num_gt):
        gt_name = gt_anno["name"][i].lower()
        if gt_name == current_cls_name:
            valid_class = 1
        else:
            valid_class = 0
        if abs(gt_anno["gt_boxes"][i][0]) <= dist_threshold:
            valid_box = 1
        else:
            valid_box = 0
        if valid_class == 1 and valid_box == 1:
            ignored_gt.append(0)
            num_valid_gt += 1
        else:
            ignored_gt.append(1)

    for i in range(num_dt):
        if dt_anno["name"][i].lower() == current_cls_name:
            valid_class = 1
        else:
            valid_class = 0
        if abs(dt_anno["dt_boxes"][i][0]) <= dist_threshold:
            valid_box = 1
        else:
            valid_box = 0
        if valid_class == 1 and valid_box == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(1)

    return num_valid_gt, ignored_gt, ignored_dt


def prepare_data(gt_annos, dt_annos, current_class, dist_threshold):
    gt_datas_list = []
    dt_datas_list = []
    ignored_gts, ignored_dets = [], []
    total_num_valid_gt = 0

    for i in range(len(gt_annos)):
        rets = clean_data(gt_annos[i], dt_annos[i], current_class, dist_threshold)
        num_valid_gt, ignored_gt, ignored_det = rets
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        total_num_valid_gt += num_valid_gt
        gt_datas = gt_annos[i]["gt_boxes"]
        dt_datas = np.concatenate(
            [dt_annos[i]["dt_boxes"], dt_annos[i]["scores"][..., np.newaxis]], axis=1)
        gt_datas_list.append(gt_datas)
        dt_datas_list.append(dt_datas)
    return (
        gt_datas_list,
        dt_datas_list,
        ignored_gts,
        ignored_dets,
        total_num_valid_gt,
    )


@numba.jit
def compute_statistics_jit(
    overlaps,
    gt_datas,
    dt_datas,
    ignored_gt,
    ignored_det,
    min_overlap,
    thresh=0,
    compute_fp=False
):
    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    dt_scores = dt_datas[:, -1]

    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size
    if compute_fp:
        for i in range(det_size):
            if dt_scores[i] < thresh:
                ignored_threshold[i] = True
    NO_DETECTION = -10000000
    tp, fp, fn = 0, 0, 0
    thresholds = np.zeros((gt_size,))
    thresh_idx = 0
    for i in range(gt_size):
        if ignored_gt[i] == 1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0
        assigned_ignored_det = False

        for j in range(det_size):
            if ignored_det[j] == 1:
                continue
            if assigned_detection[j]:
                continue
            if ignored_threshold[j]:
                continue
            overlap = overlaps[i, j]
            dt_score = dt_scores[j]
            if (
                not compute_fp
                and (overlap >= min_overlap)
                and dt_score > valid_detection
            ):
                det_idx = j
                valid_detection = dt_score
            elif (
                compute_fp
                and (overlap >= min_overlap)
                and (overlap > max_overlap or assigned_ignored_det)
                and ignored_det[j] == 0
            ):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (
                compute_fp
                and (overlap >= min_overlap)
                and (valid_detection == NO_DETECTION)
                and ignored_det[j] == 1
            ):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
        elif (valid_detection != NO_DETECTION) and (
            ignored_gt[i] == 1 or ignored_det[det_idx] == 1
        ):
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            # only a tp adds a threshold.
            tp += 1
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            assigned_detection[det_idx] = True
    if compute_fp:
        for i in range(det_size):
            if not (
                assigned_detection[i]
                or ignored_det[i] == 1
                or ignored_threshold[i]
            ):
                fp += 1

    return tp, fp, fn, thresholds[:thresh_idx]


@numba.jit
def fused_compute_statistics(
    overlaps,
    pr,
    gt_nums,
    dt_nums,
    gt_datas,
    dt_datas,
    ignored_gts,
    ignored_dets,
    min_overlap,
    thresholds,
):
    gt_num = 0
    dt_num = 0
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            overlap = overlaps[
                dt_num : dt_num + dt_nums[i], gt_num : gt_num + gt_nums[i]
            ]

            gt_data = gt_datas[gt_num : gt_num + gt_nums[i]]
            dt_data = dt_datas[dt_num : dt_num + dt_nums[i]]
            ignored_gt = ignored_gts[gt_num : gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num : dt_num + dt_nums[i]]
            tp, fp, fn, _ = compute_statistics_jit(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True,
            )
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            pr[t, 3] += 0   # original for simularity, should be removed
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]


def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]


def calculate_iou_partly(gt_annos, dt_annos, num_parts=10):
    """fast iou algorithm. this function can be used independently to
    do result analysis.
    Args:
        gt_annos: list of dict, contains ground truth annos in all frames
        dt_annos: lsit of dict, contains detection annos in all frames
        num_parts: int. a parameter for fast calculate algorithm
    """

    assert len(gt_annos) == len(dt_annos)
    total_dt_num = np.stack([len(a["dt_boxes"]) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a["gt_boxes"]) for a in gt_annos], 0)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0
    split_parts = [i for i in split_parts if i != 0]
    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx: example_idx + num_part]
        dt_annos_part = dt_annos[example_idx: example_idx + num_part]
        gt_boxes = np.concatenate(
            [a["gt_boxes"][:, [0, 1, 3, 4, 6]] for a in gt_annos_part], axis=0)
        dt_boxes = np.concatenate(
            [b["dt_boxes"][:, [0, 1, 3, 4, 6]] for b in dt_annos_part], axis=0)
        overlap_part = bev_box_overlap(gt_boxes, dt_boxes).astype(np.float64)
        parted_overlaps.append(overlap_part)
        example_idx += num_part

    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx: example_idx + num_part]
        dt_annos_part = dt_annos[example_idx: example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][
                    gt_num_idx: gt_num_idx + gt_box_num,
                    dt_num_idx: dt_num_idx + dt_box_num,
                ]
            )
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return overlaps, parted_overlaps, total_gt_num, total_dt_num


def get_eval_results(overlap_info, gt_annos, dt_annos, current_class, min_overlap, dist_threshold):
    N_SAMPLE_PTS = 41   # Defined by mAP calculation algorithm, better not change it
    precision = np.zeros((N_SAMPLE_PTS,))
    recall = np.zeros((N_SAMPLE_PTS,))
    frame_performance_stats = {}
    overlaps, split_parts, parted_overlaps, total_gt_num, total_dt_num = overlap_info
    rets = prepare_data(gt_annos, dt_annos, current_class, dist_threshold)
    (gt_datas_list, dt_datas_list, ignored_gts,
     ignored_dets, total_num_valid_gt) = rets

    thresholdss = []
    for i in range(len(gt_annos)):
        rets = compute_statistics_jit(
            overlaps[i],
            gt_datas_list[i],
            dt_datas_list[i],
            ignored_gts[i],
            ignored_dets[i],
            min_overlap=min_overlap,
            thresh=0.0,
            compute_fp=False,
        )
        tp, _, _, thresholds = rets
        # Store the number of FPs and FNs in each frame
        frame_performance_stats[i] = dt_datas_list[i].shape[0] + gt_datas_list[i].shape[0] - 2 * tp   # FP + FN
        thresholdss += thresholds.tolist()

    thresholdss = np.array(thresholdss)
    thresholds = get_thresholds(thresholdss, total_num_valid_gt, num_sample_pts=N_SAMPLE_PTS)
    thresholds = np.array(thresholds)

    pr = np.zeros([len(thresholds), 4])
    idx = 0
    for j, num_part in enumerate(split_parts):
        gt_datas_part = np.concatenate(
            gt_datas_list[idx: idx + num_part], 0
        )
        dt_datas_part = np.concatenate(
            dt_datas_list[idx: idx + num_part], 0
        )
        ignored_dets_part = np.concatenate(
            ignored_dets[idx: idx + num_part], 0
        )
        ignored_gts_part = np.concatenate(
            ignored_gts[idx: idx + num_part], 0
        )
        fused_compute_statistics(
            parted_overlaps[j],
            pr,
            total_gt_num[idx: idx + num_part],
            total_dt_num[idx: idx + num_part],
            gt_datas_part,
            dt_datas_part,
            ignored_gts_part,
            ignored_dets_part,
            min_overlap=min_overlap,
            thresholds=thresholds,
        )
        idx += num_part
    for i in range(len(thresholds)):
        recall[i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
        precision[i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])
    for i in range(len(thresholds)):
        precision[i] = np.max(
            precision[i:], axis=-1)

    sums = 0
    for i in range(0, precision.shape[-1], 4):
        sums = sums + precision[..., i]
    ap = sums / 11 * 100

    optimal_pr = 0
    optimal_rec = 0
    best_score_thres = 0
    max_pr_rec = 0
    for i, thres in enumerate(thresholds):
        sum_pr_rec = precision[i] + recall[i]
        if sum_pr_rec <= max_pr_rec:
            continue
        else:
            max_pr_rec = sum_pr_rec
            best_score_thres = thres
            optimal_pr = precision[i] * 100
            optimal_rec = recall[i] * 100

    ret_dict = {
        'avg_precision': ap,
        'optimal_precision': optimal_pr,
        'optimal_recall': optimal_rec,
        'best_score_threshold': best_score_thres,
        'frame_performance_stats': frame_performance_stats,
        'thresholds': thresholds,
        'precision': precision,
        'recall': recall
    }

    return ret_dict


def get_formatted_results(bev_range,
                          class_names,
                          gt_annos,
                          det_annos,
                          result_dir,
                          eval_cnt):
    # Initialize evaluation metrics
    os.makedirs(result_dir, exist_ok=True)
    min_overlaps = {'Car': 0.5, 'Truck': 0.5, 'Pedestrian': 0.3, 'Cyclist': 0.3}
    dist_thresholds = list(range(50, math.ceil(bev_range[3]) + 50, 50)) if bev_range[3] > 50 else [50] # Range from 50m to max detection range, step by 50
    result_str = print_str("\n================== Evaluation Results ==================")
    result_difficulty = []
    result_dict = {}

    assert len(gt_annos) == len(det_annos)
    num_examples = len(gt_annos)
    split_parts = [1] * num_examples
    overlaps, parted_overlaps, total_gt_num, total_dt_num = calculate_iou_partly(
        gt_annos, det_annos, num_parts=num_examples)
    overlap_info = overlaps, split_parts, parted_overlaps, total_gt_num, total_dt_num
    for cls in class_names:
        result_str += print_str(cls.upper(), "\t", ("{:.1f}m\t" * len(dist_thresholds)).format(*dist_thresholds))
        eval_res = []
        min_overlap = min_overlaps[cls]
        for dist_thres in dist_thresholds:
            res = get_eval_results(overlap_info, gt_annos, det_annos, cls.lower(), min_overlap, dist_thres)
            eval_res.append([res['avg_precision'],
                             res['optimal_precision'],
                             res['optimal_recall'],
                             res['best_score_threshold']])
            
            # generate t-p-r table
            with open(str(result_dir)+'/tpr.%s_%d_%d'%(cls, dist_thres, eval_cnt), 'w') as f:
                for t,p,r in zip(res['thresholds'], res['precision'], res['recall']):
                    print("%.3f,%.3f,%.3f"%(t, p, r), file = f)        
        eval_res = np.array(eval_res)
        result_difficulty.append(res['frame_performance_stats'])
        # Report the evaluation results to TensorBoard
        # Here we only report the metrics of max distance threshold to indicate the overall performance,
        # which is easier to track the model performance in TensorBoard
        result_dict[str(cls) + '/average precision'] = eval_res[-1, 0]
        result_dict[str(cls) + '/precision'] = eval_res[-1, 1]
        result_dict[str(cls) + '/recall'] = eval_res[-1, 2]
        result_dict[str(cls) + '/score threshold'] = eval_res[-1, 3]

        # Convert the results to formated string
        result_str += print_str("ap:\t", ("{:.2f}\t" * len(dist_thresholds)).format(*(eval_res[:, 0].tolist())))
        result_str += print_str("pr:\t", ("{:.2f}\t" * len(dist_thresholds)).format(*(eval_res[:, 1].tolist())))
        result_str += print_str("re:\t", ("{:.2f}\t" * len(dist_thresholds)).format(*(eval_res[:, 2].tolist())))
        result_str += print_str("th:\t", ("{:.4f}\t" * len(dist_thresholds)).format(*(eval_res[:, 3].tolist())))
        result_str += print_str("--------------------------------------------------------")
    result_str += print_str("Note:\n"
                              "'ap' stands for average precision, sampled by 41 points in PR curve by default;\n"
                              "'pr' stands for precision at recommended score threshold;\n"
                              "'re' stands for recall at recommended score threhold;\n"
                              "'th' stands for recommended score threshold that achieves optimal balance between precision and recall.")

    return result_str, result_dict

