import math
import numba
import io as sysio
import numpy as np


def wrap_to_pi_2(theta):
    while theta > math.pi / 2.0:
        theta -= math.pi
    while theta < - math.pi / 2.0:
        theta += math.pi
    return theta


def print_str(value, *arg, sstream=None):
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    print(value, *arg, file=sstream)
    return sstream.getvalue()


def bev_box_overlap(boxes, qboxes, criterion=-1):
    from kitti_utils.rotate_iou import rotate_iou_gpu_eval
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou


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
    fp_idxes = []
    fn_idxes = []
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
            fn_idxes.append(i)
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
                fp_idxes.append(i)

    return tp, fp, fn, thresholds[:thresh_idx], fp_idxes, fn_idxes


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


def get_dist_eval_result(bev_range, gt_annos, det_annos, frame_ids):
    min_overlap = 0.5
    dist_thresholds = list(range(50, math.ceil(bev_range[3]) + 50, 50)) if bev_range[3] > 50 else [50]
    use_velocity = det_annos[0]['dt_boxes'].shape[-1] == 13

    def _eval(lst_frm, gt_annos, dt_annos, last_dist_thres, dist_thres, min_overlap = min_overlap):
        # currently the range focus on front and dist_thres
        if len(gt_annos) == 0 or len(dt_annos) == 0:
            return (0,0),(0,0)
        overlaps, parted_overlaps, total_gt_num, total_dt_num = calculate_iou_partly(
            gt_annos, dt_annos)
        cntx, dx = 0, 0.0
        cnty, dy = 0, 0.0
        cntr, dr = 0, 0.0
        if use_velocity:
            cntvx, dvx = 0, 0.0
            cntvy, dvy = 0, 0.0
        for i in range(len(gt_annos)):
            bb = gt_annos[i]["gt_boxes"][:, [0, 1]] + gt_annos[i]["gt_boxes"][:, [3, 4]]/2
            bbd = dt_annos[i]["dt_boxes"][:, [0, 1]] + dt_annos[i]["dt_boxes"][:, [3, 4]]/2
            rg = gt_annos[i]["gt_boxes"][:, [6]]
            rd = dt_annos[i]["dt_boxes"][:, [6]]
            for obj_idx in range(rg.shape[0]):
                rg[obj_idx, 0] = wrap_to_pi_2(rg[obj_idx, 0])
            for obj_idx in range(rd.shape[0]):
                rd[obj_idx, 0] = wrap_to_pi_2(rd[obj_idx, 0])

            irs = np.where((bb[:,0] > last_dist_thres) & (bb[:,0] <= dist_thres))[0]
            i_in_irs, ics = np.where(overlaps[i][irs,:]>min_overlap)
            cntx += len(ics)
            cnty += len(ics)
            cntr += len(ics)
            dx += sum([abs(bb[irs[iir],0]-bbd[ic, 0]) for iir, ic in zip(i_in_irs, ics)])
            dy += sum([abs(bb[irs[iir],1]-bbd[ic,1]) for iir, ic in zip(i_in_irs, ics)])
            dr += sum([abs(rg[irs[iir],0]-rd[ic,0]) for iir, ic in zip(i_in_irs, ics)])
            if use_velocity:
                gt_velocity = gt_annos[i]['gt_boxes'][:, 7:10]
                det_velocity = dt_annos[i]['dt_boxes'][:, 10:13]
                cntvx += len(ics)
                cntvy += len(ics)
                dvx += sum([abs(gt_velocity[irs[iir],0] - det_velocity[ic,0]) for iir, ic in zip(i_in_irs, ics)])
                dvy += sum([abs(gt_velocity[irs[iir],1] - det_velocity[ic,1]) for iir, ic in zip(i_in_irs, ics)])
            for iir, ic in zip(i_in_irs, ics):
                yaw_error_list.append(abs(rg[irs[iir],0]-rd[ic,0]))
        ret = (0,0),(0,0),(0,0)
        if use_velocity:
            ret = (*ret,(0,0), (0,0))
        if cntx > 0:
            ret = (cntx,dx/cntx), (cnty, dy/cnty), (cntr,dr/cntr)
            if use_velocity:
                ret = (*ret, (cntvx, dvx/cntvx), (cntvy, dvy/cntvy))
        return ret

    aggr = []
    for cls_thres in [0.3]:
        print("=========")
        print("DistThres(clsthres=%.1f) "%cls_thres+"\t".join([str(d) for d in dist_thresholds]))
        metric = []
        last_dist_thres = 0
        for dist_thres in dist_thresholds:
            dys = [] # cnt and average diff for a frame
            dxs = []
            drs = []
            if use_velocity:
                dvxs = []
                dvys = []
            gt_annos1 = []
            det_annos1 = []
            yaw_error_list = []
            lst_frm = ''
            for frame_id, gt_anno, det_anno in zip(frame_ids, gt_annos, det_annos):
                if lst_frm == '' or frame_id == lst_frm:
                    gt_annos1.append(gt_anno)
                    det_annos1.append(det_anno)
                else:
                    cnt_dx, cnt_dy, cnt_dr, *cnt_dvs = _eval(lst_frm, gt_annos1, det_annos1, last_dist_thres, dist_thres)
                    dys.append(cnt_dy)
                    dxs.append(cnt_dx)
                    drs.append(cnt_dr)
                    if use_velocity:
                        assert len(cnt_dvs) == 2
                        cnt_dvx, cnt_dvy = cnt_dvs
                        dvys.append(cnt_dvy)
                        dvxs.append(cnt_dvx)
                    gt_annos1 = [gt_anno]
                    det_annos1 = [det_anno]
                lst_frm = frame_id
            if len(gt_annos1) > 0:
                cnt_dx, cnt_da, cnt_dr, *cnt_dvs = _eval(lst_frm, gt_annos1, det_annos1, last_dist_thres, dist_thres)
                dys.append(cnt_dy)
                dxs.append(cnt_dx)
                drs.append(cnt_dr)
                if use_velocity:
                    assert len(cnt_dvs) == 2
                    cnt_dvx, cnt_dvy = cnt_dvs
                    dvys.append(cnt_dvy)
                    dvxs.append(cnt_dvx)
            cnt = sum([cd[0] for cd in dxs])
            if cnt > 0:
                yaw_error_list.sort()
                dx = sum([cd[1]*cd[0] for cd in dxs])/sum([cd[0] for cd in dxs])
                dy = sum([cd[1]*cd[0] for cd in dys])/sum([cd[0] for cd in dys])
                dr = sum([cd[1]*cd[0] for cd in drs])/sum([cd[0] for cd in drs])
                dr_90 = yaw_error_list[int(0.9999 * len(yaw_error_list))]
                if use_velocity:
                    dvx = sum([cd[1]*cd[0] for cd in dvxs])/sum([cd[0] for cd in dvxs])
                    dvy = sum([cd[1]*cd[0] for cd in dvys])/sum([cd[0] for cd in dvys])
                    metric.append((dx, dy, dr, dr_90, dvx, dvy))
                else:
                    metric.append((dx, dy, dr, dr_90))
            else:
                if use_velocity:
                    metric.append((-1, -1, -1, -1, -1, -1))
                else:
                    metric.append((-1, -1, -1, -1))
            last_dist_thres = dist_thres
        if use_velocity:
            print("      "+"\t".join(["%.3f/%.3f/%.3f/%.3f/%.3f/%.3f"%(dx,dy,dr,dr_90,dvx,dvy) for dx,dy,dr,dr_90,dvx,dvy in metric]))
        else:
            print("      "+"\t".join(["%.3f/%.3f/%.3f/%.3f"%(dx,dy,dr,dr_90) for dx,dy,dr,dr_90 in metric]))
        aggr.append(metric)
    print("=========")
    return aggr 


# from tong
def get_x_y_r_boundary_error(gt_annos, det_annos):
    min_overlap = 0.5
    # TODO(swc): need to be confirmed
    x_thresholds = [0, 50, 100]
    y_thresholds = [-20, 0, 20]
    ranges = []
    for x_thr_idx in range(len(x_thresholds) - 1):
        for y_thr_idx in range(len(y_thresholds) - 1):
            ranges.append((x_thresholds[x_thr_idx], y_thresholds[y_thr_idx], x_thresholds[x_thr_idx + 1], y_thresholds[y_thr_idx + 1]))   

    def _eval(gt_annos, dt_annos, x_range, y_range, min_overlap = min_overlap):
        # currently the range focus on front and dist_thres
        if len(gt_annos) == 0 or len(dt_annos) == 0:
            return (0,0),(0,0)
        overlaps, parted_overlaps, total_gt_num, total_dt_num = calculate_iou_partly(
            gt_annos, dt_annos)
        is_left = y_range[1] > 0.1
        is_behind = x_range[0] < - 0.1
        cntx, dx = 0, 0.0
        cnty, dy = 0, 0.0
        cntr, dr = 0, 0.0
        for i in range(len(gt_annos)):
            gt_ct = gt_annos[i]["gt_boxes"][:, [0, 1]]
            bb = gt_annos[i]["gt_boxes"][:, [0, 1]]
            bbd = dt_annos[i]["dt_boxes"][:, [0, 1]]
            
            rg = gt_annos[i]["gt_boxes"][:, [6]]
            rd = dt_annos[i]["dt_boxes"][:, [6]]

            if is_left:
                bb[:,1] = bb[:,1] - gt_annos[i]["gt_boxes"][:, 4]/2
                bbd[:,1] = bbd[:,1] - dt_annos[i]["dt_boxes"][:, 4]/2
            else:
                bb[:,1] = bb[:,1] + gt_annos[i]["gt_boxes"][:, 4]/2
                bbd[:,1] = bbd[:,1] + dt_annos[i]["dt_boxes"][:, 4]/2

            if is_behind:
                bb[:,0] = bb[:,0] + gt_annos[i]["gt_boxes"][:, 3]/2
                bbd[:,0] = bbd[:,0] + dt_annos[i]["dt_boxes"][:, 3]/2
            else:
                bb[:,0] = bb[:,0] - gt_annos[i]["gt_boxes"][:, 3]/2
                bbd[:,0] = bbd[:,0] - dt_annos[i]["dt_boxes"][:, 3]/2

            irs = np.where((gt_ct[:,0] > x_range[0]) & (gt_ct[:,0] <= x_range[1]) &   \
                           (gt_ct[:,1] > y_range[0]) & (gt_ct[:,1] <= y_range[1]) &   \
                           (abs(gt_annos[i]["gt_boxes"][:, 6]) < 0.3))[0]

            i_in_irs, ics = np.where(overlaps[i][irs,:]>min_overlap)
            cntx += len(ics)
            cnty += len(ics)
            cntr += len(ics)
            dx += sum([abs(bb[irs[iir],0]-bbd[ic, 0]) for iir, ic in zip(i_in_irs, ics)])
            dy += sum([abs(bb[irs[iir],1]-bbd[ic,1]) for iir, ic in zip(i_in_irs, ics)])
            dr += sum([abs(rg[irs[iir],0]-rd[ic,0]) for iir, ic in zip(i_in_irs, ics)])
            for iir, ic in zip(i_in_irs, ics):
                yaw_error_list.append(abs(rg[irs[iir],0]-rd[ic,0]))
        if cntx > 0:
            return (cntx,dx/cntx), (cnty, dy/cnty), (cntr,dr/cntr)
        else:
            return (0,0),(0,0), (0,0)

    
    metric = []

    for x_thr_idx in range(len(x_thresholds) - 1):
        yaw_error_list = []
        for y_thr_idx in range(len(y_thresholds) - 1):
            x_range = [x_thresholds[x_thr_idx], x_thresholds[x_thr_idx + 1]]
            y_range = [y_thresholds[y_thr_idx], y_thresholds[y_thr_idx + 1]]
            dys = [] # cnt and average diff for a frame
            dxs = []
            drs = []
            gt_annos1 = []
            det_annos1 = []
            
            first_frame = True
            for gt_anno, det_anno in zip(gt_annos, det_annos):
                if first_frame:
                    first_frame = False
                    gt_annos1.append(gt_anno)
                    det_annos1.append(det_anno)
                else:
                    cnt_dx, cnt_dy, cnt_dr = _eval(gt_annos1, det_annos1, x_range, y_range)
                    dys.append(cnt_dy)
                    dxs.append(cnt_dx)
                    drs.append(cnt_dr)
                    gt_annos1 = [gt_anno]
                    det_annos1 = [det_anno]

            if len(gt_annos1) > 0:
                cnt_dx, cnt_dy, cnt_dr = _eval(gt_annos1, det_annos1, x_range, y_range)
                dys.append(cnt_dy)
                dxs.append(cnt_dx)
                drs.append(cnt_dr)

            cnt = sum([cd[0] for cd in dxs])
            if cnt > 0:
                yaw_error_list.sort()
                dx = sum([cd[1]*cd[0] for cd in dxs])/sum([cd[0] for cd in dxs])
                dy = sum([cd[1]*cd[0] for cd in dys])/sum([cd[0] for cd in dys])
                dr = sum([cd[1]*cd[0] for cd in drs])/sum([cd[0] for cd in drs])
                dr_90 = yaw_error_list[int(0.9999 * len(yaw_error_list))]
                metric.append((dx, dy, dr, dr_90))
            else:
                metric.append((-1, -1, -1, -1))

    return metric


def get_eval_results(overlap_info, gt_annos, dt_annos, current_class, min_overlap, dist_threshold):
    N_SAMPLE_PTS = 41   # Defined by mAP calculation algorithm, better not change it
    precision = np.zeros((N_SAMPLE_PTS,))
    recall = np.zeros((N_SAMPLE_PTS,))
    frame_performance_stats = {'precision': [], 'recall': [], 'false_num':[], 'fp_idxes':[], 'fn_idxes':[]}
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
            compute_fp=True,
        )
        tp, _, _, thresholds, fp_idxes, fn_idxes = rets
        # Store the number of FPs and FNs in each frame
        # dt_num: tp+fp
        # gt_num: tp+fn
        frame_pr = tp / dt_datas_list[i].shape[0]
        frame_re = tp / gt_datas_list[i].shape[0]
        frame_false_num = dt_datas_list[i].shape[0] + gt_datas_list[i].shape[0] - 2 * tp
        frame_performance_stats['precision'].append(frame_pr)
        frame_performance_stats['recall'].append(frame_re)
        frame_performance_stats['false_nums'].append(frame_false_num)
        frame_performance_stats['fp_idxes'].append(fp_idxes)
        frame_performance_stats['fn_idxes'].append(fn_idxes)
        thresholdss += thresholds.tolist()
    
    for k,v in frame_performance_stats.items():
        frame_performance_stats[k] = np.array(v)

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


# main
def get_formatted_results(bev_range,
                          class_names,
                          gt_annos,
                          det_annos):
    """_summary_

    Args:
        bev_range (list): _description_
        class_names (list): _description_
        gt_annos (list): _description_
        det_annos (list): _description_

    Returns:
        dict()
        'AP':
            cls_name1:
                dist_thresh1:
                    'ap':
                    'precision':
                    'recall':
                    'score_threshold':
                    'frame_performance': {'precision': [], 'recall': [], 'false_num':[], 'fp_idxes':[], 'fn_idxes':[]}
                dist_thresh2:
            cls_name2:
            .
            .
            .
        'boundary_error': [[(dx, dy, dr, dr_90),...,(dx, dy, dr, dr_90)],]
        
    """
    
    min_overlaps = {'Car': 0.5, 'Truck': 0.5, 'Pedestrian': 0.3, 'Cyclist': 0.3}
    # Range from 50m to max detection range, step by 50
    dist_thresholds = list(range(50, math.ceil(bev_range[3]) + 50, 50)) if bev_range[3] > 50 else [50] 
    result_dict = {}

    assert len(gt_annos) == len(det_annos)
    num_examples = len(gt_annos)
    split_parts = [1] * num_examples
    overlaps, parted_overlaps, total_gt_num, total_dt_num = calculate_iou_partly(
        gt_annos, det_annos, num_parts=num_examples)
    overlap_info = overlaps, split_parts, parted_overlaps, total_gt_num, total_dt_num
    
    for cls in class_names:
        min_overlap = min_overlaps[cls]
        for dist_thres in dist_thresholds:
            res = get_eval_results(overlap_info, gt_annos, det_annos, cls.lower(), min_overlap, dist_thres)
            
            result_dict['AP'][str(cls)][dist_thres] = {'ap': res['avg_precision'],
                                                       'precision': res['optimal_precision'],
                                                       'recall': res['optimal_recall'],
                                                       'score threshold': res['best_score_threshold'],
                                                       'frame_performance': res['frame_performance_stats']}
    # dist error
    boundary_error = get_x_y_r_boundary_error(gt_annos, det_annos)
    result_dict['boundary_error'] = boundary_error
    return result_dict
