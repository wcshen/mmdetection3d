import json
import os
import cv2
import numpy as np
from operator import itemgetter
from benchmark_utils import get_formatted_results
from plot_utils import plot_gt_dt_pcd

'''
pkl structure:
dt_pkl:
{
    timestamp_1:  # the type of key is float not string and 6 decimal places: 1630055632.798527
    {
        'dt_boxes': np.array([Nx7]), (x, y, z, x_size, y_size, z_size, yaw)
        'name': np.array([N]), e.g. ['Car', 'Truck',...]
        'score': np.array([N])
    },
    ...
    
    timestamp_n:
    {
        'dt_boxes': np.array([Nx7]),
        'name': np.array([N]),
        'score': np.array([N])
    }
}

gt_pkl:
{
    timestamp_1:
    {
        'gt_boxes': np.array([Nx7]), (x, y, z, x_size, y_size, z_size, yaw)
        'name': np.array([N]), e.g. ['Car', 'Truck',...]
    },
    ...
    
    timestamp_n:
    {
        'gt_boxes': np.array([Nx7]),
        'name': np.array([N]),
    }
}
'''

class PlusBenchmark:
    def __init__(self, bev_range, class_names, plot_dir='worse_100', low_quality=False, pcd_path=None):
        self.bev_range = bev_range
        self.low_quality = low_quality
        self.class_names = class_names
        self.plot_dir = plot_dir
        self.eval_res = None
        self.pcd_path = pcd_path
    
    
    def match_ts(self, dt_info, gt_info):
        dt_ts = np.array(list(dt_info.keys()))
        gt_ts = np.array(list(gt_info.keys()))
        
        if dt_ts.shape[0] != gt_ts.shape[0]:
            return False
        
        match_res = gt_ts==dt_ts
        match_num = match_res.sum()
        total_num = gt_ts.shape[0]
        if match_num < total_num:
            return False
        
        self.ts = gt_ts
        return True
        
        
    def get_infos(self, dt_json, gt_json):
        with open(dt_json, 'r') as f:
            dt_info = json.load(f, encoding='utf-8')
        with open(gt_json, 'r') as f:
            gt_info = json.load(f, encoding='utf-8')
        dt_info = sorted(dt_info.items(), key=itemgetter(0))
        gt_info = sorted(gt_info.items(), key=itemgetter(0))
        return dt_info, gt_info
        
    
    def prepare_data(self, dt_info, gt_info):
        dt_annos, gt_annos = [], []
        for ts, gt_anno in gt_info.items():
            gt_annos.append(gt_anno)
            dt_annos.append(dt_info[ts])
        self.dt_annos = dt_annos
        self.gt_annos = gt_annos
        return dt_annos, gt_annos
        
    def evaluate(self, dt_json, gt_json):
        dt_info, gt_info = self.get_infos(dt_json, gt_json)
        match_flag = self.match_ts(dt_info, gt_info)
        if not match_flag:
            return False
        dt_annos, gt_annos = self.prepare_data(dt_info, gt_info)
        eval_res = get_formatted_results(bev_range=self.bev_range,
                                         class_names=self.class_names,
                                         gt_annos=gt_annos,
                                         det_annos=dt_annos)
        self.eval_res = eval_res
        return True
    
    
    def get_eval_res(self):
        return self.eval_res
    
    
    def worse_100(self):
        # find the worse_100 samples in the farest distance and save them
        eval_res = self.eval_res
        cls_names = list(eval_res['AP'].keys())
        dist_thresh = list(eval_res['AP'][cls_names[0]].keys())[-1]
        for cls_name in cls_names:
            cls_pr_plot_dir = os.path.join(self.plot_dir, cls_name, 'pr')
            cls_re_plot_dir = os.path.join(self.plot_dir, cls_name, 're')
            os.makedirs(cls_pr_plot_dir, exist_ok=True)
            os.makedirs(cls_re_plot_dir, exist_ok=True)
            # {'precision': np [], 'recall': np, 'false_num':np, 'fp_idxes':np, 'fn_idxes':np}
            this_cls_performance = eval_res[cls_name][dist_thresh]['frame_performance']
            this_cls_pr = this_cls_performance['precision']
            this_cls_re = this_cls_performance['recall']
            pr_worse_100_idxes = this_cls_pr.argsort()[:100]
            re_worse_100_idxes = this_cls_re.argsort()[:100]
            
            # save
            for pr_worse_100_id in pr_worse_100_idxes:
                # TODO(swc): pcd_path
                pcd_path = ''
                pcd = np.fromfile(pcd_path).reshape(-1, 4)
                gt = self.gt_annos[pr_worse_100_id]
                dt = self.dt_annos[pr_worse_100_id]
                pr = this_cls_pr[pr_worse_100_id]
                re = this_cls_pr[pr_worse_100_id]
                fp_idx = this_cls_performance['fp_idxes'][pr_worse_100_id]
                fn_idx = this_cls_performance['fn_idxes'][pr_worse_100_id]
                bev_title = f"pr: {pr:.2f} re:{re:.2f}"
                bev_plot = plot_gt_dt_pcd(pcd, self.bev_range, gt, dt, fp_idx, fn_idx, bev_title)
                bev_plot_name = f"{cls_pr_plot_dir}/{pr_worse_100_id}_{self.ts[pr_worse_100_id]}.jpg"
                cv2.imwrite(bev_plot_name, bev_plot)
            
            for re_worse_100_id in re_worse_100_idxes:
                pcd_path = ''
                pcd = np.fromfile(pcd_path).reshape(-1, 4)
                gt = self.gt_annos[re_worse_100_id]
                dt = self.dt_annos[re_worse_100_id]
                pr = this_cls_pr[re_worse_100_id]
                re = this_cls_re[re_worse_100_id]
                fp_idx = this_cls_performance['fp_idxes'][re_worse_100_id]
                fn_idx = this_cls_performance['fn_idxes'][re_worse_100_id]
                bev_title = f"pr: {pr:.2f} re:{re:.2f}"
                bev_plot = plot_gt_dt_pcd(pcd, self.bev_range, gt, dt, fp_idx, fn_idx, bev_title)
                bev_plot_name = f"{cls_re_plot_dir}/{re_worse_100_id}_{self.ts[re_worse_100_id]}.jpg"
                cv2.imwrite(bev_plot_name, bev_plot)
                

def main():
    dt_json_path = ''
    gt_json_path = ''
    plus_benchmark = PlusBenchmark(bev_range=[],
                                   class_names=[])
    res = plus_benchmark.evaluate(dt_json_path, gt_json_path)
    if res:
        benckmark_res = plus_benchmark.get_eval_res()


if __name__ == "__main__":
    main()