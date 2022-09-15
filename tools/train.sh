#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES="4,5,6,7" 
function train()
{
    PORT=30000 \
    bash tools/dist_train.sh \
    configs/L4_data_models/prefusion_L4_all_class_80e_p32000_pt48_v_025.py \
    4
}

function plot_pr_curv()
{
    base_eval="/mnt/intel/jupyterhub/swc/train_log/LidarDet/L4E_data_models/prefusion_L4E_vehicle_6000_bev32_pt8/prefusion_L3_vehicle_1_1/20220914-010758/eval/checkpoint_epoch_80.pth"
    align_true="/mnt/intel/jupyterhub/swc/train_log/LidarDet/L4E_data_models/prefusion_L4E_vehicle_6000_bev32_pt16/prefusion_L3_vehicle_1_1/20220914-011331/eval/checkpoint_epoch_80.pth"
    python tools/analysis_tools/plot_prcurv.py --fds "$base_eval,$align_true" --save_plot_path $base_eval
}

while getopts "tp" arg #选项后面的冒号表示该选项需要参数
do
    case $arg in
            t)
            train
            ;;
            p)
            plot_pr_curv
            ;;
            ?)  #当有不认识的选项的时候arg为?
        echo "unkonw argument"
    exit 1
    ;;
    esac
done