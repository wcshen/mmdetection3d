#!/bin/bash

config="mmdet3d/datasets/extract_bags/data_config.yaml"
mode="training"
# remote
raw_bag_path="/mnt/intel/artifact_management/lidar/dataset"
data_path='/mnt/intel/jupyterhub/swc/datasets/L4E_extracted_data'
calib_db='/mnt/intel/jupyterhub/swc/calib_db'

echo "POINT_CLOUD_TRAINVAL: $data_path"

# subfolders = ('s1' 's2' 's3')
subfolders=('L4E_origin_0812' 'L4E_origin_1104')

for subfolder in ${subfolders[*]}; do
  echo "subfolder: $subfolder"
  mkdir -p $data_path/$subfolder/$mode
done

function extract_bags() {
    # subfolders='s1','s2','s3'
    subfolders='L4E_origin_0812','L4E_origin_1104'
    docker exec -e LD_LIBRARY_PATH=/opt/ros/melodic/lib/ swc_dev bash -c \
  "cd /home/wancheng.shen/code_hub/mmdetection3d/mmdet3d/datasets/extract_bags && python extract_bags.py main --raw_bag_path $raw_bag_path --out_path $data_path --subfolders $subfolders --calib_db $calib_db  --extract_img 1"
}

function generate_image_txt() {
    # subfolders = ('s1' 's2' 's3')
    subfolders=('hard_case_origin_data' 'side_vehicle_origin_data' 'under_tree_origin_data')
    for subfolder in ${subfolders[*]}; do
    python mmdet3d/datasets/extract_bags/generate_image_txt.py --folder $data_path/$subfolder/$mode \
        --out $data_path/$subfolder/$mode \
        --train-ratio 1.0 \
        --val-ratio 0.0 \
        --test-ratio 0.0
    done
}

function generate_pkl() {
    # subfolders = ('s1' 's2' 's3')
    subfolders=('CN_L4_origin_benchmark' 'CN_L4_origin_data' 'hard_case_origin_data' 'side_vehicle_origin_data' 'under_tree_origin_data')

    for subfolder in ${subfolders[*]}; do
    python -m mmdet3d/datasets/extract_bags/generate_pkls.py \
        create_L4_data_mm3d_infos $config $subfolder
    done
}

#选项后面的冒号表示该选项需要参数
while getopts "aedtp" arg; do
  case $arg in
  a)
    # echo "a's arg:$OPTARG" #参数存在$OPTARG中
    extract_bags
    generate_image_txt
    generate_pkl
    ;;
  e)
    extract_bags
    ;;
  d)
    detect_camera_features
    ;;
  t)
    generate_image_txt
    ;;
  p)
    generate_pkl
    ;;
  ?) #当有不认识的选项的时候arg为?
    echo "unkonw argument"
    exit 1
    ;;
  esac
done
