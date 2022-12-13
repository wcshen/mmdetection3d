#!/usr/bin/env bash
# ps -ef | grep train.py | grep rongbo | grep -v grep | awk '{print "kill -9 "$2}' | sh

export NCCL_P2P_DISABLE=1 
CUDA_VISIBLE_DEVICES="0,1,2,3" \
PORT=30000 \
bash tools/dist_train.sh \
configs/L3_data_models/pcdet_bev_fusion.py \
4 --work-dir work_dirs --extra-tag  exp_name
