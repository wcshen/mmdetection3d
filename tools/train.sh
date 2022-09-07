#!/usr/bin/env bash
export NCCL_P2P_DISABLE=1 

CUDA_VISIBLE_DEVICES="1,2,3,4,5,6" \
bash tools/dist_train.sh \
configs/transfusion_plus_pillar_L.py \
6 \
--work-dir work_dirs/plus_transfusion \
--deterministic