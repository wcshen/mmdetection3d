#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES="4,5,6,7" \
bash tools/dist_train.sh \
configs/L4_data_models/prefusion_L4_all_class_80e_p32000_pt48_v_025.py \
4