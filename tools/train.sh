#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES="5,6,7,4" \
bash tools/dist_train.sh \
configs/pointpillars/hv_pointpillars_secfpn_6x4_160e_plus-kitti-3d-4class.py\
4 \
--work-dir work_dirs/plus_pp_150_4class