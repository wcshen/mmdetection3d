#!/usr/bin/env bash
export NCCL_P2P_DISABLE=1 

config=work_dirs/plus_pp_0_100m_2class/hv_pointpillars_secfpn_6x4_160e_plus-kitti-3d-2class_0-100.py
pth=work_dirs/plus_pp_0_100m_2class/epoch_80.pth

PORT=29509 \
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
./tools/dist_test.sh \
$config $pth 8 --plus_eval
