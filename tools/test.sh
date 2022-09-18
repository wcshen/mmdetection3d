PORT=29509 \
CUDA_VISIBLE_DEVICES="0,1,2,3" \
./tools/dist_test.sh \
work_dirs/plus_pp_0_100m_2class/hv_pointpillars_secfpn_6x4_160e_plus-kitti-3d-2class_0-100.py \
work_dirs/plus_pp_0_100m_2class/epoch_80.pth \
4 --eval mAP