import argparse
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from vfe_module import PillarFeatureNet
from rpn_module import RPN


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None,
                        help='specify the config of model')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='checkpoint to start from')
    args = parser.parse_args()

    cfg = Config.fromfile(args.cfg_file)
    np.random.seed(1024)

    return args, cfg


def main():
    # Load the configs
    args, cfg = parse_config()
    grid_size = cfg.model.middle_encoder.output_shape

    max_num_pillars = cfg.model.voxel_layer.max_voxels[0]
    max_num_points_per_pillar = cfg.model.voxel_layer.max_num_points
    num_bev_features = cfg.model.middle_encoder.in_channels

    vfe_cfg = cfg.model.voxel_encoder
    backbone_cfg = cfg.model.backbone
    head_cfg = cfg.model.bbox_head
    # Build up models
    vfe_model = PillarFeatureNet(in_channels=vfe_cfg.in_channels,
                                 feat_channels=vfe_cfg.feat_channels,
                                 with_distance=vfe_cfg.with_distance,
                                 use_pcdet=vfe_cfg.use_pcdet)

    rpn_model = RPN(backbone_cfg=backbone_cfg, head_cfg=head_cfg)

    with torch.no_grad():
        checkpoint = torch.load(args.ckpt)
        model_state_disk = checkpoint['state_dict']
        
        for key, value in model_state_disk.items():
            print(key,value.size(),sep="  ")

        vfe_update_model_state = {}
        rpn_update_model_state = {}
        vfe_state_dict = vfe_model.state_dict()
        rpn_state_dict = rpn_model.state_dict()
        
        
        a,v,r = 0,0,0
        for key, val in model_state_disk.items():
            if key[14:] in vfe_state_dict and vfe_state_dict[key[14:]].shape == model_state_disk[key].shape:
                vfe_update_model_state[key[14:]] = val
                v+=1
            if key in rpn_state_dict and rpn_state_dict[key].shape == model_state_disk[key].shape:
                rpn_update_model_state[key] = val
                r+=1
            a+=1
        print(f"a:{a} v:{v} r: {r}")

        vfe_state_dict.update(vfe_update_model_state)
        vfe_model.load_state_dict(vfe_state_dict)
        vfe_model.cuda()
        vfe_model.eval()

        rpn_state_dict.update(rpn_update_model_state)
        rpn_model.load_state_dict(rpn_state_dict)
        rpn_model.cuda()
        rpn_model.eval()

        # ###################################### Convert VFE model to ONNX ######################################
        # VFE input: max_num_pillars, max_num_points_per_pillar, point_features
        # lidar_only: [1, max_num_pillars, max_num_points_per_pillar, 10]
        # prefusion: [1, max_num_pillars, max_num_points_per_pillar, 74]
        vfe_input = torch.ones(
            [1, max_num_pillars, max_num_points_per_pillar, 74], dtype=torch.float32, device=torch.device('cuda:0'))

        vfe_input_names = ['vfe_input']
        vfe_output_names = ['pillar_features']
        output_onnx_file = './tools/export_onnx/mm3d_all_prefusion_p32000_pt48_v_032_pfe.onnx'
        torch.onnx.export(vfe_model, vfe_input, output_onnx_file, verbose=False,
                          input_names=vfe_input_names, output_names=vfe_output_names)
        print("[SUCCESS] PFE model is converted to ONNX.")

        # ###################################### Convert RPN model to ONNX ######################################
        # RPN input: NCHW
        rpn_input = torch.ones(
            [1, num_bev_features, grid_size[0], grid_size[1]], dtype=torch.float32, device=torch.device('cuda:0'))
        rpn_input_names = ['spatial_features']
        rpn_output_names = ['cls_preds', 'box_preds', 'dir_cls_preds']
        output_onnx_file = './tools/export_onnx/mm3d_all_prefusion_p32000_pt48_v_032_rpn.onnx'
        torch.onnx.export(rpn_model, rpn_input, output_onnx_file, verbose=False,
                          input_names=rpn_input_names, output_names=rpn_output_names)
        print("[SUCCESS] RPN model is converted to ONNX.")


if __name__ == '__main__':
    main()
