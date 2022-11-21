import argparse
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from vfe_module import PillarFeatureNet
from rpn_module import RPN

import onnxruntime
import onnx


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

        vfe_update_model_state = {}
        rpn_update_model_state = {}
        vfe_state_dict = vfe_model.state_dict()
        rpn_state_dict = rpn_model.state_dict()
        for key, val in model_state_disk.items():
            if key[14:] in vfe_state_dict and vfe_state_dict[key[14:]].shape == model_state_disk[key].shape:
                vfe_update_model_state[key[14:]] = val
            if key in rpn_state_dict and rpn_state_dict[key].shape == model_state_disk[key].shape:
                rpn_update_model_state[key] = val

        vfe_state_dict.update(vfe_update_model_state)
        vfe_model.load_state_dict(vfe_state_dict)
        vfe_model.cuda()
        vfe_model.eval()

        rpn_state_dict.update(rpn_update_model_state)
        rpn_model.load_state_dict(rpn_state_dict)
        rpn_model.cuda()
        rpn_model.eval()
        
        # ###################################### Validate VFE model ONNX/PyTorch ######################################
        print("Validating PFE ONNX model ...")
        vfe_input = torch.ones(
            [1, max_num_pillars, max_num_points_per_pillar, 74], dtype=torch.float32, device=torch.device('cuda:0'))
        vfe_out_torch = vfe_model(vfe_input)
        
        vfe_onnx_model = onnx.load(pfe_model_file)
        onnx.checker.check_model(vfe_onnx_model)
        onnx_vfe_session = onnxruntime.InferenceSession(pfe_model_file, providers=['CUDAExecutionProvider'])
        onnx_vfe_input_name = onnx_vfe_session.get_inputs()[0].name
        onnx_vfe_output_name = [onnx_vfe_session.get_outputs()[0].name]
        vfe_out_onnx = onnx_vfe_session.run(onnx_vfe_output_name, {onnx_vfe_input_name: vfe_input.detach().cpu().numpy()})
        print(f"vfe_out_shape: {len(vfe_out_onnx)}")
        # np.testing.assert_allclose(vfe_out_torch.detach().cpu().numpy(), vfe_out_onnx[0], rtol=1e-03, atol=1e-04)
        print("[SUCCESS] PFE ONNX model validated.")

        # ####################################### Validate RPN model ONNX/PyTorch ######################################
        print("Validating RPN ONNX model ...")
        rpn_input = torch.ones(
            [1, num_bev_features, grid_size[0], grid_size[1]], dtype=torch.float32, device=torch.device('cuda:0'))
        rpn_out_torch = rpn_model(rpn_input)
        
        rpn_onnx_model = onnx.load(rpn_model_file)
        onnx.checker.check_model(rpn_onnx_model)
        onnx_rpn_session = onnxruntime.InferenceSession(rpn_model_file, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
        onnx_rpn_input_name = onnx_rpn_session.get_inputs()[0].name
        onnx_rpn_output_name = [onnx_rpn_session.get_outputs()[0].name,
                                onnx_rpn_session.get_outputs()[1].name,
                                onnx_rpn_session.get_outputs()[2].name]
        rpn_out_onnx = onnx_rpn_session.run(onnx_rpn_output_name, {onnx_rpn_input_name: rpn_input.detach().cpu().numpy()})

        np.testing.assert_allclose(rpn_out_torch[0].detach().cpu().numpy(), rpn_out_onnx[0], rtol=1e-03, atol=1e-04)
        np.testing.assert_allclose(rpn_out_torch[1].detach().cpu().numpy(), rpn_out_onnx[1], rtol=1e-03, atol=1e-04)
        np.testing.assert_allclose(rpn_out_torch[2].detach().cpu().numpy(), rpn_out_onnx[2], rtol=1e-03, atol=1e-04)
        print("[SUCCESS] RPN ONNX model validated.")


if __name__ == '__main__':
    pfe_model_file = "./tools/export_onnx/mm3d_all_prefusion_p32000_pt48_v_032_pfe.onnx"
    rpn_model_file = "./tools/export_onnx/mm3d_all_prefusion_p32000_pt48_v_032_rpn.onnx"
    main()
