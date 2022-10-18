# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings
from pathlib import Path

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

import mmdet
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor

from mmcv.ops import Voxelization
from mmdet3d.models.middle_encoders import PointPillarsScatter
from mmdet3d.models.dense_heads.anchor3d_head import Anchor3DHead

import onnxruntime
import onnx

from torch.nn import functional as F
from mmdet3d.core import bbox3d2result, voxel
import time

if mmdet.__version__ > '2.23.0':
    # If mmdet version > 2.23.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg
    
    
def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num (torch.Tensor): Actual number of points in each voxel.
        max_num (int): Max number of points in each voxel

    Returns:
        torch.Tensor: Mask indicates which points are valid inside a voxel.
    """
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(
        max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator


def point_augment(features, num_points, coors):
    vx = 0.25
    vy = 0.25
    vz = 8.0
    x_offset = vx / 2 + 0
    y_offset = vy / 2 -10
    z_offset = vz / 2 -2
    features_ls = [features]
    # Find distance of x, y, and z from cluster center
    points_mean = features[:, :, :3].sum(
        dim=1, keepdim=True) / num_points.type_as(features).view(
            -1, 1, 1)
    f_cluster = features[:, :, :3] - points_mean
    features_ls.append(f_cluster)

    # Find distance of x, y, and z from pillar center
    f_center = features[:, :, :3]
    f_center[:, :, 0] = f_center[:, :, 0] - (
        coors[:, 3].type_as(features).unsqueeze(1) * vx +
        x_offset)
    f_center[:, :, 1] = f_center[:, :, 1] - (
        coors[:, 2].type_as(features).unsqueeze(1) * vy +
        y_offset)
    f_center[:, :, 2] = f_center[:, :, 2] - (
        coors[:, 1].type_as(features).unsqueeze(1) * vz +
        z_offset)
    features_ls.append(f_center)

    # Combine together feature decorations
    features = torch.cat(features_ls, dim=-1)
    voxel_count = features.shape[1]
    mask = get_paddings_indicator(num_points, voxel_count, axis=0)
    mask = torch.unsqueeze(mask, -1).type_as(features)
    features *= mask
    return features


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--plot_result', action='store_true', help='save results')
    parser.add_argument('--plus_eval', action='store_true', help='eval one pth')
    
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def onnx_inference(cfg, data_loader):
    vfe_onnx_model = onnx.load(pfe_model_file)
    onnx.checker.check_model(vfe_onnx_model)
    onnx_vfe_session = onnxruntime.InferenceSession(pfe_model_file, providers=['CUDAExecutionProvider'])
    onnx_vfe_input_name = onnx_vfe_session.get_inputs()[0].name
    onnx_vfe_output_name = [onnx_vfe_session.get_outputs()[0].name]
    
    rpn_onnx_model = onnx.load(rpn_model_file)
    onnx.checker.check_model(rpn_onnx_model)
    onnx_rpn_session = onnxruntime.InferenceSession(rpn_model_file, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
    onnx_rpn_input_name = onnx_rpn_session.get_inputs()[0].name
    onnx_rpn_output_name = [onnx_rpn_session.get_outputs()[0].name,
                            onnx_rpn_session.get_outputs()[1].name,
                            onnx_rpn_session.get_outputs()[2].name]
    print(f"====================== {onnx_rpn_output_name}")
    
    voxel_layer = Voxelization(**cfg.model.voxel_layer)
    middle_layer = PointPillarsScatter(in_channels=64, output_shape=[80, 400])
    head = Anchor3DHead(num_classes=2,
                        in_channels=384,
                        train_cfg=cfg.model.train_cfg,
                        test_cfg=cfg.model.test_cfg,
                        feat_channels=384,
                        use_direction_classifier=True,
                        anchor_generator=cfg.model.bbox_head.anchor_generator,
                        assigner_per_size=False,
                        assign_per_class=True,
                        diff_rad_by_sin=True,
                        loss_cls=cfg.model.bbox_head.loss_cls,
                        loss_bbox=cfg.model.bbox_head.loss_bbox,
                        loss_dir=cfg.model.bbox_head.loss_dir,
                        init_cfg=None)
    onnx_results = []
    for idx, data in enumerate(data_loader):
        voxels, coors, num_points = [], [], []
        for res in data['points']:
            points = res.data[0][0]
            res_voxels, res_coors, res_num_points = voxel_layer(points)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        batch_size = coors_batch[-1, 0].item() + 1
        voxel_features = point_augment(voxels, num_points, coors_batch)
        voxel_features = voxel_features.unsqueeze(dim=0)
        # print(f"shape: {voxel_features.shape} batch size: {batch_size}")
        if voxel_features.shape[1] != 6000:
            continue
        s = time.time()
        vfe_out_onnx = onnx_vfe_session.run(onnx_vfe_output_name, {onnx_vfe_input_name: voxel_features.detach().cpu().numpy()})
        e = time.time()
        t1 = (e - s) * 1000.0
        middle_input = torch.from_numpy(vfe_out_onnx[0]).float().cuda()
        print(f"===================== middle_input: {middle_input.shape}")
        pseudo_image = middle_layer(middle_input, coors_batch, batch_size)
        s = time.time()
        rpn_out_onnx = onnx_rpn_session.run(onnx_rpn_output_name, {onnx_rpn_input_name: pseudo_image.detach().cpu().numpy()})
        e = time.time()
        t2 = (e - s) * 1000.0
        print(f"v_t: {t1:.2f} r_t: {t2:.2f} a_t: {t1+t2:.2f}")
        cls_preds = [torch.from_numpy(rpn_out_onnx[0]).float().cuda()]
        box_preds = [torch.from_numpy(rpn_out_onnx[1]).float().cuda()]
        dir_cls = [torch.from_numpy(rpn_out_onnx[2]).float().cuda()]
        
        input_metas = data['img_metas'][0].data[0]
        
        bbox_list = head.get_bboxes(cls_scores=cls_preds, 
                                    bbox_preds=box_preds, 
                                    dir_cls_preds=dir_cls, 
                                    input_metas=input_metas,
                                    rescale=True)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels, idx=idx)
            for bboxes, scores, labels in bbox_list
        ]
        print(f"========== bbox_results: {len(bbox_results)}")
        # print(bbox_results)
        # idx_out = dict(idx=i, bbox=)
        onnx_results.extend(bbox_results)
    return onnx_results

def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir or args.plot_result or args.plus_eval, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    test_dataloader_default_args = dict(
        samples_per_gpu=4, workers_per_gpu=4, dist=distributed, shuffle=False)

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    if not distributed:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, data_loader, False, args.show_dir)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        # outputs = multi_gpu_test(model, data_loader, args.tmpdir,
        #                          args.gpu_collect)
    # print(outputs[0])
    onnx_outputs = onnx_inference(cfg, data_loader)
    # print(len(onnx_outputs))
    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))
        if args.plus_eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            ckpt_name = os.path.basename(args.checkpoint)
            eval_file_tail = int(ckpt_name.split('.')[0].split('_')[1])
            base_dir = os.path.split(args.checkpoint)[0]
            save_path = os.path.join(base_dir, 'single_eval', ckpt_name)
            os.makedirs(save_path, exist_ok=True)
            plot_save_dir = os.path.join(save_path, 'plot_results')
            os.makedirs(plot_save_dir, exist_ok=True)
            eval_kwargs.update(dict(eval_file_tail=eval_file_tail,
                                    eval_result_dir=save_path,
                                    out_dir=plot_save_dir,
                                    show=args.show,
                                    test_flag=True
                                    ))
            print(dataset.evaluate(onnx_outputs, **eval_kwargs))


if __name__ == '__main__':
    pfe_model_file = './tools/export_onnx/pfe.onnx'
    rpn_model_file = './tools/export_onnx/rpn.onnx'
    main()
