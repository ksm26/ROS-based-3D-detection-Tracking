# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

from mmdeploy.apis import (extract_model, get_predefined_partition_cfg,
                           torch2onnx)
from mmdeploy.utils import (get_ir_config, get_partition_config,
                            get_root_logger, load_config)


def parse_args():
    parser = argparse.ArgumentParser(description='Export model to ONNX.')

    # nuScenes
    parser.add_argument('--deploy_cfg', 
                        default='mmdeploy/configs/mmdet3d/voxel-detection/voxel-detection_tensorrt_dynamic-nus-64x4.py',
                        help='deploy config path')
    parser.add_argument('--model_cfg', 
                        default='mmdetection3d/configs/pointpillars/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d.py',
                        help='model config path')
    parser.add_argument('--checkpoint', 
                        default='mmdetection3d/checkpoints/nuscenes/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d_20200629_050311-dcd4e090.pth',
                        help='model checkpoint path')
    parser.add_argument('--img', help='image (point-cloud) used to convert model model',
                        default='mmdetection3d/demo/data/nuscenes/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151061547455.pcd.bin')
    parser.add_argument('--work-dir',
                        default='mmdeploy/deployed_models/nuscenes/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d_20200629_050311-dcd4e090', #os.getcwd(),
                        help='the dir to save logs and models- should be same as ckpt name')

    parser.add_argument(
        '--test-img',
        default=None,
        type=str,
        nargs='+',
        help='image used to test model')

    parser.add_argument('--calib-dataset-cfg',
                        help=('dataset config path used to calibrate in int8 mode. If not '
                        'specified, it will use "val" dataset in model config instead.'),
                        default=None)
    parser.add_argument('--device',
                        help='device used for conversion', 
                        default='cuda:0')
    parser.add_argument(
        '--log-level',
        help='set log level',
        default='INFO',
        choices=list(logging._nameToLevel.keys()))
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    logger = get_root_logger(log_level=args.log_level)

    logger.info(f'torch2onnx: \n\tmodel_cfg: {args.model_cfg} '
                f'\n\tdeploy_cfg: {args.deploy_cfg}')

    os.makedirs(args.work_dir, exist_ok=True)
    # load deploy_cfg
    deploy_cfg = load_config(args.deploy_cfg)[0]
    save_file = get_ir_config(deploy_cfg)['save_file']

    torch2onnx(
        args.img,
        args.work_dir,
        save_file,
        deploy_cfg=args.deploy_cfg,
        model_cfg=args.model_cfg,
        model_checkpoint=args.checkpoint,
        device=args.device)

    # partition model
    partition_cfgs = get_partition_config(deploy_cfg)

    if partition_cfgs is not None:
        if 'partition_cfg' in partition_cfgs:
            partition_cfgs = partition_cfgs.get('partition_cfg', None)
        else:
            assert 'type' in partition_cfgs
            partition_cfgs = get_predefined_partition_cfg(
                deploy_cfg, partition_cfgs['type'])

        origin_ir_file = osp.join(args.work_dir, save_file)
        for partition_cfg in partition_cfgs:
            save_file = partition_cfg['save_file']
            save_path = osp.join(args.work_dir, save_file)
            start = partition_cfg['start']
            end = partition_cfg['end']
            dynamic_axes = partition_cfg.get('dynamic_axes', None)

            extract_model(
                origin_ir_file,
                start,
                end,
                dynamic_axes=dynamic_axes,
                save_file=save_path)
    logger.info(f'torch2onnx finished. Results saved to {args.work_dir}')


if __name__ == '__main__':
    main()
