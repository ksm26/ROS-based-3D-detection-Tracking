# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import mmcv
from mmcv import DictAction
from mmcv.parallel import MMDataParallel

from mmdeploy.apis import build_task_processor
from mmdeploy.utils.config_utils import load_config
from mmdeploy.utils.device import parse_device_id
from mmdeploy.utils.timer import TimeCounter

import torch

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDeploy test (and eval) a backend.')
    parser.add_argument('--deploy_cfg', 
                        default='mmdeploy/configs/mmdet3d/voxel-detection/voxel-detection_tensorrt_dynamic-nus-64x4.py',
                        # default='mmdeploy/configs/mmdet3d/voxel-detection/voxel-detection_tensorrt_dynamic-kitti-32x4.py',
                        help='Deploy config path')
    parser.add_argument('--model_cfg', 
                        default='mmdetection3d/configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py',
                        # default='mmdetection3d/configs/pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d.py',
                        help='Model config path')
    parser.add_argument('--model', 
        default=['mmdeploy/deployed_models/nuscenes/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d_20201021_120719-269f9dd6/end2end.engine'],
        # default=['mmdeploy/deployed_models/nuscenes/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d_20200629_050311-dcd4e090/end2end.engine'], # unable to visualize this model becoz of voxel detection issue
        type=str, nargs='+',help='Input model files.')
    parser.add_argument('--out', 
                        default='data/nuscenes_out.pkl',
                        help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--metrics', 
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the codebase and the '
        'dataset, e.g., "bbox", "segm", "proposal" for COCO, and "mAP", '
        '"recall" for PASCAL VOC in mmdet; "accuracy", "precision", "recall", '
        '"f1_score", "support" for single label dataset, and "mAP", "CP", "CR"'
        ', "CF1", "OP", "OR", "OF1" for multi-label dataset in mmcls')
    parser.add_argument('--show', default=False, action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--device', help='device used for conversion', default='cuda:0')
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
        '--metric-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--log2file',
        type=str,
        help='log evaluation results and speed to file',
        default=None)
    parser.add_argument(
        '--json-file',
        type=str,
        help='log evaluation results to json file',
        default='./results.json')
    parser.add_argument(
        '--speed-test', action='store_true', help='activate speed test')
    parser.add_argument(
        '--warmup',
        type=int,
        help='warmup before counting inference elapse, require setting '
        'speed-test first',
        default=10)
    parser.add_argument(
        '--log-interval',
        type=int,
        help='the interval between each log, require setting '
        'speed-test first',
        default=100)
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='the batch size for test, would override `samples_per_gpu`'
        'in  data config.')
    parser.add_argument(
        '--uri',
        help='Remote ipv4:port or ipv6:port for inference on edge device.')

    args = parser.parse_args()
    return args

# /home/khushdeep/miniconda3/envs/openmmlab/lib/python3.8/site-packages/mmdeploy/codebase/mmdet3d/deploy/voxel_detection.py

def main():
    args = parse_args()
    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')
    deploy_cfg_path = args.deploy_cfg
    model_cfg_path = args.model_cfg

    # load deploy_cfg
    deploy_cfg, model_cfg = load_config(deploy_cfg_path, model_cfg_path)

    # merge options for model cfg
    if args.cfg_options is not None:
        model_cfg.merge_from_dict(args.cfg_options)

    task_processor = build_task_processor(model_cfg, deploy_cfg, args.device)

    # prepare the dataset loader
    dataset_type = 'test'
    dataset = task_processor.build_dataset(model_cfg, dataset_type)
    # override samples_per_gpu that used for training
    model_cfg.data['samples_per_gpu'] = args.batch_size
    
    data_loader = task_processor.build_dataloader(
        dataset,
        samples_per_gpu=model_cfg.data.samples_per_gpu,
        workers_per_gpu=model_cfg.data.workers_per_gpu)

    # load the model of the backend
    model = task_processor.init_backend_model(args.model, uri=args.uri)

    is_device_cpu = (args.device == 'cpu')
    device_id = None if is_device_cpu else parse_device_id(args.device)

    destroy_model = model.destroy
    model = MMDataParallel(model, device_ids=[device_id])

    if hasattr(model.module, 'CLASSES'):
        model.CLASSES = model.module.CLASSES
    if args.speed_test:
        with_sync = not is_device_cpu

        with TimeCounter.activate(
                warmup=args.warmup,
                log_interval=args.log_interval,
                with_sync=with_sync,
                file=args.log2file,
                batch_size=model_cfg.data.samples_per_gpu):
            outputs = task_processor.single_gpu_test(model, data_loader,
                                                     args.show, args.show_dir)
    else:
        # outputs = task_processor.single_gpu_test(model, data_loader, args.show, args.show_dir)
        model.eval()
        results = []
        dataset = data_loader.dataset

        # prog_bar = mmcv.ProgressBar(len(dataset))
        for i, data in enumerate(data_loader):
            with torch.no_grad():
                # lidar-points : data['points'][0].data[0][0] -> tensor size[33387, 4]
                # data['points'][0].data[0][0] = lidarpoints from rosbag
                result = model(data['points'][0].data,
                                data['img_metas'][0].data, False)
            if args.show:
                model.module.show_result(
                    data,
                    result,
                    out_dir=args.out,
                    file_name=f'model_output{i}',
                    show=args.show,
                    snapshot=True,
                    score_thr=0.3)
            results.extend(result)

            # batch_size = len(result)
            # for _ in range(batch_size):
            #     prog_bar.update()
        print(results)
    
    # print(outputs)

    json_dir, _ = os.path.split(args.json_file)
    if json_dir:
        os.makedirs(json_dir, exist_ok=True)

    # task_processor.evaluate_outputs(
    #     model_cfg,
    #     outputs,
    #     dataset,
    #     args.metrics,
    #     args.out,
    #     args.metric_options,
    #     args.format_only,
    #     args.log2file,
    #     json_file=args.json_file)
    # only effective when the backend requires explicit clean-up (e.g. Ascend)
    destroy_model()

def trt_model():
    args = parse_args()
    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')
    deploy_cfg_path = args.deploy_cfg
    model_cfg_path = args.model_cfg

    # load deploy_cfg
    deploy_cfg, model_cfg = load_config(deploy_cfg_path, model_cfg_path)

    # merge options for model cfg
    if args.cfg_options is not None:
        model_cfg.merge_from_dict(args.cfg_options)

    task_processor = build_task_processor(model_cfg, deploy_cfg, args.device)

    # override samples_per_gpu that used for training
    model_cfg.data['samples_per_gpu'] = args.batch_size
    
    # load the model of the backend
    trt_model = task_processor.init_backend_model(args.model, uri=args.uri)

    is_device_cpu = (args.device == 'cpu')
    device_id = None if is_device_cpu else parse_device_id(args.device)

    trt_model = MMDataParallel(trt_model, device_ids=[device_id])

    return trt_model


if __name__ == '__main__':
    # main()
    trt = trt_model()

