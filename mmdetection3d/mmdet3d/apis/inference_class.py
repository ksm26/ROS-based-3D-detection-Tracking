# Copyright (c) OpenMMLab. All rights reserved.

from copy import deepcopy
from os import path as osp
import sys 

import mmcv
import numpy as np
import torch, rospy 
from std_msgs.msg import ColorRGBA
from pyquaternion import Quaternion
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from visualization_msgs.msg import Marker, MarkerArray

from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmdet3d.core.bbox import get_box_type
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger
from mmdet3d.core.points import get_points_type

sys.path.append("3D-Multi-Object-Tracker")
from tracker.config import cfg, cfg_from_yaml_file
from rosbag_3DMOT import Track_seq,save_seq

sys.path.append("mmdeploy")
from tools.test_rosbag import trt_model

def convert_SyncBN(config):
    """Convert config's naiveSyncBN to BN.

    Args:
         config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
    """
    if isinstance(config, dict):
        for item in config:
            if item == 'norm_cfg':
                config[item]['type'] = config[item]['type']. \
                                    replace('naiveSyncBN', 'BN')
            else:
                convert_SyncBN(config[item])
                
def init_model(config, checkpoint=None, device='cuda:0'):
    """Initialize a model from config file, which could be a 3D detector or a
    3D segmentor.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): Device to use.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    convert_SyncBN(config.model)
    config.model.train_cfg = None
    model = build_model(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = config.class_names
        if 'PALETTE' in checkpoint['meta']:  # 3D Segmentor
            model.PALETTE = checkpoint['meta']['PALETTE']
    model.cfg = config  # save the config in the model for convenience
    if device != 'cpu':
        torch.cuda.set_device(device)
    else:
        logger = get_root_logger()
        logger.warning('Don\'t suggest using CPU device. '
                       'Some functions are not supported for now.')
    model.to(device)
    model.eval()
    return model

class DetectionTracking:
    def __init__(self, args):
        """Initialize a model from config file, which could be a 3D detector or a 3D segmentor.
        Args:
            args
        """
        # Publisher
        self.pub_bbox = rospy.Publisher("/rosbag_detections", BoundingBoxArray, queue_size=1)
        self.markerIDs = rospy.Publisher("/markerID", MarkerArray, queue_size=1)

        config = args.config
        checkpoint = args.checkpoint
         
        yaml_file = args.cfg_file
        trackercfg = cfg_from_yaml_file(yaml_file, cfg)
        self.args = args
        self.Tracker = Track_seq(trackercfg)
        self.tracker_obj_label = 0 # label for car 
        self.scenenum=1

        if self.args.trt_inference:
            self.trtmodel = trt_model()
            self.trtmodel.eval()

        if isinstance(config, str):
            config = mmcv.Config.fromfile(config)
        elif not isinstance(config, mmcv.Config):
            raise TypeError('config must be a filename or Config object, '
                            f'but got {type(config)}')
        config.model.pretrained = None
        convert_SyncBN(config.model)
        config.model.train_cfg = None

        self.model = build_model(config.model, test_cfg=config.get('test_cfg'))

        if checkpoint is not None:
            checkpoint = load_checkpoint(self.model, checkpoint, map_location='cpu')
            if 'CLASSES' in checkpoint['meta']:
                self.model.CLASSES = checkpoint['meta']['CLASSES']
            else:
                self.model.CLASSES = config.class_names
            if 'PALETTE' in checkpoint['meta']:  # 3D Segmentor
                self.model.PALETTE = checkpoint['meta']['PALETTE']
        self.model.cfg = config  # save the config in the model for convenience

        if args.device != 'cpu':
            torch.cuda.set_device(args.device)
        else:
            logger = get_root_logger()
            logger.warning('Don\'t suggest using CPU device. '
                        'Some functions are not supported for now.')
        self.model.to(args.device)

        self.cfg = self.model.cfg
        self.cfg = self.cfg.copy()
        # set loading pipeline type
        self.cfg.data.test.pipeline[0].type = 'LoadPointsFromDict'

        # build the data pipeline
        self.test_pipeline = deepcopy(self.cfg.data.test.pipeline)
        self.test_pipeline = Compose(self.test_pipeline)
        self.box_type_3d, self.box_mode_3d = get_box_type(self.cfg.data.test.box_type_3d)
        self.model.eval()

    def inference_detector(self, pcd, frame_id, tf_dict):
        """Inference point cloud with the detector.

        Args:
            model (nn.Module): The loaded detector.
            pcd (np.ndarray): Point cloud.
            tf_dict (dict): tf dictionary containing transformations

        Returns:
            None.
        """
        self.frame_id = frame_id
        self.tf_dict = tf_dict
        device = next(self.model.parameters()).device
        lidar_boxes = []

        data = dict(
            box_type_3d=self.box_type_3d,
            box_mode_3d=self.box_mode_3d,
            # for ScanNet demo we need axis_align_matrix
            ann_info=dict(axis_align_matrix=np.eye(4)),
            sweeps=[],
            # set timestamp = 0
            timestamp=[0],
            img_fields=[],
            bbox3d_fields=[],
            pts_mask_fields=[],
            pts_seg_fields=[],
            bbox_fields=[],
            mask_fields=[],
            seg_fields=[])

        points_class = get_points_type('LIDAR')
        points = points_class(
            pcd, points_dim=pcd.shape[-1], attribute_dims=None)
        data['points'] = points

        data = self.test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        data = scatter(data, [device.index])[0]

        with torch.no_grad():
            if self.args.trt_inference:
                result = self.trtmodel(data['points'], data['img_metas'], False)[0]
            else:
                result = self.model(return_loss=False, rescale=True, **data)[0]
                result = result['pts_bbox']

            lidar_boxes.append( {'pred_boxes':result['boxes_3d'].tensor,
                            'pred_labels': result['labels_3d'],
                            'pred_scores': result['scores_3d']
                    })
        torch.cuda.empty_cache()
        self.tracker(lidar_boxes)

    def tracker(self,lidar_boxes):
        self.scenenum += 1
        boxes, scores = None, None
        if lidar_boxes is not None: 
            mask = lidar_boxes[0]['pred_labels'] == self.tracker_obj_label
            boxes = lidar_boxes[0]['pred_boxes'][mask].cpu().numpy()

            boxes_coord = np.append(boxes[:, 0:3], np.ones((boxes.shape[0], 1)), axis=1) 
            scores = lidar_boxes[0]['pred_scores'][mask].cpu().numpy()
            boxes_world = np.dot(self.tf_dict['ego_to_world'], boxes_coord.T).T
            boxes[:, 0:3] = boxes_world[:, 0:3]

        trackingObj, self.ids = self.Tracker.track_scene(boxes, scores, self.scenenum) 
        # self.track_obj = save_seq(trackeringObj,self.config, self.tf_dict['world_2_ego'],self.tf_dict['velo2Cam'],self.cam_P)
        self.plot_bbox_lidar(lidar_boxes)

    def plot_bbox_lidar(self,lidar_boxes):

        if self.args.score_thr is not None:
            mask = lidar_boxes[0]['pred_scores'] > self.args.score_thr
            lidar_boxes[0]['pred_boxes'] = lidar_boxes[0]['pred_boxes'][mask]
            lidar_boxes[0]['pred_scores'] = lidar_boxes[0]['pred_scores'][mask]
            lidar_boxes[0]['pred_labels'] = lidar_boxes[0]['pred_labels'][mask]

            mask_obj = (lidar_boxes[0]['pred_labels'] == self.tracker_obj_label)
            lidar_boxes[0]['pred_boxes'] = lidar_boxes[0]['pred_boxes'][mask_obj]
            lidar_boxes[0]['pred_scores'] = lidar_boxes[0]['pred_scores'][mask_obj]
            lidar_boxes[0]['pred_labels'] = lidar_boxes[0]['pred_labels'][mask_obj]

        if lidar_boxes is not None:

            num_detects = lidar_boxes[0]['pred_boxes'].shape[0]
            arr_bbox = BoundingBoxArray()
            arrIDs = MarkerArray()

            for i in range(num_detects):
                bbox = BoundingBox()
                
                bbox.header.frame_id = self.frame_id
                bbox.header.stamp = rospy.Time.now()

                bbox.pose.position.x = float(lidar_boxes[0]['pred_boxes'][i][0])
                bbox.pose.position.y = float(lidar_boxes[0]['pred_boxes'][i][1])
                bbox.pose.position.z = float(lidar_boxes[0]['pred_boxes'][i][2])
                bbox.dimensions.x = float(lidar_boxes[0]['pred_boxes'][i][3])  # width
                bbox.dimensions.y = float(lidar_boxes[0]['pred_boxes'][i][4])  # length
                bbox.dimensions.z = float(lidar_boxes[0]['pred_boxes'][i][5])  # height

                if 'nuscenes' in self.args.checkpoint:
                    q = Quaternion(axis=(0, 0, 1), radians=float(lidar_boxes[0]['pred_boxes'][i][6])-np.pi/2)
                else:
                    q = Quaternion(axis=(0, 0, 1), radians=float(lidar_boxes[0]['pred_boxes'][i][6]))

                bbox.pose.orientation.x = q.x 
                bbox.pose.orientation.y = q.y 
                bbox.pose.orientation.z = q.z  
                bbox.pose.orientation.w = q.w  

                marker = self.make_label(bbox, i)

                if int(lidar_boxes[0]['pred_labels'][i]) == self.tracker_obj_label : # change for Car label
                    arr_bbox.boxes.append(bbox)
                    arrIDs.markers.append(marker)
                    bbox.label = i
                    bbox.value = i

            arr_bbox.header.frame_id = self.frame_id
            arr_bbox.header.stamp = rospy.Time.now()
            # print(f"Scene num: {self.scenenum} Number of detections: {num_detects}")
            self.pub_bbox.publish(arr_bbox)
            self.markerIDs.publish(arrIDs)

        else:
            lidar_boxes = None

    def make_label(self, bbox, i):
        """ Helper function for generating visualization markers.
            Args:
                msg : PointCloud object
                bbox : BoundingBox object
                q : Quaternion angles
            Returns:
                Marker: A text view marker which can be published to RViz
        """
        marker = Marker()
        marker.header.frame_id= self.frame_id
        marker.header.stamp = rospy.Time.now()

        marker.ns = f"VehicleMarker"
        marker.id = self.ids[i]
        marker.type= marker.TEXT_VIEW_FACING
        marker.action = marker.ADD

        marker.scale.x = 1
        marker.scale.y = 1
        marker.scale.z = 2
        marker.pose.position.x = bbox.pose.position.x
        marker.pose.position.y = bbox.pose.position.y
        marker.pose.position.z = bbox.pose.position.z
        marker.pose.orientation.x = bbox.pose.orientation.x
        marker.pose.orientation.y = bbox.pose.orientation.y
        marker.pose.orientation.z = bbox.pose.orientation.z
        marker.pose.orientation.w = bbox.pose.orientation.w
        
        marker.color= ColorRGBA(1, 0, 0, 1)
        marker.text='ID: %d'% marker.id
        marker.lifetime = rospy.Duration(0.6)

        return marker