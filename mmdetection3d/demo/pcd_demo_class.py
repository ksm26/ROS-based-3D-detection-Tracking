#!/usr/bin/env python2
# Copyright (c) OpenMMLab. All rights reserved.
import sys, tf, rospy
import numpy as np

from argparse import ArgumentParser
from mmdet3d.apis import DetectionTracking
from numpy.lib.recfunctions import structured_to_unstructured
from sensor_msgs.msg import PointCloud2, PointField

sys.path.append("/opt/ros/melodic/lib/python2.7/dist-packages")
sys.path.append("~/../../ROS-tracker/catkin_ws/src:/opt/ros/melodic/share")
sys.path.append("~/../../mmdetection3d")

# mappings between PointField types and numpy types
type_mappings = [(PointField.INT8, np.dtype('int8')), (PointField.UINT8, np.dtype('uint8')),
                 (PointField.INT16, np.dtype('int16')),
                 (PointField.UINT16, np.dtype('uint16')), (PointField.INT32, np.dtype('int32')),
                 (PointField.UINT32, np.dtype('uint32')),
                 (PointField.FLOAT32, np.dtype('float32')), (PointField.FLOAT64, np.dtype('float64'))]
pftype_to_nptype = dict(type_mappings)
nptype_to_pftype = dict((nptype, pftype) for pftype, nptype in type_mappings)

# sizes (in bytes) of PointField types
pftype_sizes = {PointField.INT8: 1, PointField.UINT8: 1, PointField.INT16: 2, PointField.UINT16: 2,
                PointField.INT32: 4, PointField.UINT32: 4, PointField.FLOAT32: 4, PointField.FLOAT64: 8}

DUMMY_FIELD_PREFIX = '__'

class ROSDetectionTracking:
    def __init__(self):
        rospy.init_node('ros_detection_tracking')
        self.lidar_skip = 0
        self.tf_listener = tf.TransformListener()
        # Tracking
        self.tf_listener.waitForTransform('zoe/camera_front', 'zoe/velodyne', rospy.Time(0), rospy.Duration(1.0))
        (trans_camera, rot_camera) = self.tf_listener.lookupTransform('zoe/camera_front', 'zoe/velodyne',rospy.Time(0))
        self.velo2Cam = np.asarray(self.tf_listener.fromTranslationRotation(trans_camera, rot_camera))

        parser = ArgumentParser()
        parser.add_argument('--cfg_file', type=str,
                    default="3D-Multi-Object-Tracker/config/online"
                            "/rosbag.yaml",
                    help='specify the config for tracking')
        parser.add_argument('--pcd', 
            default='demo/data/kitti/kitti_000008.bin',
            help='Point cloud file')       
        parser.add_argument('--config', 
            # default='mmdetection3d/configs/pointpillars/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d.py')
            default='mmdetection3d/configs/regnet/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d.py')
        parser.add_argument('--checkpoint', 
            # default='mmdetection3d/checkpoints/nuscenes/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d_20201021_120719-269f9dd6.pth')
            default='mmdetection3d/checkpoints/nuscenes/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d_20200629_050311-dcd4e090.pth')
        parser.add_argument(
            '--device', default='cuda:0', help='Device used for inference')
        parser.add_argument(
            '--score-thr', type=float, default=0.45, help='bbox score threshold')
        parser.add_argument(
            '--trt-inference', type=bool, default=False, help='inference using TRT engine')
        parser.add_argument(
            '--out-dir', type=str, default='demo', help='dir to save results')
        parser.add_argument(
            '--show',
            default=True,
            action='store_true',
            help='show online visualization results')
        parser.add_argument(
            '--snapshot',
            default=True,
            action='store_true',
            help='whether to save online visualization results')
        self.args = parser.parse_args()
        self.detection = DetectionTracking(self.args)

        # Subscriber
        self.sub_lidar = rospy.Subscriber("zoe/velodyne_points", PointCloud2, self.lidar_callback, queue_size=1,buff_size=2**23)

        rospy.spin()

    def lidar_callback(self, msg):

        (trans, rot) = self.tf_listener.lookupTransform('zoe/world', 'zoe/velodyne', rospy.Time(0))
        (trans_inv, rot_inv) = self.tf_listener.lookupTransform('zoe/velodyne', 'zoe/world', rospy.Time(0))
        self.ego_to_world= np.asarray(self.tf_listener.fromTranslationRotation(trans,rot))
        self.world_2_ego = np.asarray(self.tf_listener.fromTranslationRotation(trans_inv,rot_inv))

        self.tf_dict = dict(
            velo2Cam=self.velo2Cam,
            ego_to_world=self.ego_to_world,
            world_2_ego=self.world_2_ego
        )

        self.lidar_skip +=1
        if self.lidar_skip % 2 ==0:

            intensity_fname = None
            for field in msg.fields:
                if field.name == "i" or field.name == "intensity":
                    intensity_fname = field.name
                
            dtype_list = self._fields_to_dtype(msg.fields, msg.point_step)
            pc_arr = np.frombuffer(msg.data, dtype_list)
            
            if intensity_fname:
                pc_arr = structured_to_unstructured(pc_arr[["x", "y", "z", intensity_fname]]).copy()
                pc_arr[:, 3] = pc_arr[:, 3] / 255
            else:
                pc_arr = structured_to_unstructured(pc_arr[["x", "y", "z"]]).copy()
                pc_arr = np.hstack((pc_arr, np.zeros((pc_arr.shape[0], 1))))

            if 'nuscenes' in self.args.checkpoint:
                pc_arr = np.hstack((pc_arr, np.zeros((pc_arr.shape[0], 1))))

            self.detection.inference_detector(pc_arr,msg.header.frame_id, self.tf_dict)

    def _fields_to_dtype(self, fields, point_step):
        '''Convert a list of PointFields to a numpy record datatype.
        '''
        offset = 0
        np_dtype_list = []
        for f in fields:
            while offset < f.offset:
                np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
                offset += 1

            dtype = pftype_to_nptype[f.datatype]
            if f.count != 1:
                dtype = np.dtype((dtype, f.count))

            np_dtype_list.append((f.name, dtype))
            offset += pftype_sizes[f.datatype] * f.count

        while offset < point_step:
            np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
            offset += 1

        return np_dtype_list

if __name__ == '__main__':
    ros_DetectionTracking = ROSDetectionTracking()