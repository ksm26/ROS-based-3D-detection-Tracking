import numpy as np
import re
import os
import cv2
from nuscenes.utils.geometry_utils import transform_matrix

from nuscenes.utils.data_classes import Box
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion

def read_image(path):
    im=cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    return im


class nuscenesTrackingDataset:
    def __init__(self,config,nusc,load_image=False,load_points=False,type=["car"]):
        self.root_path = config.dataset_path
        self.type = type
        self.config = config

        self.dataset_path = config.dataset_path
        self.detections_path = config.detections_path
        self.tracking_type = config.tracking_type
        self.verbose = config.verbose

        self.load_image = load_image
        self.load_points = load_points

        self.ob_path = config.detections_path
        self.nusc = nusc
        self.camera_intrinsic = None
        self.Ego2Cam = None
        self.all_ids = 0

    def add_total_samples(self,nbr_sample):
        self.all_ids += nbr_sample

    def __len__(self):
        return self.all_ids
    def __getitem__(self, input_sample):

        my_scene, current_token, current_sample = input_sample

        name = current_sample['token']
        # file_name = os.path.join('/media/storage/nuscenes_labels_new', str(scene_name), str(file_idx) + '.txt')
        LIDAR_data = self.nusc.get('sample_data', current_sample['data']['LIDAR_TOP'])
        camera_data = self.nusc.get('sample_data', current_sample['data']['CAM_FRONT'])
        lidar_pose_token = self.nusc.get('calibrated_sensor', LIDAR_data['calibrated_sensor_token'])
        camera_pose_token= self.nusc.get('calibrated_sensor', camera_data['calibrated_sensor_token'])
        self.camera_intrinsic = np.array(camera_pose_token['camera_intrinsic'])
        lidar_sensor_pose = transform_matrix(lidar_pose_token['translation'], Quaternion(lidar_pose_token['rotation']),
                                             inverse=True)
        self.Ego2Cam = transform_matrix(camera_pose_token['translation'],
                                              Quaternion(camera_pose_token['rotation']),
                                              inverse=True)
        if self.config.load_data == "True":
            # nusc.render_sample_data(cam_front_data['token'])
            pcl_path = os.path.join(self.config.dataset_path, LIDAR_data['filename'])
            lidar_pcl = LidarPointCloud.from_file(pcl_path)
        else:
            points = None
        if self.config.load_data == "True":
            image_path = os.path.join(self.config.dataset_path, LIDAR_data['filename'])
            image = read_image(image_path)
        else:
            image = None

        ego_pose = self.nusc.get('ego_pose', LIDAR_data['ego_pose_token'])
        # lidar_pose = self.nusc.get('calibrated_sensor', LIDAR_data['calibrated_sensor_token'])
        pose = transform_matrix(ego_pose['translation'], Quaternion(ego_pose['rotation']),
                                inverse=True)
        pose_angle = Quaternion(ego_pose['rotation']).yaw_pitch_roll[0]

        if self.ob_path is not None:
            ob_path = os.path.join(self.ob_path, name + '.txt')
            if not os.path.exists(ob_path):
                objects = np.zeros(shape=(0, 7))
                det_scores = np.zeros(shape=(0,))
            else:
                objects_list = []
                det_scores = []
                with open(ob_path) as f:
                    for each_ob in f.readlines():
                        infos = re.split(' ', each_ob)
                        if infos[0] in self.type:
                            angle_quaternion = Quaternion(infos[7:11])
                            objects_list.append(infos[1:7] + [angle_quaternion.yaw_pitch_roll[0]])
                            det_scores.append(infos[11])
                if len(objects_list)!=0:
                    objects = np.array(objects_list,np.float32)
                    # objects[:, 3:6] = cam_to_velo(objects[:, 3:6], self.V2C)[:, :3]
                    det_scores = np.array(det_scores,np.float32)
                else:
                    objects = np.zeros(shape=(0, 7))
                    det_scores = np.zeros(shape=(0,))
        else:
            objects = np.zeros(shape=(0,7))
            det_scores = np.zeros(shape=(0,))

        # if self.P2 is None:
        #     sensor = 'CAM_FRONT'
        #     cam_front_data = self.nusc.get('sample_data', current_sample['data'][sensor])
        #     #camera_intrinsic
        #     _, _, self.P2 = self.nusc.get_sample_data(cam_front_data['ego_pose_token'])

        nuscenes_data = {'camera_intrinsic':self.camera_intrinsic,
                        'Ego2Cam':self.Ego2Cam,
                        'points':points,
                        'image':image,
                        'objects':objects,
                        'det_scores':det_scores,
                        'pose':pose,
                        'lidar_sensor_pose':lidar_sensor_pose,
                         'pose_angle':pose_angle,
                         }


        return nuscenes_data
