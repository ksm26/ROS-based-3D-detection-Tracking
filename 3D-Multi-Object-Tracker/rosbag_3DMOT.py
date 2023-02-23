from tracker.tracker import Tracker3D
import time
import tqdm
import os
from tracker.config import cfg, cfg_from_yaml_file
from tracker.box_op import *
import numpy as np
import argparse

from pyquaternion import Quaternion

def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.
    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False
    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points

def translate(self, x: np.ndarray) -> None:
    """
    Applies a translation.
    :param x: <np.float: 3, 1>. Translation in x, y, z direction.
    """
    self.center += x


def rotate(center, quaternion: Quaternion) -> None:
    """
    Rotates box.
    :param quaternion: Rotation to apply.
    """
    center = np.dot(quaternion.rotation_matrix, center)
    # orientation = quaternion * self.orientation

class Track_seq():
    def __init__(self,config):
        print('Tracker initialized')
        self.config = config
        self.input_score = self.config.input_score
        self.tracking_type = self.config.tracking_type
        self.tracker = Tracker3D(box_type="OpenPCDet", tracking_features=False, config = self.config)
        self.count = 0

    def track_scene(self,objects, det_scores,timestamp):
        mask = det_scores>self.input_score
        objects = objects[mask]
        det_scores = det_scores[mask]

        bbs, ids = self.tracker.tracking(
                             objects[:,:7],
                             features=None,
                             scores=det_scores,
                             pose=None,
                             timestamp=timestamp)

        # print(f'length of dead trajectories:{self.tracker.dead_trajectories.__len__()}')
        # print(f'length of active trajectories:{self.tracker.active_trajectories.__len__()}')
        return self.tracker, ids

def save_seq(tracker,
                 config,
                 world_2_ego,
                 Ego2Cam,
                 camera_intrinsic):
    """
    saving tracking results
    Args:
        dataset: KittiTrackingDataset, Iterable dataset object
        seq_id: int, sequence id
        tracker: Tracker3D
    """
    tracking_type = config.tracking_type

    s =time.time()
    tracks = tracker.post_processing(config)
    track_obj = []
    frame_first_dict = {}
    for ob_id in tracks.keys():
        track = tracks[ob_id]

        for frame_id in track.trajectory.keys():
            ob = track.trajectory[frame_id]
            if ob.updated_state is None:
                continue
            if ob.score<config.post_score:
                continue

            if frame_id in frame_first_dict.keys():
                frame_first_dict[frame_id][ob_id]=(np.array(ob.updated_state.T),ob.score)
            else:
                frame_first_dict[frame_id]={ob_id:(np.array(ob.updated_state.T),ob.score)}

    frames_list = list(frame_first_dict)
    i = frames_list[-1]
    objects = frame_first_dict[i]

    for ob_id in objects.keys():
        updated_state,score = objects[ob_id]

        box_template = np.zeros(shape=(1,7))
        box_template[0,0:3]=updated_state[0,0:3]
        box_template[0,3:7]=updated_state[0,9:13]

        # global to lidar
        box = bb3d_corners_nuscenes(list(box_template[0, :])) # extract corners based on x,y,z,l,b,h 
        corners_ego = np.dot(world_2_ego, box)
        corners_camera = np.dot(Ego2Cam, corners_ego)
        box2d = view_points(corners_camera[:3, :], camera_intrinsic, normalize=True)

        box_center = bb3d_centres_nuscenes(corners_ego)
        box_angle =  box_template[0][6] 
        # box_angle = nuscenes_data['pose_angle']-box_template[0][6] # example from nuscences3dmot for box angle 

        track_obj.append([i, ob_id, tracking_type, box2d[0][0], box2d[0][1], box2d[0][2],
        box2d[0][3], box_center[0], box_center[1], box_center[2], box_template[0][3], box_template[0][4],
        box_template[0][5], box_angle, score])

    return track_obj

def tracking_val_seq(arg):

    yaml_file = arg.cfg_file

    config = cfg_from_yaml_file(yaml_file,cfg)

    print("\nconfig file:", yaml_file)
    print("data path: ", config.dataset_path)
    print('detections path: ', config.detections_path)
    nusc = NuScenes(version=config.nusc_version, verbose=config.verbose, dataroot=config.dataset_path)

    save_path = config.save_path                       # the results saving path

    os.makedirs(save_path,exist_ok=True)

    seq_list = config.tracking_seqs    # the tracking sequences

    print("tracking seqs: ", seq_list)

    all_time,frame_num = 0,0

    for seq_id in tqdm.trange(seq_list):
        current_scene = nusc.scene[seq_id]

        all_time = 0
        frame_num = 0
        # seq_id = seq_list[scene_no]
        dataset,tracker, this_time, this_num = track_one_seq(seq_id,config,nusc,current_scene)
        proc_time = save_one_seq(dataset,seq_id,tracker,config,nusc,current_scene)

        all_time+=this_time
        # all_time+=proc_time
        frame_num+=this_num

    print("Tracking time: ", all_time)
    print("Tracking frames: ", frame_num)
    print("Tracking FPS:", frame_num/all_time)
    print("Tracking ms:", all_time/frame_num)

    # eval_kitti()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default="config/online/pvrcnn_mot_nuscenes.yaml",
                        help='specify the config for tracking')
    args = parser.parse_args()
    tracking_val_seq(args)

