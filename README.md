# ROS-based-3D-detection-Tracking

## Installation
1) Create virtual environment
```
conda create --name rosdeploy python=3.8
conda activate rosdeploy
```

2) Install pytorch 
conda install pytorch from https://pytorch.org/ according to cuda version or lower for eg. 
```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

3) Clone this repository 
```
git clone git@github.com:ksm26/ROS-based-3D-detection-Tracking.git
```

4) Install steps 
```
pip install -U openmim
mim install mmcv-full
mim install mmdet
mim install mmsegmentation
```

5) Install [mmdetection3d](https://mmdetection3d.readthedocs.io/en/latest/getting_started.html)
```
cd mmdetection3d
pip install -e .
```
6) Install [mmdeploy](https://github.com/open-mmlab/mmdeploy/blob/master/docs/en/get_started.md)
```
cd mmdeploy 
pip install -e .
```

7) Install tf and jsk 
[tf tutorial](http://wiki.ros.org/tf/Tutorials/Writing%20a%20tf%20listener%20%28Python%29)
[Bounding box visualization](https://blog.csdn.net/weixin_36354875/article/details/125935782)
[ROS jsk_visualization](https://limhyungtae.github.io/2020-09-05-ROS-jsk_visualization-%EC%84%A4%EC%B9%98%ED%95%98%EB%8A%94-%EB%B2%95/)
[jsk_recognition_msgs](http://otamachan.github.io/sphinxros/indigo/packages/jsk_recognition_msgs.html)

```
sudo apt-get install ros-melodic-jsk-recognition-msgs 
sudo apt-get install ros-melodic-jsk-rviz-plugins
sudo apt-get install ros-melodic-jsk-rqt-plugins
sudo apt-get install ros-melodic-jsk-visualization
```
8) Install tensorrt 
```
pip install nvidia-tensorrt
python3 -m pip install --upgrade tensorrt
```
-----------------------------------------------------------------------------------------------------------------------------------------------
## Additional installation
Error with rospy.init_node(): An error with ros packages installation into conda environment.
Run the above ros_python_environment.yml using: 
```
pip install scikit-build
conda env update --file ros_python_environment.yml
```

import tf: Error\
Import tf does not work with python 3 as it is compiled for python 2.
Solution: [Recompile with python3 in catkin_workspace](https://answers.ros.org/question/326226/importerror-dynamic-module-does-not-define-module-export-function-pyinit__tf2/)\
To be able to work with python interpreter (pycharm,visual studio) add the compile path as below:
```
sys.path.append("~/Desktop/ROS-tracker/catkin_ws/src:/opt/ros/melodic/share")
```

Source catkin envrionment after tf installation and creation of new catkin workspace
```
source ~/catkin_ws/devel/setup.bash
```
-----------------------------------------------------------------------------------------------------------------------------------------------

## Benchmark and Model Zoo
Download the checkpoint for 3D-detection [model-zoo](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/model_zoo.md). This repository evaluates the performance of [Pointpillars](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/pointpillars) and [Regnet](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/regnet) models pretrained on nuScences dataset. \
Test by running a rosbag and subscribe to appropriate rostopics 
```
rosbag play rosbag_name.bag
python3 mmdetection3d/demo/pcd_demo_class.py
```

Convert the checkpoint into ONNX file and than to tensorRT .engine file
```
python3 mmdeploy/tools/deploy.py
```
ONNX file can be visualized using [NETRON](https://github.com/lutzroeder/netron).

## Implementation results on rosbags

<p align="center">
  <img src="images/2.gif" width="280" height="240" alt="Image 1 Description">
  <img src="images/3.gif" width="280" height="240" alt="Image 2 Description">
</p>

<p align="center">
  <em class="caption">Scenario 1: Detecting multiple objects in the scene</em>
  <em class="caption">Scenario 2: Accurately retaining detection and track ids</em>
</p>

<p align="center">
  <img src="images/4.gif" width="280" height="240" alt="Image 3 Description">
  <br>
  <em> Sceanrio 3 : Detecting objects in the reverse direction </em>
</p>

<p align="center">
  <img src="images/5.gif" width="700" height="300" alt="Image 4 Description">
  <br>
  <em> Sceanrio 4 : Detecting a virtual vehicle on point-cloud data </em>
</p>

## Deployment results 

<p align="center">
  <img src="images/6.gif" width="280" height="240" alt="Image 6 Description">
  <img src="images/7.gif" width="280" height="240" alt="Image 7 Description">
</p>

<p align="center">
  <em class="caption"> Scene : Pointpillar + Regnet</em>
  <em class="caption"> Scene : Pointpillar (TensorRT)</em>
</p>


| Backbone | mAP (nuScenes) | FPS | Memory (GPU) |
| --------------- | --------------- | --------------- | --------------- |
| **[Pointpillars](https://github.com/open-mmlab/mmdetection3d/tree/1.0/configs/pointpillars)**    | 39.26    | 8.88  | 3.5 Gb |
| **Pointpillars (TensorRT)**    | ----    | **15.15**    | 4.5 Gb |
| **Pointpillar + [Regnet](https://github.com/open-mmlab/mmdetection3d/tree/1.0/configs/regnet)**    | **48.2**    | 7.76    | 3.5 Gb |

## Licence
This project is released under the [Apache 2.0 license](https://github.com/ksm26/ROS-based-3D-detection-Tracking/blob/main/LICENSE).

## Reference 
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab model deployment framework.
- [3D-Multi-Object-Tracker](https://github.com/hailanyi/3D-Multi-Object-Tracker): Tracking multiple objects in 3D scene.
