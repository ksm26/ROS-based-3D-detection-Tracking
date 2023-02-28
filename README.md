# ROS-based-3D-detection-Tracking

1) Create virtual environment
```
conda create --name rosdeploy python=3.8
conda activate rosdeploy
```

2) Install pytorch 
```
conda install pytorch from https://pytorch.org/ according to cuda version or lower for eg. 
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

3) Clone this repository 
git clone git@github.com:ksm26/ROS-based-3D-detection-Tracking.git

4) Install steps 
```
pip install -U openmim
mim install mmcv-full
mim install mmdet
mim install mmsegmentation
```

5) Install mmdetection3d Documentation - https://mmdetection3d.readthedocs.io/en/latest/getting_started.html
```
cd path_to_cloned_mmdetection3d
pip install -e .
```

Download the checkpoint from 3D-detection model-zoo https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/model_zoo.md
Test by running a rosbag and subscribe to appropriate rostopics 

6) Install mmdeploy Documentation - https://github.com/open-mmlab/mmdeploy/blob/master/docs/en/get_started.md
```
cd path_to_cloned_mmdeploy 
Write - how to create .onnx and .engine file from checkpoint
```

7) Install tf and jsk 
Rostopic tf: http://wiki.ros.org/tf/Tutorials/Writing%20a%20tf%20listener%20%28Python%29
Ros implementation: https://github.com/lgsvl/second-ros

Bounding box visualization in ros:[website] https://blog.csdn.net/weixin_36354875/article/details/125935782
[Install ros jsk_visualization] https://limhyungtae.github.io/2020-09-05-ROS-jsk_visualization-%EC%84%A4%EC%B9%98%ED%95%98%EB%8A%94-%EB%B2%95/
[jsk_recognition_msgs] http://otamachan.github.io/sphinxros/indigo/packages/jsk_recognition_msgs.html
[Bbox visualization using rviz] https://blog.csdn.net/weixin_36354875/article/details/125935782
```
sudo apt-get install ros-melodic-jsk-recognition-msgs 
sudo apt-get install ros-melodic-jsk-rviz-plugins
sudo apt-get install ros-melodic-jsk-rqt-plugins
sudo apt-get install ros-melodic-jsk-visualization
```

-----------------------------------------------------------------------------------------------------------------------------------------------
Had an error with rospy.init_node(): It was an error with ros packages installation into conda environment. Have create a basic yml file for a ros environment.(File of environment) https://drive.google.com/file/d/13-W3iB0F1abQ4Db_VcEm87l_RWf15CmU/view 

Run the above yml using: conda env update --file ros_python_environment.ymlsome\directory

import tf: Error
Import tf does not work with python 3 as it is compiled for python 2. 
Solution: Recompile with python3 in catkin_workspace
https://answers.ros.org/question/326226/importerror-dynamic-module-does-not-define-module-export-function-pyinit__tf2/
To be able to work with python interpreter (pycharm,visual studio) add the compile path as below:
Example:sys.path.append("/home/khushdeep/Desktop/ROS-tracker/catkin_ws/src:/opt/ros/melodic/share")

After tf installation and creation of new catkin workspace: 
Run source ~/catkin_ws/devel/setup.bash

8) Install tensorrt 
pip install nvidia-tensorrt
python3 -m pip install --upgrade tensorrt


