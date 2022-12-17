# ICET

This repository contains code for our Iterative Closest Ellipsoidal Transform (ICET) point cloud registration algorithm. Code is structured in a ROS package to demonstrate real-time HD Map generation. Also included is a [jupyter notebook](https://github.com/mcdermatt/ICET/blob/main/src/ICET_demo.ipynb) with inline 3D visualization so our algorithm can be tested without needing to set up a ROS enviornment on your machine. 

![](https://github.com/mcdermatt/ICET/blob/main/demo1.gif)


## Compile

Use the following commands to download and compile ICET
```
cd ~/catkin_ws/src
git clone https://github.com/mcdermatt/ICET
cd ..
catkin_make
```
This package was developed and tested with ROS Noetic

## Real-time HD Map generation

Begin by sourcing your workspace and running a roscore
```
source ~/catkin_ws/devel/setup.bash
roscore
```
Uncomment the chunk of code in lines 86-130 of the fake_lidar node that corresponds to the the desired filetype of your point cloud data.

Rename 'fn1' to match your point clouds files.

Open a new terminal and run the launch file
```
source ~/catkin_ws/devel/setup.bash
roslaunch ICET simple_mapping.launch
```

HD Maps are published as PointCloud2 messages in the "/hd_map" topic which can be viewed via RViz

A "snail trail" showing the trajectory of the vehicle is also published as a PointCloud2 message "/snail_trail"

Estimated transforms between frames and the associated covarince estimates output by ICET are published to /relative_transform and  /relative_covariance respectively 

## Cite ICET

Thank you for citing our work if you have used any of our code: 

[Mitigating Shadows in LIDAR Scan Matching Using Spherical Voxels](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9928328) 
```
@ARTICLE{SphericalICET,
  author={McDermott, Matthew and Rife, Jason},
  journal={IEEE Robotics and Automation Letters}, 
  title={Mitigating Shadows in LIDAR Scan Matching Using Spherical Voxels}, 
  year={2022},
  volume={7},
  number={4},
  pages={12363-12370},
  doi={10.1109/LRA.2022.3216987}}
}
```

[DNN Filter for Bias Reduction in Distribution-to-Distribution Scan Matching](https://arxiv.org/pdf/2211.04047.pdf) 
```
@article{mcdermott2022dnn,
  title={DNN Filter for Bias Reduction in Distribution-to-Distribution Scan Matching},
  author={McDermott, Matthew and Rife, Jason},
  journal={arXiv preprint arXiv:2211.04047},
  year={2022}
}
```

[Enhanced Laser-Scan Matching with Online Error Estimation for Highway and Tunnel Driving](https://arxiv.org/pdf/2207.14674.pdf) 
```
@inproceedings{2DICET,
  title={Enhanced Laser-Scan Matching with Online Error Estimation for Highway and Tunnel Driving},
  author={McDermott, Matthew and Rife, Jason},
  booktitle={Proceedings of the 2022 International Technical Meeting of The Institute of Navigation},
  pages={643--654},
  year={2022}
}
```
