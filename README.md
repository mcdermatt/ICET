# ICET

This repository contains code for our Iterative Closest Ellipsoidal Transform (ICET) point cloud registration algorithm. C++ code is provided in a ROS package to demonstrate real-time HD Map generation. Also included is a python implementation of ICET and interactive [jupyter notebook](https://github.com/mcdermatt/ICET/blob/main/python/ICET_demo.ipynb) with inline 3D visualization so our algorithm can be tested without needing to set up a ROS enviornment on your machine. 

![](https://github.com/mcdermatt/ICET/blob/main/figures/demo3.gif)

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

The included fake_lidar node provides an interface for loading popular autonomous driving datasets from a .csv, .bin, or .npy format and publishing point cloud messages in the same format as the Velodyne VLP-32 sensor. Modify the fake_lidar.py file to point towards the correct local directory containing your dataset and run this node in a new terminal with:

```
source ~/catkin_ws/devel/setup.bash
rosrun icet fake_lidar 
```  

Alternatively, the included package can be run directly off a ROSBAG using the following command. Make sure that the simpleMapMaker.cpp code is subscribed to the correct topic!
```
source ~/catkin_ws/devel/setup.bash
rosbag play my_bag.bag
```

A real time odometry and mapping demo can be run with the following node in another terminal:
```
source ~/catkin_ws/devel/setup.bash
rosrun icet map_maker_node
```

HD Maps are published as PointCloud2 messages in the "/hd_map" topic which can be viewed via RViz

A "snail trail" showing the trajectory of the vehicle is also published as a PointCloud2 message "/snail_trail"

Changes in pose between subsequent scans are broadcast as ROS Transforms and can be viewed with:

```
rostopic echo /tf
```

![](https://github.com/mcdermatt/ICET/blob/main/figures/map1.jpg)

An OpenGL visualizer is included which can be helpful for tuning the C++ implementation of ICET for new sensors and enviornments. Exaple use is demonstrated in the icetViz.cpp file. 

![](https://github.com/mcdermatt/ICET/blob/main/figures/cppviz.jpg)


## Cite ICET

Thank you for citing our work if you have used any of our code: 

[ICET Online Accuracy Characterization for Geometry-Based Laser Scan Matching](https://navi.ion.org/content/navi/71/2/navi.647.full.pdf) 
```
@article {ICET,
  author = {McDermott, Matthew and Rife, Jason},
  title = {ICET Online Accuracy Characterization for Geometry-Based Laser Scan Matching},
  volume = {71},
  number = {2},
  elocation-id = {navi.647},
  year = {2024},
  doi = {10.33012/navi.647},
  publisher = {Institute of Navigation},
  issn = {0028-1522},
  URL = {https://navi.ion.org/content/71/2/navi.647},
  eprint = {https://navi.ion.org/content/71/2/navi.647.full.pdf},
  journal = {NAVIGATION: Journal of the Institute of Navigation}
}
```
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

![](https://github.com/mcdermatt/ICET/blob/main/demo2.gif)

