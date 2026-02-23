# ICET

This repository contains code for our Iterative Closest Ellipsoidal Transform (ICET) point cloud registration algorithm. C++ code is provided in a ROS package to demonstrate real-time HD Map generation. Also included is a python implementation of ICET and interactive [jupyter notebook](https://github.com/mcdermatt/ICET/blob/main/python/ICET_demo.ipynb) with inline 3D visualization so our algorithm can be tested without needing to set up a ROS enviornment on your machine. 

![](https://github.com/mcdermatt/ICET/blob/main/figures/demo3.gif)

## Build

Use the following commands to download and compile ICET in a ROS2 workspace:
```bash
cd ~/ros2_ws/src
git clone https://github.com/mcdermatt/ICET
cd ~/ros2_ws
colcon build --packages-select icet
source install/setup.bash
```

### Dependencies

- ROS2 (tested with Jazzy)
- `rclcpp`, `sensor_msgs`, `nav_msgs`, `geometry_msgs`
- `tf2_ros`, `tf2_geometry_msgs`
- `pcl_conversions`, `pcl_ros`
- Eigen3

Install missing dependencies with:
```bash
sudo apt-get install -y freeglut3-dev
rosdep install --from-paths src --ignore-src -r -y
```

## Real-time HD Map generation

![](https://github.com/mcdermatt/ICET/blob/main/figures/map1.jpg)

Play a recorded bag in one terminal:
```bash
source ~/ros2_ws/install/setup.bash
ros2 bag play my_bag/
```

Run the map maker node in another terminal:
```bash
source ~/ros2_ws/install/setup.bash
ros2 run icet map_maker_node
```

HD Maps are published as PointCloud2 messages on the `/hd_map` topic which can be viewed via RViz2.

A "snail trail" showing the trajectory of the vehicle is also published as a PointCloud2 message on `/snail_trail_topic`.

Alternatively, an odometry node can be run which outputs transforms and accurate error bounds to a standard odometry message:

```bash
ros2 run icet odometry_node
```

Or use the included launch file (auto-loads config YAML):
```bash
ros2 launch icet icet_launch.py
```

## Configuration

Algorithm parameters are loaded from `config/icet_config.yaml` at runtime — **no rebuild needed** when you change them. To load parameters when running a node directly:

```bash
ros2 run icet odometry_node --ros-args --params-file ~/ros2_ws/src/ICET/config/icet_config.yaml
```

Or with the launch file (parameters are loaded automatically):
```bash
ros2 launch icet icet_launch.py
```

Key parameters:
| Parameter | Default | Description |
|---|---|---|
| `num_bins_phi` | 24 | Elevation voxels |
| `num_bins_theta` | 75 | Azimuthal voxels |
| `run_length` | 7 | ICET iterations per scan pair |
| `min_points` | 25 | Minimum points per voxel |
| `cluster_thresh` | 0.1 | Radial jump threshold (m) |
| `cluster_buff` | 0.1 | Radial buffer (m) |
| `min_distance` | 0.2 | Filter points closer than this (m) |
| `trans_thresh` | 3.0 | Max translation per scan (m) |
| `rot_thresh` | 0.5 | Max rotation per scan (rad) |

## Real-time OpenGL Visualization

An odometry node with a built-in OpenGL visualizer shows point clouds, spherical voxel bounds, and covariance ellipsoids updating in real time:

```bash
ros2 run icet odometry_viz_node --ros-args --params-file ~/ros2_ws/src/ICET/config/icet_config.yaml
```

![](https://github.com/mcdermatt/ICET/blob/main/figures/cppviz.png)

Controls:
- **Left-click drag** — rotate
- **Middle-click drag** — pan
- **Scroll wheel** — zoom
- **Arrow keys** — zoom in/out
- **Esc** — quit

## Benchmarking

Runtime for scan registration is dependant on sensor resolution, computer hardware, voxel resolution, and the number of iterations the algorithm is allowed to run. In our testing, we were able to achieve a optimal registration accuracy with a mean registration time of 25ms for 64-channel data running for 7 iterations on a Ryzen 5800x CPU and with a resolution of 75 azimuthal voxels and 24 elevation voxels.   

Raspberry pi 5 comfortably runs at 45ms for 32-channel data with 7 iterations and a resolution of 75 azimuthal voxels and 24 elevation voxels.

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

