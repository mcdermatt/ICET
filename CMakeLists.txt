cmake_minimum_required(VERSION 3.8)
project(icet)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# Eigen alignment
add_definitions(-DEIGEN_ALIGN_16)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
# freeglut3-dev provides the GLUT headers/libs
find_package(GLUT REQUIRED)

# Generate custom messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Num.msg"
  DEPENDENCIES builtin_interfaces
)

# Include directories
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)
add_definitions(${PCL_DEFINITIONS})

# ---- ICET core library (no ROS dependency) ----
add_library(icet_core
  src/icet.cpp
  src/utils.cpp
  src/ThreadPool.cpp
)
target_include_directories(icet_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(icet_core
  ${PCL_LIBRARIES}
  tbb
)

# ---- Visualization library (OpenGL) ----
add_library(icet_viz
  src/visualization.cpp
)
target_include_directories(icet_viz PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(icet_viz
  OpenGL::GL
  OpenGL::GLU
  GLEW::GLEW
  GLUT::GLUT
)

# ---- Odometry Node ----
add_executable(odometry_node src/odometry.cpp)
target_link_libraries(odometry_node icet_core ${PCL_LIBRARIES} tbb)
ament_target_dependencies(odometry_node
  rclcpp sensor_msgs nav_msgs geometry_msgs
  tf2 tf2_ros tf2_geometry_msgs pcl_conversions
)

# ---- Map Maker Node ----
add_executable(map_maker_node src/simpleMapMaker.cpp)
target_link_libraries(map_maker_node icet_core ${PCL_LIBRARIES} tbb)
ament_target_dependencies(map_maker_node
  rclcpp sensor_msgs nav_msgs geometry_msgs
  tf2 tf2_ros tf2_geometry_msgs pcl_conversions
)

# ---- Odometry + Visualization Node ----
add_executable(odometry_viz_node src/odometryViz.cpp)
target_link_libraries(odometry_viz_node icet_core icet_viz ${PCL_LIBRARIES} tbb)
ament_target_dependencies(odometry_viz_node
  rclcpp sensor_msgs nav_msgs geometry_msgs
  tf2 tf2_ros tf2_geometry_msgs pcl_conversions
)

# Install targets
install(TARGETS odometry_node map_maker_node odometry_viz_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS icet_core icet_viz
  EXPORT icet_coreTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Export
ament_export_targets(icet_coreTargets HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_export_libraries(icet_core icet_viz)
ament_export_dependencies(PCL)

ament_package()
