cmake_minimum_required(VERSION 3.0.2)
project(icet)

find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  std_msgs
  message_generation
  nav_msgs
  tf
  tf2
  tf2_ros
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(PCL REQUIRED COMPONENTS common io)

find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  Num.msg
)
catkin_package(
 INCLUDE_DIRS include
 LIBRARIES icet
 CATKIN_DEPENDS rospy roscpp std_msgs tf2 tf2_ros message_runtime
 DEPENDS system_lib
)

# Set optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

#NEW -- add library to allow other packages to use ICET stuff-------
add_library(icet
  src/icet.cpp 
  src/utils.cpp
  src/ThreadPool.cpp
)
target_link_libraries(icet
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  tbb
)

# Make sure the library is exported properly
add_dependencies(icet ${catkin_EXPORTED_TARGETS})
#------------------------------------------------------------------

## Specify additional locations of header files
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/csv-parser/single_include
)

add_definitions(${PCL_DEFINITIONS})

add_executable(map_maker_node 
                src/simpleMapMaker.cpp 
                src/icet.cpp 
                src/utils.cpp
                src/ThreadPool.cpp)
target_link_libraries(map_maker_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  tbb
)


add_executable(odometry_node 
                src/odometry.cpp 
                src/icet.cpp 
                src/utils.cpp
                src/ThreadPool.cpp)
target_link_libraries(odometry_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  tbb
)
