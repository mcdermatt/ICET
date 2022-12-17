#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
import numpy as np
from ICET.msg import Num #custom message type
import std_msgs.msg as std_msgs
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
import sensor_msgs
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
import tensorflow #as tf #pretty dumb naming convention smh

import sys
# import tf 
import tf_conversions
import tf2_ros
import geometry_msgs.msg

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3

#limit GPU memory ---------------------------------------------------------------------
# if you don't include this TensorFlow WILL eat up all your VRAM and make rviz run poorly
gpus = tensorflow.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
  try:
    memlim = 4*1024
    tensorflow.config.experimental.set_virtual_device_configuration(gpus[0], [tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=memlim)])
  except RuntimeError as e:
    print(e)
#--------------------------------------------------------------------------------------

from ICET_spherical import ICET #my point cloud registration algorithm

#TODO: don't output anything if LIDAR just reset
#      clear queue???

class ScanMatcher():
    """scanmatcher node subscribes to /point_cloud topic and attemts to 
        register sequential point clouds. 

        ICET can only run at ~2-3 Hz so it will miss some point cloud frames

        Publishes 6dof transformation with associated frame IDs"""
    
    def __init__(self, scan_topic="raw_point_cloud"):

        rospy.init_node('scanmatcher', anonymous=True)

        self.scan_sub = rospy.Subscriber(scan_topic, PointCloud2, self.on_scan)
        self.etc_sub = rospy.Subscriber('lidar_info', Num, self.get_info)
        self.TPub = rospy.Publisher('relative_transform', Floats, queue_size = 10) #simple array output
        self.SigmaPub = rospy.Publisher('relative_covariance', Floats, queue_size = 10)
        #for publishing corrected point clouds with moving objects removed
        self.pcPub = rospy.Publisher('static_point_cloud', PointCloud2, queue_size = 1)


        #tf uses "broadcasters" instead of publishers
        # self.broadcaster = tf2_ros.StaticTransformBroadcaster() #tf static transform: don't change over time (not what we want!)
        self.broadcaster = tf2_ros.TransformBroadcaster() #tf transform: assumed to change over time

        #TEST~~~~~~~
        self.odom_pub = rospy.Publisher("odom", Odometry, queue_size=50)
        self.odom_broadcaster = tf2_ros.TransformBroadcaster()
        #~~~~~~~~~~~

        r = 100 #not going to be able to actually run this fast, but no harm in setting at 10 Hz
        self.rate = rospy.Rate(r)

        self.remove_moving_objects = False
        # self.remove_moving_objects = True


        self.reset()

    def reset(self):
        self.keyframe_scan = None #init scans1 and 2
        self.new_scan = None
        self.x0 = tensorflow.constant([0., 0., 0., 0., 0., 0.])

    def get_info(self, data):
        """ Gets Lidar info from custom Num msg """
        self.scan_data = data

        # if self.scan_data.frame < 3:
        #     self.reset()

        if self.scan_data.restart == True:
            self.reset()

    def on_scan(self, scan):
        """Finds the transformation between current and previous frames"""

        #convert new point cloud msg to np array
        gen = point_cloud2.read_points(scan, skip_nans=True, field_names=("x", "y", "z"))
        xyz = []
        for p in gen:
            xyz.append(p)
        self.pc_xyz = np.array(xyz)
        # print("newscan_xyz", np.shape(self.pc_xyz)) #fixed size for Ouster dataset

        #init
        if self.keyframe_scan is not None:
            if self.new_scan is None:
                self.new_scan = self.pc_xyz
                self.new_scan_idx = self.scan_data.frame
                print("new_scan_idx = ", self.new_scan_idx, "\n")
        else:
            self.keyframe_scan = self.pc_xyz
            self.keyframe_idx = self.scan_data.frame
            print("keyframe_idx = ", self.keyframe_idx)

        if self.keyframe_scan is not None and self.new_scan is not None:

            it = ICET(cloud1 = self.keyframe_scan, cloud2 = self.new_scan, fid = 50, niter = 5, 
                draw = False, group = 2, RM = self.remove_moving_objects, DNN_filter = False, x0 = self.x0)

            self.X = it.X
            self.pred_stds = it.pred_stds 
            self.new_scan_static_points = it.cloud2_static  #hold on to non-moving points

            self.x0 = self.X    #set inital conditions for next registration 

            #publish estimated local transform and error covariance
            # [x, y, z, phi, theta, psi, idx_keyframe, idx_newframe]
            T_msg = np.append(self.X.numpy(), (self.keyframe_idx, self.new_scan_idx)) 
                #produces very weired bug when publishing np_msg -> just use Floats
            print("\n T_msg", T_msg)
            self.TPub.publish(T_msg)
            sigma_msg = np.append(self.pred_stds, (self.keyframe_idx, self.new_scan_idx))
            print("\n sigma_msg", sigma_msg)
            self.SigmaPub.publish(sigma_msg)


            #broadcast transform using tf message--------------------------------------- 
            #DEBUG WITH <rosrun rqt_tf_tree rqt_tf_tree>

            t = geometry_msgs.msg.TransformStamped()
            t.header.stamp = rospy.Time.now()

            #set parent frame
            t.header.frame_id = "map" #(set as origin)
            # t.header.frame_id = "world" #test
            # t.header.frame_id = "last_frame" #TODO-- set as previously stamped frame in this ICET thread

            child_name = 'child_tf_frame'
            t.child_frame_id = child_name
            t.transform.translation.x = self.X[0]
            t.transform.translation.y = self.X[1]
            t.transform.translation.z = self.X[2]
            q = tf_conversions.transformations.quaternion_from_euler(self.X[3], self.X[4], self.X[5])
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]
            # print(t)
            self.broadcaster.sendTransform(t) #shares the transform but doesn't actually publish? (need to manually add pub node in launch file)

            #ODOMETRY MESSAGES
            odom = Odometry()
            odom.header.stamp = rospy.Time.now()
            odom.header.frame_id = "map"
            # set the position
            odom_quat = tf_conversions.transformations.quaternion_from_euler(self.X[3], self.X[4], self.X[5])
            odom.pose.pose = Pose(Point(self.X[0], self.X[1], self.X[2]), Quaternion(*odom_quat))
            # set the velocity
            odom.child_frame_id = "base_link"
            # odom.twist.twist = Twist(Vector3(vx, vy, 0), Vector3(0, 0, vth))
            # publish the message
            self.odom_pub.publish(odom)

            #----------------------------------------------------------------------------

            #publish non-moving points in new scan
            #   (overridden by self.keyframe_scan if ICET isn't running moving object suppression)
            self.pcPub.publish(point_cloud(it.cloud2_static, 'map'))

            print("\n keyframe idx = ", self.keyframe_idx)
            print("new_scan idx = ", self.new_scan_idx)
            self.keyframe_scan = self.new_scan
            self.keyframe_idx = self.new_scan_idx
            self.new_scan = self.pc_xyz
            self.new_scan_idx = self.scan_data.frame

def point_cloud(points, parent_frame):
    """ Creates a point cloud message.
    Args:
        points: Nx3 array of xyz positions (m) and rgba colors (0..1)
        parent_frame: frame in which the point cloud is defined
    Returns:
        sensor_msgs/PointCloud2 message
    """
    ros_dtype = PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize

    data = points.astype(dtype).tobytes()

    fields = [PointField(
        name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
        for i, n in enumerate('xyzrgba')]

    header = std_msgs.Header(frame_id=parent_frame, stamp=rospy.Time.now())

    return PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * 3),
        row_step=(itemsize * 3 * points.shape[0]),
        data=data
    )

if __name__ == '__main__':
    m = ScanMatcher()

    while not rospy.is_shutdown():

        rospy.spin()