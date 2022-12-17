#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
import numpy as np
from ICET.msg import Num #using custom message type
from rospy_tutorials.msg import Floats
import std_msgs.msg as std_msgs
import sensor_msgs
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField

from utils import R_tf

class MapMaker():
    """ simplified mapmaker node uses sequential transformations output by <scanmatcher> to combine multiple point clouds
        into single HD map."""
    
    def __init__(self, scan_topic="static_point_cloud"):

        rospy.init_node('mapmaker', anonymous=True)

        self.scan_sub = rospy.Subscriber(scan_topic, PointCloud2, self.on_scan)
        self.etc_sub = rospy.Subscriber('lidar_info', Num, self.get_info)
        self.mapPub = rospy.Publisher('hd_map', PointCloud2, queue_size = 10)
        self.snailPub = rospy.Publisher('snail_trail', PointCloud2, queue_size = 1)
        self.downsample_size = 100_000 #10_000 #size of each sub-cloud

        #subscribe to local transformation estimates output by ScanMatcher
        self.Tsub = rospy.Subscriber('relative_transform', Floats, self.on_transform) 

        self.map_xyz = np.array([[0., 0., 0.]])
        self.snail_trail = np.array([[0., 0., 0.]])

        self.global_pose = np.array([0.,0.,0., 0., 0., 0.])
        self.rot = 1 #debug

        r = 100
        self.rate = rospy.Rate(r)

    def point_cloud(self, points, parent_frame):
        """ Creates a point cloud message.
        https://gist.github.com/pgorczak/5c717baa44479fa064eb8d33ea4587e0
        Args:
            points: Nx7 array of xyz positions (m) and rgba colors (0..1)
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

    def update_map(self):
        """add new scan to HD map"""

        # self.newscan_xyz = self.newscan_xyz[self.newscan_xyz[:,2] > -1.65] #remove ground plane
        self.newscan_xyz = self.newscan_xyz[np.random.choice(len(self.newscan_xyz), size = self.downsample_size)]  #downsample

        #add two clouds together to update Map--------------------------
        #keep vehicle in center of frame~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~      
        #[x, y, z, phi, theta, psi, idx_keyframe, idx_newframe]
        trans = self.local_estimate[:3]
        print("\n trans:", trans)
        rot = R_tf(-np.array(self.local_estimate[3:6])).numpy()
        print("rot", rot)
        transformed_old_map = (self.map_xyz - trans).dot(rot) #trans then rotate needed for ICET transform outputs
        self.map_xyz = np.append(transformed_old_map, self.newscan_xyz, axis = 0)
        self.mapPub.publish(self.point_cloud(self.map_xyz, 'map'))
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # #transform new PC so map stays in place ~~~~~~~~~~~~~~~~~~~~~~~
        # self.global_pose[3:] += self.local_estimate[3:6] #update angles
        # self.global_pose[:3] += np.array(self.local_estimate[:3]).dot(R_tf(self.global_pose[3:]).numpy()) #update xyz

        # print("\n global_pose", self.global_pose)
        # # self.rot = R_tf(np.array(self.local_estimate[3:6])).numpy().dot(self.rot) #??
        # self.rot = R_tf(self.global_pose[3:])
        # print("rot", self.rot)

        # # transformed_new_scan = (self.newscan_xyz + self.global_pose[:3]).dot(self.rot)
        # transformed_new_scan = (self.newscan_xyz.dot(self.rot) + self.global_pose[:3]) #almost works??

        # self.map_xyz = np.append(self.map_xyz, transformed_new_scan, axis = 0)
        # self.mapPub.publish(self.point_cloud(self.map_xyz, 'map'))
        # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


        #---------------------------------------------------------------

    def update_snail_trail(self):
        """ update trail of points that visualize the vehicle COM over each frame """

        # [x, y, z, phi, theta, psi, idx_keyframe, idx_newframe]
        trans = self.local_estimate[:3]
        rot = R_tf(-np.array(self.local_estimate[3:6])).numpy()
        transformed_old_trail = (self.snail_trail - trans).dot(rot)
        self.snail_trail = np.append(transformed_old_trail, np.zeros([1,3]), axis = 0)

        #keep vehicle in center of frame~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~      
        self.snailPub.publish(self.point_cloud(self.snail_trail, 'map')) #coordinate frame = map
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        #stay centered at starting location~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # self.snailPub.publish(self.point_cloud(self.snail_trail + self.global_pose[:3], 'map')) #coordinate frame = map
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    def get_info(self, data):
        """ Gets Lidar info from custom Num msg """

        # rospy.loginfo("got data!")
        self.scan_data = data
        print("frame idx:", data.frame)

        if self.scan_data.restart == True:
            print("restart: ", self.scan_data.restart)
            self.map_xyz = np.array([[0., 0., 0.]])  #Clear map
            self.mapPub.publish(self.point_cloud(self.map_xyz, 'map')) #publish updated map
            self.snail_trail = np.array([[0., 0., 0.]])  #Clear snail trail
            self.snailPub.publish(self.point_cloud(self.snail_trail, 'map')) #coordinate frame = map
            # self.global_pose = np.array([0.,0.,0.])
            self.global_pose = np.array([0.,0.,0., 0., 0., 0.])


    def on_transform(self, local_estimate):
        """called when ScanMatcher node publishes local transformation estimate"""

        self.local_estimate = local_estimate.data
        print("self.local_estimate", self.local_estimate)

    def on_scan(self, scan):
        # https://answers.ros.org/question/202787/using-pointcloud2-data-getting-xy-points-in-python/

        gen = point_cloud2.read_points(scan, skip_nans=True, field_names=("x", "y", "z"))
        # self.xyz_generator = gen

        xyz = []
        for p in gen:
            xyz.append(p)
        self.newscan_xyz = np.array(xyz)
        # print("newscan_xyz", np.shape(self.newscan_xyz))

        # rospy.loginfo("Got scan!")
        # print(self.newscan_xyz[:10])

        self.update_map()
        self.update_snail_trail()

if __name__ == '__main__':
    m = MapMaker()

    while not rospy.is_shutdown():

        rospy.spin()