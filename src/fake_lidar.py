#!/usr/bin/env python3
import rospy # import rospy, the package that lets us use ros in python. 
from ICET.msg import Num #using custom message type
import numpy as np
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
import pandas as pd
import std_msgs.msg as std_msgs
import sensor_msgs
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
import trimesh #when using KITTI_CARLA dataset
from utils import R_tf
import pykitti
import tensorflow as tf
from time import sleep

#limit GPU memory ---------------------------------------------------------------------
# if you don't include this TensorFlow WILL eat up all your VRAM and make rviz run poorly
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
  try:
    memlim = 4*1024
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memlim)])
  except RuntimeError as e:
    print(e)
#--------------------------------------------------------------------------------------


"""script to publish custom LIDAR point cloud messages"""

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

def main():

    # publish numpy array 
    # pcNpyPub = rospy.Publisher('numpy_cloud', numpy_msg(Floats), queue_size = 1)

    # traditional pointcloud2 msg
    pcPub = rospy.Publisher('raw_point_cloud', PointCloud2, queue_size = 1)

    # publish custom message with additional LIDAR info
    etcPub = rospy.Publisher('lidar_info', Num, queue_size=1) #This publisher can hold 10 msgs that haven't been sent yet.  
    rospy.init_node('LidarScanner', anonymous=False)
    # r = 10 #real time (too fast to work at full resolution)
    # r = 5 #slower real time (many sensors run at 5Hz)
    r = 2
    # r = 1
    rate = rospy.Rate(r) # hz
    
    start_time = rospy.get_time()

    while not rospy.is_shutdown(): # While there's a roscore

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # # use simulated point clouds stored in .csv I generated for a previous publication 
        # # https://github.com/mcdermatt/ASAR/tree/main/v3/spherical_paper/MC_trajectories
        # runlen = 40 #this many frames in the dataset
        # # idx = int(r*rospy.get_time()%runlen) + 1 #use ROS timestamp as seed for scan idx
        # # idx = int(2*r*rospy.get_time()%runlen) + 1 #use ROS timestamp as seed for scan idx
        # idx = int(r*(rospy.get_time() - start_time)%runlen) + 1 #test
        # print(idx)
        # fn = "/home/derm/ASAR/v3/spherical_paper/MC_trajectories/scene1_scan" + str(idx) + ".txt"
        # pcNpy = np.loadtxt(fn)
        # pcNpy += 0.02*np.random.randn(np.shape(pcNpy)[0], 3) #simulate sensor noise (important)
        # rot = R_tf(tf.constant([0., 0., idx*0.05])) #simulate constant turning
        # pcNpy = pcNpy @ rot.numpy() 

        # #use KITTI_CARLA synthetic LIDAR data
        # idx = int(r*rospy.get_time()%400) + 2300 #use ROS timestamp as seed for scan idx
        # fn = '/home/derm/KITTICARLA/dataset/Town01/generated/frames/frame_%04d.ply' %(idx)
        # print("Publishing", fn)
        # dat1 = trimesh.load(fn)
        # pcNpy = dat1.vertices
        # # pcNpy = pcNpy[pcNpy[:,2] > -1.5] #debug

        # # use KITTI raw dataset
        # basedir = '/media/derm/06EF-127D2/KITTI'
        # date = '2011_09_26'
        # drive = '0005' #traffic circle, city, 150
        # # drive = '0027' #straght highway, dense forest
        # # drive = '0095' #dense residential, 260
        # dataset = pykitti.raw(basedir, date, drive)
        # runlen = 150
        # # idx = int(r*rospy.get_time()%150) + 1
        # idx = int(r*(rospy.get_time() - start_time)%runlen) + 1
        # velo1 = dataset.get_velo(idx) # Each scan is a Nx4 array of [x,y,z,reflectance]
        # pcNpy = velo1[:,:3]
        # print("Publishing", basedir + '/' + date + '/' + drive + '/' + str(idx))

        #use Ouster sample dataset (from high fidelity 128-channel sensor), .pcap -> .csv 719 frames total
        runlen = 615 #715
        idx = int(r*(rospy.get_time() - start_time)%runlen) + 100 #+ 1
        fn1 = "/media/derm/06EF-127D2/Ouster/csv/pcap_out_" + '%06d.csv' %(idx)
        print("Publishing", fn1)
        df1 = pd.read_csv(fn1, sep=',', skiprows=[0])
        pcNpy = df1.values[:,8:11]*0.001 #1st sensor return (what we want)
        # pcNpy = df1.values[:,11:14]*0.001 #2nd sensor return (not useful here)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # pcNpyPub.publish(pcNpy)                #publish point cloud as numpy_msg (rospy)
        pcPub.publish(point_cloud(pcNpy, 'map')) #publish point cloud as point_cloud2 message

        # publish custom <Num> message type~~~~~~~~~~~~~~~~~
        msg = Num()
        msg.frame = idx
        # if idx == 1: 
        if idx < 3: #for debug
            msg.restart = True
        else:
            msg.restart = False

        #for debug: provide prior knowledge of ground truth for simulated dummy data
        msg.true_transform = [0.0, 0.0, 0.0, 0.0, 0.0, 0.00] #[x, y, z, r, p, y]

        status_str =  "Frame # " + str(idx) + " Lidar Timestamp %s" % rospy.get_time() # 
        msg.status = status_str

        # Publish the string message and log it (optional)
        # rospy.loginfo(msg)
        etcPub.publish(msg)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        # sleep so we don't publish way too many messages per second. This also gives us a chance to process incoming messages (but this node has no incoming messages)
        rate.sleep()

        if idx == runlen:
            print("\n resetting LIDAR sensor")
            sleep(5)
            print("\n done")
            start_time = rospy.get_time()

if __name__ == '__main__': # this runs when the file is run
    try: # the try catch block allows this code to easily be interrupted without hanging
        main() # run the main function
    except rospy.ROSInterruptException:
        pass