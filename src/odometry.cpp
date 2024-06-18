#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/buffer.h>  // Add this line for tf2_ros::Buffer
#include <pcl/registration/icp.h>
#include <Eigen/Dense>
#include <random>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Quaternion.h>
#include <tf/tf.h>
#include "icet.h"
#include "utils.h"

using namespace std;
using namespace Eigen;


class OdometryNode {
public:
    OdometryNode() : nh_("~"), initialized_(false) {
        // Set up ROS subscribers and publishers
        pointcloud_sub_ = nh_.subscribe("/velodyne_points", 10, &OdometryNode::pointcloudCallback, this); //use when connected to Velodyne VLP-16
        // pointcloud_sub_ = nh_.subscribe("/os1_cloud_node/points", 10, &OdometryNode::pointcloudCallback, this); //use when connected to Ouster OS1 sensor
        // pointcloud_sub_ = nh_.subscribe("/raw_point_cloud", 10, &OdometryNode::pointcloudCallback, this); //use with fake_lidar node
        odom_pub = nh_.advertise<nav_msgs::Odometry>("/odom", 50);

        X0.resize(6);
        X0 << 0., 0., 0., 0., 0., 0.; 

        ros::Rate rate(10);
    }

    void pointcloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
        auto before = std::chrono::system_clock::now();
        auto beforeMs = std::chrono::time_point_cast<std::chrono::milliseconds>(before);
        frameCount++;

        // Convert PointCloud2 to PCL PointCloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *pcl_cloud);

        if (!initialized_) {
            // On the first callback, initialize prev_pcl_cloud_
            prev_pcl_cloud_ = pcl_cloud;
            initialized_ = true;
            prev_pcl_matrix = convertPCLtoEigen(prev_pcl_cloud_);
            return;
        }

        // Convert PCL PointCloud to Eigen::MatrixXf
        Eigen::MatrixXf pcl_matrix = convertPCLtoEigen(pcl_cloud);

        // Filter out points less than distance 'd' from the origin
        float minD = 2.;
        vector<int> not_too_close_idxs;
        for (int i = 0; i < pcl_matrix.rows(); i++){
            float distance = pcl_matrix.row(i).norm();
            if (distance > minD){
                not_too_close_idxs.push_back(i);
            }
        }
        Eigen::MatrixXf filtered_pcl_matrix(not_too_close_idxs.size(), 3);
        for (std::size_t i = 0; i < not_too_close_idxs.size(); i++){
            filtered_pcl_matrix.row(i) = pcl_matrix.row(not_too_close_idxs[i]);
        }
        pcl_matrix = filtered_pcl_matrix;

        // RUN UPDATED ICET CODE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        int run_length = 7;
        int numBinsPhi = 24;
        int numBinsTheta = 75; 
        ICET it(prev_pcl_matrix, pcl_matrix, run_length, X0, numBinsPhi, numBinsTheta);
        Eigen::VectorXf X = it.X;
        cout << "soln: " << endl << X << endl;
        cout << "1-sigma error bounds:" << endl << it.pred_stds << endl;
        //seed initial estimate for next iteration
        // X0 << 0., 0., 0., 0., 0., 0.; 
        X0 << X[0], X[1], X[2], X[3], X[4], X[5]; 
        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        // Convert back to ROS PointCloud2 and publish the aligned point cloud
        MatrixXf rot_mat = utils::R(X[3], X[4], X[5]);
        Eigen::RowVector3f trans(X[0], X[1], X[2]);

        prev_pcl_matrix = pcl_matrix;

        //broadcast transform that relates /velodyne frame to /map frame ~~~~~~~~~~~~~~~~~~~~~
        //convert ICET output X for scan i to homogenous transformation matrix
        Eigen::Matrix4f X_homo_i = Eigen::Matrix4f::Identity();
        X_homo_i.block<3, 3>(0, 0) = rot_mat;
        X_homo_i.block<3, 1>(0, 3) = trans.transpose();

        //update accumulated transfrom
        X_homo = X_homo * X_homo_i;
        std::cout << "X: \n" << X_homo <<endl;

        // Create the odometry message
        nav_msgs::Odometry odom;
        // ros::Time current_time = msg->header.stamp;  // Use lidar scan timestamp --> issues with using old rosbag data
        ros::Time current_time = ros::Time::now(); //use roscore computer timestamp 
        odom.header.stamp = current_time;
        odom.header.frame_id = "map";         // Parent frame 
        odom.child_frame_id = "velodyne";     // Child frame
        // odom.child_frame_id = "/os1_lidar";     // Child frame

        // Set the translation and rotation in the transform message
        odom.pose.pose.position.x = X_homo(0,3);
        odom.pose.pose.position.y = X_homo(1,3);
        odom.pose.pose.position.z = X_homo(2,3);
        // Convert Eigen rotation matrix to quaternion
        Eigen::Matrix3f rotationMatrix = X_homo.topLeftCorner(3, 3);
        Eigen::Quaternionf quaternion(rotationMatrix);
        odom.pose.pose.orientation.x = quaternion.x();
        odom.pose.pose.orientation.y = quaternion.y();
        odom.pose.pose.orientation.z = quaternion.z();
        odom.pose.pose.orientation.w = quaternion.w();

        // Set the pose covariance matrix
        for (int i = 0; i < 36; ++i) {
            odom.pose.covariance[i] = 0.0;
        }
        odom.pose.covariance[0] = it.pred_stds[0];  // x
        odom.pose.covariance[7] = it.pred_stds[1];  // y
        odom.pose.covariance[14] = it.pred_stds[2]; // z
        odom.pose.covariance[21] = it.pred_stds[3]; // roll
        odom.pose.covariance[28] = it.pred_stds[4]; // pitch
        odom.pose.covariance[35] = it.pred_stds[5]; // yaw


        //publish the velocity -- (assumes 10Hz LIDAR sensor)
        odom.twist.twist.linear.x = 10*X[0];
        odom.twist.twist.linear.y = 10*X[1];
        odom.twist.twist.linear.z = 10*X[2];
        odom.twist.twist.angular.x = 10*X[3];
        odom.twist.twist.angular.y = 10*X[4];
        odom.twist.twist.angular.z = 10*X[5];

        odom_pub.publish(odom);

        // Broadcast the transform
        geometry_msgs::TransformStamped odom_trans;
        odom_trans.header.stamp = current_time;
        odom_trans.header.frame_id = "map";
        odom_trans.child_frame_id = "velodyne";
        // Set the translation and rotation in the transform message
        odom_trans.transform.translation.x = X_homo(0,3);
        odom_trans.transform.translation.y = X_homo(1,3);
        odom_trans.transform.translation.z = X_homo(2,3);
        odom_trans.transform.rotation.x = quaternion.x();
        odom_trans.transform.rotation.y = quaternion.y();
        odom_trans.transform.rotation.z = quaternion.z();
        odom_trans.transform.rotation.w = quaternion.w();

        odom_broadcaster.sendTransform(odom_trans);

        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        auto after1 = std::chrono::system_clock::now();
        auto after1Ms = std::chrono::time_point_cast<std::chrono::milliseconds>(after1);
        auto elapsedTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(after1Ms - beforeMs).count();
        std::cout << "Registered scans in: " << elapsedTimeMs << " ms" << std::endl;

        // rate.sleep(); //do I need this??
    }

    Eigen::VectorXf X0;

private:
    ros::NodeHandle nh_;
    ros::Subscriber pointcloud_sub_;
    ros::Publisher odom_pub;
    tf::TransformBroadcaster odom_broadcaster;
    pcl::PointCloud<pcl::PointXYZ>::Ptr prev_pcl_cloud_;
    tf2_ros::Buffer tfBuffer_;
    bool initialized_;
    int frameCount = 0;
    Eigen::MatrixXf prev_pcl_matrix;

    //init variable to hold cumulative homogenous transform
    Eigen::Matrix4f X_homo = Eigen::Matrix4f::Identity(); 

    Eigen::MatrixXf convertPCLtoEigen(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pcl_cloud) {
        Eigen::MatrixXf eigen_matrix(pcl_cloud->size(), 3);
        for (size_t i = 0; i < pcl_cloud->size(); ++i) {
            eigen_matrix.row(i) << pcl_cloud->points[i].x, pcl_cloud->points[i].y, pcl_cloud->points[i].z;
        }
        return eigen_matrix;
    }

    sensor_msgs::PointCloud2 convertEigenToROS(const Eigen::MatrixXf& eigenPointCloud, const std_msgs::Header& header) {
    pcl::PointCloud<pcl::PointXYZ> pclPointCloud;
        // Assuming each row of the Eigen matrix represents a point (x, y, z)
        for (int i = 0; i < eigenPointCloud.rows(); ++i) {
            pcl::PointXYZ point;
            point.x = eigenPointCloud(i, 0);
            point.y = eigenPointCloud(i, 1);
            point.z = eigenPointCloud(i, 2);
            pclPointCloud.push_back(point);
        }

        // Convert PCL point cloud to ROS message
        sensor_msgs::PointCloud2 rosPointCloud;
        pcl::toROSMsg(pclPointCloud, rosPointCloud);
        rosPointCloud.header = header;  // Set the header from the input

        return rosPointCloud;
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "odometry_node");
    OdometryNode odometry_node;
    ros::spin();
    return 0;
}
