// ICET Odometry node with real-time OpenGL visualization
// Runs GLUT on the main thread, ROS spinning in a background thread.
// After each scan registration, the visualization is updated with the
// latest point clouds, voxel bounds, and covariance ellipsoids.

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <Eigen/Dense>
#include <random>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <thread>
#include <mutex>
#include <GL/glew.h>
#include <GL/glut.h>
#include "icet.h"
#include "utils.h"
#include "visualization.h"

using namespace std;
using namespace Eigen;

// Global visualization instance and mutex for thread-safe updates
visualization* viz = nullptr;
std::mutex vizMutex;
bool vizDataReady = false;

// Viz data to be swapped in on the GLUT thread
Eigen::MatrixXf vizPoints1, vizPoints2, vizClusterBounds;
std::vector<Eigen::Vector3f> vizE1Means, vizE2Means;
std::vector<Eigen::Matrix3f> vizE1Covs, vizE2Covs;
std::vector<float> vizE1Alphas, vizE2Alphas;


class OdometryVizNode : public rclcpp::Node {
public:
    OdometryVizNode() : Node("odometry_viz_node"), initialized_(false) {
        // Declare parameters (loaded from config YAML at runtime)
        this->declare_parameter("num_bins_phi", 24);
        this->declare_parameter("num_bins_theta", 75);
        this->declare_parameter("run_length", 7);
        this->declare_parameter("min_points", 25);
        this->declare_parameter("cluster_thresh", 0.1);
        this->declare_parameter("cluster_buff", 0.1);
        this->declare_parameter("min_distance", 0.2);
        this->declare_parameter("trans_thresh", 3.0);
        this->declare_parameter("rot_thresh", 0.5);

        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/velodyne_points", 10,
            std::bind(&OdometryVizNode::pointcloudCallback, this, std::placeholders::_1));

        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/odom", 50);
        odom_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        X0.resize(6);
        X0 << 0., 0., 0., 0., 0., 0.; 
    }

    void pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        auto before = std::chrono::system_clock::now();
        auto beforeMs = std::chrono::time_point_cast<std::chrono::milliseconds>(before);
        frameCount++;

        pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *pcl_cloud);

        if (!initialized_) {
            prev_pcl_cloud_ = pcl_cloud;
            initialized_ = true;
            prev_pcl_matrix = convertPCLtoEigen(prev_pcl_cloud_);
            return;
        }

        Eigen::MatrixXf pcl_matrix = convertPCLtoEigen(pcl_cloud);

        // Read parameters
        float minD = this->get_parameter("min_distance").as_double();
        int numBinsPhi = this->get_parameter("num_bins_phi").as_int();
        int numBinsTheta = this->get_parameter("num_bins_theta").as_int();
        int run_length = this->get_parameter("run_length").as_int();
        int min_points = this->get_parameter("min_points").as_int();
        float cluster_thresh = this->get_parameter("cluster_thresh").as_double();
        float cluster_buff = this->get_parameter("cluster_buff").as_double();
        float trans_thresh = this->get_parameter("trans_thresh").as_double();
        float rot_thresh = this->get_parameter("rot_thresh").as_double();

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

        // RUN ICET
        ICET it(prev_pcl_matrix, pcl_matrix, run_length, X0, numBinsPhi, numBinsTheta, min_points, cluster_thresh, cluster_buff);
        Eigen::VectorXf X = it.X;
        cout << "soln: " << endl << X << endl;
        cout << "1-sigma error bounds:" << endl << it.pred_stds << endl;
        X0 << 0., 0., 0., 0., 0., 0.; 

        // Outlier rejection
        if (abs(X[0]) > trans_thresh || abs(X[1]) > trans_thresh || abs(X[2]) > trans_thresh
           || abs(X[3]) > rot_thresh || abs(X[4]) > rot_thresh || abs(X[5]) > rot_thresh){
            RCLCPP_WARN(this->get_logger(), "Outlier registration rejected");
            X[0] = 0; X[1] = 0; X[2] = 0;
            X[3] = 0; X[4] = 0; X[5] = 0;
        }

        MatrixXf rot_mat = utils::R(X[3], X[4], X[5]);
        Eigen::RowVector3f trans(X[0], X[1], X[2]);

        // Update visualization data (thread-safe swap)
        {
            std::lock_guard<std::mutex> lock(vizMutex);
            vizPoints1 = it.points1;
            vizPoints2 = it.points2;
            vizClusterBounds = it.clusterBounds;
            vizE1Means = it.ellipsoid1Means;
            vizE1Covs = it.ellipsoid1Covariances;
            vizE1Alphas = it.ellipsoid1Alphas;
            vizE2Means = it.ellipsoid2Means;
            vizE2Covs = it.ellipsoid2Covariances;
            vizE2Alphas = it.ellipsoid2Alphas;
            vizDataReady = true;
        }
        // Request GLUT to redraw
        glutPostRedisplay();

        prev_pcl_matrix = pcl_matrix;

        // Accumulate transform
        Eigen::Matrix4f X_homo_i = Eigen::Matrix4f::Identity();
        X_homo_i.block<3, 3>(0, 0) = rot_mat;
        X_homo_i.block<3, 1>(0, 3) = trans.transpose();
        X_homo = X_homo * X_homo_i;
        std::cout << "X: \n" << X_homo << endl;

        // Publish odometry
        nav_msgs::msg::Odometry odom;
        rclcpp::Time current_time = this->now();
        odom.header.stamp = current_time;
        odom.header.frame_id = "map";
        odom.child_frame_id = "velodyne";

        odom.pose.pose.position.x = X_homo(0,3);
        odom.pose.pose.position.y = X_homo(1,3);
        odom.pose.pose.position.z = X_homo(2,3);
        Eigen::Matrix3f rotationMatrix = X_homo.topLeftCorner(3, 3);
        Eigen::Quaternionf quaternion(rotationMatrix);
        odom.pose.pose.orientation.x = quaternion.x();
        odom.pose.pose.orientation.y = quaternion.y();
        odom.pose.pose.orientation.z = quaternion.z();
        odom.pose.pose.orientation.w = quaternion.w();

        for (int i = 0; i < 36; ++i) odom.pose.covariance[i] = 0.0;
        odom.pose.covariance[0] = it.pred_stds[0];
        odom.pose.covariance[7] = it.pred_stds[1];
        odom.pose.covariance[14] = it.pred_stds[2];
        odom.pose.covariance[21] = it.pred_stds[3];
        odom.pose.covariance[28] = it.pred_stds[4];
        odom.pose.covariance[35] = it.pred_stds[5];

        odom.twist.twist.linear.x = 10*X[0];
        odom.twist.twist.linear.y = 10*X[1];
        odom.twist.twist.linear.z = 10*X[2];
        odom.twist.twist.angular.x = 10*X[3];
        odom.twist.twist.angular.y = 10*X[4];
        odom.twist.twist.angular.z = 10*X[5];

        odom_pub_->publish(odom);

        // Broadcast TF
        geometry_msgs::msg::TransformStamped odom_trans;
        odom_trans.header.stamp = current_time;
        odom_trans.header.frame_id = "map";
        odom_trans.child_frame_id = "velodyne";
        odom_trans.transform.translation.x = X_homo(0,3);
        odom_trans.transform.translation.y = X_homo(1,3);
        odom_trans.transform.translation.z = X_homo(2,3);
        odom_trans.transform.rotation.x = quaternion.x();
        odom_trans.transform.rotation.y = quaternion.y();
        odom_trans.transform.rotation.z = quaternion.z();
        odom_trans.transform.rotation.w = quaternion.w();
        odom_broadcaster_->sendTransform(odom_trans);

        auto after1 = std::chrono::system_clock::now();
        auto after1Ms = std::chrono::time_point_cast<std::chrono::milliseconds>(after1);
        auto elapsedTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(after1Ms - beforeMs).count();
        std::cout << "Registered scans in: " << elapsedTimeMs << " ms" << std::endl;
    }

    Eigen::VectorXf X0;

private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> odom_broadcaster_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr prev_pcl_cloud_;
    bool initialized_;
    int frameCount = 0;
    Eigen::MatrixXf prev_pcl_matrix;
    Eigen::Matrix4f X_homo = Eigen::Matrix4f::Identity(); 

    Eigen::MatrixXf convertPCLtoEigen(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pcl_cloud) {
        Eigen::MatrixXf eigen_matrix(pcl_cloud->size(), 3);
        for (size_t i = 0; i < pcl_cloud->size(); ++i) {
            eigen_matrix.row(i) << pcl_cloud->points[i].x, pcl_cloud->points[i].y, pcl_cloud->points[i].z;
        }
        return eigen_matrix;
    }
};

// GLUT idle callback: check if new viz data is ready and swap it in
void idleCallback() {
    std::lock_guard<std::mutex> lock(vizMutex);
    if (vizDataReady && viz) {
        viz->points1 = vizPoints1;
        viz->points2 = vizPoints2;
        viz->clusterBounds = vizClusterBounds;
        viz->ellipsoid1Means = vizE1Means;
        viz->ellipsoid1Covariances = vizE1Covs;
        viz->ellipsoid1Alphas = vizE1Alphas;
        viz->ellipsoid2Means = vizE2Means;
        viz->ellipsoid2Covariances = vizE2Covs;
        viz->ellipsoid2Alphas = vizE2Alphas;
        vizDataReady = false;
        glutPostRedisplay();
    }
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    // Create the viz (this initializes GLUT/GLEW)
    viz = new visualization();

    // Set GLUT idle function to poll for new data
    glutIdleFunc(idleCallback);

    // Spin ROS in a background thread
    auto node = std::make_shared<OdometryVizNode>();
    std::thread ros_thread([&node]() {
        rclcpp::spin(node);
    });
    ros_thread.detach();

    // Run GLUT main loop on the main thread (required by OpenGL)
    glutMainLoop();

    rclcpp::shutdown();
    delete viz;
    return 0;
}
