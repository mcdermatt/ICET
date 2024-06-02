#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <Eigen/Dense>
#include "icet.h"
#include "utils.h"

using namespace std;
using namespace Eigen;

class ScanRegistrationNode {
public:
    ScanRegistrationNode() : nh_("~") {
        // Set up ROS subscribers and publishers
        pointcloud_sub_ = nh_.subscribe("/os1_cloud_node/points", 10, &ScanRegistrationNode::pointcloudCallback, this);
        // pointcloud_sub_ = nh_.subscribe("/velodyne_points", 10, &ScanRegistrationNode::pointcloudCallback, this);
        aligned_pointcloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/aligned_pointcloud_topic", 1);
        snail_trail_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/snail_trail_topic", 1);

        // Initialize the previous point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr prev_point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        prev_pcl_cloud_ = prev_point_cloud;
        snailTrail.resize(1,3);
        snailTrail.row(0) << 0., 0., 0.;
    }

    void pointcloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
        auto before = std::chrono::system_clock::now();
        auto beforeMs = std::chrono::time_point_cast<std::chrono::milliseconds>(before);

        frameCount++;

        // Convert PointCloud2 to PCL PointCloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *pcl_cloud);

        if (pcl_cloud->empty()) {
            ROS_WARN("Received an empty point cloud");
            return;       
        }
        Eigen::MatrixXf pcl_matrix = convertPCLtoEigen(pcl_cloud);

        if (prev_pcl_cloud_->empty()) {
            ROS_WARN("Previous point cloud is empty, skipping this frame");
            prev_pcl_cloud_ = pcl_cloud; // Update the previous point cloud
            return;
        }
        Eigen::MatrixXf prev_pcl_matrix = convertPCLtoEigen(prev_pcl_cloud_); 

        // ros::Duration(0.005).sleep(); // Wait for 5 milliseconds

        try{
            // NEW UPDATED ICET CODE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            int run_length = 7;
            int numBinsPhi = 24; //24 for ouster, 18 for 32 channel sensor
            int numBinsTheta = 75;  //75; 
            Eigen::VectorXf X0;
            X0.resize(6);
            X0 << 0., 0., 0., 0., 0., 0.; //set initial estimate
            ICET it(prev_pcl_matrix, pcl_matrix, run_length, X0, numBinsPhi, numBinsTheta);
            Eigen::VectorXf X = it.X;
            cout << "odometry estimate for frame " << frameCount << ":" << endl << X << endl;
            //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            // Update the previous point cloud for the next iteration
            prev_pcl_cloud_ = pcl_cloud;

            // Convert back to ROS PointCloud2 and publish the aligned point cloud
            sensor_msgs::PointCloud2 aligned_cloud_msg;
            MatrixXf rot_mat = utils::R(X[3], X[4], X[5]);
            Eigen::RowVector3f trans(X[0], X[1], X[2]);

            Eigen:MatrixXf scan2_in_scan1_frame = (pcl_matrix * rot_mat.inverse()).rowwise() - trans;
            
            //update snail trail
            snailTrail = (snailTrail * rot_mat.inverse()).rowwise() - trans;
            Eigen::VectorXf new_row(3);
            new_row << 0., 0., 0.;
            int currentRows = snailTrail.rows();
            snailTrail.conservativeResize(currentRows + 1, Eigen::NoChange);
            snailTrail.row(currentRows) = new_row;
            // cout << "snail trail: " << endl << snailTrail << endl;

            // Create a header for the ROS message
            std_msgs::Header header;
            header.stamp = ros::Time::now();
            // header.frame_id = "velodyne";  // Set frame ID
            header.frame_id = "/os1_lidar";  // Set frame ID
            sensor_msgs::PointCloud2 rosPointCloud = convertEigenToROS(scan2_in_scan1_frame, header);
            aligned_pointcloud_pub_.publish(rosPointCloud);

            sensor_msgs::PointCloud2 snailPC = convertEigenToROS(snailTrail, header);
            snail_trail_pub_.publish(snailPC);
        }
        catch (const std::exception& e) {
            ROS_ERROR("Exception caught during ICET processing: %s", e.what());
            return;
        } catch (...) {
            ROS_ERROR("Unknown exception caught during ICET processing");
            return;
        }

        auto after1 = std::chrono::system_clock::now();
        auto after1Ms = std::chrono::time_point_cast<std::chrono::milliseconds>(after1);
        auto elapsedTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(after1Ms - beforeMs).count();
        std::cout << "Registered scans in: " << elapsedTimeMs << " ms" << std::endl;
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber pointcloud_sub_;
    ros::Publisher aligned_pointcloud_pub_;
    ros::Publisher snail_trail_pub_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr prev_pcl_cloud_;
    int frameCount = 0;
    Eigen::MatrixXf snailTrail;

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
    ros::init(argc, argv, "scan_registration_node");
    ScanRegistrationNode scan_registration_node;
    ros::spin();
    return 0;
}
