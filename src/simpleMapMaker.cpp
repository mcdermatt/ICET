#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/buffer.h>  // Add this line for tf2_ros::Buffer
#include <pcl/registration/icp.h>
#include <Eigen/Dense>
#include <random>
#include "icet.h"
#include "utils.h"

using namespace std;
using namespace Eigen;

//FIFO data structure to hold on to point clouds in large HD Map 
class EigenQueue {
public:
    EigenQueue(int maxSize, int numCols) 
        : maxSize(maxSize), numCols(numCols), matrix(maxSize, numCols), pos(0), filled(false) {}

    //just add point to queue
    void enqueue(const Eigen::VectorXf& row) {
        if (row.size() != numCols) {
            throw std::invalid_argument("Row size does not match the number of columns in the matrix");
        }
        matrix.row(pos) = row;
        pos = (pos + 1) % maxSize;
        if (pos == 0) filled = true;
    }

    //given rotation and translation-- add new scan and transform matrix
    void add_new_scan(Eigen::MatrixXf newScan, Eigen::RowVector3f trans, MatrixXf rot_mat){
        // const Eigen::VectorXf & row
        for (int i = 0; i < newScan.rows(); i++){
            enqueue(newScan.row(i));
        }
        // matrix = (matrix * rot_mat.inverse()).rowwise() - trans; //test
        matrix = (matrix.rowwise() - trans) * rot_mat.inverse(); //old
    }

    Eigen::MatrixXf getQueue() const {
        if (!filled) {
            return matrix.topRows(pos);
        }
        Eigen::MatrixXf result(maxSize, numCols);
        result << matrix.bottomRows(maxSize - pos), matrix.topRows(pos);
        return result;
    }

private:
    int maxSize;
    int numCols;
    Eigen::MatrixXf matrix;
    int pos;
    bool filled;
};

class MapMakerNode {
public:
    MapMakerNode() : nh_("~"), initialized_(false) ,  q(600'000,3) {
        // Set up ROS subscribers and publishers
        pointcloud_sub_ = nh_.subscribe("/velodyne_points", 10, &MapMakerNode::pointcloudCallback, this); //use when connected to Velodyne VLP-16
        // pointcloud_sub_ = nh_.subscribe("/os1_cloud_node/points", 10, &MapMakerNode::pointcloudCallback, this); //use when connected to Ouster OS1 sensor
        // pointcloud_sub_ = nh_.subscribe("/raw_point_cloud", 10, &MapMakerNode::pointcloudCallback, this); //use with fake_lidar node
        aligned_pointcloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/hd_map", 1);
        snail_trail_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/snail_trail_topic", 1);

        X0.resize(6);
        X0 << 0., 0., 0., 0., 0., 0.; 

        snailTrail.resize(1,3);
        snailTrail.row(0) << 0., 0., 0.;
    }

    void pointcloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
        auto before = std::chrono::system_clock::now();
        auto beforeMs = std::chrono::time_point_cast<std::chrono::milliseconds>(before);
        frameCount++;

        // Convert PointCloud2 to PCL PointCloud
        boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *pcl_cloud);

        if (!initialized_) {
            // On the first callback, initialize prev_pcl_cloud_
            prev_pcl_cloud_ = pcl_cloud;
            initialized_ = true;
            prev_pcl_matrix = convertPCLtoEigen(prev_pcl_cloud_); //test
            return;
        }

        // Convert PCL PointCloud to Eigen::MatrixXf
        Eigen::MatrixXf pcl_matrix = convertPCLtoEigen(pcl_cloud);

        // Filter out points less than distance 'd' from the origin
        float minD = 0.2;
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
        X0 << 0., 0., 0., 0., 0., 0.; 
        // X0 << X[0], X[1], X[2], X[3], X[4], X[5]; 
        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        // Convert back to ROS PointCloud2 and publish the aligned point cloud
        sensor_msgs::PointCloud2 aligned_cloud_msg;
        MatrixXf rot_mat = utils::R(X[3], X[4], X[5]);
        Eigen::RowVector3f trans(X[0], X[1], X[2]);
        // MatrixXf rot_mat = utils::R(-X[3], -X[4], -X[5]); //test
        // Eigen::RowVector3f trans(-X[0], -X[1], -X[2]); //test

        //pass pcl_matrix directly to map queue
        // q.add_new_scan(pcl_matrix, trans, rot_mat);

        //downsample pcl_matrix before passing to map queue
        int downsampleSize = 2'000; //10'000
        std::size_t originalSize = pcl_matrix.rows();
        std::vector<std::size_t> indices(originalSize);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), gen);
        //account for situations where sensor is occluded and numpts << downsampleSize
        Eigen::MatrixXf downsampledMatrix(std::min(downsampleSize, static_cast<int>(pcl_matrix.size()) / 3), pcl_matrix.cols());
        for (std::size_t i = 0; i < downsampleSize; ++i) {
            downsampledMatrix.row(i) = pcl_matrix.row(indices[i]);
        }
        q.add_new_scan(downsampledMatrix, trans, rot_mat);

        prev_pcl_matrix = pcl_matrix;
        mapPC = q.getQueue();

        //"broadcast" transform that relates /velodyne frame to /map frame ~~~~~~~~~~~~~~~~~~~~~
        //convert ICET output X for scan i to homogenous transformation matrix
        Eigen::Matrix4f X_homo_i = Eigen::Matrix4f::Identity();
        X_homo_i.block<3, 3>(0, 0) = rot_mat;
        X_homo_i.block<3, 1>(0, 3) = trans.transpose();

        //update accumulated transfrom
        X_homo = X_homo * X_homo_i;
        std::cout << "X: \n" << X_homo <<endl;

        // Create a geometry_msgs::TransformStamped message
        geometry_msgs::TransformStamped transformStamped;

        // Set the frame IDs
        transformStamped.header.frame_id = "map";         // Parent frame 
        transformStamped.child_frame_id = "velodyne";     // Child frame
        // transformStamped.child_frame_id = "/os1_lidar";     // Child frame
        // transformStamped.child_frame_id = "sensor";     // Child frame

        // Convert Eigen rotation matrix to quaternion
        // Eigen::Matrix3f rotationMatrix = rot_mat.topLeftCorner(3, 3); //X_i
        Eigen::Matrix3f rotationMatrix = X_homo.topLeftCorner(3, 3); //X
        Eigen::Quaternionf quaternion(rotationMatrix);

        // Set the translation and rotation in the transform message
        //X_i
        // transformStamped.transform.translation.x = trans(0);
        // transformStamped.transform.translation.y = trans(1);
        // transformStamped.transform.translation.z = trans(2);
        // X
        transformStamped.transform.translation.x = X_homo(0,3);
        transformStamped.transform.translation.y = X_homo(1,3);
        transformStamped.transform.translation.z = X_homo(2,3);
        transformStamped.transform.rotation.x = quaternion.x();
        transformStamped.transform.rotation.y = quaternion.y();
        transformStamped.transform.rotation.z = quaternion.z();
        transformStamped.transform.rotation.w = quaternion.w();

        // Set the timestamp
        // std::cout << msg->header.stamp << endl;
        // transformStamped.header.stamp = msg->header.stamp;  // Use lidar scan timestamp --> issues with using old rosbag data
        transformStamped.header.stamp = ros::Time::now(); //use PC timestamp 

        // Broadcast the transform
        broadcaster_.sendTransform(transformStamped);

        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // Create a header for the ROS message
        std_msgs::Header header; 
        header.stamp = ros::Time::now(); //use current timestamp 
        // header.stamp = msg->header.stamp; //use lidar scan timestamp --> issues with using rosbag data?
        header.frame_id = "map";  // Set frame ID
        // sensor_msgs::PointCloud2 rosPointCloud = convertEigenToROS(prev_pcl_matrix, header);
        sensor_msgs::PointCloud2 rosPointCloud = convertEigenToROS(mapPC, header);
        aligned_pointcloud_pub_.publish(rosPointCloud);

        //update snail trail
        snailTrail = (snailTrail * rot_mat.inverse()).rowwise() - trans;
        Eigen::VectorXf new_row(3);
        new_row << 0., 0., 0.;
        int currentRows = snailTrail.rows();
        snailTrail.conservativeResize(currentRows + 1, Eigen::NoChange);
        snailTrail.row(currentRows) = new_row;

        sensor_msgs::PointCloud2 snailPC = convertEigenToROS(snailTrail, header);
        snail_trail_pub_.publish(snailPC);

        auto after1 = std::chrono::system_clock::now();
        auto after1Ms = std::chrono::time_point_cast<std::chrono::milliseconds>(after1);
        auto elapsedTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(after1Ms - beforeMs).count();
        std::cout << "Registered scans in: " << elapsedTimeMs << " ms" << std::endl;
    }

    Eigen::VectorXf X0;

private:
    ros::NodeHandle nh_;
    ros::Subscriber pointcloud_sub_;
    ros::Publisher snail_trail_pub_;
    ros::Publisher aligned_pointcloud_pub_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr prev_pcl_cloud_;
    tf2_ros::Buffer tfBuffer_;
    tf2_ros::TransformBroadcaster broadcaster_;
    bool initialized_;
    int frameCount = 0;
    Eigen::MatrixXf snailTrail;
    Eigen::MatrixXf prev_pcl_matrix;
    Eigen::MatrixXf mapPC;
    EigenQueue q;
    std::random_device rd; //init for RNG
    std::mt19937 gen;      

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
    ros::init(argc, argv, "simple_map_maker_node");
    MapMakerNode simple_map_maker_node;
    ros::spin();
    return 0;
}
