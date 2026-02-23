#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <tf2_ros/transform_broadcaster.h>
#include <pcl/registration/icp.h>
#include <Eigen/Dense>
#include <random>
#include <geometry_msgs/msg/transform_stamped.hpp>
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

class MapMakerNode : public rclcpp::Node {
public:
    MapMakerNode() : Node("map_maker_node"), initialized_(false), q(600'000, 3), gen(rd()) {
        // Declare parameters (loaded from config YAML at runtime)
        this->declare_parameter("num_bins_phi", 24);
        this->declare_parameter("num_bins_theta", 75);
        this->declare_parameter("run_length", 12);
        this->declare_parameter("min_points", 25);
        this->declare_parameter("cluster_thresh", 0.1);
        this->declare_parameter("cluster_buff", 0.1);
        this->declare_parameter("min_distance", 0.2);
        this->declare_parameter("trans_thresh", 3.0);
        this->declare_parameter("rot_thresh", 0.5);

        // Set up ROS subscribers and publishers
        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/velodyne_points", 10,
            std::bind(&MapMakerNode::pointcloudCallback, this, std::placeholders::_1));

        aligned_pointcloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/hd_map", 1);
        snail_trail_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/snail_trail_topic", 1);

        broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        X0.resize(6);
        X0 << 0., 0., 0., 0., 0., 0.; 

        snailTrail.resize(1,3);
        snailTrail.row(0) << 0., 0., 0.;
    }

    void pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        auto before = std::chrono::system_clock::now();
        auto beforeMs = std::chrono::time_point_cast<std::chrono::milliseconds>(before);
        frameCount++;

        // Convert PointCloud2 to PCL PointCloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>());
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

        // RUN ICET ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ICET it(prev_pcl_matrix, pcl_matrix, run_length, X0, numBinsPhi, numBinsTheta, min_points, cluster_thresh, cluster_buff);
        Eigen::VectorXf X = it.X;
        cout << "soln: " << endl << X << endl;
        cout << "1-sigma error bounds:" << endl << it.pred_stds << endl;
        X0 << 0., 0., 0., 0., 0., 0.; 
        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        //look for jumps in acceleration to prevent solution from exploding
        if (abs(X[0]) > trans_thresh || abs(X[1]) > trans_thresh || abs(X[2]) > trans_thresh
           || abs(X[3]) > rot_thresh || abs(X[4]) > rot_thresh || abs(X[5]) > rot_thresh){
            X[0] = 0;
            X[1] = 0;
            X[2] = 0;
            X[3] = 0;
            X[4] = 0;
            X[5] = 0;
        }

        // Convert back to ROS PointCloud2 and publish the aligned point cloud
        MatrixXf rot_mat = utils::R(X[3], X[4], X[5]);
        Eigen::RowVector3f trans(X[0], X[1], X[2]);

        //downsample pcl_matrix before passing to map queue
        int downsampleSize = 2'000;
        std::size_t originalSize = pcl_matrix.rows();
        std::vector<std::size_t> indices(originalSize);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), gen);
        //account for situations where sensor is occluded and numpts << downsampleSize
        Eigen::MatrixXf downsampledMatrix(std::min(downsampleSize, static_cast<int>(pcl_matrix.size()) / 3), pcl_matrix.cols());
        for (std::size_t i = 0; i < static_cast<std::size_t>(downsampleSize); ++i) {
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
        geometry_msgs::msg::TransformStamped transformStamped;

        // Set the frame IDs
        transformStamped.header.frame_id = "map";         // Parent frame 
        transformStamped.child_frame_id = "velodyne";     // Child frame

        // Convert Eigen rotation matrix to quaternion
        Eigen::Matrix3f rotationMatrix = X_homo.topLeftCorner(3, 3);
        Eigen::Quaternionf quaternion(rotationMatrix);

        // Set the translation and rotation in the transform message
        transformStamped.transform.translation.x = X_homo(0,3);
        transformStamped.transform.translation.y = X_homo(1,3);
        transformStamped.transform.translation.z = X_homo(2,3);
        transformStamped.transform.rotation.x = quaternion.x();
        transformStamped.transform.rotation.y = quaternion.y();
        transformStamped.transform.rotation.z = quaternion.z();
        transformStamped.transform.rotation.w = quaternion.w();

        // Set the timestamp
        rclcpp::Time current_time = this->now();
        transformStamped.header.stamp = current_time;

        // Broadcast the transform
        broadcaster_->sendTransform(transformStamped);

        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // Create a header for the ROS message
        std_msgs::msg::Header header; 
        header.stamp = this->now();
        header.frame_id = "map";
        sensor_msgs::msg::PointCloud2 rosPointCloud = convertEigenToROS(mapPC, header);
        aligned_pointcloud_pub_->publish(rosPointCloud);

        //update snail trail
        snailTrail = (snailTrail * rot_mat.inverse()).rowwise() - trans;
        Eigen::VectorXf new_row(3);
        new_row << 0., 0., 0.;
        int currentRows = snailTrail.rows();
        snailTrail.conservativeResize(currentRows + 1, Eigen::NoChange);
        snailTrail.row(currentRows) = new_row;

        sensor_msgs::msg::PointCloud2 snailPC = convertEigenToROS(snailTrail, header);
        snail_trail_pub_->publish(snailPC);

        auto after1 = std::chrono::system_clock::now();
        auto after1Ms = std::chrono::time_point_cast<std::chrono::milliseconds>(after1);
        auto elapsedTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(after1Ms - beforeMs).count();
        std::cout << "Registered scans in: " << elapsedTimeMs << " ms" << std::endl;
    }

    Eigen::VectorXf X0;

private:

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr snail_trail_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr aligned_pointcloud_pub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> broadcaster_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr prev_pcl_cloud_;
    bool initialized_;
    int frameCount = 0;
    Eigen::MatrixXf snailTrail;
    Eigen::MatrixXf prev_pcl_matrix;
    Eigen::MatrixXf mapPC;
    EigenQueue q;
    std::random_device rd;
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

    sensor_msgs::msg::PointCloud2 convertEigenToROS(const Eigen::MatrixXf& eigenPointCloud, const std_msgs::msg::Header& header) {
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
        sensor_msgs::msg::PointCloud2 rosPointCloud;
        pcl::toROSMsg(pclPointCloud, rosPointCloud);
        rosPointCloud.header = header;  // Set the header from the input

        return rosPointCloud;
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MapMakerNode>());
    rclcpp::shutdown();
    return 0;
}
