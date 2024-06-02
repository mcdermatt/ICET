//utils.h 
#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Dense>
#include <string>

namespace utils {
    
    Eigen::MatrixXf loadPointCloudCSV(std::string filename, std::string datasetType = "csv");

    Eigen::MatrixXf cartesianToSpherical(const Eigen::MatrixXf& cartesianPoints);

    Eigen::MatrixXf sphericalToCartesian(const Eigen::MatrixXf& sphericalPoints);

    Eigen::Matrix3f R(float phi, float theta, float psi);

}

#endif 