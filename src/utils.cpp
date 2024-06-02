#include <Eigen/Dense>
#include <string>
#include <fstream>
#include "csv.hpp"
#include "utils.h"

using namespace std;
using namespace Eigen;

namespace utils{

    MatrixXf loadPointCloudCSV(string filename, string datasetType){
        // Open the CSV file
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open the CSV file." << std::endl;
        }

        if (datasetType == "ouster"){
            // Parse and process the CSV file, skipping the first row
            csv::CSVReader reader(file, csv::CSVFormat().header_row(1).trim({}));
            
            // Initialize a vector to store rows temporarily
            std::vector<Eigen::Vector3f> rows;

            csv::CSVRow row;
            reader.read_row(row); // Skip the first row
            csv::CSVRow secondRow;
            reader.read_row(secondRow); // Skip the second row
            // Parse and process the CSV file
            // Iterate over rows and fill the vector
            for (csv::CSVRow& currentRow : reader) {
                // Assuming three columns in each row
                Eigen::Vector3f rowData;
                rowData << static_cast<float>(currentRow[8].get<int>()),
                        static_cast<float>(currentRow[9].get<int>()),
                        static_cast<float>(currentRow[10].get<int>());

                // Append the row to the vector
                rows.push_back(rowData);
            }

            // Close the file before processing the vector
            file.close();

            // Preallocate memory for the dataMatrix
            Eigen::MatrixXf dataMatrix(rows.size(), 3);

            // Copy the data from the vector to the dataMatrix
            for (size_t i = 0; i < rows.size(); ++i) {
                dataMatrix.row(i) = rows[i]/1000;
            }
            // points = dataMatrix;
            // return points;
            return dataMatrix;    // this seems to work?

            // // Extract every nth point
            // int n = 4;
            // Eigen::MatrixXf points = dataMatrix.block(0, 0, dataMatrix.rows() / n, dataMatrix.cols());
            // return points;
        }

        // for loading generic point cloud data provided in xyz
        else{
            //tab delimeter
            csv::CSVReader reader(file, csv::CSVFormat().delimiter('\t'));
            std::vector<Eigen::Vector3f> rows;
            for (csv::CSVRow& currentRow : reader) {
                // Assuming three columns in each row
                Eigen::Vector3f rowData;
                rowData << stof(currentRow[0].get<>()),
                        stof(currentRow[1].get<>()),
                        stof(currentRow[2].get<>());
                // Append the row to the vector
                rows.push_back(rowData);
            }
            // Close the file before processing the vector
            file.close();

            // Preallocate memory for the dataMatrix
            Eigen::MatrixXf dataMatrix(rows.size(), 3);

            // Copy the data from the vector to the dataMatrix
            for (size_t i = 0; i < rows.size(); ++i) {
                dataMatrix.row(i) = rows[i];
            }
            return dataMatrix; 
        }


    }

    MatrixXf cartesianToSpherical(const MatrixXf& cartesianPoints){
        // Ensure that the input matrix has 3 columns (X, Y, Z coordinates)
        assert(cartesianPoints.cols() == 3);

        // MatrixXf spherical_coords(cartesianPoints.rows(), 3);

        // // Extract x, y, z components
        // ArrayXf x = cartesianPoints.col(0).array();
        // ArrayXf y = cartesianPoints.col(1).array();
        // ArrayXf z = cartesianPoints.col(2).array();

        // // Compute r
        // ArrayXf r = (x.square() + y.square() + z.square()).sqrt();

        // // Compute theta (atan2(y, x)), ensuring it's in the range [0, 2pi)
        // ArrayXf theta = y.binaryExpr(x, [](float yi, float xi) { return std::atan2(yi, xi); });
        // theta = theta.unaryExpr([](float angle) { return angle < 0.0f ? angle + 2.0f * float(M_PI) : angle; });

        // // Compute phi (acos(z / r)), guard against division by zero
        // ArrayXf phi = (z / r).min(1.0f).max(-1.0f);  // clamp the values to [-1, 1] to avoid domain errors
        // phi = phi.unaryExpr([](float ratio) { return std::acos(ratio); });

        // // Assign to spherical_coords
        // spherical_coords.col(0) = r.matrix();
        // spherical_coords.col(1) = theta.matrix();
        // spherical_coords.col(2) = phi.matrix();

        // return spherical_coords;

        // Compute norms (radial distance)
        VectorXf r = cartesianPoints.rowwise().norm();

        // Compute azimuthal angle (theta) and elevation angle (phi)
        VectorXf theta(cartesianPoints.rows());
        VectorXf phi(cartesianPoints.rows());
        for (int i = 0; i < cartesianPoints.rows(); ++i) {
            theta(i) = std::atan2(cartesianPoints(i, 1), cartesianPoints(i, 0));
            if (theta(i) < 0.0) {
                theta(i) += 2.0 * M_PI;
            }
            phi(i) = std::acos(cartesianPoints(i, 2) / r(i));
        }

        // Combine r, theta, phi into a new matrix
        MatrixXf sphericalPoints(cartesianPoints.rows(), 3);
        sphericalPoints << r, theta, phi;

        // Replace NaN values with a default value
        sphericalPoints = (sphericalPoints.array().isNaN()).select(1000.0, sphericalPoints);

        return sphericalPoints;

        // VectorXf x = cartesianPoints.col(0);
        // VectorXf y = cartesianPoints.col(1);
        // VectorXf z = cartesianPoints.col(2);
        // VectorXf r = cartesianPoints.rowwise().norm();

        // // Compute azimuthal angle (theta)
        // VectorXf theta = VectorXf::Zero(cartesianPoints.rows());
        // for (int i = 0; i < cartesianPoints.rows(); ++i) {
        //     theta(i) = std::atan2(y(i), x(i));
        //     if (theta(i) < 0.0) {
        //         theta(i) += 2.0 * M_PI;
        //     }
        // }

        // // Compute elevation angle (phi)
        // VectorXf phi = VectorXf::Zero(cartesianPoints.rows());
        // for (int i = 0; i < cartesianPoints.rows(); ++i) {
        //     phi(i) = std::acos(z(i) / r(i));
        //     // std::cout << "phi(i): \n" << phi(i) << "\n";
        // }

        // // Combine r, theta, phi into a new matrix
        // MatrixXf sphericalPoints(cartesianPoints.rows(), 3);
        // sphericalPoints << r, theta, phi;

        // // Replace NaN or -NaN values with [0, 0, 0]
        // for (int i = 0; i < sphericalPoints.rows(); ++i) {
        //     for (int j = 0; j < sphericalPoints.cols(); ++j) {
        //         if (std::isnan(sphericalPoints(i, j))) {
        //             sphericalPoints(i, j) = 1000.0;
        //         }
        //     }
        // }

        // return sphericalPoints;
    }

    Eigen::MatrixXf sphericalToCartesian(const Eigen::MatrixXf& sphericalPoints) {
        // Ensure that the input matrix has 3 columns (r, theta, phi)
        assert(sphericalPoints.cols() == 3);

        // Extract r, theta, phi columns
        Eigen::VectorXf r = sphericalPoints.col(0);
        Eigen::VectorXf theta = sphericalPoints.col(1);
        Eigen::VectorXf phi = sphericalPoints.col(2);

        // Convert spherical coordinates to Cartesian coordinates
        Eigen::MatrixXf cartesianPoints(sphericalPoints.rows(), 3);

        for (int i = 0; i < sphericalPoints.rows(); ++i) {
            float x = r(i) * sin(phi(i)) * cos(theta(i));
            float y = r(i) * sin(phi(i)) * sin(theta(i));
            float z = r(i) * cos(phi(i));

            cartesianPoints.row(i) << x, y, z;
        }

        return cartesianPoints;
    }

    Eigen::Matrix3f R(float phi, float theta, float psi){
        //given body frame xyz euler angles [phi, theta, psi], return 3x3 rotation matrix
        MatrixXf mat(3,3); 
        mat << cos(theta)*cos(psi), sin(psi)*cos(phi)+sin(phi)*sin(theta)*cos(psi), sin(phi)*sin(psi)-sin(theta)*cos(phi)*cos(psi),
            -sin(psi)*cos(theta), cos(phi)*cos(psi)-sin(phi)*sin(theta)*sin(psi), sin(phi)*cos(psi)+sin(theta)*sin(psi)*cos(phi),
            sin(theta), -sin(phi)*cos(theta), cos(phi)*cos(theta);

        return mat;
    }

}