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
        assert(cartesianPoints.cols() == 3);
        int N = cartesianPoints.rows();

        // Vectorized: compute all norms at once
        Eigen::ArrayXf r = cartesianPoints.rowwise().norm().array();

        // Vectorized: atan2 and acos over entire arrays
        Eigen::ArrayXf theta = cartesianPoints.col(1).array().binaryExpr(
            cartesianPoints.col(0).array(),
            [](float y, float x) { return std::atan2(y, x); });
        theta = (theta < 0.f).select(theta + 2.f * static_cast<float>(M_PI), theta);

        Eigen::ArrayXf phi = (cartesianPoints.col(2).array() / r).unaryExpr(
            [](float v) { return std::acos(v); });

        // Combine and replace NaN
        MatrixXf sphericalPoints(N, 3);
        sphericalPoints.col(0) = r.matrix();
        sphericalPoints.col(1) = theta.matrix();
        sphericalPoints.col(2) = phi.matrix();
        sphericalPoints = (sphericalPoints.array().isNaN()).select(1000.0, sphericalPoints);

        return sphericalPoints;
    }

    Eigen::MatrixXf sphericalToCartesian(const Eigen::MatrixXf& sphericalPoints) {
        assert(sphericalPoints.cols() == 3);

        // Vectorized: extract and compute all trig at once
        Eigen::ArrayXf r = sphericalPoints.col(0).array();
        Eigen::ArrayXf theta = sphericalPoints.col(1).array();
        Eigen::ArrayXf phi = sphericalPoints.col(2).array();

        Eigen::ArrayXf sinPhi = phi.sin();
        Eigen::ArrayXf cosPhi = phi.cos();
        Eigen::ArrayXf sinTheta = theta.sin();
        Eigen::ArrayXf cosTheta = theta.cos();

        Eigen::MatrixXf cartesianPoints(sphericalPoints.rows(), 3);
        cartesianPoints.col(0) = (r * sinPhi * cosTheta).matrix();
        cartesianPoints.col(1) = (r * sinPhi * sinTheta).matrix();
        cartesianPoints.col(2) = (r * cosPhi).matrix();

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