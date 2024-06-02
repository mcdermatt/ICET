#include <iostream>
#include <GL/glew.h>
#include <GL/glut.h>
#include <thread>
#include <chrono>
#include "visualization.h"
#include "utils.h"
#include "icet.h"
using namespace std;

//script for running C++ ICET with OpenGL visualization

int main(){

    auto before = std::chrono::system_clock::now();
    auto beforeMs = std::chrono::time_point_cast<std::chrono::milliseconds>(before);

    // Load Ouster Sample Dataset
    std::string csvFilePath1 = "sample_data/pcap_out_000261.csv";
    std::string csvFilePath2 = "sample_data/pcap_out_000262.csv";
    string datasetType = "ouster";

    Eigen::MatrixXf scan1 = utils::loadPointCloudCSV(csvFilePath1, datasetType);
    Eigen::MatrixXf scan2 = utils::loadPointCloudCSV(csvFilePath2, datasetType);

    int run_length = 7;
    int numBinsPhi = 24;
    int numBinsTheta = 75; 
    Eigen::VectorXf X0;
    X0.resize(6);
    X0 << 1., 0., 0., 0., 0., 0.; //set initial estimate

    ICET it(scan1, scan2, run_length, X0, numBinsPhi, numBinsTheta);
    Eigen::VectorXf X = it.X;
    cout << "soln: " << endl << X;

    visualization viz;
    viz.points1 = it.points1;
    viz.points2 = it.points2;
    viz.clusterBounds = it.clusterBounds; //draw spherical voxel bounds
    //set covariance ellipsoids
    viz.ellipsoid1Means = it.ellipsoid1Means;
    viz.ellipsoid1Covariances = it.ellipsoid1Covariances;
    viz.ellipsoid1Alphas = it.ellipsoid1Alphas;
    viz.ellipsoid2Means = it.ellipsoid2Means;
    viz.ellipsoid2Covariances = it.ellipsoid2Covariances;
    viz.ellipsoid2Alphas = it.ellipsoid2Alphas;
    
    auto afterAll = std::chrono::system_clock::now();
    auto afterAllMs = std::chrono::time_point_cast<std::chrono::milliseconds>(afterAll);
    auto elapsedTimeAllMs = std::chrono::duration_cast<std::chrono::milliseconds>(afterAllMs - beforeMs).count();
    std::cout << "Whole process took: " << elapsedTimeAllMs << " ms" << std::endl;

    viz.display();
    glutMainLoop();
    return 0;
}