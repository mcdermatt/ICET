#ifndef ICET_H
#define ICET_H

#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <iostream>
#include <vector>
#include <array>
#include <string>
#include "csv.hpp"
#include <fstream>
#include <limits>
#include <algorithm> 
#include <execution>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>
#include <queue>
#include "ThreadPool.h"
#include "utils.h"

class ICET{
public:
    ICET(Eigen::MatrixXf& scan1, Eigen::MatrixXf& scan2, int runlen, 
    Eigen::VectorXf X0, int num_bins_phi, int num_bins_theta, int n = 25,
    float thresh = 0.1, float buff = 0.1);
    ~ICET();

    //avoid using static methods so we can run multiple ICETs at once?
    void step();

    void fitScan1();
    void prepScan2();
    void fitScan2();

    std::vector<std::vector<std::vector<int>>> sortSphericalCoordinates(Eigen::MatrixXf sphericalCoords);

    std::pair<float, float> findCluster(const Eigen::MatrixXf& sphericalCoords, int n, float thresh, float buff);

    Eigen::MatrixXf filterPointsInsideCluster(const Eigen::MatrixXf& selectedPoints, const Eigen::MatrixXf& clusterBounds);

    Eigen::MatrixXi testSigmaPoints(const Eigen::MatrixXf& selectedPoints, const Eigen::MatrixXf& lims);

    void fitCells1(const std::vector<int>& indices, int theta, int phi);

    std::tuple<Eigen::MatrixXf, Eigen::MatrixXf> fitCells2(const std::vector<int>& indices1, const std::vector<int>& indices2, int theta, int phi);

    void parallelFitCells2(const std::vector<std::vector<std::vector<int>>>& pointIndices1,
                       const std::vector<std::vector<std::vector<int>>>& pointIndices2,
                       int numBinsPhi, int numBinsTheta);

    // Precomputed trig passed directly to avoid redundant sin/cos calls per voxel
    Eigen::MatrixXf get_H(Eigen::Vector3f mu, float sp, float cp, float st, float ct, float ss, float cs);

    std::tuple<Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf> checkCondition(Eigen::MatrixXf HTWH);

    // Helper to convert flat index
    inline int voxelIdx(int theta, int phi) const { return theta * numBinsPhi + phi; }

    //algorithm params
    int rl;
    int numBinsPhi;  
    int numBinsTheta;
    int n; 
    float thresh;
    float buff; 

    Eigen::MatrixXf points1;
    Eigen::MatrixXf points1Spherical;
    Eigen::MatrixXf points2;
    Eigen::MatrixXf points2_OG;
    Eigen::MatrixXf points2Spherical;
    Eigen::MatrixXf clusterBounds;
    Eigen::MatrixXf testPoints;
    Eigen::MatrixXf HTWH_i;
    Eigen::MatrixXf HTWdz_i; 
    Eigen::VectorXf pred_stds;

    // Flat vectors indexed by voxelIdx(theta, phi) — replaces nested std::map
    std::vector<Eigen::Matrix3f> sigma1;
    std::vector<Eigen::Matrix3f> sigma2;
    std::vector<Eigen::Vector3f> mu1;
    std::vector<Eigen::Vector3f> mu2;
    std::vector<Eigen::Matrix3f> L;
    std::vector<Eigen::Matrix3f> U;
    // Track which voxels have valid scan1 fits
    std::vector<bool> voxelOccupied;

    std::vector<std::vector<std::vector<int>>> pointIndices1;
    std::vector<std::vector<std::vector<int>>> pointIndices2;

    Eigen::VectorXf X; //global solution vector
    Eigen::VectorXf dx; //linear perterbation to X solved for during each iteration

    //for viz
    std::vector<Eigen::Vector3f> ellipsoid1Means;
    std::vector<Eigen::Matrix3f> ellipsoid1Covariances;
    std::vector<float> ellipsoid1Alphas;
    std::vector<Eigen::Vector3f> ellipsoid2Means;
    std::vector<Eigen::Matrix3f> ellipsoid2Covariances;
    std::vector<float> ellipsoid2Alphas;

private:

    std::string status; 

    ThreadPool pool;
    std::vector<std::future<void>> futures;

};

#endif