#include <Eigen/Dense>
#include <random>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include "csv.hpp"
#include "csv.hpp"
#include <fstream>
#include <cmath>
#include <limits>
#include <algorithm>
#include <map>
#include <execution>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>
#include <queue>
#include "ThreadPool.h"
#include "utils.h"
#include "icet.h"

using namespace Eigen;
using namespace std;

// Constructor implementation
ICET::ICET(MatrixXf& scan1, MatrixXf& scan2, int runlen, Eigen::VectorXf X0,
           int num_bins_phi, int num_bins_theta) : points1(scan1), points2(scan2), rl(runlen), X(X0), 
           numBinsPhi(num_bins_phi), numBinsTheta(num_bins_theta), pool(4) {

    // init hyperparameters for spherical voxels
    n = 25; //50; // min size of the cluster
    thresh = 0.3; // 0.1 indoor, 0.3 outdoor; // Jump threshold for beginning and ending radial clusters
    buff = 0.5; // 0.1 indoor, outdoor 0.5; //buffer to add to inner and outer cluster range (helps attract nearby distributions)

    points2_OG = points2;
    HTWH_i.resize(6,6);
    HTWdz_i.resize(6,1);
    pred_stds = Eigen::VectorXf(6);
    pred_stds.setZero();

    clusterBounds.resize(numBinsPhi*numBinsTheta,6);
    testPoints.resize(numBinsPhi*numBinsTheta*6,3);

    fitScan1();
    prepScan2();
    for (int iter=0; iter<rl; iter++){fitScan2();}

    // // if we want to threadpool fit gaussians in scan2, we can't draw ellipsoids for scan2 
    // //update visualization for scan2 ellipsoids
    // for (const auto& outerPair : sigma2) {
    //     for (const auto& innerPair : outerPair.second) {
    //         ellipsoid2Covariances.push_back(innerPair.second);
    //     }
    // }
    // for (const auto& outerPair : mu2) {
    //     for (const auto& innerPair : outerPair.second) {
    //         ellipsoid2Means.push_back(innerPair.second);
    //         ellipsoid2Alphas.push_back(0.3f);
    //     }
    // }

}

ICET::~ICET() {
    // Destructor implementation
}

void ICET::fitScan1(){

    // auto beforesort = std::chrono::system_clock::now();
    // auto beforesortms = std::chrono::time_point_cast<std::chrono::milliseconds>(beforesort);

    points1Spherical = utils::cartesianToSpherical(points1);

    // auto aftersort = std::chrono::system_clock::now();
    // auto aftersortms = std::chrono::time_point_cast<std::chrono::milliseconds>(aftersort);
    // auto ets = std::chrono::duration_cast<std::chrono::milliseconds>(aftersortms - beforesortms).count();
    // std::cout << "c2s took: " << ets << " ms" << std::endl;

    // Sort sphericalCoords based on radial distance
    vector<int> index(points1Spherical.rows());
    iota(index.begin(), index.end(), 0);
    // sort(index.begin(), index.end(), [&](int a, int b) {
    sort(std::execution::par, index.begin(), index.end(), [&](int a, int b) {
        return points1Spherical(a, 0) < points1Spherical(b, 0); // Sort by radial distance
    });
    for (int i = 0; i < points1Spherical.rows(); i++) {
        if (index[i] != i) {
            points1Spherical.row(i).swap(points1Spherical.row(index[i]));
            std::swap(index[i], index[index[i]]); // Update the index
        }
    }

    //get spherical coordiantes and fit gaussians to points from first scan 
    pointIndices1 = sortSphericalCoordinates(points1Spherical);

    // // Define a lambda function to wrap the member function fitCells1
    // // not feasible to use threadpool since we need concurrent writes to U and L?
    // auto task = [this](const std::vector<int>& indices, int theta, int phi) {
    //     this->fitCells1(indices, theta, phi);
    // };

    int count = 0;
    for (int phi = 0; phi < numBinsPhi; phi++){
        for (int theta = 0; theta< numBinsTheta; theta++){
            // Retrieve the point indices inside angular bin
            const vector<int>& indices = pointIndices1[theta][phi];
            // futures.push_back(pool.enqueue(task, indices, theta, phi)); //run multithread
            fitCells1(indices, theta, phi); //no multithreading
        }
    }
    // // Wait for all tasks to complete
    // for (auto &fut : futures) {
    //     fut.get();
    // }
}

void ICET::fitCells1(const vector<int>& indices, int theta, int phi){
    float innerDistance;
    float outerDistance;


    // only calculate inner/outer bounds if there are a sufficient number of points in the spike 
    if (indices.size() >= n) {

        // Use the indices to access the corresponding rows in sortedPointsSpherical
        MatrixXf selectedPoints = MatrixXf::Zero(indices.size(), points1Spherical.cols());
        for (int i = 0; i < indices.size(); ++i) { 
            selectedPoints.row(i) = points1Spherical.row(indices[i]);
            // cout << i;
        }

        // find inner and outer bounds for each theta/phi bin
        pair<float, float> clusterDistances = findCluster(selectedPoints, n, thresh, buff);
        innerDistance = clusterDistances.first;
        outerDistance = clusterDistances.second;

        //convert [desiredPhi][desiredTheta] to azimMin, azimMax, elevMin, elevMax
        float azimMin_i =  (static_cast<float>(theta) / numBinsTheta) * (2 * M_PI) ;
        float azimMax_i =  (static_cast<float>(theta+1) / numBinsTheta) * (2 * M_PI) ;
        float elevMin_i =  (static_cast<float>(phi) / numBinsPhi) * (M_PI) ;
        float elevMax_i =  (static_cast<float>(phi+1) / numBinsPhi) * (M_PI) ;
        //hold on to these values
        clusterBounds.row(numBinsTheta*phi + theta) << azimMin_i, azimMax_i, elevMin_i, elevMax_i, innerDistance, outerDistance;

        // find points from first scan inside voxel bounds and fit gaussians to each cluster
        MatrixXf filteredPoints = filterPointsInsideCluster(selectedPoints, clusterBounds.row(numBinsTheta*phi + theta));
        if (outerDistance > 0.1 && filteredPoints.size() >= n){
            MatrixXf filteredPointsCart = utils::sphericalToCartesian(filteredPoints);
            Eigen::VectorXf mean = filteredPointsCart.colwise().mean();
            Eigen::MatrixXf centered = filteredPointsCart.rowwise() - mean.transpose();
            Eigen::MatrixXf covariance = (centered.adjoint() * centered) / static_cast<float>(filteredPointsCart.rows() - 1);

            // cout << endl;
            // cout << "selectedPoints: " << selectedPoints.rows() << endl;
            // for (int i = 0; i < 10 && i <selectedPoints.rows();i++){
            //     cout << selectedPoints.row(i) << endl;
            // }
            // cout << "filteredPoints: " << endl << filteredPoints.rows() << endl; 
            // for (int i = 0; i < 10 && i <filteredPoints.rows();i++){
            //     cout << filteredPoints.row(i) << endl;
            // }
            // cout << "filteredPointsCart: " << endl << filteredPointsCart.rows() << endl; 
            // cout << " covariance: " << endl << covariance << endl;

            //hold on to means and covariances of clusters from scan1
            sigma1[theta][phi] = covariance;
            mu1[theta][phi] = mean;

            // get U and L ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigensolver(covariance);
            Eigen::Vector3f eigenvalues = eigensolver.eigenvalues().real();
            Eigen::Matrix3f eigenvectors = eigensolver.eigenvectors().real();
            U[theta][phi] = eigenvectors.transpose();

            // create 6 2-sigma test points for each cluster and test to see if they fit inside the voxel
            MatrixXf axislen(3,3);
            axislen << eigenvalues[0], 0, 0,
                        0, eigenvalues[1], 0,
                        0, 0, eigenvalues[2];
            axislen = 2.0 * axislen.array().sqrt(); //theoretically should be *2 not *3 but this seems to work better

            MatrixXf rotated = axislen * U[theta][phi].transpose();

            Eigen::MatrixXf sigmaPoints(6,3);
            //converges faster on Ouster dataset, but won't work in simulated tunnel
            sigmaPoints.row(0) = mu1[theta][phi] + rotated.row(0).transpose(); //most compact axis
            sigmaPoints.row(1) = mu1[theta][phi] - rotated.row(0).transpose();
            sigmaPoints.row(2) = mu1[theta][phi] + rotated.row(1).transpose(); //middle
            sigmaPoints.row(3) = mu1[theta][phi] - rotated.row(1).transpose();
            sigmaPoints.row(4) = mu1[theta][phi] + rotated.row(2).transpose(); //largest axis
            sigmaPoints.row(5) = mu1[theta][phi] - rotated.row(2).transpose();

            // find out which test points fall inside the voxel bounds
            Eigen::MatrixXf sigmaPointsSpherical = utils::cartesianToSpherical(sigmaPoints);
            MatrixXi sigmaPointsInside = testSigmaPoints(sigmaPointsSpherical, clusterBounds.row(numBinsTheta*phi + theta));
            
            //see if each axis contains at least one test point within voxel
            if ((sigmaPointsInside.array() == 0).any() || (sigmaPointsInside.array() == 1).any()){
                L[theta][phi].row(0) << 1, 0, 0; 
            } 
            else{
                L[theta][phi].row(0) << 0, 0, 0;
                testPoints.row(6*(numBinsTheta*phi + theta)) = sigmaPoints.row(0).transpose();
                testPoints.row(6*(numBinsTheta*phi + theta)+1) = sigmaPoints.row(1).transpose();
            }
            if ((sigmaPointsInside.array() == 2).any() || (sigmaPointsInside.array() == 3).any()){
                L[theta][phi].row(1) << 0, 1, 0; 
            } 
            else{
                L[theta][phi].row(1) << 0, 0, 0;
                testPoints.row(6*(numBinsTheta*phi + theta)+2) = sigmaPoints.row(2).transpose();
                testPoints.row(6*(numBinsTheta*phi + theta)+3) = sigmaPoints.row(3).transpose();
            }
            if ((sigmaPointsInside.array() == 4).any() || (sigmaPointsInside.array() == 5).any()){
                L[theta][phi].row(2) << 0, 0, 1; 
            } 
            else{
                L[theta][phi].row(2) << 0, 0, 0;
                testPoints.row(6*(numBinsTheta*phi + theta)+4) = sigmaPoints.row(4).transpose();
                testPoints.row(6*(numBinsTheta*phi + theta)+5) = sigmaPoints.row(5).transpose();
            }
            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            //update for drawing
            float alpha1 = 0.3f;
            ellipsoid1Means.push_back(mean);
            ellipsoid1Covariances.push_back(covariance);
            ellipsoid1Alphas.push_back(alpha1);
        }
    }
    // use 0 value as a flag for unoccupied voxels
    else{
        innerDistance = 0;
        outerDistance = 0;
        float azimMin_i =  (static_cast<float>(theta) / numBinsTheta) * (2 * M_PI) ;
        float azimMax_i =  (static_cast<float>(theta+1) / numBinsTheta) * (2 * M_PI) ;
        float elevMin_i =  (static_cast<float>(phi) / numBinsPhi) * (M_PI) ;
        float elevMax_i =  (static_cast<float>(phi+1) / numBinsPhi) * (M_PI) ;      
        clusterBounds.row(numBinsTheta*phi + theta) << azimMin_i, azimMax_i, elevMin_i, elevMax_i, innerDistance, outerDistance;
    }
}

void ICET::prepScan2(){

    // //apply x0 here to get slightly better radial sorting (since we only do it once for scan2)
    // MatrixXf rot_mat = utils::R(X[3], X[4], X[5]); 
    // Eigen::RowVector3f trans(X[0], X[1], X[2]);
    // points2 = points2_OG.rowwise() + trans;
    // points2 = points2 * rot_mat;

    //sort radially only once at begninning of process
    points2Spherical = utils::cartesianToSpherical(points2);
    vector<int> index(points2Spherical.rows());
    iota(index.begin(), index.end(), 0);
    sort(std::execution::par, index.begin(), index.end(), [&](int a, int b) {
        return points2Spherical(a, 0) < points2Spherical(b, 0); // Sort by radial distance
    });
    for (int i = 0; i < points2Spherical.rows(); i++) {
        if (index[i] != i) {
            points2Spherical.row(i).swap(points2Spherical.row(index[i]));
            std::swap(index[i], index[index[i]]); // Update the index
        }
    }
    points2_OG = utils::sphericalToCartesian(points2Spherical);

}

std::tuple<Eigen::MatrixXf, Eigen::MatrixXf> ICET::fitCells2(const std::vector<int>& indices1, const std::vector<int>& indices2, int theta, int phi){

    Eigen::MatrixXf HTWdz_j;
    HTWdz_j.resize(6,1);
    HTWdz_j.setZero();
    Eigen::MatrixXf HTWH_j;
    HTWH_j.resize(6,6);
    HTWH_j.setZero();

    // only fit gaussians if there enough points from both scans 1 and 2 in the cell 
    // if ((indices2.size() > n) && (indices1.size() > n)) {
    if ((indices2.size() > n) && (indices1.size() > n) && (clusterBounds.row(numBinsTheta*phi + theta)[5] > 1)) { //U and L won't exist if no useful bin in scan1
        // Use the indices to access the corresponding rows in sortedPointsSpherical
        // when not re-sorting by radial distance after each update of X
        MatrixXf selectedPoints2 = MatrixXf::Zero(indices2.size(), points2Spherical.cols());
        for (int i = 0; i < indices2.size(); ++i) {
            selectedPoints2.row(i) = points2Spherical.row(indices2[i]);
        }

        // find points from first scan inside voxel bounds and fit gaussians to each cluster
        MatrixXf filteredPoints2 = filterPointsInsideCluster(selectedPoints2, clusterBounds.row(numBinsTheta*phi + theta));

        // only carry on if there are enough points from scan2 actually inside the radial bounds
        if (filteredPoints2.size()/3 > n){
            MatrixXf filteredPointsCart2 = utils::sphericalToCartesian(filteredPoints2);
            Eigen::VectorXf mean = filteredPointsCart2.colwise().mean();
            Eigen::MatrixXf centered = filteredPointsCart2.rowwise() - mean.transpose();
            Eigen::MatrixXf covariance = (centered.adjoint() * centered) / static_cast<float>(filteredPointsCart2.rows() - 1);

            // //hold on to means and covariances of clusters-- can't do this and threadpool
            // sigma2[theta][phi] = covariance;
            // mu2[theta][phi] = mean;
            
            //add contributions to HTWH
            // Get noise components
            Eigen::MatrixXf R_noise(3,3);
            R_noise << (sigma1[theta][phi] / (indices1.size() - 1)) + (covariance / (indices2.size()-1)); //supposed to be this
            // use projection matrix to remove extended directions
            R_noise = L[theta][phi] * U[theta][phi].transpose() * R_noise * U[theta][phi] * L[theta][phi].transpose(); //was this in python

            // pinv noise to get weighting
            Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXf> cod(R_noise);
            Eigen::MatrixXf W = cod.pseudoInverse();

            //get H matrix for voxel j
            Eigen::Vector3f angs;
            angs << X[3], X[4], X[5];
            Eigen::MatrixXf H_j = get_H(mean, angs); //working correctly (same output as python code)

            //suppress rows of H corresponding to overly extended directions
            Eigen::MatrixXf H_z = L[theta][phi] * U[theta][phi].transpose() * H_j;

            //put together HTWH for voxel j and contribute to total HTWH_i (for all voxels of current iteration)
            HTWH_j = H_z.transpose() * W * H_z;

            // get compact residuals between means of distributions from scans 1 and 2
            Eigen::Vector3f z1 = L[theta][phi] * U[theta][phi].transpose() * mu1[theta][phi];
            Eigen::Vector3f z2 = L[theta][phi] * U[theta][phi].transpose() * mean;
            Eigen::Vector3f dz = z2-z1;
            HTWdz_j = H_z.transpose() * W * dz;

        }
    }
    
    return std::make_tuple(HTWH_j, HTWdz_j);
}

void ICET::parallelFitCells2(const std::vector<std::vector<std::vector<int>>>& pointIndices1,
                             const std::vector<std::vector<std::vector<int>>>& pointIndices2,
                             int numBinsPhi, int numBinsTheta) {

    // Create a vector to hold futures
    std::vector<std::future<std::tuple<Eigen::MatrixXf, Eigen::MatrixXf>>> futures;

    // Iterate over cells and submit tasks to the thread pool
    for (int phi = 0; phi < numBinsPhi; phi++) {
        for (int theta = 0; theta < numBinsTheta; theta++) {
            const std::vector<int>& indices1 = pointIndices1[theta][phi];
            const std::vector<int>& indices2 = pointIndices2[theta][phi];

            // Submit task to thread pool
            futures.push_back(pool.enqueue(&ICET::fitCells2, this, std::cref(indices1), std::cref(indices2), theta, phi));
        }
    }

    // Wait for all tasks to complete and aggregate results
    for (auto& future : futures) {
        auto result = future.get();
        HTWH_i += std::get<0>(result);
        HTWdz_i += std::get<1>(result);
    }
}

void ICET::fitScan2(){

    // apply transformation to points2 (takes ~0.4ms)
    MatrixXf rot_mat = utils::R(X[3], X[4], X[5]); 
    Eigen::RowVector3f trans(X[0], X[1], X[2]);
    points2 = points2_OG.rowwise() + trans;
    points2 = points2 * rot_mat;
    // points2 = points2_OG;

    // It is inefficient to construct the full (H^T W H) matrix direclty since W is very sparse
    // Instead we sum contributions from each voxel to a single 6x6 matrix to avoid memory inefficiency   
    HTWH_i.setZero();
    // Similarly, we accumulate contributions to (H^T W dz) from each voxel
    HTWdz_i.setZero();

    points2Spherical = utils::cartesianToSpherical(points2);
    pointIndices2 = sortSphericalCoordinates(points2Spherical); 

    // //fit gaussians (single thread)
    // for (int phi = 0; phi < numBinsPhi; phi++){
    //     for (int theta = 0; theta< numBinsTheta; theta++){
    //         // Retrieve the point indices inside angular bin
    //         const vector<int>& indices1 = pointIndices1[theta][phi];
    //         const vector<int>& indices2 = pointIndices2[theta][phi];

    //         //get cell j's contribution to HTWH and HTWdz for iteration i
    //         std::tuple<Eigen::MatrixXf, Eigen::MatrixXf> result = fitCells2(indices1, indices2, theta, phi);
    //         Eigen::MatrixXf HTWH_j = std::get<0>(result);
    //         Eigen::MatrixXf HTWdz_j = std::get<1>(result);
    //         HTWH_i += HTWH_j;
    //         HTWdz_i += HTWdz_j;
    //     }
    // }

    //fit gaussians (thread pool)
    parallelFitCells2(pointIndices1, pointIndices2, numBinsPhi, numBinsTheta);

    //update noise matrix
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXf> temp(HTWH_i);
    Eigen::MatrixXf noise_mat = temp.pseudoInverse();
    pred_stds[0] = sqrt(abs(noise_mat(0,0)));
    pred_stds[1] = sqrt(abs(noise_mat(1,1)));
    pred_stds[2] = sqrt(abs(noise_mat(2,2)));
    pred_stds[3] = sqrt(abs(noise_mat(3,3)));
    pred_stds[4] = sqrt(abs(noise_mat(4,4)));
    pred_stds[5] = sqrt(abs(noise_mat(5,5)));
    // //Check condition for HTWH to suppress globally ambiguous components ~~~~~~~~~~~

    auto result = checkCondition(HTWH_i);
    MatrixXf L2 = get<0>(result);
    MatrixXf lam = get<1>(result);
    MatrixXf U2 = get<2>(result);

    // dx = (pinv(L2 * lam * U2.T) * L2 * U2.T() ) * HTWdz_i;
    // get pseudoinverse of inner parts
    Eigen::MatrixXf innards = L2 * lam * U2.transpose(); 
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXf> cod(innards);
    Eigen::MatrixXf inverted_innards = cod.pseudoInverse();
    dx = (inverted_innards * L2 * U2.transpose() ) * HTWdz_i;
    
    // std::cout << "dx: \n " << dx << endl;
    X += dx;
    // std::cout << "X: \n " << X << endl;

}

void ICET::step(){
    cout << "rl: " << rl << endl;
    rl--;
}

tuple<MatrixXf, MatrixXf, MatrixXf> ICET::checkCondition(MatrixXf HTWH){
    //function for checking condition number of HTWH
    //      if not enough information present (i.e. any compoenents are globally ambiguous) create additional axis pruning matrix

    // L2 = identity matrix which keeps non-extended axis of solution [n, 6]
    //      n = number of non-globally ambiguous axis 
    // lam = diagonal eigenvalue matrix [6,6]
    // U2 = rotation matrix to transform for L2 pruning [6, 6]

    //higher than this threshold and there is not enough information about a solution component to invert HTWH 
    float cutoff = 1e6;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eigensolver(HTWH);
    Eigen::MatrixXf U2 = eigensolver.eigenvectors().real();
    // Eigen::MatrixXf U2 = eigensolver.eigenvectors().real().transpose(); //test
    Eigen::VectorXf eigenvalues = eigensolver.eigenvalues().real();

    // std::cout << "\n eigenvalues: \n" << eigenvalues << endl;
    float condition = eigenvalues(5) / eigenvalues(0);
    // std::cout << "\n OG condition: " << condition << endl;

    MatrixXf L2(6,6);
    L2.setIdentity();

    //chop off the top row of the eye matrix until condition drops below desired threshold
    int eyecount = 1;
    while (std::abs(condition) > cutoff){

        // ~~~~~~~~~~~~~ For easier integration with ROS odometry msg types ~~~~~~~~~~~~~~~~~~~~~~~
        // inflate axis in solution that correspond to conditioning issues in intverting matrix
        //   (TODO: project this properly across 6 dimensions instead of just linearly combining components)
        // Ex of how this currently works:
        //   In case of long tunnel roughly aligned with Y axis:
        //   U2.T = [[0.025, 0.95, 0.0001, 0.0001, 0.00001]
        //           [...]]
        //   We are inflating the y component of the pred std 
        pred_stds += U2.transpose().row(eyecount-1);
        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        L2.block(0, 0, L2.rows() - 1, L2.cols()) = L2.block(1, 0, L2.rows() - 1, L2.cols());
        L2.conservativeResize(L2.rows() - 1, Eigen::NoChange);
        condition = eigenvalues(5) / eigenvalues(eyecount);
        eyecount++;
    }

    Eigen::MatrixXf lam = eigenvalues.asDiagonal();

    return make_tuple(L2, lam, U2);

}

MatrixXf ICET::get_H(Eigen::Vector3f mu, Eigen::Vector3f angs){

    float phi = angs[0];
    float theta = angs[1];
    float psi = angs[2];    

    MatrixXf H(3,6);
    MatrixXf eye(3,3);
    eye << -1, 0, 0,
             0, -1, 0,
             0, 0, -1;
    H.block(0,0,3,3) << eye;

    // deriv of R() wrt phi.dot(mu)
    Eigen::MatrixXf Jx(3,3);
    Jx << 0., (-sin(psi)*sin(phi) + cos(phi)*sin(theta)*cos(psi)), (cos(phi)*sin(psi) + sin(theta)*sin(phi)*cos(psi)),
          0., (-sin(phi)*cos(psi) - cos(phi)*sin(theta)*sin(psi)), (cos(phi)*cos(psi) - sin(theta)*sin(psi)*sin(phi)), 
          0., (-cos(phi)*cos(theta)), (-sin(phi)*cos(theta));
    Jx = Jx * mu;
    H.block(0, 3, 3, 1) = Jx;

    // deriv of R() wrt theta.dot(mu)
    Eigen::MatrixXf Jy(3,3);
    Jy << (-sin(theta)*cos(psi)), (cos(theta)*sin(phi)*cos(psi)), (-cos(theta)*cos(phi)*cos(psi)),
          (sin(psi)*sin(theta)), (-cos(theta)*sin(phi)*sin(psi)), (cos(theta)*sin(psi)*cos(phi)),
          (cos(theta)), (sin(phi)*sin(theta)), (-sin(theta)*cos(phi));
    Jy = Jy * mu;
    H.block(0, 4, 3, 1) = Jy;

    // deriv of R() wrt psi.dot(mu)
    Eigen::MatrixXf Jz(3,3);
    Jz << (-cos(theta)*sin(psi)), (cos(psi)*cos(phi) - sin(phi)*sin(theta)*sin(psi)), (cos(psi)*sin(phi) + sin(theta)*cos(phi)*sin(psi)),
         (-cos(psi)*cos(theta)), (-sin(psi)*cos(phi) - sin(phi)*sin(theta)*cos(psi)), (-sin(phi)*sin(psi) + sin(theta)*cos(psi)*cos(phi)),
         0., 0., 0.;
    Jz = Jz * mu;
    H.block(0, 5, 3, 1) = Jz;

    return H;
}

vector<vector<vector<int>>> ICET::sortSphericalCoordinates(Eigen::MatrixXf sphericalCoords) {
    // Create a 2D vector of vectors to store point indices in each bin
    vector<vector<vector<int>>> pointIndices(numBinsTheta, vector<vector<int>>(numBinsPhi));

    // Iterate through each spherical coordinate
    for (int i = 0; i < sphericalCoords.rows(); ++i) {
        // Extract phi and theta values
        float theta = sphericalCoords(i, 1);
        float phi = sphericalCoords(i, 2);

        // Calculate bin indices
        int binTheta = static_cast<int>((theta / (2 * M_PI)) * numBinsTheta) % numBinsTheta;
        int binPhi = static_cast<int>((phi / M_PI) * numBinsPhi) % numBinsPhi;

        // Store the point index in the corresponding bin
        pointIndices[binTheta][binPhi].push_back(i);
    }

    // Return the vector of point indices
    return pointIndices;
}

pair<float, float> ICET::findCluster(const MatrixXf& sphericalCoords, int n, float thresh, float buff) {
    int numPoints = sphericalCoords.rows();

    float innerDistance = 0.0;
    float outerDistance = 0.0;
    vector<Vector3f> localPoints;

    for (int i = 0; i < numPoints; i++) {
        Vector3f point = sphericalCoords.row(i);

        // Check if the point is within the threshold of the last point
        if (!localPoints.empty() && std::abs(localPoints.back()(0) - point(0)) <= thresh) {
            // Add the point to the current cluster
            localPoints.push_back(point);
        } else {
            // Check if the cluster is sufficiently large
            if (localPoints.size() >= n) {
                // Found a sufficiently large cluster
                innerDistance = localPoints.front()(0) - buff;
                outerDistance = localPoints.back()(0) + buff;
                // cout << "Found cluster - Inner Distance: " << innerDistance << ", Outer Distance: " << outerDistance << endl;
                return {innerDistance, outerDistance};
            } else {
                // Reset the cluster if it's not large enough
                localPoints.clear();
                // Add the current point to start a new cluster
                localPoints.push_back(point);
            }
        }
    }
    // Check for the last cluster at the end of the loop
    if (localPoints.size() >= n) {
        // innerDistance = localPoints.front()(0);
        // outerDistance = localPoints.back()(0);
        if (localPoints.front()(0) !=0){
            innerDistance = localPoints.front()(0) - buff;
            outerDistance = localPoints.back()(0) + buff;
            // cout << "Found cluster - Inner Distance: " << innerDistance << ", Outer Distance: " << outerDistance << endl;
            return {innerDistance, outerDistance};
        }
        else{
            return {0.0, 0.0};
        }
    }

    return {innerDistance, outerDistance};
}

MatrixXf ICET::filterPointsInsideCluster(const MatrixXf& selectedPoints, const MatrixXf& lims) {
    int numPoints = selectedPoints.rows();
    int numClusters = lims.rows();

    // cout << "numClusters: " << numClusters << endl;

    MatrixXf filteredPoints(numPoints, 3);
    int filteredRowCount = 0;

    // for (int i = 0; i < numClusters; i++) {
    float azimMin = lims(0, 0);
    float azimMax = lims(0, 1);
    float elevMin = lims(0, 2);
    float elevMax = lims(0, 3);
    float innerDistance = lims(0, 4);
    float outerDistance = lims(0, 5);

    for (int j = 0; j < numPoints; j++) {
        float azim = selectedPoints(j, 1);
        float elev = selectedPoints(j, 2);
        float r = selectedPoints(j, 0);

        // Check if the point is within the cluster bounds
        if (azim >= azimMin && azim <= azimMax &&
            elev >= elevMin && elev <= elevMax &&
            r >= innerDistance && r <= outerDistance) {
            // Add the point to the filteredPoints matrix
            filteredPoints.row(filteredRowCount++) = selectedPoints.row(j);
        }

        //TODO: was able to do this in old ICET codebase but this seems to break things now
        //      is there a reason selectedPoints isn't sorted?
        // // If the current point is beyond the outer distance, break the inner loop
        // if (r > outerDistance) {
        //     break;
        // }
    }
    // }

    // Resize the matrix to remove unused rows
    filteredPoints.conservativeResize(filteredRowCount, 3);

    return filteredPoints;
}

MatrixXi ICET::testSigmaPoints(const MatrixXf& selectedPoints, const MatrixXf& clusterBounds) {
    int numPoints = selectedPoints.rows();
    int numClusters = clusterBounds.rows();

    // Vector to store indices of filtered points
    vector<int> filteredIndices;

    for (int i = 0; i < numClusters; i++) {
        float azimMin = clusterBounds(i, 0);
        float azimMax = clusterBounds(i, 1);
        float elevMin = clusterBounds(i, 2);
        float elevMax = clusterBounds(i, 3);
        float innerDistance = clusterBounds(i, 4);
        float outerDistance = clusterBounds(i, 5);

        for (int j = 0; j < numPoints; j++) {
            float azim = selectedPoints(j, 1);
            float elev = selectedPoints(j, 2);
            float r = selectedPoints(j, 0);

            // Check if the point is within the cluster bounds
            if (azim >= azimMin && azim <= azimMax &&
                elev >= elevMin && elev <= elevMax &&
                r >= innerDistance && r <= outerDistance) {
                // Add the index to the filteredIndices vector
                filteredIndices.push_back(j);
            }

            // If the current point is beyond the outer distance, break the inner loop
            if (r > outerDistance) {
                break;
            }
        }
    }

    // Create a matrix from the indices
    MatrixXi filteredIndicesMatrix(filteredIndices.size(), 1);
    for (size_t i = 0; i < filteredIndices.size(); i++) {
        filteredIndicesMatrix(i, 0) = filteredIndices[i];
    }

    return filteredIndicesMatrix;
}