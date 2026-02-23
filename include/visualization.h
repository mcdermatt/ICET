#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include <GL/glew.h>
#include <GL/glut.h>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <array>
#include <string>

class visualization {
public:
    visualization();
    ~visualization();

    void initializeOpenGL();
    void render();
    static void display();
    static void reshape(int w, int h);
    static void specialKeys(int key, int x, int y);
    static void mouse(int button, int state, int x, int y);
    static void motion(int x, int y);
    static void keyboard(unsigned char key, int x, int y);    

    Eigen::MatrixXf points1;
    Eigen::MatrixXf points2;    
    Eigen::MatrixXf clusterBounds;
    std::vector<Eigen::Vector3f> ellipsoid1Means;
    std::vector<Eigen::Matrix3f> ellipsoid1Covariances;
    std::vector<float> ellipsoid1Alphas;

    std::vector<Eigen::Vector3f> ellipsoid2Means;
    std::vector<Eigen::Matrix3f> ellipsoid2Covariances;
    std::vector<float> ellipsoid2Alphas;

private:
    void handleReshape(int w, int h);
    void handleSpecialKeys(int key, int x, int y);
    void handleMouse(int button, int state, int x, int y);
    void handleMotion(int x, int y);
    void handleKeyboard(unsigned char key, int x, int y);

    GLdouble cameraDistance;
    GLdouble cameraAngleX;
    GLdouble cameraAngleY;
    GLdouble centerX;
    GLdouble centerY;
    GLdouble centerZ;
    double cameraX, cameraY, cameraZ;

    int lastMouseX;
    int lastMouseY;
    int lastPanX;
    int lastPanY;
    bool isPanning;
    bool isRotating;

    void drawPoints1(); 
    void drawPoints2();
    void drawFrustum(GLfloat azimuthalMin, GLfloat azimuthalMax,
                     GLfloat elevationMin, GLfloat elevationMax, 
                     GLfloat innerDistance, GLfloat outerDistance);
    void drawEllipsoid(const Eigen::Vector3f& mean, 
                       const Eigen::Matrix3f& covariance, 
                       GLfloat alpha, const std::array<float, 3>& color);

    void drawCroppedSphere(GLfloat radius, GLfloat azimuthalMin, GLfloat azimuthalMax,
                       GLfloat elevationMin, GLfloat elevationMax, GLint slices, GLint stacks); 

};

#endif // VISUALIZATION_H
