#include <GL/glew.h>
#include <GL/glut.h>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include "visualization.h"

using namespace Eigen;
using namespace std;

// Static pointer to hold the instance of the visualization object
visualization* visualizationInstance = nullptr;

visualization::visualization() {
    cameraAngleX = 45.0;
    cameraAngleY = 45.0;
    centerX = 0.;
    centerY = 0.; 
    centerZ = 0.;
    cameraX = 0.;
    cameraY = 0.;
    cameraZ = 0.;
    lastMouseX = 0;
    lastMouseY = 0;
    lastPanX = 0;
    lastPanY = 0;
    isPanning = false;
    isRotating = false;
    cameraDistance = 10.0;

    initializeOpenGL();
    visualizationInstance = this;
}

visualization::~visualization() {
}

void visualization::initializeOpenGL() {
    int argc = 1;
    char *argv[1] = {(char *)"program"};
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutCreateWindow("ICET Visualizer");
    glutReshapeWindow(1200, 800);

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutSpecialFunc(specialKeys);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutKeyboardFunc(keyboard);

    GLenum err = glewInit();
    if (GLEW_OK != err) {
        std::cerr << "Error: GLEW initialization failed: " << glewGetErrorString(err) << std::endl;
    }
}

void visualization::display(){
    if (visualizationInstance){
        visualizationInstance->render();
    }
}

void visualization::reshape(int w, int h) {
    if (visualizationInstance) {
        visualizationInstance->handleReshape(w, h);
    }
}

void visualization::specialKeys(int key, int x, int y) {
    if (visualizationInstance) {
        visualizationInstance->handleSpecialKeys(key, x, y);
    }
}

void visualization::mouse(int button, int state, int x, int y) {
    if (visualizationInstance) {
        visualizationInstance->handleMouse(button, state, x, y);
    }
}

void visualization::motion(int x, int y) {
    if (visualizationInstance) {
        visualizationInstance->handleMotion(x, y);
    }
}

void visualization::keyboard(unsigned char key, int x, int y) {
    if (visualizationInstance) {
        visualizationInstance->handleKeyboard(key, x, y);
    }
}

void visualization::handleReshape(int w, int h) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, static_cast<GLdouble>(w) / h, 0.1, 1000.0);
    glMatrixMode(GL_MODELVIEW);
}

void visualization::handleSpecialKeys(int key, int x, int y) {
    if (key == GLUT_KEY_UP) {
        cameraDistance -= 0.5;
    } else if (key == GLUT_KEY_DOWN) {
        cameraDistance += 0.5;
    }
    glutPostRedisplay();
}

void visualization::handleMouse(int button, int state, int x, int y) {
    // Scroll wheel
    if (button == 3) {
        cameraDistance *= 0.95;
        glutPostRedisplay();
    } else if (button == 4) {
        cameraDistance *= 1.05;
        glutPostRedisplay();
    }

    if (button == GLUT_MIDDLE_BUTTON && state == GLUT_DOWN) {
        lastMouseX = x;
        lastMouseY = y;
        isPanning = true;
    }
    if (button == GLUT_MIDDLE_BUTTON && state == GLUT_UP) {
        isPanning = false;
    }

    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        lastMouseX = x;
        lastMouseY = y;
        isRotating = true;
    }
    if (button == GLUT_LEFT_BUTTON && state == GLUT_UP) {
        isRotating = false;
    }
}

void visualization::handleMotion(int x, int y) {
    int dx = x - lastMouseX;
    int dy = y - lastMouseY;

    if (isPanning) {
        float factor = 0.01f;
        float rightX = -cos(cameraAngleY * M_PI / 180.0);
        float rightY = sin(cameraAngleY * M_PI / 180.0);
        float rightZ = 0.0f;
        float upX = sin(cameraAngleY * M_PI / 180.0) * sin(cameraAngleX * M_PI / 180.0);
        float upY = cos(cameraAngleY * M_PI / 180.0) * sin(cameraAngleX * M_PI / 180.0);
        float upZ = cos(cameraAngleX * M_PI / 180.0);
        centerX -= dx * factor * rightX + dy * factor * upX;
        centerY -= dx * factor * rightY + dy * factor * upY;
        centerZ += dx * factor * rightZ + dy * factor * upZ;

        glutPostRedisplay();
    }

    if (isRotating) {
        cameraAngleY += dx * 0.1;
        cameraAngleX += dy * 0.1;

        if (cameraAngleX > 89.0f) cameraAngleX = 89.0f;
        if (cameraAngleX < -89.0f) cameraAngleX = -89.0f;

        glutPostRedisplay();
    }

    lastMouseX = x;
    lastMouseY = y;
}

void visualization::handleKeyboard(unsigned char key, int x, int y) {
    if (key == 27) {
        exit(0);
    }
}

void visualization::render() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    GLdouble camX = cameraDistance * sin(cameraAngleY * M_PI / 180.0) * cos(cameraAngleX * M_PI / 180.0);
    GLdouble camY = cameraDistance * cos(cameraAngleY * M_PI / 180.0) * cos(cameraAngleX * M_PI / 180.0);
    GLdouble camZ = cameraDistance * sin(cameraAngleX * M_PI / 180.0);

    camX += centerX;
    camY += centerY;
    camZ += centerZ;

    gluLookAt(camX, camY, camZ, centerX, centerY, centerZ, 0.0, 0.0, 1.0);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Draw points
    glColor3f(1.0, 1.0, 1.0);
    glEnable(GL_POINT_SMOOTH);
    drawPoints1();
    glDisable(GL_POINT_SMOOTH);

    glColor3f(1.0, 1.0, 1.0);
    glEnable(GL_POINT_SMOOTH);
    drawPoints2();
    glDisable(GL_POINT_SMOOTH);

    // Draw all voxels
    glColor3f(0.8, 0.8, 1.0);
    glEnable(GL_LINE_SMOOTH);
    for (int i = 0; i < clusterBounds.rows(); i++){
        if (clusterBounds(i,5) != 0){
            drawFrustum(clusterBounds(i,0), clusterBounds(i,1), clusterBounds(i,2), clusterBounds(i,3), clusterBounds(i,4), clusterBounds(i,5));
        }
    }

    // Draw ellipsoids
    array<float, 3> color1 = {0.1, 0.1, 1.0};
    for (size_t i = 0; i < ellipsoid1Means.size(); i++) {
        drawEllipsoid(ellipsoid1Means[i], ellipsoid1Covariances[i], ellipsoid1Alphas[i], color1);
    }
    array<float, 3> color2 = {1., 0.1, 0.1};
    for (size_t i = 0; i < ellipsoid2Means.size(); i++) {
        drawEllipsoid(ellipsoid2Means[i], ellipsoid2Covariances[i], ellipsoid2Alphas[i], color2);
    }

    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        std::cerr << "OpenGL Error: " << gluErrorString(error) << std::endl;
    }

    glutSwapBuffers();
}

void visualization::drawPoints1() {
    glPointSize(3.0f);
    glColor4f(0.5, 0.5, 1.0, 0.5);

    glBegin(GL_POINTS);
    for (int i = 0; i < points1.rows(); ++i) {
        glVertex3f(points1(i, 0), points1(i, 1), points1(i, 2));
    }
    glEnd();
}

void visualization::drawPoints2() {
    glPointSize(3.0f);
    glColor4f(1., 0.5, 0.5, 0.5);
    glBegin(GL_POINTS);
    for (int i = 0; i < points2.rows(); ++i) {
        glVertex3f(points2(i, 0), points2(i, 1), points2(i, 2));
    }
    glEnd();
}

void visualization::drawCroppedSphere(GLfloat radius, GLfloat azimuthalMin, GLfloat azimuthalMax,
                       GLfloat elevationMin, GLfloat elevationMax, GLint slices, GLint stacks) {

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);

    for (GLint i = 0; i < slices; ++i) {
        glBegin(GL_TRIANGLE_STRIP);
        for (GLint j = 0; j <= stacks; ++j) {
            GLfloat azimuthalAngle1 = azimuthalMin + (azimuthalMax - azimuthalMin) * i / slices;
            GLfloat elevationAngle1 = elevationMin + (elevationMax - elevationMin) * j / stacks;

            GLfloat azimuthalAngle2 = azimuthalMin + (azimuthalMax - azimuthalMin) * (i + 1) / slices;
            GLfloat elevationAngle2 = elevationMin + (elevationMax - elevationMin) * j / stacks;

            GLfloat x1 = radius * sin(elevationAngle1) * cos(azimuthalAngle1);
            GLfloat y1 = radius * sin(elevationAngle1) * sin(azimuthalAngle1);
            GLfloat z1 = radius * cos(elevationAngle1);

            GLfloat x2 = radius * sin(elevationAngle2) * cos(azimuthalAngle2);
            GLfloat y2 = radius * sin(elevationAngle2) * sin(azimuthalAngle2);
            GLfloat z2 = radius * cos(elevationAngle2);

            glVertex3f(x1, y1, z1);
            glVertex3f(x2, y2, z2);
        }
        glEnd();
    }
}

void visualization::drawFrustum(GLfloat azimuthalMin, GLfloat azimuthalMax, 
                                GLfloat elevationMin, GLfloat elevationMax,
                                GLfloat innerDistance, GLfloat outerDistance) {

    glBegin(GL_LINES);

    GLfloat points[8][3];

    // Near surface
    points[0][0] = innerDistance * std::sin(elevationMin) * std::cos(azimuthalMin);
    points[0][1] = innerDistance * std::sin(elevationMin) * std::sin(azimuthalMin);
    points[0][2] = innerDistance * std::cos(elevationMin);

    points[1][0] = innerDistance * std::sin(elevationMin) * std::cos(azimuthalMax);
    points[1][1] = innerDistance * std::sin(elevationMin) * std::sin(azimuthalMax);
    points[1][2] = innerDistance * std::cos(elevationMin);

    points[2][0] = innerDistance * std::sin(elevationMax) * std::cos(azimuthalMax);
    points[2][1] = innerDistance * std::sin(elevationMax) * std::sin(azimuthalMax);
    points[2][2] = innerDistance * std::cos(elevationMax);

    points[3][0] = innerDistance * std::sin(elevationMax) * std::cos(azimuthalMin);
    points[3][1] = innerDistance * std::sin(elevationMax) * std::sin(azimuthalMin);
    points[3][2] = innerDistance * std::cos(elevationMax);

    // Far surface
    points[4][0] = outerDistance * std::sin(elevationMin) * std::cos(azimuthalMin);
    points[4][1] = outerDistance * std::sin(elevationMin) * std::sin(azimuthalMin);
    points[4][2] = outerDistance * std::cos(elevationMin);

    points[5][0] = outerDistance * std::sin(elevationMin) * std::cos(azimuthalMax);
    points[5][1] = outerDistance * std::sin(elevationMin) * std::sin(azimuthalMax);
    points[5][2] = outerDistance * std::cos(elevationMin);

    points[6][0] = outerDistance * std::sin(elevationMax) * std::cos(azimuthalMax);
    points[6][1] = outerDistance * std::sin(elevationMax) * std::sin(azimuthalMax);
    points[6][2] = outerDistance * std::cos(elevationMax);

    points[7][0] = outerDistance * std::sin(elevationMax) * std::cos(azimuthalMin);
    points[7][1] = outerDistance * std::sin(elevationMax) * std::sin(azimuthalMin);
    points[7][2] = outerDistance * std::cos(elevationMax);

    // Connecting lines
    for (int i = 0; i < 4; ++i) {
        glVertex3fv(points[i]);
        glVertex3fv(points[i + 4]);
    }

    glEnd();

    // Draw partial sphere on the inner and outer surfaces
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA_SATURATE, GL_ONE);
    glDepthMask(GL_FALSE);

    glColor4f(0.8, 0.8, 1.0, 0.7); 
    drawCroppedSphere(innerDistance, azimuthalMin, azimuthalMax, elevationMin, elevationMax, 12, 12);
    glColor4f(0.8, 0.8, 1.0, 0.7);  
    drawCroppedSphere(outerDistance, azimuthalMin, azimuthalMax, elevationMin, elevationMax, 12, 12);

    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
    glDisable(GL_DEPTH_TEST);
}

void visualization::drawEllipsoid(const Eigen::Vector3f& mean, const Eigen::Matrix3f& covariance, GLfloat alpha, const std::array<float, 3>& color) {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigensolver(covariance);
    Eigen::Vector3f eigenvalues = eigensolver.eigenvalues().real();
    Eigen::Matrix3f eigenvectors = eigensolver.eigenvectors().real();

    glPushMatrix();
    glTranslatef(mean(0), mean(1), mean(2));

    // Apply rotation based on eigenvectors
    Eigen::Matrix4f rotationMatrix;
    rotationMatrix << eigenvectors(0, 0), eigenvectors(0, 1), eigenvectors(0, 2), 0,
                      eigenvectors(1, 0), eigenvectors(1, 1), eigenvectors(1, 2), 0,
                      eigenvectors(2, 0), eigenvectors(2, 1), eigenvectors(2, 2), 0,
                      0, 0, 0, 1;
    glMultMatrixf(rotationMatrix.data());

    // Apply non-uniform scaling based on eigenvalues
    glScalef(2 * sqrt(eigenvalues(0)), 2 * sqrt(eigenvalues(1)), 2 * sqrt(eigenvalues(2)));

    glColor4f(color[0], color[1], color[2], alpha);
    glutSolidSphere(1.0, 30, 30);

    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
    glDisable(GL_DEPTH_TEST);
    glPopMatrix();
}
