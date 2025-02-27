#pragma once

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

class NormalRefine
{
public:

	// RANSAC (0,0,0)
	bool ransacFitPlaneThroughOrigin(Eigen::MatrixX3d pcc, int maxIterations, Eigen::Vector3d & norm);

	bool checkNormalDirection(Eigen::MatrixX3d normal_origin, Eigen::MatrixX3d normals, Eigen::MatrixX3d & new_normals);

};