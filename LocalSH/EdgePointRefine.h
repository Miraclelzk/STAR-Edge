#pragma once

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <stack>

class EdgePointRefine
{
public:

	bool edgePointRe(std::vector<bool>& flag, std::vector<std::vector<int>>& neighboor, Eigen::MatrixX3d& pos, Eigen::MatrixX3d& new_normals, double mu,  Eigen::MatrixX3d & NewEdgePoints);

};