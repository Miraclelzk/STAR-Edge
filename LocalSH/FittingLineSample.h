#pragma once

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <stack>

class FittingLineSample
{
public:

    /**
    * @briefPC
    *
    * @param pcc
	* @param projectedPcc
	* @param projected_2d
    *
    */
	bool projectPointCloud(const Eigen::MatrixX3d pcc, Eigen::MatrixX3d & projectedPcc, Eigen::MatrixX2d& projected_2d);

	/**
	* @brief
	*
	* @param projected_2d
	* @param hullIndicespc
	*
	*/
	bool convexHull(const Eigen::MatrixX2d projected_2d, std::vector<int> & hullIndices);

	bool fitCurve(const Eigen::MatrixX3d pccHull, int numSamples,int degree, Eigen::MatrixX3d & sampleCurve);

};