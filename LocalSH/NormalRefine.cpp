#include <nanoflann.hpp>

#include "NormalRefine.h"
#include <iostream>
#include <random>
#include <omp.h>

// RANSAC (0,0,0)
bool NormalRefine::ransacFitPlaneThroughOrigin(Eigen::MatrixX3d pcc, int maxIterations, Eigen::Vector3d & bestNormal)
{
	//Eigen::Vector3d bestNormal;
	int maxInliers = 0;

	for (int i = 0; i < pcc.rows(); ++i) {
		for (int j = i + 1; j < pcc.rows(); ++j) {
			Eigen::Vector3d p1 = pcc.row(i);
			Eigen::Vector3d p2 = pcc.row(j);

			//
			Eigen::Vector3d p0(0.0, 0.0, 0.0);

			//
			Eigen::Vector3d v1 = p1 - p0;
			Eigen::Vector3d v2 = p2 - p0;
			Eigen::Vector3d normal = v1.cross(v2);

			if (normal.norm() == 0) continue;

			normal.normalize();

			//
			int inliers = 0;
			for (int k = 0; k < pcc.rows(); ++k) {
				Eigen::Vector3d pointk = pcc.row(k);
				if (std::abs(normal.dot(pointk)) / normal.norm() < 0.05)
				{
					++inliers;
				}
			}

			//
			if (inliers > maxInliers) {
				maxInliers = inliers;
				bestNormal = normal;
			}
		}
	}

	return true;
}


bool NormalRefine::checkNormalDirection(Eigen::MatrixX3d normal_origin, Eigen::MatrixX3d normals, Eigen::MatrixX3d & new_normals)
{
	//
#pragma omp parallel for schedule(dynamic, 40)
	for (int i = 0; i < normal_origin.rows(); ++i) 
	{
		Eigen::Vector3d normal = normals.row(i);
		Eigen::Vector3d origin_normal = normal_origin.row(i);

		//
		double cosAngle = normal.dot(origin_normal) / normal.norm() * origin_normal.norm();

		//9 (cosAngle < 0)
		if (cosAngle < 0) {
			new_normals.row(i) = -normal;
		}
		else
		{
			new_normals.row(i) = normal;
		}
	}
	return true;
}