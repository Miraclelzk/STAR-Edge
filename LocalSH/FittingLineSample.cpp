#include <nanoflann.hpp>
#include "FittingLineSample.h"
#include <unsupported/Eigen/Splines>

bool FittingLineSample::projectPointCloud(const Eigen::MatrixX3d pcc, Eigen::MatrixX3d& projectedPcc, Eigen::MatrixX2d& projected_2d)
{
	//）
	Eigen::Vector3d centroid = pcc.colwise().mean();

	//
	Eigen::MatrixX3d centered = pcc.rowwise() - centroid.transpose();

	//
	Eigen::Matrix3d covariance = (centered.adjoint() * centered) / double(pcc.rows() - 1);

	//Eige
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigSolver(covariance, Eigen::ComputeEigenvectors);

	//
	Eigen::Matrix3d eigVectors = eigSolver.eigenvectors();
	Eigen::Vector3d eigValues = eigSolver.eigenvalues();

	//
	Eigen::Vector3d normal = eigVectors.col(0); //

	//
	//Eigen::VectorXd distances = (centered * normal).array();
	Eigen::VectorXd distances = centered * normal;
	//Eigen::MatrixX3d ttt(distances.asDiagonal());

	//Eigen::MatrixX3d xxx = pcc - (ttt * normal.transpose());

	//
	//projectedPcc = pcc - (distances.asDiagonal() * normal.transpose());
	//broadcastin
	//projectedPcc = pcc - (distances.asDiagonal() * normal).transpose();

	projectedPcc = pcc;
	//
	for (int i = 0; i < pcc.rows(); ++i) {
		projectedPcc.row(i) = pcc.row(i) - distances(i) * normal.transpose();
	}

	//
	Eigen::Vector3d second_axis = eigVectors.col(1); //
	Eigen::Vector3d third_axis = eigVectors.col(2);  //

	//
	
	for (int i = 0; i < projectedPcc.rows(); ++i) {
		Eigen::Vector3d point = projectedPcc.row(i);
		double x = point.dot(second_axis);
		double y = point.dot(third_axis);
		projected_2d.row(i) << x, y;
	}
	//std::vector<int> hullIndices;
	//convexHull(projected_2d, hullIndices);

	////pccHul
	//Eigen::MatrixX3d pccHull(hullIndices.size(), 3);
	//for (int i = 0; i < hullIndices.size(); ++i)
	//{
	//	pccHull.row(i) = pcc.row(hullIndices[i]);
	//}
	//
	//
	//
	//
	//int numSamples = 1000; // Example number of samples
	//Eigen::MatrixX3d sampleCurve(numSamples,3);

	//fitCurve(pccHull, numSamples,sampleCurve);

	return true;
}


// Graha
bool FittingLineSample::convexHull(const Eigen::MatrixX2d projected_2d, std::vector<int> & hullIndices)
{
	std::vector<Eigen::Vector2d> pointsVec(projected_2d.rows());
	for (int i = 0; i < projected_2d.rows(); ++i) {
		pointsVec[i] = projected_2d.row(i);
	}


	// Create an index array and initialize it
	std::vector<int> indices(pointsVec.size());
	for (int i = 0; i < indices.size(); ++i) {
		indices[i] = i;
	}


	//
	int minY = pointsVec[0].y(), minIndex = 0;
	for (int i = 1; i < pointsVec.size(); ++i) {
		if (pointsVec[i].y() < minY || (pointsVec[i].y() == minY && pointsVec[i].x() < pointsVec[minIndex].x())) 
		{
			minY = pointsVec[i].y();
			minIndex = i;
		}
	}
	std::swap(pointsVec[0], pointsVec[minIndex]);
	
	Eigen::Vector2d p0 = pointsVec[0];
	//std::swap(indices[0], indices[minIndex]);
	////
	//std::sort(pointsVec.begin() + 1, pointsVec.end(), [&p0](const Eigen::Vector2d& p1, const Eigen::Vector2d& p2) 
	//{
	//	double orientation = (p1.y() - p0.y()) * (p2.x() - p1.x()) - (p1.x() - p0.x()) * (p2.y() - p1.y());
	//	if (orientation == 0)
	//		return (p1.x()*p1.x() + p1.y()*p1.y()) < (p2.x()*p2.x() + p2.y()*p2.y()); //
	//	return orientation < 0;
	//});

	std::sort(indices.begin() + 1, indices.end(), [&pointsVec, &p0](int i, int j) 
	{
		const Eigen::Vector2d& p1 = pointsVec[i];
		const Eigen::Vector2d& p2 = pointsVec[j];
		/*double angle1 = atan2(p1.y() - p0.y(), p1.x() - p0.x());
		double angle2 = atan2(p2.y() - p0.y(), p2.x() - p0.x());
		return angle1 < angle2;*/
		double orientation = (p1.y() - p0.y()) * (p2.x() - p1.x()) - (p1.x() - p0.x()) * (p2.y() - p1.y());
		if (orientation == 0)
			return (p1.x()*p1.x() + p1.y()*p1.y()) < (p2.x()*p2.x() + p2.y()*p2.y()); //
		return orientation < 0;
	});

	// Find the positions of 0 and minValue
	int zeroIndex = -1;
	int minValueIndex = -1;

	for (int i = 0; i < indices.size(); ++i) {
		if (indices[i] == 0 && zeroIndex == -1) {
			zeroIndex = i;
		}
		else if (indices[i] == minIndex && minValueIndex == -1) {
			minValueIndex = i;
		}
	}

	// Swap the positions of 0 and minValue if both are found
	if (zeroIndex != -1 && minValueIndex != -1) {
		std::swap(indices[zeroIndex], indices[minValueIndex]);
	}

	std::vector<Eigen::Vector2d> pointssort1 (projected_2d.rows());
	for (int i = 0; i < pointsVec.size(); ++i)
	{
		pointssort1[i] = projected_2d.row(indices[i]);
	}

	//
	std::stack<Eigen::Vector2d> hullStack;
	//std::vector<int> hullIndices;
	hullStack.push(pointssort1[0]);
	hullStack.push(pointssort1[1]);
	hullIndices.push_back(indices[0]);
	hullIndices.push_back(indices[1]);
	for (int i = 2; i < pointssort1.size(); ++i) 
	{
		while (hullStack.size() > 1 && ((hullStack.top().y() - pointssort1[i].y()) * (pointssort1[i].x() - hullStack.top().x()) -
			(hullStack.top().x() - pointssort1[i].x()) * (pointssort1[i].y() - hullStack.top().y())) > 0)
		{
			hullStack.pop();
		}
		hullStack.push(pointssort1[i]);
		hullIndices.push_back(indices[i]);
	}

	////
	////std::vector<Eigen::Vector2d> convexHullPoints;
	//while (!hullStack.empty()) {
	//	convexHullPoints.push_back(hullStack.top());
	//	hullStack.pop();
	//}
	//std::reverse(convexHullPoints.begin(), convexHullPoints.end());
	return true;
}


bool FittingLineSample::fitCurve(const Eigen::MatrixX3d pccHull, int numSamples, int degree_input, Eigen::MatrixX3d & sampleCurve)
{
	///
	//typedef Eigen::Spline<double, 3> Spline3d;
	//int degree = 3;  //
	//auto spline = Eigen::SplineFitting<Spline3d>::Interpolate(pccHull.transpose(), degree);

	//
	int max_degree = degree_input;

	//
	typedef Eigen::Spline<double, 3> Spline3d;
	Spline3d spline;

	//（）
	int degree;
	for (degree = max_degree; degree > 0; --degree) {
		if (pccHull.rows() >= degree + 1) {
			spline = Eigen::SplineFitting<Spline3d>::Interpolate(pccHull.transpose(), degree);
			break;
		}
	}

	//
	//int num_samples = 1000; //
	//MatrixXd sampled_points(num_samples, 3); // (x, y, z)

	double t_min = 0.0;
	double t_max = 1.0;
	double t_increment = (t_max - t_min) / (numSamples - 1);

	double t = t_min;
	for (int i = 0; i < numSamples; ++i) {
		Eigen::Vector3d point = spline(t);
		sampleCurve.row(i) = point; //
		t += t_increment;
	}

	return true;
}


