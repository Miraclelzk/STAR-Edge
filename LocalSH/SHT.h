#pragma once

#include <iostream>
#include <Eigen/Dense>
#include <vector>


class SHT
{
public:

	//s2ki
	void s2_semi_memo(int _bw, double* data, std::vector<std::complex<double>>& coeffs);

	void s2_semi_memo_inv(int _bw, std::vector<std::complex<double>>* coeffs, double* data);

	
	void s2_Earth_forward(int _lat, int _lon, double* data,
		int _bw, std::vector<std::complex<double>>* coeffs);
	void s2_Earth_Inverse(int _bw, std::vector<std::complex<double>>* coeffs,
		int _lat, int _lon, double* data);

    //util
    void ToSphericalCoords(Eigen::Vector3d P, double* phi, double* theta);

    double computeGeodesicLines(double theta1, double phi1, double theta2, double phi2);

    int getIndexOfSH(int m, int l, int bw);
};