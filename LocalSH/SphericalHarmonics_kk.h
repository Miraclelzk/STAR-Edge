#pragma once
//#define _USE_MATH_DEFINES


#include <complex>
#include <vector>
#include <math.h>
class SphericalHarmonics_kk
{
public:
	SphericalHarmonics_kk();
	~SphericalHarmonics_kk();

public:

	int GetCoefficientCount(int order);

	double DoubleFactorial(int x);

	double Factorial(int x);

	//
	// 1. l >= 0
	// 2. 0 <= m <= l
	// 3. -1 <= x <= 1
	double EvalLegendrePolynomial(int l, int m, double x);

	std::complex<double> EvalSHcomplex(int l, int m, double phi, double theta);
	double EvalSH(int l, int m, double phi, double theta);

	
	void SHT_f_complex(double *rdata, double *idata,double* weight,int lat,int lon,
		double *rcoeffs, double *icoeffs, int bw);
	void SHT_f(double *rdata, double *rcoeffs, int bw);

	void SHT_inv_complex(double *rcoeffs, double *icoeffs,int bw,
		double *rdata, double *idata, int lat, int lon);
	void SHT_inv(double *rcoeffs,double *rdata, int bw);
	
	
	void makeweight(int bw,int lat,int lon, double *EarthWeight);

	
	int GetIndex(int l, int m) 
	{
		return l * (l + 1) + m;
	}
};

