#include "SphericalHarmonics_kk.h"

SphericalHarmonics_kk::SphericalHarmonics_kk()
{
}

SphericalHarmonics_kk::~SphericalHarmonics_kk()
{
}

int SphericalHarmonics_kk::GetCoefficientCount(int order) 
{
	return (order + 1) * (order + 1);
}

double SphericalHarmonics_kk::DoubleFactorial(int x) {

	double s = 1.0;
	double n = x;
	while (n > 1.0) {
		s *= n;
		n -= 2.0;
	}
	return s;
}

double SphericalHarmonics_kk::Factorial(int x) {

	double s = 1.0;
	for (int n = 2; n <= x; n++) {
		s *= n;
	}
	return s;
}

// Evaluate the associated Legendre polynomial of degree @l and order @m at
// coordinate @x. The inputs must satisfy:

// 1. l >= 0
// 2. 0 <= m <= l
// 3. -1 <= x <= 1
// See http://en.wikipedia.org/wiki/Associated_Legendre_polynomials
//
// This implementation is based off the approach described in [1],
// instead of computing Pml(x) directly, Pmm(x) is computed. Pmm can be
// lifted to Pmm+1 recursively until Pml is found
double SphericalHarmonics_kk::EvalLegendrePolynomial(int l, int m, double x) {

	// Compute Pmm(x) = (-1)^m(2m - 1)!!(1 - x^2)^(m/2), 
	//where !! is the double factorial.
	double pmm = 1.0;
	// P00 is defined as 1.0, do don't evaluate Pmm unless we know m > 0
	if (m > 0) {

		double sign = (m % 2 == 0 ? 1 : -1);
		pmm = sign * DoubleFactorial(2 * m - 1) * pow(1 - x * x, m / 2.0);
	}


	if (l == m) {
		// Pml is the same as Pmm so there's no lifting to higher bands needed
		//PmPm
		return pmm;
	}

	// Compute P^m_m+1(x) = x(2m + 1)Pmm(x)
	double pmm1 = x * (2 * m + 1) * pmm;
	if (l == m + 1) {
		// Pml is the same as Pmm+1 so we are done as well
		return pmm1;
	}

	// Use the last two computed bands to lift up to the next band until l is
	// reached, using the recurrence relationship:
	//l
	// Pml(x) = (x(2l - 1)Pml-1 - (l + m - 1)Pml-2) / (l - m)
	for (int n = m + 2; n <= l; n++) 
	{
		double pmn = (x * (2 * n - 1) * pmm1 - (n + m - 1) * pmm) / (n - m);
		pmm = pmm1;
		pmm1 = pmn;
	}
	// Pmm1 at the end of the above loop is equal to Pml
	return pmm1;
}


std::complex<double> SphericalHarmonics_kk::EvalSHcomplex(int l, int m, double phi, double theta)
{

	
	//sqrt((2l+1)/(4*PI)*(l-m)!/(l+m)!)
	double kml = sqrt((2.0 * l + 1) * Factorial(l - abs(m))/
		(4.0 * M_PI * Factorial(l + abs(m))));
	double kml0 = sqrt((2.0 * l + 1) / (4.0 * M_PI));

	double real;
	double img;
	
	if (m > 0) 
	{

		
		double leg = EvalLegendrePolynomial(l, m, cos(theta));
		real =  kml * cos(m * phi) *leg;
		img =  kml * sin(m * phi) *leg;

	}
	else if (m < 0) 
	{
		
		
		double sign = (m % 2 == 0 ? 1 : -1);
		double leg = EvalLegendrePolynomial(l, -m, cos(theta));
		real = sign *kml * cos(m * phi) *leg;
		img = sign*kml * sin(m * phi) *leg;

	}
	else 
	{
		
		double leg = EvalLegendrePolynomial(l, 0, cos(theta));
		real = kml0 * cos(0) *leg;
		img = kml0 * sin(0) *leg;
	}
	std::complex<double> result(real,img);
	return result;
}

double SphericalHarmonics_kk::EvalSH(int l, int m, double phi, double theta) {

	
	double kml = sqrt((2.0 * l + 1) * Factorial(l - abs(m)) /
		(4.0 * M_PI * Factorial(l + abs(m))));
	double kml0 = sqrt((2.0 * l + 1) / (4.0 * M_PI));
	if (m > 0) 
	{
		return pow(-1,m)*sqrt(2.0) * kml * cos(m * phi) *
			EvalLegendrePolynomial(l, m, cos(theta));
	}
	else if (m < 0) 
	{
		return pow(-1, m)*sqrt(2.0) * kml * sin(-m * phi) *
			EvalLegendrePolynomial(l, -m, cos(theta));
	}
	else {
		return kml0 * EvalLegendrePolynomial(l, 0, cos(theta));
	}
}


void SphericalHarmonics_kk::SHT_f(double *rdata,double *rcoeffs, int bw)
{

	for (int i = 0; i < bw*bw;i++)
	{
		rcoeffs[i] = 0.0;
	}


	int size = 2 * bw;
	int n = size*size;
	double golden = (sqrt(5) - 1.0) / 2.0;
	
	for (int i = 0; i < n; i++)
	{
		double d = (1.0 - (double)n) / 2.0 + (double)i;
		//0-PI
		double theta = asin(2.0 * d / (double)n) + M_PI / 2.0;
		//0-2PI
		double phi = 2.0 * M_PI*(d*golden - floor(d*golden));

		
		for (int l = 0; l < bw; l++)
		{
			for (int m = -l; m <= l; m++)
			{
				double sh = EvalSH(l, m, phi, theta);
				rcoeffs[GetIndex(l, m)] += rdata[i] * sh;
			}
		}
	}
	
	

	double weight = 4.0 * M_PI / (double)n;
	for (unsigned int i = 0; i < bw*bw; i++) 
	{
		rcoeffs[i] *= weight;
	}

}


void SphericalHarmonics_kk::SHT_inv(double *rcoeffs, double *rdata, int bw)
{

	
	int size = 2 * bw;
	int n = size*size;
	double golden = (sqrt(5) - 1.0) / 2.0;
	
	for (int i = 0; i < n; i++)
	{
		double d = (1.0 - (double)n) / 2.0 + (double)i;
		//0-PI
		double theta = asin(2.0 * d / (double)n) + M_PI / 2.0;
		//0-2PI
		double phi = 2.0 * M_PI*(d*golden - floor(d*golden));


		rdata[i] = 0.0;
		for (int l = 0; l < bw; l++) {
			for (int m = -l; m <= l; m++) {
				rdata[i] += EvalSH(l, m, phi, theta) * rcoeffs[GetIndex(l, m)];
			}
		}

	}
}
void SphericalHarmonics_kk::SHT_f_complex(double *rdata, double *idata, double* weight, int lat, int lon, double *rcoeffs, double *icoeffs, int bw)
{

	for (int i = 0; i < bw*bw; i++)
	{
		rcoeffs[i] = 0.0;
		icoeffs[i] = 0.0;
	}

	

	//int _size = 2 * bw;

	double thetaStep = M_PI / static_cast<double>(lat);
	double phiStep = 2 * M_PI / static_cast<double>(lon);

	double thetatemp;
	double phitemp;
	//theta
	for (int i = 0; i < lat; i++)
	{
		//phi
		for (int j = 0; j < lon; j++)
		{
			thetatemp = i*thetaStep + thetaStep / 2;
			phitemp = j*phiStep + phiStep / 2;

			for (int l = 0; l < bw; l++)
			{
				for (int m = -l; m <= l; m++)
				{

					std::complex<double> sh = EvalSHcomplex(l, m, phitemp, thetatemp);
					rcoeffs[GetIndex(l, m)] += (rdata[i*lon + j] * sh.real() - idata[i*lon + j] * sh.imag())*weight[i];
					icoeffs[GetIndex(l, m)] += (idata[i*lon + j] * sh.real() + rdata[i*lon + j] * sh.imag())*weight[i];
				}
			}

		}
	}


}

void SphericalHarmonics_kk::SHT_inv_complex(double *rcoeffs, double *icoeffs, int bw,
	double *rdata, double *idata, int lat, int lon)
{

	
	//int _size = 2 * bw;

	double thetaStep = M_PI / static_cast<double>(lat);
	double phiStep = 2 * M_PI / static_cast<double>(lon);

	double thetatemp;
	double phitemp;
	//theta
	for (int i = 0; i < lat; i++)
	{
		//phi
		for (int j = 0; j < lon; j++)
		{
			thetatemp = i*thetaStep + thetaStep / 2;
			phitemp = j*phiStep + phiStep / 2;

			rdata[i*lon + j] = 0.0;
			idata[i*lon + j] = 0.0;
			for (int l = 0; l < bw; l++)
			{
				for (int m = -l; m <= l; m++)
				{
					std::complex<double> sh = EvalSHcomplex(l, m, phitemp, thetatemp);
					rdata[i*lon + j] = rdata[i*lon + j] + (sh.real() * rcoeffs[GetIndex(l, m)] - (-sh.imag()) * icoeffs[GetIndex(l, m)]);
					idata[i*lon + j] = idata[i*lon + j] + (sh.real() * icoeffs[GetIndex(l, m)] + (-sh.imag()) * rcoeffs[GetIndex(l, m)]);
				}
			}
		}
	}

}

void SphericalHarmonics_kk::makeweight(int bw, int lat, int lon, double *EarthWeight)
{
	

	//int bw2 = bw * 2;

	int j, k;
	double fudge;
	double tmpsum;

	//theta =(2*j+1)*fudge
	fudge = M_PI / ((double)(2 * lat));

	for (j = 0; j < lat; j++)
	{
		tmpsum = 0.0;
		
		for (k = 0; k < bw; k++)
			tmpsum += 1. / ((double)(2 * k + 1)) * sin((double)((2 * k + 1))*(2 * j + 1)*fudge);
		tmpsum *= sin((double)(2 * j + 1)*fudge);
		
		tmpsum *= 2.*M_PI*4. / ((double)lat)/ ((double)lon);

		EarthWeight[j] = tmpsum;
	}

}

