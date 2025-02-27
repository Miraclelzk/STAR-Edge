#include "SHT.h"


// ---- s2kit ----
#include "fftw3.h"
extern "C" {
#include "s2kit/FST_semi_memo.h"
#include "s2kit/cospml.h"
#include "s2kit/weights.h"
#include "s2kit/util.h"
}
// ---- sh-kk ----
#include "SphericalHarmonics_kk.h"


void SHT::s2_semi_memo(int _bw, double* data, std::vector<std::complex<double>>& coeffs)
{
    int _size = 2 * _bw;
    int cutoff = _bw; // seminaive all orders // TODO?
    DataFormat data_format = COMPLEX; 

    double* workspace = (double*)malloc(sizeof(double) * ((8 * (_bw * _bw)) + (7 * _bw)));

	int te = Reduced_Naive_TableSize(_bw, cutoff) + Reduced_SpharmonicTableSize(_bw, cutoff);
    double* seminaive_naive_tablespace = (double*)malloc(
        sizeof(double) * (Reduced_Naive_TableSize(_bw, cutoff) + Reduced_SpharmonicTableSize(_bw, cutoff)));
    
    // Generating seminaive_naive tables
    double** seminaive_naive_table = SemiNaive_Naive_Pml_Table(_bw, cutoff, seminaive_naive_tablespace, workspace);

    double* weights = (double*)malloc(sizeof(double) * 4 * _bw);
    double* rdata = (double*)malloc(sizeof(double) * (_size * _size));
    double* idata = (double*)malloc(sizeof(double) * (_size * _size));

    // Make DCT plan. Note that I will be using the GURU interface to execute these plans within the routines
    fftw_plan DCT_plan = fftw_plan_r2r_1d(2 * _bw, weights, rdata, FFTW_REDFT10, FFTW_ESTIMATE);

    // fftw "preamble"
    // Note that FFT plan places the output in a transposed array
    int rank = 1;
    //fftw_iodim dims[rank];
	fftw_iodim dims[1];
    dims[0].n = 2 * _bw;
    dims[0].is = 1;
    dims[0].os = 2 * _bw;

    int howmany_rank = 1;
    //fftw_iodim howmany_dims[howmany_rank];
	fftw_iodim howmany_dims[1];
    howmany_dims[0].n = 2 * _bw;
    howmany_dims[0].is = 2 * _bw;
    howmany_dims[0].os = 1;

    fftw_plan FFT_plan = fftw_plan_guru_split_dft(rank, dims, howmany_rank, howmany_dims, rdata, idata, workspace,
                                                  workspace + (4 * _bw * _bw), FFTW_ESTIMATE);

    //
    GenerateWeightsForDLT(_bw, weights);
    
	for (int i = 0; i < _size*_size; i++)
	{
		rdata[i] = data[i];
		idata[i] = 0.0;
	}

    double* rcoeffs = (double*)malloc(sizeof(double) * (_bw * _bw));
    double* icoeffs = (double*)malloc(sizeof(double) * (_bw * _bw));

    // forward spherical transform
    FSTSemiMemo(rdata, idata, rcoeffs, icoeffs, _bw, seminaive_naive_table, workspace, data_format, cutoff, &DCT_plan,
                &FFT_plan, weights);

    for (int i = 0; i < _bw*_bw; i++)
	{
		std::complex<double> temp(rcoeffs[i], icoeffs[i]);
		coeffs.push_back(temp);
	}

    fftw_destroy_plan(FFT_plan);
    fftw_destroy_plan(DCT_plan);

	free(workspace);
	free(seminaive_naive_table);
	free(seminaive_naive_tablespace);
	free(weights);
	free(icoeffs);
	free(rcoeffs);
	free(idata);
	free(rdata);

}



void SHT::s2_semi_memo_inv(int _bw, std::vector<std::complex<double>>* coeffs, double* data)
{
    int _size = 2 * _bw;
    int cutoff = _bw; // seminaive all orders
    DataFormat data_format = COMPLEX;

    double* workspace = (double*)malloc(sizeof(double) * ((8 * (_bw * _bw)) + (10 * _bw)));
    double* seminaive_naive_tablespace = (double*)malloc(
        sizeof(double) * (Reduced_Naive_TableSize(_bw, cutoff) + Reduced_SpharmonicTableSize(_bw, cutoff)));
    double* trans_seminaive_naive_tablespace = (double*)malloc(
        sizeof(double) * (Reduced_Naive_TableSize(_bw, cutoff) + Reduced_SpharmonicTableSize(_bw, cutoff)));

    // precompute the Legendres (that's what memo suffix for)
    // Generating seminaive_naive tables
    double** seminaive_naive_table = SemiNaive_Naive_Pml_Table(_bw, cutoff, seminaive_naive_tablespace, workspace);

    // Generating trans_seminaive_naive tables
    double** trans_seminaive_naive_table = Transpose_SemiNaive_Naive_Pml_Table(
        seminaive_naive_table, _bw, cutoff, trans_seminaive_naive_tablespace, workspace);

    double* weights = (double*)malloc(sizeof(double) * 4 * _bw);
    double* rdata = (double*)malloc(sizeof(double) * (_size * _size));
    double* idata = (double*)malloc(sizeof(double) * (_size * _size));

    // Make inverse DCT plan. Note that I will be using the GURU interface to execute these plans within the routines
    fftw_plan inv_DCT_plan = fftw_plan_r2r_1d(2 * _bw, weights, rdata, FFTW_REDFT01, FFTW_ESTIMATE);

    // fftw "preamble"
    // Note that FFT plans assumes that I'm working with a transposed array, e.g. the inputs for a length 2*bw transform
    // are placed every 2*bw apart, the output will be consecutive entries in the array

	int rank = 1;
	//fftw_iodim dims[rank];
	fftw_iodim dims[1];
    dims[0].n = 2 * _bw;
    dims[0].is = 2 * _bw;
    dims[0].os = 1;

	int howmany_rank = 1;
	//fftw_iodim howmany_dims[howmany_rank];
	fftw_iodim howmany_dims[1];
    howmany_dims[0].n = 2 * _bw;
    howmany_dims[0].is = 1;
    howmany_dims[0].os = 2 * _bw;

    fftw_plan inv_FFT_plan = fftw_plan_guru_split_dft(rank, dims, howmany_rank, howmany_dims, rdata, idata, workspace,
                                                      workspace + (4 * _bw * _bw), FFTW_ESTIMATE);

    GenerateWeightsForDLT(_bw, weights);

    double* rcoeffs = (double*)malloc(sizeof(double) * (_bw * _bw));
    double* icoeffs = (double*)malloc(sizeof(double) * (_bw * _bw));

    // read coefficients
	for (int i = 0; i < _bw*_bw; i++)
	{
		/* first the real part of the coefficient */
		rcoeffs[i] = coeffs->at(i).real();
		icoeffs[i] = coeffs->at(i).imag();
	}

    // inverse spherical transform
    InvFSTSemiMemo(rcoeffs, icoeffs, rdata, idata, _bw, trans_seminaive_naive_table, workspace, data_format, cutoff,
                   &inv_DCT_plan, &inv_FFT_plan);

    for (int i = 0; i < _size*_size; i++)
	{
		data[i] = rdata[i];
		//idata[i];
	}

    fftw_destroy_plan(inv_FFT_plan);
    fftw_destroy_plan(inv_DCT_plan);

    free(trans_seminaive_naive_table);
    free(seminaive_naive_table);
    free(trans_seminaive_naive_tablespace);
    free(seminaive_naive_tablespace);
    free(workspace);
    free(weights);
    free(idata);
    free(rdata);
    free(icoeffs);
    free(rcoeffs);

}


void SHT::s2_Earth_forward(int _lat, int _lon, double* data, int _bw, std::vector<std::complex<double>>* coeffs)
{
	int _size = _lat*_lon;

	double *rdata, *idata, *weight;
	double *rcoeffs, *icoeffs;

	rdata = new double[_size];
	idata = new double[_size];
	weight = new double[_size];

	rcoeffs = new double[_bw*_bw];
	icoeffs = new double[_bw*_bw];


	
	for (int i = 0; i < _size; i++)
	{
		rdata[i] = data[i];
		idata[i] = 0.0;
	}

	SphericalHarmonics_kk Workspace;
	
	Workspace.makeweight(_bw, _lat, _lon, weight);
	Workspace.SHT_f_complex(rdata, idata, weight, _lat, _lon, rcoeffs, icoeffs, _bw);


	
	for (int i = 0; i < _bw*_bw; i++)
	{
		std::complex<double> temp(rcoeffs[i], icoeffs[i]);
		coeffs->push_back(temp);
	}

	delete[] rdata;
	delete[] idata;
	delete[] rcoeffs;
	delete[] icoeffs;
}

void SHT::s2_Earth_Inverse(int _bw, std::vector<std::complex<double>>* coeffs, int _lat, int _lon, double* data)
{
	int _size = _lat*_lon;

	double *rcoeffs, *icoeffs, *rdata, *idata;

	
	rcoeffs = new double[_bw*_bw];
	icoeffs = new double[_bw*_bw];
	rdata = new double[_size];
	idata = new double[_size];

	
	for (int i = 0; i < _bw*_bw; i++)
	{
		rcoeffs[i] = coeffs->at(i).real();
		icoeffs[i] = coeffs->at(i).imag();
	}

	SphericalHarmonics_kk Workspace;
	Workspace.SHT_inv_complex(rcoeffs, icoeffs, _bw, rdata, idata, _lat, _lon);

	for (int i = 0; i < _size; i++)
	{
		data[i] = rdata[i];
		//idata[i];
	}

	delete[] rdata;
	delete[] idata;
	delete[] rcoeffs;
	delete[] icoeffs;
}

//xytheta phi
void SHT::ToSphericalCoords(Eigen::Vector3d P, double* theta, double* phi)
{
	double rtemp = P.norm();

	//[0,pi]
	double thetatemp = acos(P(2) / rtemp); 
	//[0,2pi]
	double phitemp;
	//phitemp = atan(P.y / P.x);  / (-pi/2  pi/2)
	if (P(0) > 0)
	{
		if (P(1) >= 0)
		{
			phitemp = atan(P(1) / P(0));
		}
		else
		{
			phitemp = atan(P(1) / P(0)) + 2 * M_PI;
		}
	}
	else if (P(0) == 0)
	{
		if (P(1) > 0)
		{
			phitemp = M_PI / 2;
		}
		else
		{
			phitemp = 3 * M_PI / 2;
		}
	}
	else if (P(0) < 0)
	{
		phitemp = atan(P(1) / P(0)) + M_PI;
	}

	*theta = thetatemp;
	*phi = phitemp;
}

double SHT::computeGeodesicLines(double theta1, double phi1, double theta2, double phi2)
{

    // thet
    double temp= sin(theta1)*sin(theta2)*cos(phi2 - phi1) + cos(theta1)*cos(theta2);
    if (temp > 1)
        return 0;
    else if (temp < -1)
        return M_PI;
    else
        return acos(temp);
}

int SHT::getIndexOfSH(int m, int l, int bw)
{
    return IndexOfHarmonicCoeff(m, l, bw);
}
