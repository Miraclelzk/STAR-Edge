#include <nanoflann.hpp>

#include "LocalSHFeature.h"
#include "SHT.h"
#include <FittingLineSample.h>
#include <iomanip>
#include <fstream>
#include "NormalRefine.h"
#include <omp.h>

std::mutex mtx;

bool LocalSHFeature::LocalSH(Eigen::MatrixX3d& pcc, int _bw, std::vector<double> &Descriptor)
{
	
	int lat = _bw * 2;
	int lon = _bw * 2;
	int n = lat * lon;


	std::vector<std::vector<std::vector<int>>> m_index(lat, std::vector<std::vector<int>>(lon));

	double _kernel = lat;

	double h = M_PI / _kernel;

	double radius = 3.0 * h * 1.0;
	
	double thetaStep = M_PI / static_cast<double>(lat);
	double phiStep = 2.0 * M_PI / static_cast<double>(lon);

	//theta
	for (int ii = 0; ii < lat; ii++)
	{
		//phi
		for (int jj = 0; jj < lon; jj++)
		{
			
			double theta = ii * thetaStep + thetaStep / 2;
			double phi = jj * phiStep + phiStep / 2;
            Eigen::Vector3d P1(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));

            //
			for (unsigned k = 0; k < pcc.rows(); k++)
			{
				Eigen::Vector3d P2 = pcc.row(k);

				double temp = (P2 - P1).norm();

				if (temp> radius)
				{
					continue;
				}
				
				m_index[ii][jj].push_back(k);
			}
		}
	}

	double thetatemp;
	double phitemp;

	std::vector<double> data(n);

    SHT sht_tool;

	//theta
	for (int ii = 0; ii < lat; ii++)
	{
		//phi
		for (int jj = 0; jj < lon; jj++)
		{

			
			double theta = ii * thetaStep + thetaStep / 2;
			double phi = jj * phiStep + phiStep / 2;
			
			double ddd = 0;
			double ddd_max = 0;

			for (int k = 0; k < m_index[ii][jj].size(); k++)
			{
				
				Eigen::Vector3d norm = pcc.row(m_index[ii][jj][k]);

				
				double rtemp = norm.norm();
				if (rtemp < 0.0000000001)
				{
					continue;
				}
				sht_tool.ToSphericalCoords(norm, &thetatemp, &phitemp);

				double GeodesicLines = sht_tool.computeGeodesicLines(thetatemp, phitemp, theta, phi);


				double x = GeodesicLines / h;
				
				double K = 1.0 / sqrt(2.0*M_PI)*exp(-0.5*x*x);
				
				double D = K / h / pcc.rows();

				
				ddd = ddd + D;
				if (ddd_max < D)
				{
					ddd_max = D;
				}
			}

			//
			//data[ii*lon + jj] = ddd;
			//
			data[ii*lon + jj] = ddd_max;
			//
			//data[ii*lon + jj] = m_index[ii][jj].size();
		}
	}

	std::vector<std::complex<double>> coeffs;
	{
		std::lock_guard<std::mutex> lock(mtx);
		sht_tool.s2_semi_memo(_bw, data.data(), coeffs);
		//s2_Earth_forward(lat, lon, data, _bw, coeffs);
	}

	for (int l = 0; l < _bw; l++)
	{
		double D = 0.0;
		for (int m = -l; m <= l; m++)
		{
			int dummy;
			dummy = sht_tool.getIndexOfSH(m, l, _bw);

			std::complex<double> temp = coeffs.at(dummy);
			D = D + temp.real()*temp.real() + temp.imag()*temp.imag();
		}
		Descriptor[l] = sqrt(D);
	}

	return true;
}

bool LocalSHFeature::ComLocalSHFeatures(Eigen::MatrixX3d& pc, Eigen::MatrixX3d& pos, int _bw, float radius, Eigen::MatrixXd& Descs)
{

    using kd_tree = nanoflann::KDTreeEigenMatrixAdaptor< Eigen::MatrixX3d >;

    //build de kd_tree
    kd_tree tree(3, pc, 10 /* max leaf */ );
	
    tree.index_->buildIndex();

    Descs.setZero(); //

	int progress = 0;
  

	//int omp_cnt = 0;
	//omp_set_num_threads(24);
//#pragma omp parallel for schedule(dynamic, 20)
    
    for (int j = 0; j < pos.rows(); ++j)
    {
		//omp_cnt++;
        
        const Eigen::Vector3d& pt_query = pos.row(j);
        const float search_radius = radius * radius;

        std::vector<nanoflann::ResultItem<Eigen::Index, double>> ret_matches;

        size_t spCount = tree.index_->radiusSearch(&pt_query[0], search_radius , ret_matches);
        //ret_matches[i].first
        //ret_matches[i].second dist
			

        
        if (spCount < 6)
            continue;
        
        
        Eigen::MatrixX3d pcc;
        pcc.resize(pc.rows(), 3);

        int currentRow = 0; //

        Eigen::Vector3d P1 = pos.row(j);
        //
        for (int jj = 0; jj < spCount; jj++)
        {
            
            size_t pointindex = ret_matches[jj].first;
			double dist = std::sqrt(ret_matches[jj].second);

            if (dist < radius/3.0)
                continue;
            
            Eigen::Vector3d P2 = pc.row(pointindex);

            Eigen::Vector3d norm = (P2 - P1) / (P2 - P1).norm();

            pcc.row(currentRow) = norm;
            currentRow++;
        }

        //
        pcc.conservativeResize(currentRow, Eigen::NoChange);

        //
        std::vector<double> m_Descriptor;
		if (!LocalSH(pcc, _bw, m_Descriptor))
		{
			//return false;
			int a = 1;
		}
        //Descs
        for(int k = 0; k < m_Descriptor.size(); k++)
        {
            Descs(j, k) = m_Descriptor[k];
        }

		if (progressCallback)
		{
			progressCallback(++progress);
		}
    }

    return true;
}


bool LocalSHFeature::ComLocalSHFeatures_standard(Eigen::MatrixX3d pcc, int _bw, Eigen::VectorXd& oneDescs)
{
	//
	std::vector<double> m_Descriptor;
	if (!LocalSH(pcc, _bw, m_Descriptor))
	{
		//return false;
		int a = 1;
	}
	//Descs
	for (int k = 0; k < m_Descriptor.size(); k++)
	{
		oneDescs(k) = m_Descriptor[k];
	}
	return true;
}

bool LocalSHFeature::ComLocalSHFeatures_knn_upsample(Eigen::MatrixX3d& pc, Eigen::MatrixX3d& pos, int _bw, int k, int numSamples, Eigen::MatrixXd& Descs, Eigen::MatrixX3d& normals, std::vector<std::vector<int>> & neighboor)
{
	using kd_tree = nanoflann::KDTreeEigenMatrixAdaptor< Eigen::MatrixX3d >;

	//build de kd_tree
	kd_tree tree(3, pc, 10 /* max leaf */);

	tree.index_->buildIndex();

	Descs.setZero(); //
	normals.setZero();
	int progress = 0;

	// int maxThreads = omp_get_max_threads();
    // std::cout << "Maximum number of threads: " << maxThreads << std::endl;

    // //
    // omp_set_num_threads(maxThreads);


	std::cout << "LSHF-start" << std::endl;
	
	
	#pragma omp parallel for schedule(dynamic, 40)
	for (size_t j = 0; j < pos.rows(); ++j)
	{
		// std::cout << "Iteration " << j << " is being processed by thread " << omp_get_thread_num() << std::endl;
		
		const Eigen::Vector3d& pt_query = pos.row(j);
		
		std::vector<Eigen::MatrixX3d::Index> pointIdxSearch(k);
		std::vector<double> pointSquaredDistance(k);


		size_t spCount = tree.index_->knnSearch(&pt_query[0], k, &pointIdxSearch[0], &pointSquaredDistance[0]);
		//ret_matches[i].first
		//ret_matches[i].second dist

		
		if (spCount < 6)
			continue;

		
		Eigen::MatrixX3d pcc;
		pcc.resize(pc.rows(), 3);

		int currentRow = 0; //

		Eigen::Vector3d P1 = (pos.row(j));
		//
		for (int jj = 0; jj < spCount; jj++)
		{
			
			size_t pointindex = pointIdxSearch[jj];
			double dist = std::sqrt(pointSquaredDistance[jj]);
			neighboor[j].push_back(pointIdxSearch[jj]);
			if (dist == 0)
			{
				continue;
			}

			Eigen::Vector3d P2 = pc.row(pointindex);

			Eigen::Vector3d norm = (P2 - P1) / (P2 - P1).norm();

			pcc.row(currentRow) = norm;
			currentRow++;
		}

		//
		pcc.conservativeResize(currentRow, Eigen::NoChange);

		//-------------------------------------
		
		FittingLineSample FLS;

		
		Eigen::MatrixX3d projectedPcc;
		Eigen::MatrixX2d projected_2d(pcc.rows(), 2);
		FLS.projectPointCloud(pcc, projectedPcc, projected_2d);

		
		std::vector<int> hullIndices;
		FLS.convexHull(projected_2d, hullIndices);

		//pccHul
		Eigen::MatrixX3d pccHull(hullIndices.size() + 1, 3);
		for (int i = 0; i < hullIndices.size(); ++i)
		{
			pccHull.row(i) = pcc.row(hullIndices[i]);
		}
		
		pccHull.row(hullIndices.size()) = pcc.row(hullIndices[0]);

		
		int numSamples = 30; // Example number of samples
		int degree = 1;
		Eigen::MatrixX3d sampleCurve(numSamples, 3);//

		FLS.fitCurve(pccHull, numSamples, degree, sampleCurve);


		Eigen::MatrixX3d sampleCurveQ;
		sampleCurveQ.resize(sampleCurve.rows(), 3);

		//
		for (int jj = 0; jj < sampleCurve.rows(); jj++)
		{		
			Eigen::Vector3d P2 = sampleCurve.row(jj);

			Eigen::Vector3d norm = P2 / P2.norm();

			sampleCurveQ.row(jj) = norm;
		}

		Eigen::Vector3d normal;
		NormalRefine NR;
		NR.ransacFitPlaneThroughOrigin(sampleCurveQ, 5000, normal);
		normals.row(j) = normal;

		//-----------------
		//
		std::vector<double> m_Descriptor(_bw);
		if (!LocalSH(sampleCurveQ, _bw, m_Descriptor))
		{
			continue;
		}
		// //Descs
		for (int k = 0; k < m_Descriptor.size(); k++)
		{
			Descs(j, k) = m_Descriptor[k];
		}

		if (progressCallback)
		{
			progressCallback(++progress);
		}
	}

	std::cout << "LSHF-end" << std::endl;

	return true;

}


bool LocalSHFeature::ComLocalSHFeatures_knn_nosample(Eigen::MatrixX3d& pc, Eigen::MatrixX3d& pos, int _bw, int k, int numSamples, Eigen::MatrixXd& Descs, Eigen::MatrixX3d& normals, std::vector<std::vector<int>> & neighboor)
{

	using kd_tree = nanoflann::KDTreeEigenMatrixAdaptor< Eigen::MatrixX3d >;

	//build de kd_tree
	kd_tree tree(3, pc, 10 /* max leaf */);

	tree.index_->buildIndex();

	Descs.setZero(); //
	normals.setZero();
	int progress = 0;


	
	for (size_t j = 0; j < pos.rows(); ++j)
	{
		
		const Eigen::Vector3d& pt_query = pos.row(j);

		std::vector<Eigen::MatrixX3d::Index> pointIdxSearch(k);
		std::vector<double> pointSquaredDistance(k);


		size_t spCount = tree.index_->knnSearch(&pt_query[0], k, &pointIdxSearch[0], &pointSquaredDistance[0]);
		//ret_matches[i].first
		//ret_matches[i].second dist

		
		if (spCount < 6)
			continue;

		
		Eigen::MatrixX3d pcc;
		pcc.resize(pc.rows(), 3);

		int currentRow = 0; //

		Eigen::Vector3d P1 = (pos.row(j));
		//
		for (int jj = 0; jj < spCount; jj++)
		{
			
			size_t pointindex = pointIdxSearch[jj];
			double dist = std::sqrt(pointSquaredDistance[jj]);

			neighboor[j].push_back(pointIdxSearch[jj]);

			if (dist == 0)
			{
				continue;
			}

			Eigen::Vector3d P2 = pc.row(pointindex);

			Eigen::Vector3d norm = (P2 - P1) / (P2 - P1).norm();

			pcc.row(currentRow) = norm;
			currentRow++;
		}

		//
		pcc.conservativeResize(currentRow, Eigen::NoChange);
		
		//-----------------
		//
		std::vector<double> m_Descriptor;
		if (!LocalSH(pcc, _bw, m_Descriptor))
		{
			return false;
		}
		//Descs
		for (int k = 0; k < m_Descriptor.size(); k++)
		{
			Descs(j, k) = m_Descriptor[k];
		}

		if (progressCallback)
		{
			progressCallback(++progress);
		}
	}

	return true;

}



bool LocalSHFeature::ComOneLocalSHFeatures(Eigen::MatrixX3d& pc, Eigen::Vector3d pos, int _bw, float radius, Eigen::MatrixX3d& pcc, double*& data, std::vector<std::complex<double>>& coeffs)
{
	using kd_tree = nanoflann::KDTreeEigenMatrixAdaptor< Eigen::MatrixX3d >;

	//build de kd_tree
	kd_tree tree(3, pc, 10 /* max leaf */);

	tree.index_->buildIndex();

	int progress = 0;

	
	const Eigen::Vector3d& pt_query = pos;
	const float search_radius = radius * radius;

	std::vector<nanoflann::ResultItem<Eigen::Index, double>> ret_matches;

	size_t spCount = tree.index_->radiusSearch(&pt_query[0], search_radius, ret_matches);
	//ret_matches[i].first
	//ret_matches[i].second dist

	if (spCount == 0)
		return false;

	
	//Eigen::MatrixX3d pcc;
	pcc.resize(pc.rows(), 3);

	int currentRow = 0; //

	Eigen::Vector3d P1 = pos;
	//
	for (int jj = 0; jj < spCount; jj++)
	{
		
		size_t pointindex = ret_matches[jj].first;
		double dist = std::sqrt(ret_matches[jj].second);

		if (dist < radius / 3.0)
			continue;

		Eigen::Vector3d P2 = pc.row(pointindex);

		Eigen::Vector3d norm = (P2 - P1) / (P2 - P1).norm();

		pcc.row(currentRow) = norm;
		currentRow++;
	}
	if (currentRow == 0)
		return false;

	//
	pcc.conservativeResize(currentRow, Eigen::NoChange);

	
	int lat = _bw * 2;
	int lon = _bw * 2;
	int n = lat * lon;

	
	std::vector<std::vector<std::vector<int>>> m_index(lat, std::vector<std::vector<int>>(lon));

	double _kernel = lat;
	
	double h = M_PI / _kernel * 2;

	double R = 2 * h * 1.0;

	double thetaStep = M_PI / static_cast<double>(lat);
	double phiStep = 2.0 * M_PI / static_cast<double>(lon);

	//theta
	for (int ii = 0; ii < lat; ii++)
	{
		//phi
		for (int jj = 0; jj < lon; jj++)
		{
			
			double theta = ii * thetaStep + thetaStep / 2;
			double phi = jj * phiStep + phiStep / 2;
			Eigen::Vector3d P1(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));

			//
			for (unsigned k = 0; k < pcc.rows(); k++)
			{
				Eigen::Vector3d P2 = pcc.row(k);

				double temp = (P2 - P1).norm();

				if (temp > R)
				{
					continue;
				}
				
				m_index[ii][jj].push_back(k);
			}
		}
	}

	double thetatemp;
	double phitemp;

	data = new double[n];

	SHT sht_tool;

	

	//theta
	for (int ii = 0; ii < lat; ii++)
	{
		//phi
		for (int jj = 0; jj < lon; jj++)
		{

			
			double theta = ii * thetaStep + thetaStep / 2;
			double phi = jj * phiStep + phiStep / 2;
			
			double ddd = 0;
			double ddd_max = 0;

			for (int k = 0; k < m_index[ii][jj].size(); k++)
			{
				//theta phi
				Eigen::Vector3d norm = pcc.row(m_index[ii][jj][k]);

				
				double rtemp = norm.norm();
				if (rtemp < 0.0000000001)
				{
					continue;
				}
				sht_tool.ToSphericalCoords(norm, &thetatemp, &phitemp);

				double GeodesicLines = sht_tool.computeGeodesicLines(thetatemp, phitemp, theta, phi);

				double x = GeodesicLines / h;

				double K = 0;
				
				K = 1.0 / sqrt(2.0*M_PI)*exp(-0.5*x*x);
				//Epanechniko
				//if (std::abs(x) <= 1)
				//	K = 3.0 / 4.0 * (1 - x * x);

				double D = K / h / pcc.rows();

				
				ddd = ddd + D;
				if (ddd_max < D )
				{
					ddd_max = D;
				}
			}
			//
			//data[ii*lon + jj] = ddd;
			//
			data[ii*lon + jj] = ddd_max;
			//
			//data[ii*lon + jj] = m_index[ii][jj].size();
		}
	}


	sht_tool.s2_semi_memo(_bw, data, coeffs);
	//s2_Earth_forward(lat, lon, data, _bw, coeffs);

	return true;
}

bool LocalSHFeature::ComOneLocalSHFeatures_knn(Eigen::MatrixX3d& pc, Eigen::Vector3d pos, int _bw, int k, Eigen::MatrixX3d& pcc, double*& data, std::vector<std::complex<double>>& coeffs)
{
	using kd_tree = nanoflann::KDTreeEigenMatrixAdaptor< Eigen::MatrixX3d >;

	//build de kd_tree
	kd_tree tree(3, pc, 10 /* max leaf */);

	tree.index_->buildIndex();

	int progress = 0;

	//knn
	const Eigen::Vector3d& pt_query = pos;
	std::vector<Eigen::MatrixX3d::Index> pointIdxSearch(k);
	std::vector<double> pointSquaredDistance(k);


	size_t spCount = tree.index_->knnSearch(&pt_query[0], k, &pointIdxSearch[0], &pointSquaredDistance[0]);
	//ret_matches[i].first
	//ret_matches[i].second dist

	if (spCount == 0)
		return false;

	
	//Eigen::MatrixX3d pcc;
	pcc.resize(pc.rows(), 3);

	int currentRow = 0; //

	Eigen::Vector3d P1 = pos;
	//
	for (int jj = 0; jj < spCount; jj++)
	{
		
		size_t pointindex = pointIdxSearch[jj];
		double dist = std::sqrt(pointSquaredDistance[jj]);
		if (dist == 0)
		{
			continue;
		}

		Eigen::Vector3d P2 = pc.row(pointindex);

		Eigen::Vector3d norm = (P2 - P1) / (P2 - P1).norm();

		pcc.row(currentRow) = norm;
		currentRow++;
	}
	if (currentRow == 0)
		return false;

	//
	pcc.conservativeResize(currentRow, Eigen::NoChange);

	int lat = _bw * 2;
	int lon = _bw * 2;
	int n = lat * lon;


	std::vector<std::vector<std::vector<int>>> m_index(lat, std::vector<std::vector<int>>(lon));

	double _kernel = lat;

	double h = M_PI / _kernel * 2;

	double R = 2 * h * 1.0;

	double thetaStep = M_PI / static_cast<double>(lat);
	double phiStep = 2.0 * M_PI / static_cast<double>(lon);

	//theta
	for (int ii = 0; ii < lat; ii++)
	{
		//phi
		for (int jj = 0; jj < lon; jj++)
		{
			
			double theta = ii * thetaStep + thetaStep / 2;
			double phi = jj * phiStep + phiStep / 2;
			Eigen::Vector3d P1(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));

			//
			for (unsigned k = 0; k < pcc.rows(); k++)
			{
				Eigen::Vector3d P2 = pcc.row(k);

				double temp = (P2 - P1).norm();

				if (temp > R)
				{
					continue;
				}
				//pc
				m_index[ii][jj].push_back(k);
			}
		}
	}

	double thetatemp;
	double phitemp;

	data = new double[n];

	SHT sht_tool;



	//theta
	for (int ii = 0; ii < lat; ii++)
	{
		//phi
		for (int jj = 0; jj < lon; jj++)
		{

			
			double theta = ii * thetaStep + thetaStep / 2;
			double phi = jj * phiStep + phiStep / 2;
			
			double ddd = 0;
			double ddd_max = 0;

			for (int k = 0; k < m_index[ii][jj].size(); k++)
			{
				//theta phi
				Eigen::Vector3d norm = pcc.row(m_index[ii][jj][k]);

				
				double rtemp = norm.norm();
				if (rtemp < 0.0000000001)
				{
					continue;
				}
				sht_tool.ToSphericalCoords(norm, &thetatemp, &phitemp);

				double GeodesicLines = sht_tool.computeGeodesicLines(thetatemp, phitemp, theta, phi);

				double x = GeodesicLines / h;

				double K = 0;
				
				K = 1.0 / sqrt(2.0*M_PI)*exp(-0.5*x*x);
				//Epanechniko
				//if (std::abs(x) <= 1)
				//	K = 3.0 / 4.0 * (1 - x * x);

				double D = K / h / pcc.rows();

				
				ddd = ddd + D;
				if (ddd_max < D)
				{
					ddd_max = D;
				}
			}
			//
			//data[ii*lon + jj] = ddd;
			//
			data[ii*lon + jj] = ddd_max;
			//
			//data[ii*lon + jj] = m_index[ii][jj].size();
		}
	}

	sht_tool.s2_semi_memo(_bw, data, coeffs);
	//s2_Earth_forward(lat, lon, data, _bw, coeffs);

	return true;
}

void LocalSHFeature::ComCoeffsToDesc(std::vector<std::complex<double>>*& coeffs, std::vector<double> &Descriptor)
{
	size_t _bw = std::sqrt(coeffs->size());

	SHT sht_tool;
	for (int l = 0; l < _bw; l++)
	{
		double D = 0.0;
		for (int m = -l; m <= l; m++)
		{
			int dummy;
			dummy = sht_tool.getIndexOfSH(m, l, _bw);

			std::complex<double> temp = coeffs->at(dummy);
			D = D + temp.real() * temp.real() + temp.imag() * temp.imag();
		}
		Descriptor.push_back(sqrt(D));
	}
}



